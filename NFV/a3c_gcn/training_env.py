"""EO-backed training environment for a node-only A3C-GCN policy.

The database is read once to create a snapshot.  Episodes mutate an in-memory
copy of that snapshot, so training never changes EO's real database.  Link
mapping follows EO's directed path-table semantics: paths are keyed by
``(source, destination)`` and the first hop-sorted path whose every directed
edge has enough residual bandwidth is selected.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


NodeId = int
Edge = Tuple[int, int]


@dataclass
class DatabaseSnapshot:
    nodes: Dict[NodeId, Dict[str, float]]
    edges: Dict[Edge, float]
    paths: Dict[Edge, List[Dict[str, object]]]

    @classmethod
    def from_connection(cls, conn, max_paths_per_pair: int = 5) -> "DatabaseSnapshot":
        """Read an EO snapshot without loading the entire path table.

        EO's ``link_table`` can contain millions of enumerated paths.  Training
        only needs a small hop-sorted candidate set for each directed endpoint
        pair, matching EO's first-feasible-path behavior.
        """
        if max_paths_per_pair <= 0:
            raise ValueError("max_paths_per_pair must be positive")
        cursor = conn.cursor()
        cursor.execute("SELECT node_id, cpu, pos_x, pos_y FROM node_single_table")
        nodes = {
            int(node_id): {
                "cpu": float(cpu),
                "max_cpu": float(cpu),
                "pos": float(pos_x),
                "pos2": float(pos_y),
            }
            for node_id, cpu, pos_x, pos_y in cursor.fetchall()
        }

        cursor.execute(
            "SELECT source_node, destination_node, bw_resource "
            "FROM link_single_table"
        )
        edges = {
            (int(source), int(target)): float(bandwidth)
            for source, target, bandwidth in cursor.fetchall()
        }

        paths: Dict[Edge, List[Dict[str, object]]] = {}
        node_ids = sorted(nodes)
        pair_count = len(node_ids) * max(len(node_ids) - 1, 0)
        completed = 0
        query = (
            "SELECT hop_count, link_path FROM link_table "
            "WHERE source_node=%s AND destination_node=%s "
            "ORDER BY hop_count ASC, link_id ASC LIMIT %s"
        )
        for source in node_ids:
            for target in node_ids:
                if source == target:
                    continue
                cursor.execute(query, (source, target, int(max_paths_per_pair)))
                candidates = []
                for hop, raw_path in cursor.fetchall():
                    parsed = json.loads(raw_path) if isinstance(raw_path, str) else raw_path
                    edge_path = [(int(edge[0]), int(edge[1])) for edge in parsed]
                    candidates.append({"hop": int(hop), "path": edge_path})
                if candidates:
                    paths[(source, target)] = candidates
                completed += 1
                if completed % 100 == 0 or completed == pair_count:
                    print(
                        f"[database] loaded path candidates for {completed}/{pair_count} "
                        f"directed node pairs",
                        flush=True,
                    )
        cursor.close()
        return cls(nodes=nodes, edges=edges, paths=paths)


@dataclass
class Observation:
    x: np.ndarray
    edge_index: np.ndarray
    node_ids: Tuple[NodeId, ...]
    action_mask: np.ndarray
    current_virtual_node: NodeId


@dataclass
class ActiveEmbedding:
    node_mapping: Dict[NodeId, NodeId]
    node_usage: Dict[NodeId, float]
    edge_usage: Dict[Edge, float]
    link_mapping: List[Dict[str, object]]


def load_events(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return sorted(payload.get("events", []), key=lambda event: (event["time"], -event["type"]))


def load_vnet(path: str) -> Tuple[List[dict], List[dict]]:
    """Load EO's GML-like VNR file without importing realtime scripts."""
    nodes: List[dict] = []
    edges: List[dict] = []
    current = None
    section = None
    pos_count = 0
    with open(path, "r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("node ["):
                current, section, pos_count = {}, "node", 0
                continue
            if line.startswith("edge ["):
                current, section = {}, "edge"
                continue
            if section and line.startswith("]"):
                (nodes if section == "node" else edges).append(current)
                current, section = None, None
                continue
            if not section or not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            key, value = parts
            value = value.strip().replace('"', "")
            if section == "node":
                if key == "id":
                    current["id"] = int(value)
                elif key == "pos":
                    current["pos_x" if pos_count == 0 else "pos_y"] = float(value)
                    pos_count += 1
                elif key in {"cpu", "max_cpu"}:
                    current[key] = float(value)
            else:
                if key in {"source", "target"}:
                    current[key] = int(value)
                elif key in {"bw", "max_bw"}:
                    current[key] = float(value)
    return nodes, edges


class EOTrainingEnvironment:
    """Sequential node-placement environment with EO link mapping at terminal."""

    feature_dim = 11

    def __init__(
        self,
        snapshot: DatabaseSnapshot,
        distance_limit: Optional[float] = 10.0,
        reusable: bool = False,
        intermediate_reward: float = 0.01,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
        quality_weight: float = 0.5,
    ) -> None:
        self.initial = copy.deepcopy(snapshot)
        self.distance_limit = distance_limit
        self.reusable = reusable
        self.intermediate_reward = float(intermediate_reward)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.quality_weight = float(quality_weight)
        self.reset()

    def reset(self) -> None:
        self.nodes = copy.deepcopy(self.initial.nodes)
        self.edges = copy.deepcopy(self.initial.edges)
        self.paths = copy.deepcopy(self.initial.paths)
        self.active: Dict[int, ActiveEmbedding] = {}
        self._clear_request()

    def _clear_request(self) -> None:
        self.vnr_id: Optional[int] = None
        self.virtual_nodes: List[dict] = []
        self.virtual_edges: List[dict] = []
        self.virtual_order: List[int] = []
        self.current_index = 0
        self.node_mapping: Dict[int, int] = {}
        self.candidates: Dict[int, List[int]] = {}

    def begin_request(
        self,
        vnr_id: int,
        virtual_nodes: Sequence[Mapping[str, float]],
        virtual_edges: Sequence[Mapping[str, float]],
        virtual_order: Optional[Sequence[int]] = None,
    ) -> Optional[Observation]:
        self.vnr_id = int(vnr_id)
        self.virtual_nodes = [dict(node) for node in virtual_nodes]
        self.virtual_edges = [dict(edge) for edge in virtual_edges]
        ids = [int(node["id"]) for node in self.virtual_nodes]
        self.virtual_order = list(map(int, virtual_order)) if virtual_order else ids
        if set(self.virtual_order) != set(ids) or len(self.virtual_order) != len(ids):
            raise ValueError("virtual_order must contain every virtual node exactly once")
        self.current_index = 0
        self.node_mapping = {}
        self.candidates = self._build_candidates()
        return self.observation() if self.virtual_order else None

    @property
    def current_virtual_node(self) -> int:
        return self.virtual_order[self.current_index]

    def observation(self) -> Observation:
        current = self.current_virtual_node
        node_ids = tuple(sorted(self.nodes))
        node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
        graph_edges = sorted(
            edge for edge in self.edges if edge[0] in node_to_index and edge[1] in node_to_index
        )
        edge_index = np.asarray(
            [[node_to_index[u] for u, _ in graph_edges], [node_to_index[v] for _, v in graph_edges]],
            dtype=np.int64,
        )
        if not graph_edges:
            edge_index = np.empty((2, 0), dtype=np.int64)
        return Observation(
            x=self._features(node_ids, current),
            edge_index=edge_index,
            node_ids=node_ids,
            action_mask=self._action_mask(node_ids, current),
            current_virtual_node=current,
        )

    def fail_current_request(self, reason: str) -> Tuple[None, float, bool, dict]:
        info = {"success": False, "reason": reason, "node_mapping": dict(self.node_mapping)}
        self._clear_request()
        return None, self.failure_reward, True, info

    def step(self, physical_node: int) -> Tuple[Optional[Observation], float, bool, dict]:
        observation = self.observation()
        physical_node = int(physical_node)
        try:
            action_index = observation.node_ids.index(physical_node)
        except ValueError:
            return self.fail_current_request("unknown physical node")
        if not observation.action_mask[action_index]:
            return self.fail_current_request("masked or infeasible node action")

        self.node_mapping[self.current_virtual_node] = physical_node
        self.current_index += 1
        if self.current_index < len(self.virtual_order):
            next_observation = self.observation()
            if not next_observation.action_mask.any():
                return self.fail_current_request("no directed-feasible action")
            return next_observation, self.intermediate_reward, False, {
                "success": None,
                "node_mapping": dict(self.node_mapping),
            }

        link_mapping = self._map_virtual_links()
        if link_mapping is False:
            return self.fail_current_request("EO directed link mapping failed")

        embedding = self._commit(link_mapping)
        reward = self._success_reward(link_mapping)
        info = {
            "success": True,
            "node_mapping": dict(embedding.node_mapping),
            "link_mapping": copy.deepcopy(embedding.link_mapping),
            "link_cost": sum(float(item["bw"]) * int(item["hop"]) for item in link_mapping),
        }
        self._clear_request()
        return None, reward, True, info

    def release(self, vnr_id: int) -> bool:
        embedding = self.active.pop(int(vnr_id), None)
        if embedding is None:
            return False
        for node_id, amount in embedding.node_usage.items():
            capacity = float(self.initial.nodes[node_id]["max_cpu"])
            self.nodes[node_id]["cpu"] = min(capacity, self.nodes[node_id]["cpu"] + amount)
        for edge, amount in embedding.edge_usage.items():
            self.edges[edge] = min(self.initial.edges[edge], self.edges[edge] + amount)
        return True

    def _build_candidates(self) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for virtual in self.virtual_nodes:
            virtual_id = int(virtual["id"])
            required_cpu = float(virtual.get("cpu", 0.0))
            candidates = []
            for node_id, substrate in self.nodes.items():
                if float(substrate["cpu"]) < required_cpu:
                    continue
                if self.distance_limit is not None and all(
                    key in virtual for key in ("pos_x", "pos_y")
                ):
                    distance = math.hypot(
                        float(virtual["pos_x"]) - float(substrate["pos"]),
                        float(virtual["pos_y"]) - float(substrate["pos2"]),
                    )
                    if distance > self.distance_limit:
                        continue
                candidates.append(int(node_id))
            result[virtual_id] = candidates
        return result

    def _action_mask(self, node_ids: Sequence[int], current: int) -> np.ndarray:
        virtual_by_id = {int(node["id"]): node for node in self.virtual_nodes}
        required_cpu = float(virtual_by_id[current].get("cpu", 0.0))
        used_nodes = set(self.node_mapping.values())
        mask = np.zeros(len(node_ids), dtype=bool)
        incident = [
            edge for edge in self.virtual_edges
            if int(edge["source"]) == current or int(edge["target"]) == current
        ]
        for index, node_id in enumerate(node_ids):
            if node_id not in self.candidates.get(current, []):
                continue
            if float(self.nodes[node_id]["cpu"]) < required_cpu:
                continue
            if not self.reusable and node_id in used_nodes:
                continue
            feasible = True
            for edge in incident:
                source, target = int(edge["source"]), int(edge["target"])
                bandwidth = float(edge.get("bw", 0.0))
                if source == current and target in self.node_mapping:
                    feasible = self._has_path(node_id, self.node_mapping[target], bandwidth)
                elif target == current and source in self.node_mapping:
                    feasible = self._has_path(self.node_mapping[source], node_id, bandwidth)
                if not feasible:
                    break
            mask[index] = feasible
        return mask

    def _has_path(self, source: int, target: int, bandwidth: float) -> bool:
        if source == target:
            return True
        for candidate in self.paths.get((source, target), []):
            if all(self.edges.get(tuple(edge), 0.0) >= bandwidth for edge in candidate["path"]):
                return True
        return False

    def _map_virtual_links(self):
        used: Dict[Edge, float] = {}
        result: List[Dict[str, object]] = []
        for virtual_edge in self.virtual_edges:
            v_source, v_target = int(virtual_edge["source"]), int(virtual_edge["target"])
            source, target = self.node_mapping[v_source], self.node_mapping[v_target]
            bandwidth = float(virtual_edge.get("bw", 0.0))
            if source == target:
                result.append({"source": source, "target": target, "bw": bandwidth, "hop": 0, "linkpath": []})
                continue
            chosen = None
            for candidate in self.paths.get((source, target), []):
                path = [tuple(edge) for edge in candidate["path"]]
                if all(self.edges.get(edge, 0.0) - used.get(edge, 0.0) >= bandwidth for edge in path):
                    chosen = (candidate, path)
                    break
            if chosen is None:
                return False
            candidate, path = chosen
            for edge in path:
                used[edge] = used.get(edge, 0.0) + bandwidth
            result.append({
                "source": source,
                "target": target,
                "bw": bandwidth,
                "hop": int(candidate["hop"]),
                "linkpath": path,
            })
        return result

    def _commit(self, link_mapping: Sequence[Mapping[str, object]]) -> ActiveEmbedding:
        virtual_by_id = {int(node["id"]): node for node in self.virtual_nodes}
        node_usage: Dict[int, float] = {}
        for virtual_id, node_id in self.node_mapping.items():
            amount = float(virtual_by_id[virtual_id].get("cpu", 0.0))
            node_usage[node_id] = node_usage.get(node_id, 0.0) + amount
        edge_usage: Dict[Edge, float] = {}
        for item in link_mapping:
            for edge in item["linkpath"]:
                edge = tuple(edge)
                edge_usage[edge] = edge_usage.get(edge, 0.0) + float(item["bw"])
        for node_id, amount in node_usage.items():
            self.nodes[node_id]["cpu"] -= amount
        for edge, amount in edge_usage.items():
            self.edges[edge] -= amount
        embedding = ActiveEmbedding(
            node_mapping=dict(self.node_mapping),
            node_usage=node_usage,
            edge_usage=edge_usage,
            link_mapping=[dict(item) for item in link_mapping],
        )
        self.active[int(self.vnr_id)] = embedding
        return embedding

    def _success_reward(self, link_mapping: Sequence[Mapping[str, object]]) -> float:
        weighted_hops = sum(float(item["bw"]) * int(item["hop"]) for item in link_mapping)
        total_bandwidth = sum(float(item["bw"]) for item in link_mapping)
        max_hops = max(len(self.nodes) - 1, 1)
        normalized_cost = weighted_hops / max(total_bandwidth * max_hops, 1.0)
        quality = 1.0 - min(max(normalized_cost, 0.0), 1.0)
        return self.success_reward + self.quality_weight * quality

    def _features(self, node_ids: Sequence[int], current: int) -> np.ndarray:
        out_free = {node: 0.0 for node in node_ids}
        in_free = {node: 0.0 for node in node_ids}
        out_capacity = {node: 0.0 for node in node_ids}
        in_capacity = {node: 0.0 for node in node_ids}
        for (source, target), bandwidth in self.edges.items():
            if source in out_free:
                out_free[source] += bandwidth
            if target in in_free:
                in_free[target] += bandwidth
        for (source, target), bandwidth in self.initial.edges.items():
            if source in out_capacity:
                out_capacity[source] += bandwidth
            if target in in_capacity:
                in_capacity[target] += bandwidth

        virtual_by_id = {int(node["id"]): node for node in self.virtual_nodes}
        virtual = virtual_by_id[current]
        v_out = sum(float(edge.get("bw", 0.0)) for edge in self.virtual_edges if int(edge["source"]) == current)
        v_in = sum(float(edge.get("bw", 0.0)) for edge in self.virtual_edges if int(edge["target"]) == current)
        cpu_scale = max(max(float(node["max_cpu"]) for node in self.initial.nodes.values()), 1.0)
        bw_scale = max(max(out_capacity.values(), default=1.0), max(in_capacity.values(), default=1.0), 1.0)
        remaining = max(len(self.virtual_order) - self.current_index - 1, 0) / max(len(self.virtual_order), 1)
        selected = set(self.node_mapping.values())
        rows = []
        for node_id in node_ids:
            rows.append([
                float(self.nodes[node_id]["cpu"]) / cpu_scale,
                float(self.initial.nodes[node_id]["max_cpu"]) / cpu_scale,
                out_free[node_id] / bw_scale,
                out_capacity[node_id] / bw_scale,
                in_free[node_id] / bw_scale,
                in_capacity[node_id] / bw_scale,
                float(node_id in selected),
                float(virtual.get("cpu", 0.0)) / cpu_scale,
                v_out / bw_scale,
                v_in / bw_scale,
                remaining,
            ])
        return np.asarray(rows, dtype=np.float32)


def resolve_vnet_path(vnet_dir: str, filename_pattern: str, vnr_id: int) -> str:
    return str(Path(vnet_dir) / filename_pattern.format(vnr_id=int(vnr_id)))
