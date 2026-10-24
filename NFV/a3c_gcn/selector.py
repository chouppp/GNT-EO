"""Frozen actor inference; selecting nodes never commits any resources."""

import copy
from collections import Counter

import torch
from torch_geometric.data import Data

from .training_env import EOTrainingEnvironment
from .training_model import DirectionalA3CGCNActor


class NodeSelector:
    def __init__(self, checkpoint, snapshot, distance_limit=10.0):
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # actor_best is a plain state_dict, not an Actor-Critic checkpoint.
        embedding_dim = int(weights["head.0.weight"].shape[0])
        self.actor = DirectionalA3CGCNActor(11, embedding_dim)
        self.actor.load_state_dict(weights, strict=True)
        self.actor.eval()
        self.env = EOTrainingEnvironment(snapshot, distance_limit=distance_limit, reusable=False)
        self.failure_reason = None

    @torch.inference_mode()
    def select_nodes(self, vnr_id, virtual_nodes, virtual_edges, nodes, edges):
        self.failure_reason = None
        # Keep initial capacities/scales fixed; only refresh residual resources.
        self.env.nodes = copy.deepcopy(nodes)
        self.env.edges = dict(edges)
        observation = self.env.begin_request(vnr_id, virtual_nodes, virtual_edges)
        if observation is None:
            self.failure_reason = "empty virtual network"
            return None
        while observation is not None:
            if not observation.action_mask.any():
                self.failure_reason = "no directed-feasible action"
                return None
            data = Data(
                x=torch.as_tensor(observation.x, dtype=torch.float32),
                edge_index=torch.as_tensor(observation.edge_index, dtype=torch.long),
            )
            data.batch = torch.zeros(len(observation.node_ids), dtype=torch.long)
            logits = self.actor(data).reshape(-1)
            mask = torch.as_tensor(observation.action_mask, dtype=torch.bool)
            index = int(logits.masked_fill(~mask, -torch.inf).argmax())
            self.env.node_mapping[self.env.current_virtual_node] = observation.node_ids[index]
            self.env.current_index += 1
            if self.env.current_index == len(self.env.virtual_order):
                return dict(self.env.node_mapping)
            # Do NOT call env.step(): it also maps links and commits resources.
            observation = self.env.observation()


def map_directed_links(node_mapping, virtual_edges, residual_edges, paths):
    """EO first-hop-sorted feasible database path; reserve BW within this VNR.

    Paths are database candidates, never newly computed shortest paths. Reverse
    edges are independent. The input dictionaries are not mutated.
    """
    used = Counter()
    results = []
    for edge in virtual_edges:
        src, dst = node_mapping[int(edge["source"])], node_mapping[int(edge["target"])]
        bw = float(edge.get("bw", 0.0))
        candidates = paths.get((src, dst), [])
        if src == dst:
            candidates = [{"hop": 0, "path": []}]
        for candidate in candidates:
            path = [tuple(e) for e in candidate["path"]]
            multiplicities = Counter(path)
            if all(e in residual_edges and residual_edges[e] - used[e] >= bw * n
                   for e, n in multiplicities.items()):
                bottleneck = min((residual_edges[e] - used[e] for e in path), default=0.0)
                for e, n in multiplicities.items():
                    used[e] += bw * n
                results.append({"source": src, "target": dst, "bw": bw,
                                "hop": int(candidate["hop"]), "path_bottleneck": bottleneck,
                                "linkpath": [list(e) for e in path]})
                break
        else:
            return False
    return results


def resource_usage(mapping, virtual_nodes, links):
    cpu, bandwidth = Counter(), Counter()
    for node in virtual_nodes:
        cpu[mapping[int(node["id"])]] += float(node.get("cpu", 0.0))
    for link in links:
        for edge in link["linkpath"]:
            bandwidth[tuple(edge)] += float(link["bw"])
    return dict(cpu), dict(bandwidth)
