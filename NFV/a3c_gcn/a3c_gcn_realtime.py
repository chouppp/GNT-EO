"""EO-style mapping experiment without traffic simulation.

Default: snapshot-only. --write-db: allocate/release CPU and directed bandwidth.

Run from project root with python -m NFV.a3c_gcn.a3c_gcn_realtime --config ...
"""

import argparse
from NFV.project_paths import project_path, resolve_config_paths
import copy
import json
from NFV.db_config import connect_database, load_database_config
import time
from collections import Counter
from pathlib import Path

import yaml

from .selector import NodeSelector, map_directed_links, resource_usage
from .training_env import DatabaseSnapshot, load_vnet, resolve_vnet_path


def read_resources(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT node_id,cpu,max_cpu,pos_x,pos_y FROM node_single_table")
        nodes = {int(i): {"cpu": float(c), "max_cpu": float(m), "pos": float(x), "pos2": float(y)}
                 for i, c, m, x, y in cursor.fetchall()}
        cursor.execute("SELECT source_node,destination_node,bw_resource,bw_max_resource FROM link_single_table")
        rows = cursor.fetchall()
        edges = {(int(u), int(v)): float(b) for u, v, b, m in rows}
        capacities = {(int(u), int(v)): float(m) for u, v, b, m in rows}
    return nodes, edges, capacities


def apply_resources(conn, cpu, bandwidth, release=False):
    """Single transaction for CPU and directed BW; no partial VNR allocation.

    Requires transactional (InnoDB) tables. Unlike legacy helpers, no clipping
    to zero is permitted on allocation: a stale-resource conflict is rejected.
    """
    conn.begin()
    try:
        with conn.cursor() as cursor:
            for node, amount in sorted(cpu.items()):
                if amount == 0:
                    continue
                if release:
                    cursor.execute("UPDATE node_single_table SET cpu=cpu+%s WHERE node_id=%s",
                                   (amount, node))
                else:
                    cursor.execute("UPDATE node_single_table SET cpu=cpu-%s WHERE node_id=%s AND cpu >= %s",
                                   (amount, node, amount))
                if cursor.rowcount != 1:
                    raise RuntimeError(f"Node resource conflict: {node}")
            for (u, v), amount in sorted(bandwidth.items()):
                if amount == 0:
                    continue
                if release:
                    cursor.execute("UPDATE link_single_table SET bw_resource=bw_resource+%s "
                                   "WHERE source_node=%s AND destination_node=%s", (amount, u, v))
                else:
                    cursor.execute("UPDATE link_single_table SET bw_resource=bw_resource-%s "
                                   "WHERE source_node=%s AND destination_node=%s AND bw_resource >= %s",
                                   (amount, u, v, amount))
                if cursor.rowcount != 1:
                    raise RuntimeError(f"Directed link resource conflict: {(u, v)}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def event_windows(events, window_size):
    """Match nfv_realtime: leaves first within each window, then arrivals."""
    ordered = sorted(events, key=lambda event: event["time"])
    if not ordered:
        return
    position = 0
    for start in range(0, int(ordered[-1]["time"]) + window_size, window_size):
        group = []
        while position < len(ordered) and ordered[position]["time"] < start + window_size:
            group.append(ordered[position])
            position += 1
        yield start + window_size, ([e for e in group if int(e["type"]) == 0]
                                    + [e for e in group if int(e["type"]) == 1])


def new_run_directory(root):
    """Use algorithm/1, algorithm/2, ... without overwriting previous runs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    number = max((int(p.name) for p in root.iterdir() if p.name.isdigit()), default=0) + 1
    while True:
        output = root / str(number)
        try:
            output.mkdir()
            return output
        except FileExistsError:
            number += 1


def write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def mapping_record(sequence, event, links):
    """Same success-only schema as EO's vne_mapping.yaml."""
    return {
        "id": int(sequence), "vnr_ind": int(event["v_net_id"]),
        "event_id": int(event["id"]), "event_time": float(event["time"]),
        "links": [{"source": int(link["source"]), "target": int(link["target"]),
                   "bw": float(link["bw"]),
                   "linkpath": [[int(u), int(v)] for u, v in link["linkpath"]]}
                  for link in links],
    }


def resource_record(now, nodes, edges, capacities, acceptance):
    def usage(used, total):
        ratio = used / total if total > 0 else 0.0
        return {"used": float(used), "total": float(total),
                "usage_ratio": ratio, "usage_percent": round(ratio * 100, 4)}
    cpu_total = sum(n["max_cpu"] for n in nodes.values())
    bw_total = sum(capacities.values())
    return {"time": float(now),
            "bandwidth": usage(bw_total - sum(edges.values()), bw_total),
            "cpu": usage(cpu_total - sum(n["cpu"] for n in nodes.values()), cpu_total),
            "acceptance_rate": float(acceptance),
            "domain_temperatures": []}  # Not computed, never fabricated as zero.


def run(config, conn, write_db=False):
    config = resolve_config_paths(config)
    window_size = int(config.get("time_window", 100))
    if window_size <= 0:
        raise ValueError("time_window must be positive")
    with open(config["event_file"], encoding="utf-8") as stream:
        events = (yaml.safe_load(stream) or {}).get("events", [])
    for event in events:
        int(event["id"])  # Require original event IDs for EO-compatible records.
        if int(event["type"]) not in (0, 1) or float(event["time"]) < 0:
            raise ValueError(f"Invalid event: {event}")
    # Fail before any database writes when a VNR input is missing/malformed.
    vnets = {}
    for event in events:
        if int(event["type"]) == 1:
            ident = int(event["v_net_id"])
            path = resolve_vnet_path(config["vnet_dir"], config["filename_pattern"], ident)
            vnets[ident] = load_vnet(path)
            vnodes, vedges = vnets[ident]
            ids = {int(n["id"]) for n in vnodes}
            if len(ids) != len(vnodes) or any(float(n.get("cpu", 0)) < 0 for n in vnodes):
                raise ValueError(f"Invalid virtual nodes: {path}")
            if any(int(e["source"]) not in ids or int(e["target"]) not in ids
                   or float(e.get("bw", 0)) < 0 for e in vedges):
                raise ValueError(f"Invalid virtual edges: {path}")
    nodes, edges, capacities = read_resources(conn)
    if not nodes or any(abs(n["cpu"] - n["max_cpu"]) > 1e-6 for n in nodes.values()) or any(
        abs(edges[e] - capacities[e]) > 1e-6 for e in edges
    ):
        raise ValueError("Start from a full-capacity experiment database. No automatic reset is performed.")
    limit = config.get("max_paths_per_pair", 100)
    print(f"[startup] loading database path candidates: limit={limit}", flush=True)
    if limit is None:
        from NFV.util.normal_method import load_all_paths_numpy
        raw = load_all_paths_numpy(conn)
        paths = {pair: [{"hop": int(hop), "path": [tuple(e) for e in path]}
                        for hop, path in zip(record["hop"], record["path"])]
                 for pair, record in raw.items()}
    else:
        paths = DatabaseSnapshot.from_connection(conn, int(limit)).paths
    snapshot = DatabaseSnapshot(copy.deepcopy(nodes), dict(edges), paths)
    selector = NodeSelector(config["checkpoint"], snapshot, config.get("distance_limit", 10.0))
    output = new_run_directory(config["output_dir"])
    config_record = dict(config, write_db=write_db)
    config_record["database"] = {k: v for k, v in config.get("database", {}).items() if k != "password"}
    (output / "run_config.yaml").write_text(yaml.safe_dump(config_record, allow_unicode=True), encoding="utf-8")
    print(f"[experiment] mode={'LIVE DATABASE WRITES' if write_db else 'snapshot-only'}, output={output}", flush=True)

    if write_db:
        with conn.cursor() as cursor:
            cursor.execute("SELECT TABLE_NAME,ENGINE FROM information_schema.TABLES "
                           "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME IN ('node_single_table','link_single_table')")
            engines = dict(cursor.fetchall())
        if len(engines) != 2 or any(str(e).lower() != "innodb" for e in engines.values()):
            raise ValueError("Live experiment requires InnoDB resource tables")

    active, mapping_records, window_records = {}, [], []
    write_yaml(output / "vne_mapping.yaml", mapping_records)
    write_yaml(output / "resource_usage.yaml", {"records": window_records})
    failures = Counter()
    total = successes = 0
    status = "completed"

    def release(ident):
        if ident not in active:
            return
        record = active[ident]
        if write_db:
            apply_resources(conn, record["cpu"], record["bandwidth"], release=True)
        else:
            for node, amount in record["cpu"].items():
                nodes[node]["cpu"] += amount
            for edge, amount in record["bandwidth"].items():
                edges[edge] += amount
        del active[ident]

    try:
        for now, group in event_windows(events, window_size):
            for event in group:
                ident = int(event["v_net_id"])
                if int(event["type"]) == 0:
                    release(ident)
                    continue
                total += 1
                started = time.perf_counter()
                reason, mapping, links = None, None, None
                if ident in active:
                    reason = "duplicate active VNR ID"
                else:
                    if write_db:
                        nodes, edges, _ = read_resources(conn)
                    vnodes, vedges = vnets[ident]
                    mapping = selector.select_nodes(ident, vnodes, vedges, nodes, edges)
                    if mapping is None:
                        reason = selector.failure_reason
                    else:
                        links = map_directed_links(mapping, vedges, edges, paths)
                        if links is False:
                            reason = "EO directed link mapping failed"
                if reason is None:
                    cpu, bandwidth = resource_usage(mapping, vnodes, links)
                    if write_db:
                        apply_resources(conn, cpu, bandwidth)
                    else:
                        for node, amount in cpu.items():
                            nodes[node]["cpu"] -= amount
                        for edge, amount in bandwidth.items():
                            edges[edge] -= amount
                    # Register immediately so finally can release committed resources.
                    active[ident] = {"cpu": cpu, "bandwidth": bandwidth}
                    successes += 1
                else:
                    failures[reason] += 1
                record = {"event_id": int(event.get("id", total)), "vnr_id": ident,
                          "event_time": float(event["time"]), "success": reason is None,
                          "failure_reason": reason, "node_mapping": mapping, "link_mapping": links,
                          "mapping_seconds": time.perf_counter() - started}
                if reason is None:
                    mapping_records.append(mapping_record(successes, event, links))
                    write_yaml(output / "vne_mapping.yaml", mapping_records)
                with (output / "mapping_results.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[mapping] requests={total}, successes={successes}, acceptance={successes/total:.4f}", flush=True)
            if write_db:
                nodes, edges, _ = read_resources(conn)
            window_records.append(resource_record(now, nodes, edges, capacities, successes/max(total, 1)))
            write_yaml(output / "resource_usage.yaml", {"records": window_records})
    except BaseException:
        status = "interrupted_or_failed"
        raise
    finally:
        # EO's leave-first windows can encounter leave before arrival for a short
        # lived request. Release any remaining allocations at experiment exit.
        remaining = list(active)
        cleanup_errors = []
        for ident in remaining:
            try:
                release(ident)
            except Exception as exc:
                cleanup_errors.append(f"VNR {ident}: {exc}")
        summary = {"status": "cleanup_failed" if cleanup_errors else status, "requests": total, "successes": successes,
                   "acceptance_ratio": successes/max(total, 1), "failure_reasons": dict(failures),
                   "checkpoint": config["checkpoint"], "event_file": config["event_file"],
                   "write_db": write_db, "event_policy": "window_leaves_first",
                   "traffic_simulation": False,
                   "max_paths_per_pair": limit, "active_before_cleanup": len(remaining),
                   "cleanup_errors": cleanup_errors, "output_dir": str(output)}
        (output / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if cleanup_errors:
        raise RuntimeError("Database cleanup incomplete; inspect experiment_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("mapping_config.example.yaml")))
    parser.add_argument("--write-db", action="store_true", help="Enable CPU/BW allocation and release only; no traffic simulation")
    args = parser.parse_args()
    with open(project_path(args.config), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    import torch
    torch.set_num_threads(max(1, int(config.get("inference_threads", 1))))
    # Record the effective shared configuration, never legacy YAML credentials.
    config["database"] = load_database_config()
    conn = connect_database(autocommit=True)
    try:
        run(config, conn, args.write_db)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
