"""CLI: python -m NFV.a3c_gcn.train --config <yaml>."""

import argparse
from NFV.project_paths import project_path
from NFV.db_config import connect_database

import yaml

from .trainer import TrainConfig, train
from .training_env import DatabaseSnapshot


def main():
    parser = argparse.ArgumentParser(description="Train EO's node-only A3C-GCN policy")
    parser.add_argument("--config", required=True, help="training YAML file")
    args = parser.parse_args()
    with open(project_path(args.config), "r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}

    db = payload.pop("database", {}) or {}
    max_paths_per_pair = int(db.get("max_paths_per_pair", 5))
    print("[startup] connecting to EO database", flush=True)
    connection = connect_database()
    try:
        print(
            f"[startup] loading node/link snapshot and at most "
            f"{max_paths_per_pair} paths per directed pair",
            flush=True,
        )
        snapshot = DatabaseSnapshot.from_connection(
            connection, max_paths_per_pair=max_paths_per_pair
        )
    finally:
        connection.close()
    print(
        f"[startup] snapshot ready: {len(snapshot.nodes)} nodes, "
        f"{len(snapshot.edges)} directed links, {len(snapshot.paths)} path pairs",
        flush=True,
    )
    config = TrainConfig(**payload)
    print(
        f"[training] starting {config.num_workers} workers for "
        f"{config.episodes} episodes",
        flush=True,
    )
    summary = train(snapshot, config)
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
