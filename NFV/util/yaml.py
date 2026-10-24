# Save the final result as YAML.
import os

import yaml






# Record memory and bandwidth usage for each time interval.
def record_resource_usage_yaml(
        conn,
        current_time,
        yaml_path="resource_usage.yaml",
        domain_temperatures=None,
        acceptance_rate=0.0,
):
    """Record current resource utilization and temperatures of five domains in YAML."""

    with conn.cursor() as cursor:

        cursor.execute("""
            SELECT
                SUM(bw_max_resource - bw_resource) AS used_bw,
                SUM(bw_max_resource) AS total_bw
            FROM link_single_table
        """)
        used_bw, total_bw = cursor.fetchone()

        cursor.execute("""
            SELECT
                SUM(max_cpu - cpu) AS used_cpu,
                SUM(max_cpu) AS total_cpu
            FROM node_single_table
        """)
        used_cpu, total_cpu = cursor.fetchone()

    used_bw = float(used_bw or 0)
    total_bw = float(total_bw or 0)
    used_cpu = float(used_cpu or 0)
    total_cpu = float(total_cpu or 0)

    bw_ratio = used_bw / total_bw if total_bw > 0 else 0
    cpu_ratio = used_cpu / total_cpu if total_cpu > 0 else 0

    record = {
        "time": float(current_time),

        "bandwidth": {
            "used": used_bw,
            "total": total_bw,
            "usage_ratio": bw_ratio,
            "usage_percent": round(bw_ratio * 100, 4)
        },

        "cpu": {
            "used": used_cpu,
            "total": total_cpu,
            "usage_ratio": cpu_ratio,
            "usage_percent": round(cpu_ratio * 100, 4)
        },
        "acceptance_rate": float(acceptance_rate),
    }

    if domain_temperatures is not None:
        record["domain_temperatures"] = [
            {
                "domain_id": int(item["domain_id"]),
                "T": float(item["T"]) if item.get("T") is not None else None,
                "F": float(item["F"]) if item.get("F") is not None else None,
                "H": float(item["H"]) if item.get("H") is not None else None
            }
            for item in domain_temperatures
        ]

    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    data.setdefault("records", [])
    data["records"].append(record)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    temp_msg = ""
    if domain_temperatures is not None:
        temp_msg = ", Temp=[" + ", ".join(
            f"D{item['domain_id']}={item.get('T')}"
            for item in domain_temperatures
        ) + "]"

    print(
        f"[ResourceUsage] "
        f"time={current_time}, "
        f"BW={bw_ratio * 100:.2f}%, "
        f"CPU={cpu_ratio * 100:.2f}%"
        f"{temp_msg}"
    )