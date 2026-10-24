import yaml
# This appears to be unused.

def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def save_yaml(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )


def add_event_leave_time(
    vne_mapping_path="vne_mapping.yaml",
    vnr_event_path="vnr_events.yaml",
    output_path="vne_mapping_with_leave.yaml"
):
    # 1. Read the file.
    vne_mapping = load_yaml(vne_mapping_path)
    vnr_events = load_yaml(vnr_event_path)

    # 2. Automatically unwrap vnr_events if enclosed in an events field.
    if isinstance(vnr_events, dict):
        if "events" in vnr_events:
            vnr_events = vnr_events["events"]
        else:
            raise ValueError("vnr_events.yaml 是 dict，但没有找到 events 字段")

    # 3. Build an id-to-event dictionary.
    event_dict = {}

    for event in vnr_events:
        if not isinstance(event, dict):
            continue

        event_id = event.get("id")

        if event_id is not None:
            event_dict[event_id] = event

    # 4. Add event_leave_time to vne_mapping.
    for record in vne_mapping:
        if not isinstance(record, dict):
            continue

        arrival_event_id = record.get("event_id")

        if arrival_event_id is None:
            record["event_leave_time"] = None
            continue

        leave_event_id = arrival_event_id + 1
        leave_event = event_dict.get(leave_event_id)

        if leave_event is not None and leave_event.get("type") == 0:
            record["event_leave_time"] = leave_event.get("time")
        else:
            record["event_leave_time"] = None

    # 5. Save the results.
    save_yaml(vne_mapping, output_path)

    print(f"✅ 已完成 event_leave_time 补充，保存到：{output_path}")


if __name__ == "__main__":
    add_event_leave_time(
        vne_mapping_path="vne_mapping.yaml",
        vnr_event_path="vnr_events.yaml",
        output_path="vne_mapping_with_leave.yaml"
    )