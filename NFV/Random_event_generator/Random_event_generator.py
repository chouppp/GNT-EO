import numpy as np
import yaml
from typing import List
from NFV.virtual_net import VirtualNetworkEvent

def save_event_config_yaml(
    events: List[VirtualNetworkEvent],
    output_path: str,
    arrival_rate: float,
    fixed_lifetime : float,
    max_time: float,
    seed: int
):
    config = {
        "simulation": {
            "arrival_rate": arrival_rate,
            "lifetime_mean": fixed_lifetime ,
            "max_time": max_time,
            "seed": seed
        },
        "events": [
            {
                "id": e.id,
                "type": e.type,
                "v_net_id": e.v_net_id,
                "time": e.time
            }
            for e in events
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
def generate_vnr_events(
    arrival_rate: float,
    fixed_lifetime: float,
    max_time: float,
    vnr_id_range: tuple = (1, 1000),
    seed: int = None
) -> List[VirtualNetworkEvent]:
    """
    Generate VNR arrival and leave events using
    a Poisson arrival process with fixed lifetime.
    Ensures no duplicate VNR IDs on arrival.
    """
    if seed is not None:
        np.random.seed(seed)

    low_id, high_id = vnr_id_range
    assert low_id <= high_id, "Invalid vnr_id_range"

    available_ids = list(range(low_id, high_id + 1))
    np.random.shuffle(available_ids)  # Shuffle the order.
    id_index = 0  # Iterator over available IDs.

    events = []
    current_time = 0.0
    event_id = 0

    while current_time < max_time and id_index < len(available_ids):
        # Exponential inter-arrival time (Poisson process)
        inter_arrival = np.random.exponential(1.0 / arrival_rate)
        current_time += inter_arrival

        if current_time >= max_time:
            break

        # Assign unique VNR IDs.
        vnr_id = available_ids[id_index]
        id_index += 1

        # ---------- arrival event ----------
        arrival_event = VirtualNetworkEvent(
            id=event_id,
            type=1,
            v_net_id=vnr_id,
            time=round(float(current_time), 4)
        )
        events.append(arrival_event)
        event_id += 1

        # Fixed lifetime.
        leave_time = current_time + fixed_lifetime
        leave_event = VirtualNetworkEvent(
            id=event_id,
            type=0,
            v_net_id=vnr_id,
            time=round(float(leave_time), 4)
        )
        events.append(leave_event)
        event_id += 1

    # Sort by time; arrival before leave if same time
    events.sort(key=lambda e: (e.time, -e.type))
    return events

if __name__ == "__main__":
    arrival_rate = 0.05
    fixed_lifetime  = 1000
    max_time = 8000
    seed = 3055

    events = generate_vnr_events(
        arrival_rate=arrival_rate,
        fixed_lifetime =fixed_lifetime ,
        max_time=max_time,
        vnr_id_range=(1, 4000),   # Set the ID range here.
        seed=seed
    )

    save_event_config_yaml(
        events,
        output_path="vnr_events10.yaml",
        arrival_rate=arrival_rate,
        fixed_lifetime =fixed_lifetime ,
        max_time=max_time,
        seed=seed
    )

    for e in events[:8]:
        print(e)