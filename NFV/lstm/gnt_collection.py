"""Chronological collection helpers; importing this module never accesses a database."""
from pathlib import Path
import math


def window_schedule(events, start, width=100):
    """Events at a sampling time are processed before the snapshot at that time."""
    items = []
    for index, event in enumerate(events):
        time = float(event["time"])
        if not math.isfinite(time) or not start <= time < start + width:
            raise ValueError("Event outside current window")
        if event["type"] not in (0, 1):
            raise ValueError("Unknown event type")
        items.append((time, 0, index, event))
    items.extend((start+offset, 1, offset, None) for offset in range(5, 36, 5))
    return [(time, event) for time, _, _, event in sorted(items)]


def prepare_run(run):
    """Reserve a fresh paired dataset before simulation opens its DB connection."""
    if not str(run).isalnum():
        raise ValueError("GNT_PREFIX_RUN must be alphanumeric")
    root = Path(__file__).resolve().parent / "GNT" / "collected"
    path = root / str(run)
    path.mkdir(parents=True, exist_ok=False)
    return path


def domain_snapshot(calculator, domains, prev_tms):
    # Read-only calculation: do not advance flow or update decision-time history.
    return [dict(domain_id=i, T=result["T"], F=result["F"], H=result["H"])
            for i, nodes in enumerate(domains)
            for result in [calculator.compute_domain_temperature(nodes, prev_tm=prev_tms[i])]]
