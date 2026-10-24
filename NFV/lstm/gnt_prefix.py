"""Strict 35%-window observations. Never infer sample times from window-end logs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import gnt_dataset as ds

OBSERVATIONS = ds.ROOT / "GNT" / "observations"


def validate_record(record):
    if record.get("schema") != "gnt-observation-v1":
        raise ValueError("Expected genuine gnt-observation-v1 records, not synthetic substeps")
    end = float(record["window_end"])
    time = float(record["sample_time"])
    processed = float(record["latest_processed_event_time"])
    if not np.all(np.isfinite([end, time, processed])) or end % 100:
        raise ValueError("Invalid observation time")
    if not end - 100 < time <= end - 65:
        raise ValueError("Observation must be within the first 35% of the window")
    if processed > time:
        raise ValueError("Future events have already affected this observation")
    domains = record["domains"]
    if len(domains) != 5 or {d["domain_id"] for d in domains} != set(range(5)):
        raise ValueError("Exactly five unique domains are required")
    state = np.zeros((4, 5))
    for d in domains:
        f, h, t = d["F"], d["H"], d["T"]
        if not np.all(np.isfinite([f, h])) or min(f, h) < 0:
            raise ValueError("Invalid domain F/H")
        if t is not None and not np.isfinite(t):
            raise ValueError("Invalid domain T")
        state[:, d["domain_id"]] = [np.log1p(f), h,
            0 if t is None else np.clip(t, -ds.LIMIT, ds.LIMIT)/ds.LIMIT, float(t is not None)]
    return end, time, state.reshape(-1)


def record_observation(run, window_end, sample_time, latest_processed_event_time, domains,
                       observation_dir=OBSERVATIONS):
    """Call with a real snapshot; caller must supply the actual processed-event watermark.

    Does not advance the simulator or recompute T. Preserve its temperature definition.
    Call at start+5,10,...,35, BEFORE processing any event later than sample_time.
    """
    if not str(run).isalnum():
        raise ValueError("Run ID must be alphanumeric")
    record = dict(schema="gnt-observation-v1", run=str(run), window_end=float(window_end),
                  sample_time=float(sample_time), latest_processed_event_time=float(latest_processed_event_time),
                  domains=domains)
    validate_record(record)
    content = json.dumps(record, allow_nan=False)
    path = (Path(observation_dir)/str(run)/"prefix35.jsonl").resolve()
    if not path.is_relative_to(ds.ROOT):
        raise ValueError("Observations must stay inside lstm")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(content + "\n")
    return path


def build_samples(raw_path, observation_path, run, start=500., end=2000.):
    # Reuse the exact original target definition, including duplicate-time averaging.
    if not np.all(np.isfinite([start, end])) or start % 100 or end % 100 or end < start:
        raise ValueError("Invalid evaluation range")
    times, states, target_hash = ds.read_states(raw_path)
    labels = dict(zip(times, states))
    path = Path(observation_path)
    if not path.is_file():
        raise ValueError(f"Run {run}: missing genuine observations: {path}. "
                         "Old window-end logs cannot supply first-35% observations.")
    content = path.read_bytes()
    samples = {}
    for line in content.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("run") != str(run):
            raise ValueError("Observation run does not match target run")
        window, time, state = validate_record(record)
        key = (window, time)
        if key in samples:
            raise ValueError(f"Duplicate observation {key}; do not append multiple experiments to one run")
        samples[key] = state
    targets = np.arange(start, end+.01, 100.)
    x, y, observation_times = [], [], []
    for t in targets:
        required = t-100+np.arange(5., 36., 5.)
        missing = [float(s) for s in required if (t, s) not in samples]
        if missing or t not in labels:
            raise ValueError(f"Run {run}, window {t}: missing observations {missing} "
                             f"or target={t not in labels}; no interpolation from future states")
        x.append([samples[t, s] for s in required])
        y.append(labels[t][2])
        observation_times.append(required)
    source = json.dumps({"target": str(Path(raw_path).resolve()), "observations": str(path.resolve())})
    digest = hashlib.sha256((target_hash+hashlib.sha256(content).hexdigest()).encode()).hexdigest()
    return ds.Samples(str(run), source, digest, targets, np.asarray(observation_times),
                      np.asarray(x), np.asarray(y))


def load_runs(raw_base, observation_dir, runs, start=500., end=2000.):
    if not runs or len(set(runs)) != len(runs):
        raise ValueError("Run IDs must be nonempty and unique")
    return {r: build_samples(Path(raw_base)/r/"resource_usage.yaml",
                            Path(observation_dir)/r/"prefix35.jsonl", r, start, end) for r in runs}
