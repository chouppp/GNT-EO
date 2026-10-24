"""Raw-log GNT dataset: past windows as inputs and the next window as the label. Reads logs only."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
RAW_BASE = ROOT.parent / "result_storage" / "EO"
LIMIT = 10000.0
DOMAINS = 5
FEATURE_NAMES = [f"{field}_domain_{d}" for field in ("log1p_F", "H", "clipped_T_scaled", "T_valid") for d in range(DOMAINS)]


@dataclass
class Samples:
    run: str
    source: str
    source_sha256: str
    times: np.ndarray
    history_times: np.ndarray
    x: np.ndarray
    y: np.ndarray


def read_states(path):
    """For duplicate times, clip each domain before averaging to preserve the original picture.py ground truth."""
    path = Path(path)
    content = path.read_bytes()
    data = yaml.load(content.decode("utf-8-sig"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    records = data if isinstance(data, list) else data.get("records", []) if isinstance(data, dict) else []
    grouped = {}
    for record in records:
        if not isinstance(record, dict) or record.get("time") is None:
            continue
        t = float(record["time"])
        if not np.isfinite(t) or t < 100 or not np.isclose(t % 100, 0):
            continue
        domains = record.get("domain_temperatures")
        if domains is None:
            domains = record.get("domains")
        if domains is None:
            domains = record.get("temperature_domains")
        state = np.zeros((4, DOMAINS), dtype=np.float64)
        for d in domains or []:
            if not isinstance(d, dict) or d.get("domain_id") is None:
                continue
            index = int(d["domain_id"])
            if index not in range(DOMAINS):
                raise ValueError(f"{path}: unexpected domain {index}")
            f, h, temp = d.get("F", 0.), d.get("H", 0.), d.get("T")
            if not np.isfinite(f) or not np.isfinite(h) or f < 0 or h < 0:
                raise ValueError(f"{path}: invalid F/H at {t}")
            if temp is not None and not np.isfinite(temp):
                raise ValueError(f"{path}: invalid T at {t}")
            state[:, index] = [np.log1p(f), h, 0. if temp is None else np.clip(temp, -LIMIT, LIMIT)/LIMIT,
                               float(temp is not None)]
        grouped.setdefault(t, []).append(state)
    if not grouped:
        raise ValueError(f"No recorded temperature states in {path}")
    times = np.array(sorted(grouped), dtype=np.float64)
    states = np.stack([np.mean(grouped[t], axis=0) for t in times])
    return times, states, hashlib.sha256(content).hexdigest()


def build_samples(path, run, history=3, start=500., end=2000.):
    if history < 1 or not np.isfinite(start) or not np.isfinite(end) or end < start:
        raise ValueError("Invalid history length or evaluation range")
    times, states, digest = read_states(path)
    by_time = {float(t): s for t, s in zip(times, states)}
    target_times = np.arange(start, end + .01, 100., dtype=float)
    if not np.isclose(start % 100, 0) or not np.isclose(end % 100, 0):
        raise ValueError("start/end must be multiples of 100")
    x, y, histories = [], [], []
    for t in target_times:
        past = t - 100*np.arange(history, 0, -1)
        missing = [float(v) for v in [*past, t] if v not in by_time]
        if missing:
            raise ValueError(f"Run {run}: missing recorded windows {missing}; no filling or target replacement")
        x.append(np.stack([by_time[p].reshape(-1) for p in past]))
        y.append(by_time[t][2].copy())
        histories.append(past)
    if not len(x):
        raise ValueError("No samples")
    return Samples(str(run), str(Path(path).resolve()), digest, target_times, np.asarray(histories),
                   np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))


def load_runs(raw_base, runs, history=3, start=500., end=2000.):
    if len(runs) != len(set(runs)) or not runs:
        raise ValueError("Run IDs must be nonempty and unique")
    return {r: build_samples(Path(raw_base)/r/"resource_usage.yaml", r, history, start, end) for r in runs}


def concatenate(datasets, runs):
    return np.concatenate([datasets[r].x for r in runs]), np.concatenate([datasets[r].y for r in runs])


def fit_scaler(x):
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1))
    return mean, np.where(std < 1e-8, 1., std)


def transform(x, mean, std):
    return ((x-mean)/std).astype(np.float32)


def baselines(x):
    """Historical temperature features occupy columns 10:15 and strictly precede the target time."""
    return {"zero": np.zeros((len(x), DOMAINS)), "last_value": x[:, -1, 10:15].copy(),
            "history_mean": x[:, :, 10:15].mean(axis=1)}


def make_folds(runs, mode="holdout", train=None, validation=None, test=None):
    """Default split: runs 1..5 for fitting, 6 for validation, 7..8 for testing. loro is leave-one-run-out cross-validation only."""
    if len(set(runs)) != len(runs):
        raise ValueError("Duplicate run IDs")
    if mode == "holdout":
        fit = list(train if train is not None else runs[:-3])
        val = list(validation if validation is not None else runs[-3:-2])
        held = list(test if test is not None else runs[-2:])
        groups = [set(fit), set(val), set(held)]
        if any(not g for g in groups) or any(len(g) != len(v) for g,v in zip(groups,[fit,val,held])):
            raise ValueError("Fit/validation/test must be nonempty and unique")
        if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2] or set.union(*groups)!=set(runs):
            raise ValueError("Fit/validation/test must partition all runs without overlap")
        return [{"name":"holdout", "fit":fit, "validation":val, "test":held}]
    if mode != "loro" or len(runs)<3 or any(v is not None for v in (train,validation,test)):
        raise ValueError("loro needs >=3 runs and no explicit split lists")
    return [{"name":f"test_{r}", "test":[r], "validation":[runs[(i+1)%len(runs)]],
             "fit":[v for v in runs if v not in (r, runs[(i+1)%len(runs)])]} for i,r in enumerate(runs)]
