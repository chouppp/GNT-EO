from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Basic configuration.
# =========================

BASE_DIR = str(_PROJECT_ROOT / 'NFV/result_storage')

OUTPUT_DIR = str(_PROJECT_ROOT / 'NFV/figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. Select the runs to analyze.
# =========================

SELECTED_RUNS = [str(i) for i in range(1, 10)]


# =========================
# 3. Time-axis sampling settings.
# =========================

MAX_TIME = 3000

# Configure the first time point separately.
FIRST_TIME = 100

# Subsequent samples start at 500 and occur every 500 time units.
TIME_INTERVAL = 500

# Final time axis: 100, 500, 1000, 1500, ...
TARGET_TIMES = np.concatenate((
    np.array([FIRST_TIME]),
    np.arange(TIME_INTERVAL, MAX_TIME + 1, TIME_INTERVAL)
))

# "previous": use the latest bandwidth-utilization record before the target time.
# "linear": linear interpolation.
RESAMPLE_METHOD = "previous"


# =========================
# 4. Algorithm configuration interface.
# =========================

ALGORITHMS = {
    "GNT-EO": {
        "type": "multi_run",
        "path_template": os.path.join(
            BASE_DIR, "EO", "{run}", "resource_usage.yaml"
        ),
        "run_dirs": SELECTED_RUNS,
        "marker": "o",
    },
    "GA-STP": {
        "type": "multi_run",
        "path_template": os.path.join(
            BASE_DIR, "GA", "{run}", "resource_usage.yaml"
        ),
        "run_dirs": SELECTED_RUNS,
        "marker": "^",
    },
    "NORD": {
        "type": "multi_run",
        "path_template": os.path.join(
            BASE_DIR, "NORD", "{run}", "resource_usage.yaml"
        ),
        "run_dirs": SELECTED_RUNS,
        "marker": "s",
    },
    "D-ViNE": {
        "type": "multi_run",
        "path_template": os.path.join(
            BASE_DIR, "Dvine", "{run}", "resource_usage.yaml"
        ),
        "run_dirs": SELECTED_RUNS,
        "marker": "D",
    },
    "CEVNE": {
        "type": "multi_run",
        "path_template": os.path.join(BASE_DIR, "CEVNE", "{run}", "resource_usage.yaml"),
        "run_dirs": SELECTED_RUNS,
        "marker": "v",  # Diamond.
    },
    "A3C+GCN": {
    "type": "multi_run",
    "path_template": os.path.join(
        BASE_DIR, "A3C_GCN", "{run}", "resource_usage.yaml"
    ),
    "run_dirs": SELECTED_RUNS,
    "marker": "*",       # Five-pointed star.
    "color": "#222222",  # Black.
    },
    # Use the following format to add algorithms.
    # "New-Algorithm": {
    #     "type": "multi_run",
    #     "path_template": os.path.join(
    #         BASE_DIR, "New-Algorithm", "{run}", "resource_usage.yaml"
    #     ),
    #     "run_dirs": SELECTED_RUNS,
    #     "marker": "v",
    # },
}


# =========================
# 5. Bandwidth-utilization plot settings.
# =========================

Y_LABEL = "Average Link Resource Utilization"
TITLE = ""
OUTPUT_NAME = "bandwidth_utilization_errorbar.pdf"

# bandwidth.usage_ratio lies in [0, 1].
Y_LIM = (0.0, 1.0)

# When reading usage_percent instead, adjust the y-axis limits as follows.
# Y_LIM = (0, 100)


# =========================
# 6. Read bandwidth utilization from YAML.
# =========================

def load_resource_usage_yaml(file_path):
    """Read time and bandwidth.usage_ratio.
        Accept either a top-level record list or a dictionary containing a records field."""
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        print(f"[WARNING] Empty YAML file: {file_path}")
        return None

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("records", [])
    else:
        print(f"[WARNING] Unsupported YAML format: {file_path}")
        return None

    if not records:
        print(f"[WARNING] No records found in: {file_path}")
        return None

    rows = []

    for item in records:
        if not isinstance(item, dict):
            continue

        time_value = item.get("time")
        bandwidth_data = item.get("bandwidth", {})

        if not isinstance(bandwidth_data, dict):
            continue

        bandwidth_ratio = bandwidth_data.get("usage_ratio")

        if time_value is None or bandwidth_ratio is None:
            continue

        rows.append({
            "time": float(time_value),
            "bandwidth_ratio": float(bandwidth_ratio),
        })

    if not rows:
        print(
            f"[WARNING] No valid bandwidth.usage_ratio records in: {file_path}"
        )
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")
    return df


# =========================
# 7. Read multiple runs of one algorithm.
# =========================

def load_algorithm_runs(algo_name, algo_config):
    runs = {}

    if algo_config["type"] == "multi_run":
        for run in algo_config["run_dirs"]:
            file_path = algo_config["path_template"].format(run=run)
            df = load_resource_usage_yaml(file_path)

            if df is not None:
                runs[run] = df

    elif algo_config["type"] == "single_file":
        df = load_resource_usage_yaml(algo_config["file_path"])

        if df is not None:
            runs["single"] = df

    else:
        raise ValueError(f"Unknown algorithm type: {algo_config['type']}")

    if not runs:
        print(f"[WARNING] No valid data loaded for algorithm: {algo_name}")

    return runs


# =========================
# 8. Sample at the specified time points.
# =========================

def resample_series_to_target_times(series, target_times, method="previous"):
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.dropna()

    if series.empty:
        return pd.Series(index=target_times, dtype=float)

    target_index = pd.Index(target_times, name="time")

    if method == "previous":
        sampled = series.reindex(target_index, method="ffill")
    elif method == "linear":
        combined_index = series.index.union(target_index)
        sampled = (
            series.reindex(combined_index)
            .sort_index()
            .interpolate(method="index")
            .reindex(target_index)
        )
    else:
        raise ValueError(f"Unknown resample method: {method}")

    return sampled


# =========================
# 9. Calculate means and error bars.
# =========================

def compute_mean_and_error(runs, error_type="ci95"):
    """error_type: 'std' for standard deviation, 'se' for standard error, or 'ci95' for a 95% confidence interval."""
    series_list = []

    for run_name, df in runs.items():
        df = df[df["time"] <= MAX_TIME]

        if df.empty:
            print(f"[WARNING] Run {run_name} has no data before {MAX_TIME}")
            continue

        series = df.set_index("time")["bandwidth_ratio"]
        sampled = resample_series_to_target_times(
            series,
            target_times=TARGET_TIMES,
            method=RESAMPLE_METHOD
        )
        sampled.name = run_name
        series_list.append(sampled)

    if not series_list:
        return None, None, None

    merged = pd.concat(series_list, axis=1)
    merged = merged.dropna(axis=0, how="all")

    if merged.empty:
        return None, None, None

    time = merged.index.to_numpy(dtype=float)
    values = merged.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1)
    error = np.zeros_like(mean)

    for i in range(values.shape[0]):
        row_values = values[i, :]
        row_values = row_values[~np.isnan(row_values)]

        if len(row_values) <= 1:
            error[i] = 0.0
            continue

        std = np.std(row_values, ddof=1)
        n = len(row_values)

        if error_type == "std":
            error[i] = std
        elif error_type == "se":
            error[i] = std / np.sqrt(n)
        elif error_type == "ci95":
            error[i] = 1.96 * std / np.sqrt(n)
        else:
            raise ValueError(f"Unknown error_type: {error_type}")

    return time, mean, error


# =========================
# 10. Plot bandwidth utilization with error bars.
# =========================

def plot_bandwidth_utilization_with_errorbar(error_type="ci95"):
    plt.figure(figsize=(8, 5))

    for algo_name, algo_config in ALGORITHMS.items():
        runs = load_algorithm_runs(algo_name, algo_config)
        time, mean, error = compute_mean_and_error(
            runs,
            error_type=error_type
        )

        if time is None:
            continue

        plt.errorbar(
            time,
            mean,
            yerr=error,
            marker=algo_config.get("marker", "o"),
            markersize=7,
            linewidth=2,
            capsize=4,
            label=algo_name
        )

    plt.xlabel("Time[Time slots]")
    plt.ylabel(Y_LABEL)
    plt.title(TITLE)
    plt.xlim(0, MAX_TIME)
    plt.xticks(TARGET_TIMES)

    if Y_LIM is not None:
        plt.ylim(Y_LIM)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"[INFO] Figure saved to: {output_path}")


# =========================
# 11. Main program.
# =========================

if __name__ == "__main__":
    plot_bandwidth_utilization_with_errorbar(
        error_type="ci95"  # Options: "std", "se", "ci95".
    )