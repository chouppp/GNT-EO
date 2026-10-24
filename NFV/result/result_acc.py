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

BASE_DIR = str(_PROJECT_ROOT / 'NFV/result_storage')   # Root data directory.

OUTPUT_DIR = str(_PROJECT_ROOT / 'NFV/figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. Select the runs to analyze.
# =========================
# For example, analyze runs 1, 2, and 3 only.
# SELECTED_RUNS = ["1", "2", "3"]
#
# To analyze runs 1 through 7:
# SELECTED_RUNS = [str(i) for i in range(1, 8)]

SELECTED_RUNS = [str(i) for i in range(1, 10)]


# =========================
# 3. Time-axis sampling settings.
# =========================
# Limit the x-axis to 8000.
MAX_TIME = 7000

# Configure the first time point separately, e.g. 100.
FIRST_TIME = 100

# Subsequent samples start at 500 and occur every 500 time units.
TIME_INTERVAL = 500

# Final time axis: 100, 500, 1000, 1500, ...
TARGET_TIMES = np.concatenate((
    np.array([FIRST_TIME]),
    np.arange(TIME_INTERVAL, MAX_TIME + 1, TIME_INTERVAL)
))

# Sampling method:
# "previous": use the most recent acceptance rate before the target time; suitable for cumulative rates.
# "linear": linear interpolation.
RESAMPLE_METHOD = "previous"


# =========================
# 4. Algorithm configuration interface.
# =========================
# Current default directory structure:
# result_storage/
# ├── EO/1/resource_usage.yaml
# ├── EO/2/resource_usage.yaml
# ├── GA/1/resource_usage.yaml
# └── NORD/1/resource_usage.yaml

ALGORITHMS = {
    "GNT-EO": {
        "type": "multi_run",
        "path_template": os.path.join(BASE_DIR, "EO", "{run}", "resource_usage.yaml"),
        "run_dirs": SELECTED_RUNS,
        "marker": "o",  # Circle.
    },

    "GA-STP": {
        "type": "multi_run",
        "path_template": os.path.join(BASE_DIR, "GA", "{run}", "resource_usage.yaml"),
        "run_dirs": SELECTED_RUNS,
        "marker": "^",  # Triangle.
    },

    "NORD": {
        "type": "multi_run",
        "path_template": os.path.join(BASE_DIR, "NORD", "{run}", "resource_usage.yaml"),
        "run_dirs": SELECTED_RUNS,
        "marker": "s",  # Square.
    },
    "D-ViNE": {
        "type": "multi_run",
        "path_template": os.path.join(BASE_DIR, "Dvine", "{run}", "resource_usage.yaml"),
        "run_dirs": SELECTED_RUNS,
        "marker": "D",  # Diamond.
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
    # "D-ViNE": {
    #     "type": "multi_run",
    #     "path_template": os.path.join(BASE_DIR, "D-ViNE", "{run}", "resource_usage.yaml"),
    #     "run_dirs": SELECTED_RUNS,
    # },

}


# =========================
# 5. Acceptance-rate plot settings.
# =========================

Y_LABEL = "Acceptance Ratio"
TITLE = ""
OUTPUT_NAME = "acceptance_ratio_errorbar.pdf"
Y_LIM = (0.4, 1.05)


# =========================
# 6. Read acceptance rates from YAML.
# =========================

def load_resource_usage_yaml(file_path):
    """Read a single resource_usage.yaml and extract only time and acceptance_rate."""
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        print(f"[WARNING] Empty YAML file: {file_path}")
        return None

    records = data.get("records", [])

    if not records:
        print(f"[WARNING] No records found in: {file_path}")
        return None

    rows = []

    for item in records:
        time_value = item.get("time")
        acceptance_rate = item.get("acceptance_rate")

        if time_value is None or acceptance_rate is None:
            continue

        rows.append({
            "time": float(time_value),
            "acceptance_rate": float(acceptance_rate),
        })

    if not rows:
        print(f"[WARNING] No valid acceptance_rate records in: {file_path}")
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")

    return df


# =========================
# 7. Read multiple runs of one algorithm.
# =========================

def load_algorithm_runs(algo_name, algo_config):
    """Return a run-to-DataFrame mapping: {'1': df1, '2': df2, ...}."""
    runs = {}

    if algo_config["type"] == "multi_run":
        path_template = algo_config["path_template"]
        run_dirs = algo_config["run_dirs"]

        for run in run_dirs:
            file_path = path_template.format(run=run)
            df = load_resource_usage_yaml(file_path)

            if df is not None:
                runs[run] = df

    elif algo_config["type"] == "single_file":
        file_path = algo_config["file_path"]
        df = load_resource_usage_yaml(file_path)

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
    """Sample one run's acceptance-rate series at the requested times.
        method='previous' uses the latest prior record; 'linear' uses linear interpolation."""
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
    """error_type: 'std' for standard deviation, 'se' for standard error, 'ci95' for a 95% confidence interval.
        Return time, mean, error."""
    series_list = []

    for run_name, df in runs.items():
        df = df[df["time"] <= MAX_TIME]

        if df.empty:
            print(f"[WARNING] Run {run_name} has no data before MAX_TIME={MAX_TIME}")
            continue

        s = df.set_index("time")["acceptance_rate"]
        s.name = run_name

        sampled = resample_series_to_target_times(
            s,
            target_times=TARGET_TIMES,
            method=RESAMPLE_METHOD
        )
        sampled.name = run_name
        series_list.append(sampled)

    if not series_list:
        return None, None, None

    merged = pd.concat(series_list, axis=1)

    # Retain time points with at least one observed run value.
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
# 10. Plot acceptance rates with error bars.
# =========================

def plot_acceptance_rate_with_errorbar(error_type="ci95"):
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

    # Limit the x-axis to 8000.
    plt.xlim(0, MAX_TIME)

    # Match sampling times: 100, 500, 1000, 1500, ...
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
    plot_acceptance_rate_with_errorbar(
        error_type="ci95"   # Options: "std", "se", "ci95".
    )
