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
# Maximum displayed x-axis time.
MAX_TIME = 6000

# Configure the first time point separately.
FIRST_TIME = 1000

# Subsequent samples start at 500 and occur every 500 time units.
TIME_INTERVAL = 500

# Final time axis: 100, 500, 1000, 1500, ...
TARGET_TIMES = np.concatenate((
    np.array([FIRST_TIME]),
    np.arange(TIME_INTERVAL, MAX_TIME + 1, TIME_INTERVAL)
))

# Sampling method:
# "previous": use the latest mean network temperature before the target time.
# "linear": linear interpolation.
RESAMPLE_METHOD = "previous"


# =========================
# 4. Temperature-domain settings.
# =========================
# Average over four domains here: domain_id = 0, 1, 2, 3.
# For five domains, use [0, 1, 2, 3, 4].
DOMAIN_IDS = [0, 1, 2, 3,4]

# Clipping limits for extreme temperatures.
# Clip values above 80000 to 80000 and values below -10000 to -10000.
TEMP_CLIP_MIN = -10000.0
TEMP_CLIP_MAX = 10000.0


# =========================
# 5. Algorithm configuration interface.
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
        "marker": "o",
    },

    "GA-STP": {
        "type": "multi_run",
        "path_template": os.path.join(BASE_DIR, "GA", "{run}", "resource_usage.yaml"),
        "run_dirs": SELECTED_RUNS,
        "marker": "^",
    },



    # Use the following format to add algorithms.
    # "D-ViNE": {
    #     "type": "multi_run",
    #     "path_template": os.path.join(BASE_DIR, "D-ViNE", "{run}", "resource_usage.yaml"),
    #     "run_dirs": SELECTED_RUNS,
    # },
}


# =========================
# 6. Mean network-temperature plot settings.
# =========================

Y_LABEL = "Average GNT"
TITLE = ""
OUTPUT_NAME = "average_network_temperature_errorbar.pdf"

# Temperature can be negative or highly variable; leave the y-axis unrestricted by default.
# For example, set Y_LIM = (-1e6, 1e6) to restrict the range.
Y_LIM = (-4000, 3000)


# =========================
# 7. Helper: calculate mean network temperature for one record.
# =========================

def clip_temperature(t_value):
    """Clip temperatures to [TEMP_CLIP_MIN, TEMP_CLIP_MAX]."""
    return float(np.clip(float(t_value), TEMP_CLIP_MIN, TEMP_CLIP_MAX))


def calc_average_temperature(domain_temperatures, domain_ids=DOMAIN_IDS):
    """Calculate mean network temperature for the specified domain_ids.
        Treat None/null T and missing domains as zero; clip extremes to [TEMP_CLIP_MIN, TEMP_CLIP_MAX]."""
    if not domain_temperatures:
        return 0.0

    temp_dict = {}
    for domain in domain_temperatures:
        domain_id = domain.get("domain_id")
        t_value = domain.get("T")

        if domain_id is None:
            continue

        if t_value is None:
            t_value = 0.0

        temp_dict[int(domain_id)] = clip_temperature(t_value)

    values = [temp_dict.get(domain_id, 0.0) for domain_id in domain_ids]
    return float(np.mean(values))


# =========================
# 8. Read mean network temperature from YAML.
# =========================

def load_resource_usage_yaml(file_path):
    """Read time and average_temperature from one resource_usage.yaml file."""
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
        domain_temperatures = item.get("domain_temperatures")

        # Support alternative field names.
        if domain_temperatures is None:
            domain_temperatures = item.get("domains")
        if domain_temperatures is None:
            domain_temperatures = item.get("temperature_domains")

        if time_value is None:
            continue

        avg_temperature = calc_average_temperature(domain_temperatures)

        rows.append({
            "time": float(time_value),
            "average_temperature": avg_temperature,
        })

    if not rows:
        print(f"[WARNING] No valid temperature records in: {file_path}")
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")

    return df


# =========================
# 9. Read multiple runs of one algorithm.
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
# 10. Sample at the specified time points.
# =========================

def resample_series_to_target_times(series, target_times, method="previous"):
    """Sample one run's mean-temperature series at the specified times.
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
# 11. Calculate means and error bars.
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

        s = df.set_index("time")["average_temperature"]
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
# 12. Plot mean network temperature with error bars.
# =========================

def plot_average_temperature_with_errorbar(error_type="ci95"):
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

    # Displayed x-axis range.
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
# 13. Main program.
# =========================

if __name__ == "__main__":
    plot_average_temperature_with_errorbar(
        error_type="ci95"   # Options: "std", "se", "ci95".
    )