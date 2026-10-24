from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


# ============================================================
# 1. Basic configuration
# ============================================================
BASE_DIR = str(_PROJECT_ROOT / 'NFV/result_storage')
OUTPUT_DIR = str(_PROJECT_ROOT / 'NFV/figures')
ERROR_TYPE = "ci95"  # "std", "se", or "ci95"
IQR_MULTIPLIER = 1.5

# The JSONL file under every run folder has the unified name:
# iteration_history.jsonl
ALGORITHMS = {
    "GNT-EO": {
        "folder": "EO_R",
        "run_dirs": ["1", "2", "3", "4", "5","6","7"],
        "color": "#1f77b4",
    },
    "GNT-GA": {
        "folder": "GA_R",
        "run_dirs": ["1", "2", "3", "4", "5","6","7"],
        "color": "#ff7f0e",
    },
    "GNT-PSO": {
        "folder": "PSO_R",
        "run_dirs": ["1", "2", "3", "4", "5","6","7"],
        "color": "#2ca02c",
    },
}

METRICS = {
    "last_improvement_iteration": {
        "title": "",
        "ylabel": "Last Improvement Iteration",
        "output_name": "convergence_iteration_comparison.pdf",
    },
    "time_to_final_best_seconds": {
        "title": "",
        "ylabel": "Time (s)",
        "output_name": "time_to_final_best_comparison.pdf",
    },
    "final_fitness": {
        "title": "",
        "ylabel": "Fitness Value",
        "output_name": "final_best_fitness_comparison.pdf",
    },
}


# ============================================================
# 2. Read JSONL files
# ============================================================


def get_history_path(algo_folder, run):
    """Return the exact JSONL path for one algorithm run."""
    return os.path.join(
        BASE_DIR,
        algo_folder,
        str(run),
        "iteration_history.jsonl",
    )


def get_resource_usage_path(algo_folder, run):
    """Return the resource usage YAML path for one algorithm run."""
    return os.path.join(
        BASE_DIR,
        algo_folder,
        str(run),
        "resource_usage.yaml",
    )


def load_valid_records(file_path):
    """Read one JSONL file and remove invalid records."""
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARNING] Invalid JSON at line {line_no}: {file_path}")
                continue

            # GA/PSO logs may have no success field; retain them.
            # Explicitly failed records are excluded.
            if item.get("success") is False:
                continue

            # Ignore the whole record when final_fitness is null.
            final_fitness = item.get("final_fitness")
            if final_fitness is None:
                continue

            try:
                final_fitness = float(final_fitness)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(final_fitness):
                continue

            records.append(item)

    return records


def extract_metric_values(records, metric_name):
    """Extract all finite numeric values of one metric from one run."""
    values = []

    for item in records:
        value = item.get(metric_name)

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            values.append(value)

    return np.asarray(values, dtype=float)


def remove_iqr_outliers(values, multiplier=IQR_MULTIPLIER):
    """Remove values outside [Q1 - 1.5IQR, Q3 + 1.5IQR]."""
    if len(values) < 4:
        return values, 0

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    filtered = values[
        (values >= q1 - multiplier * iqr)
        & (values <= q3 + multiplier * iqr)
    ]

    return filtered, len(values) - len(filtered)


def load_final_acceptance_rate(yaml_path):
    """Return acceptance_rate at the final time slot of one run."""
    if not os.path.isfile(yaml_path):
        print(f"[WARNING] File not found: {yaml_path}")
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(f"[WARNING] YAML read failed: {yaml_path}\n{exc}")
        return None

    valid_rows = []

    for item in data.get("records", []):
        try:
            time_value = float(item.get("time"))
            acceptance_rate = float(item.get("acceptance_rate"))
        except (TypeError, ValueError):
            continue

        if np.isfinite(time_value) and np.isfinite(acceptance_rate):
            valid_rows.append((time_value, acceptance_rate))

    if not valid_rows:
        print(f"[WARNING] No valid acceptance_rate in: {yaml_path}")
        return None

    return max(valid_rows, key=lambda row: row[0])[1]


# ============================================================
# 3. Statistical calculation
# ============================================================
def compute_error(values, error_type):
    """Calculate the error bar across repeated runs."""
    if len(values) <= 1:
        return 0.0

    std = np.std(values, ddof=1)

    if error_type == "std":
        return std

    if error_type == "se":
        return std / np.sqrt(len(values))

    if error_type == "ci95":
        return 1.96 * std / np.sqrt(len(values))

    raise ValueError(f"Unknown ERROR_TYPE: {error_type}")


def summarize_algorithms():
    """
    First calculate the mean of each metric within each run, then calculate
    the overall mean and error bar across repeated runs.
    """
    summary = {metric: [] for metric in METRICS}

    for algo_name, config in ALGORITHMS.items():
        run_means = {metric: [] for metric in METRICS}

        for run in config["run_dirs"]:
            file_path = get_history_path(config["folder"], run)

            if not os.path.isfile(file_path):
                print(f"[WARNING] File not found: {file_path}")
                continue

            records = load_valid_records(file_path)
            print(f"[INFO] {algo_name}, run {run}: {len(records)} valid records")

            for metric_name in METRICS:
                values = extract_metric_values(records, metric_name)

                if len(values) == 0:
                    print(f"  [WARNING] Run {run} has no valid {metric_name}")
                    continue

                if metric_name == "final_fitness":
                    values, removed_count = remove_iqr_outliers(values)

                    if removed_count > 0:
                        print(
                            f"  [INFO] Run {run}, final_fitness: "
                            f"removed {removed_count} IQR outlier(s)"
                        )

                    if len(values) == 0:
                        continue

                # Each run contributes one value, preventing runs with more
                # JSONL records from receiving a larger weight.
                run_means[metric_name].append(float(np.mean(values)))

        print(f"\n{algo_name}")

        for metric_name in METRICS:
            values = np.asarray(run_means[metric_name], dtype=float)

            if len(values) == 0:
                print(f"  [WARNING] No valid {metric_name} values")
                summary[metric_name].append((algo_name, np.nan, np.nan, 0))
                continue

            mean_value = float(np.mean(values))
            error_value = float(compute_error(values, ERROR_TYPE))

            print(f"  {metric_name} run means: {values}")
            print(
                f"  {metric_name}: mean={mean_value:.6f}, "
                f"error={error_value:.6f}, n={len(values)}"
            )

            summary[metric_name].append(
                (algo_name, mean_value, error_value, len(values))
            )

    return summary


def print_average_acceptance_rates():
    """Print final acceptance ratio mean and error bar across runs."""
    print("\n" + "=" * 60)
    print("Final Acceptance Ratio Summary")
    print("=" * 60)

    for algo_name, config in ALGORITHMS.items():
        values = []

        for run in config["run_dirs"]:
            yaml_path = get_resource_usage_path(config["folder"], run)
            acceptance_rate = load_final_acceptance_rate(yaml_path)

            if acceptance_rate is not None:
                values.append(acceptance_rate)

        values = np.asarray(values, dtype=float)

        if len(values) == 0:
            print(f"{algo_name}: no valid acceptance-rate data")
            continue

        print(
            f"{algo_name}: mean acceptance ratio = {np.mean(values):.4f} "
            f"± {compute_error(values, ERROR_TYPE):.4f} (n={len(values)})"
        )


# ============================================================
# 4. Plot three independent figures
# ============================================================
def plot_single_metric(metric_name, metric_rows):
    """Create one independent PDF figure for one metric."""
    setting = METRICS[metric_name]

    rows = [row for row in metric_rows if np.isfinite(row[1])]

    if not rows:
        print(f"[WARNING] No valid data for figure: {metric_name}")
        return

    names = [row[0] for row in rows]
    means = np.asarray([row[1] for row in rows], dtype=float)
    errors = np.asarray([row[2] for row in rows], dtype=float)

    positions = np.arange(len(names))
    colors = [ALGORITHMS[name]["color"] for name in names]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    ax.bar(
        positions,
        means,
        yerr=errors,
        color=colors,
        width=0.58,
        capsize=5,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_title(setting["title"], fontsize=16)
    ax.set_ylabel(setting["ylabel"], fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)

    if metric_name == "final_fitness":
        ax.set_ylim(bottom=0)


    fig.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, setting["output_name"])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"[INFO] Figure saved to: {output_path}")


# ============================================================
# 5. Main program
# ============================================================
if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print_average_acceptance_rates()

    statistics_summary = summarize_algorithms()

    for metric_name in METRICS:
        plot_single_metric(
            metric_name,
            statistics_summary[metric_name],
        )