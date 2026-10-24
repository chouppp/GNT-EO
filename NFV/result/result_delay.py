from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Basic configuration
# ============================================================
BASE_DIR = str(_PROJECT_ROOT / 'NFV/result_storage')
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
OUTPUT_NAME = "optimizer_ablation_mean_comparison.pdf"

# "std": standard deviation; "se": standard error; "ci95": 95% CI
ERROR_TYPE = "ci95"

# Only final_fitness uses IQR filtering.
IQR_MULTIPLIER = 1.5

# Every run folder uses the unified filename: iteration_history.jsonl
ALGORITHMS = {
    "GNT-EO2": {
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
        "title": "Convergence Iteration",
        "ylabel": "Last Improvement Iteration",
    },
    "time_to_final_best_seconds": {
        "title": "Time to Final Best",
        "ylabel": "Time (s)",
    },
    "final_fitness": {
        "title": "Final Best Fitness",
        "ylabel": "Fitness Value",
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


def load_valid_records(file_path):
    """
    Read one JSONL file.

    Records with success=False, final_fitness=None,
    or non-finite final_fitness are ignored completely.
    """
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

            # Explicitly failed records are ignored.
            # GA/PSO logs without a success field are retained.
            if item.get("success") is False:
                continue

            final_fitness = item.get("final_fitness")

            # Ignore the whole record if final_fitness is null.
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
    """Extract finite numeric values of one metric from one run."""
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
    """
    Remove outliers according to the IQR rule:
    [Q1 - multiplier * IQR, Q3 + multiplier * IQR].

    Returns:
        filtered_values, removed_count
    """
    if len(values) < 4:
        return values, 0

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    filtered_values = values[
        (values >= lower_bound) &
        (values <= upper_bound)
    ]

    removed_count = len(values) - len(filtered_values)

    return filtered_values, removed_count


# ============================================================
# 3. Statistical calculation
# ============================================================
def compute_error(values, error_type):
    """Calculate error bars across repeated runs."""
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
    Calculate the mean metric value within every repeated run first.
    Then calculate the overall mean and confidence interval across runs.
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

                # Remove IQR outliers only for final fitness.
                if metric_name == "final_fitness":
                    values, removed_count = remove_iqr_outliers(values)

                    if removed_count > 0:
                        print(
                            f"  [INFO] Run {run}, final_fitness: "
                            f"removed {removed_count} IQR outlier(s)"
                        )

                if len(values) == 0:
                    print(
                        f"  [WARNING] Run {run} has no remaining "
                        f"valid {metric_name} values after filtering"
                    )
                    continue

                # Each repeated experiment has equal weight.
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

            print(
                f"  {metric_name}: mean={mean_value:.6f}, "
                f"error={error_value:.6f}, n={len(values)}"
            )

            summary[metric_name].append(
                (algo_name, mean_value, error_value, len(values))
            )

    return summary


# ============================================================
# 4. Plot three mean comparisons
# ============================================================
def plot_mean_comparison(summary):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))

    for ax, (metric_name, setting) in zip(axes, METRICS.items()):
        rows = [
            row
            for row in summary[metric_name]
            if np.isfinite(row[1])
        ]

        ax.set_title(setting["title"])
        ax.set_ylabel(setting["ylabel"])
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)

        if not rows:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_xticks([])
            continue

        names = [row[0] for row in rows]
        means = np.asarray([row[1] for row in rows], dtype=float)
        errors = np.asarray([row[2] for row in rows], dtype=float)
        counts = [row[3] for row in rows]

        positions = np.arange(len(names))
        colors = [ALGORITHMS[name]["color"] for name in names]

        bars = ax.bar(
            positions,
            means,
            yerr=errors,
            color=colors,
            width=0.58,
            capsize=5,
            edgecolor="black",
            linewidth=0.7,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(names)

        offset = max(
            float(np.max(errors)),
            float(np.max(means)) * 0.015,
        )

        for bar, mean_value, error_value, sample_count in zip(
            bars,
            means,
            errors,
            counts,
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean_value + error_value + offset,
                f"{mean_value:.3g}\n(n={sample_count})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.text(
        0.5,
        0.01,
        "Bars: mean; error bars: 95% confidence interval",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# ============================================================
# 5. Main program
# ============================================================
if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    statistics_summary = summarize_algorithms()
    figure = plot_mean_comparison(statistics_summary)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"\n[INFO] Figure saved to: {output_path}")