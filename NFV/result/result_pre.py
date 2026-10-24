#!/usr/bin/env python3
"""Plot initialization-quality metrics with 95% confidence intervals.

Each YAML record corresponds to one VNR deployment attempt.  Records are
grouped automatically by ``init_mode`` (e.g., ``random`` and
``temperature_guided``), so the same script can compare the two initialization
methods after their YAML files are listed in INPUT_YAML_FILES.
"""

from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import t


# Replace these paths with your two experiment-result files.  It is also valid
# to put both methods into one YAML file; in that case only list that one file.
INPUT_YAML_FILES = [
    str(_PROJECT_ROOT / 'NFV/result_storage/EO/pre/initialization_ablation_metrics.yaml'),
    str(_PROJECT_ROOT / 'NFV/result_storage/EO/pre/initialization_ablation_metrics2.yaml'),
]

OUTPUT_DIR = Path(str(_PROJECT_ROOT / 'NFV/figures'))
SUMMARY_FILE = OUTPUT_DIR / "initialization_ablation_95ci_summary.yaml"

CONFIDENCE_LEVEL = 0.95
# Fitness values above this threshold are penalty-dominated outliers and are
# excluded from the two fitness plots/statistics.  The feasible-solution rate
# is never filtered.
FITNESS_OUTLIER_THRESHOLD = 10000.0
BAR_WIDTH = 0.42
BAR_CENTER_SPACING = 0.62
# ``capsize`` is the half-length of each horizontal error-bar cap, in points.
# It is chosen so the cap span is visually close to the narrow bar width.
ERROR_BAR_CAPSIZE = 28

METRICS = {
    "initial_average_fitness": {
        "title": "Initial Average Fitness",
        "ylabel": "Fitness ",
        # Failed mappings do not have a meaningful average fitness (null).
        "include_failed": False,
    },
    "initial_best_fitness": {
        "title": "Initial Best Fitness",
        "ylabel": "Fitness ",
        # Failed mappings do not have a meaningful best fitness (null).
        "include_failed": False,
    },
    "initial_feasible_solution_rate": {
        "title": "Initial Feasible-Solution Rate",
        "ylabel": "Feasible-solution rate",
        # Retain failed mappings: their recorded rate is correctly zero.
        "include_failed": True,
    },
}

DISPLAY_NAMES = {
    "random": "Random initialization",
    "temperature_guided": "GNT initialization",
}


def read_records(paths):
    """Read and merge all YAML records, rejecting nonexistent input paths."""
    records = []
    missing = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        records.extend(data.get("records", []))

    if missing:
        print("Warning: the following input files were not found:")
        for path in missing:
            print(f"  {path}")
    if not records:
        raise FileNotFoundError(
            "No YAML records were read. Please update INPUT_YAML_FILES."
        )
    return records


def is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean_and_ci95(values):
    """Return mean and two-sided Student-t 95% CI half-width."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    half_width = float(t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1) * sem)
    return mean, half_width


def collect_statistics(records):
    modes = []
    for record in records:
        mode = record.get("init_mode", "unknown")
        if mode not in modes:
            modes.append(mode)

    summary = {mode: {} for mode in modes}
    for mode in modes:
        mode_records = [r for r in records if r.get("init_mode", "unknown") == mode]
        for metric, settings in METRICS.items():
            values = []
            excluded_outlier_count = 0
            for record in mode_records:
                if not settings["include_failed"] and not record.get("mapping_success", False):
                    continue
                value = record.get(metric)
                if is_number(value):
                    value = float(value)
                    if "fitness" in metric and value > FITNESS_OUTLIER_THRESHOLD:
                        excluded_outlier_count += 1
                        continue
                    values.append(value)

            if not values:
                raise ValueError(f"No valid '{metric}' values for init_mode='{mode}'.")
            mean, ci95 = mean_and_ci95(values)
            summary[mode][metric] = {
                "n": len(values),
                "mean": mean,
                "ci95_half_width": ci95,
                "excluded_outlier_count": excluded_outlier_count,
            }
    return modes, summary


def plot_statistics(modes, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for metric, settings in METRICS.items():
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        means = [summary[mode][metric]["mean"] for mode in modes]
        ci95 = [summary[mode][metric]["ci95_half_width"] for mode in modes]
        x = np.arange(len(modes)) * BAR_CENTER_SPACING

        bars = ax.bar(
            x, means, width=BAR_WIDTH, yerr=ci95, capsize=ERROR_BAR_CAPSIZE,
            color=[colors[i % len(colors)] for i in range(len(modes))],
            edgecolor="black", linewidth=0.8,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.set_title(settings["title"])
        ax.set_ylabel(settings["ylabel"])
        ax.set_xticks(x, [DISPLAY_NAMES.get(mode, mode) for mode in modes], rotation=0)
        # Plot the original values and use zero as the baseline.
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        fig.tight_layout()
        output_file = OUTPUT_DIR / f"{metric}_95ci.pdf"
        fig.savefig(output_file, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {output_file}")


def main():
    records = read_records(INPUT_YAML_FILES)
    modes, summary = collect_statistics(records)
    plot_statistics(modes, summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "confidence_level": CONFIDENCE_LEVEL,
                "fitness_outlier_threshold": FITNESS_OUTLIER_THRESHOLD,
                "metrics": summary,
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Numerical summary saved to: {SUMMARY_FILE}")
    for mode in modes:
        for metric in ("initial_average_fitness", "initial_best_fitness"):
            result = summary[mode][metric]
            print(
                f"{mode} | {metric}: n={result['n']}, "
                f"excluded_outliers={result['excluded_outlier_count']}"
            )


if __name__ == "__main__":
    main()