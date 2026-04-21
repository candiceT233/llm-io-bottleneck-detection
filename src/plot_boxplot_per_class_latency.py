"""
Boxplot of latency per class for each method.
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

STRATEGY_LABELS = {
    "rule_based": "Rule-Based",
    "classify_only": "Classify-Only",
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain-of-Thought",
}

STRATEGY_COLORS = {
    "rule_based": "#4C72B0",
    "classify_only": "#DD8452",
    "zero_shot": "#55A868",
    "few_shot": "#C44E52",
    "chain_of_thought": "#8172B2",
}

STRATEGY_ORDER = ["rule_based", "classify_only", "zero_shot", "few_shot", "chain_of_thought"]

CLASSES = [
    "checkpointing_overhead",
    "compute_bound",
    "data_skew",
    "io_interference",
    "lock_contention",
    "metadata_contention",
    "network_io_bottleneck",
    "read_bandwidth_saturation",
    "serialized_io",
    "staging_inefficiency",
    "storage_bandwidth_saturation",
]

CLASS_ABBREV = {
    "checkpointing_overhead": "Checkpoint",
    "compute_bound": "Compute",
    "data_skew": "Data Skew",
    "io_interference": "IO Interf.",
    "lock_contention": "Lock",
    "metadata_contention": "Metadata",
    "network_io_bottleneck": "Network IO",
    "read_bandwidth_saturation": "Read BW",
    "serialized_io": "Serial IO",
    "staging_inefficiency": "Staging",
    "storage_bandwidth_saturation": "Storage BW",
}


def load_results():
    data = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        with open(path) as f:
            d = json.load(f)
        strategy = d["strategy"]
        data[strategy] = d["results"]
    return data


def plot_latency_boxplot(ax, results, title, color):
    data_by_class = []
    labels = []
    for c in CLASSES:
        durations = [r["duration_s"] for r in results if r["ground_truth"] == c]
        data_by_class.append(durations)
        labels.append(CLASS_ABBREV[c])

    bp = ax.boxplot(
        data_by_class,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Zoom y-axis to IQR region so boxes aren't squished by outliers.
    # Compute the 5th–95th percentile across all data points to set the limit.
    all_vals = [v for group in data_by_class for v in group]
    if all_vals:
        q05 = np.percentile(all_vals, 5)
        q95 = np.percentile(all_vals, 95)
        margin = (q95 - q05) * 0.3 or 0.05
        ax.set_ylim(max(0, q05 - margin), q95 + margin + 0.35)

    ax.set_xticks(range(1, len(CLASSES) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Latency (s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def main():
    all_results = load_results()

    # --- Latency boxplots: one PDF per strategy ---
    for strategy in STRATEGY_ORDER:
        if strategy not in all_results:
            continue
        results = all_results[strategy]
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_latency_boxplot(
            ax, results,
            f"Latency per Class — {STRATEGY_LABELS[strategy]}",
            STRATEGY_COLORS[strategy],
        )
        plt.tight_layout()
        fname = f"05_latency_boxplot_{strategy}.pdf"
        fig.savefig(os.path.join(PLOTS_DIR, fname))
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()
