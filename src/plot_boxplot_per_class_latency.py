"""
Boxplot of latency per class for each strategy.
Both models (GPT-4.1-mini and Claude Haiku 4.5) are superposed within each strategy plot,
with the two models shown side-by-side for every class.
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

MODEL_COLORS = {
    "gpt-4.1-mini":    "#4C72B0",
    "claude-haiku-4-5": "#DD8452",
}
MODEL_LABELS = {
    "gpt-4.1-mini":    "GPT-4.1-mini",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}
MODEL_ORDER = ["gpt-4.1-mini", "claude-haiku-4-5"]


def load_results():
    """Return dict keyed by (model, strategy) → list of result dicts."""
    data = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        key = (d.get("model", "unknown"), d["strategy"])
        data[key] = d["results"]
    return data


def _draw_boxes(ax, positions, data_by_class, color, width=0.3):
    bp = ax.boxplot(
        data_by_class,
        positions=positions,
        widths=width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0),
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    return bp


def plot_strategy(strategy, all_results):
    n_classes = len(CLASSES)
    # base x positions for each class (1-indexed, spread by 2 so models don't overlap)
    base = np.arange(1, n_classes + 1) * 2.0
    gap = 0.35   # half-distance between the two model boxes

    fig, ax = plt.subplots(figsize=(22, 8))

    models_present = [m for m in MODEL_ORDER if (m, strategy) in all_results]

    all_vals = []
    legend_handles = []

    for i, model in enumerate(models_present):
        results = all_results[(model, strategy)]
        data_by_class = [
            [r["duration_s"] for r in results if r["ground_truth"] == c]
            for c in CLASSES
        ]
        offset = (i - (len(models_present) - 1) / 2) * gap * 2
        positions = base + offset
        color = MODEL_COLORS.get(model, "gray")
        _draw_boxes(ax, positions, data_by_class, color)
        all_vals.extend(v for group in data_by_class for v in group)
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=6, alpha=0.7,
                       label=MODEL_LABELS.get(model, model))
        )

    # y-axis zoom to 5th–95th percentile so outliers don't squish the boxes
    if all_vals:
        q05, q95 = np.percentile(all_vals, 5), np.percentile(all_vals, 95)
        margin = (q95 - q05) * 0.3 or 0.05
        ax.set_ylim(max(0, q05 - margin), q95 + margin + 0.35)

    ax.set_xticks(base)
    ax.set_xticklabels([CLASS_ABBREV[c] for c in CLASSES], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Latency (s)", fontsize=11)
    ax.set_title(f"Latency per Class — {STRATEGY_LABELS[strategy]}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    plt.tight_layout()
    fname = f"05_latency_boxplot_{strategy}.pdf"
    fig.savefig(os.path.join(PLOTS_DIR, fname))
    plt.close(fig)
    print(f"Saved {fname}")


def main():
    all_results = load_results()
    for strategy in STRATEGY_ORDER:
        if not any((m, strategy) in all_results for m in MODEL_ORDER):
            continue
        plot_strategy(strategy, all_results)


if __name__ == "__main__":
    main()
