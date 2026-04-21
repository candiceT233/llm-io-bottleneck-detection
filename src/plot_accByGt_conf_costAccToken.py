"""
Generates analysis plots from the JSON result files in data/results/.
Plots: accuracy by gt_confidence, cost vs accuracy Pareto.
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


def load_all_results():
    datasets = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        strategy = d["strategy"]
        datasets[strategy] = d
    return datasets


def all_classes(datasets):
    classes = set()
    for d in datasets.values():
        for r in d["results"]:
            classes.add(r["ground_truth"])
    return sorted(classes)


# ── Plot 2: Accuracy by gt_confidence ─────────────────────────────────────────

def plot_accuracy_by_gt_confidence(datasets):
    tiers = ["high", "medium", "low"]
    strategies = list(datasets.keys())
    x = np.arange(len(tiers))
    width = 0.15
    offsets = np.linspace(-(len(strategies) - 1) / 2, (len(strategies) - 1) / 2, len(strategies)) * width

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, strategy in enumerate(strategies):
        accs = []
        for tier in tiers:
            subset = [r for r in datasets[strategy]["results"] if r.get("gt_confidence") == tier]
            acc = sum(r["correct"] for r in subset) / len(subset) * 100 if subset else float("nan")
            accs.append(acc)
        bars = ax.bar(x + offsets[i], accs, width, label=STRATEGY_LABELS.get(strategy, strategy),
                      color=STRATEGY_COLORS.get(strategy, None))
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{acc:.2f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tiers])
    ax.set_xlabel("Ground Truth Confidence Tier")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Accuracy by Ground Truth Confidence Tier", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "02_accuracy_by_gt_confidence.pdf")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Plot 4: Cost vs Accuracy  ───────────────────────────────────────────

def plot_cost_accuracy_pareto(datasets):
    points = []
    for strategy, d in datasets.items():
        results = d["results"]
        classes = sorted(set(r["ground_truth"] for r in results))
        f1s = []
        for cls in classes:
            tp = sum(1 for r in results if r["ground_truth"] == cls and r["predicted"] == cls)
            fp = sum(1 for r in results if r["ground_truth"] != cls and r["predicted"] == cls)
            fn = sum(1 for r in results if r["ground_truth"] == cls and r["predicted"] != cls)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
            f1s.append(f1)
        macro_f1 = np.mean(f1s)
        avg_tokens = np.mean([r["token_usage"]["total_tokens"] for r in results])
        points.append((strategy, avg_tokens, macro_f1))

    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy, tokens, f1 in points:
        color = STRATEGY_COLORS.get(strategy, "gray")
        ax.scatter(tokens, f1, s=120, color=color, zorder=5)
        ax.annotate(STRATEGY_LABELS.get(strategy, strategy),
                    (tokens, f1), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, color=color)

    ax.set_xlabel("Avg Tokens / Snapshot", fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Cost vs. Accuracy Pareto\n(lower-left = cheaper, upper-left = better)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    ax.annotate("", xy=(0.05, 0.95), xytext=(0.15, 0.85), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.text(0.05, 0.97, "ideal", transform=ax.transAxes, color="green", fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "04_cost_accuracy_pareto.pdf")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    datasets = load_all_results()
    print(f"Loaded strategies: {list(datasets.keys())}")

    plot_accuracy_by_gt_confidence(datasets)
    plot_cost_accuracy_pareto(datasets)

    print(f"\nAll plots saved to {PLOTS_DIR}")
