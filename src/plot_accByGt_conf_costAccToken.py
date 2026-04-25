"""
Generates analysis plots from the JSON result files in data/results/.
Plots: accuracy by gt_confidence (one chart per model), cost vs accuracy Pareto (all models combined).
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

# Marker per model so the Pareto chart is readable without colour clash
MODEL_MARKERS = {
    "gpt-4.1-mini": "o",
    "claude-haiku-4-5": "s",
}
MODEL_LABELS = {
    "gpt-4.1-mini": "GPT-4.1-mini",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}


def load_all_results():
    """Return dict keyed by (model, strategy) → result dict."""
    datasets = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        model = d.get("model", "unknown")
        strategy = d["strategy"]
        key = (model, strategy)
        # keep the most recent file if duplicates exist (sorted gives oldest first)
        datasets[key] = d
    return datasets


def models_in(datasets):
    return sorted(set(m for m, _ in datasets))


# ── Plot 2: Accuracy by gt_confidence — one chart per model ───────────────────

def plot_accuracy_by_gt_confidence(datasets):
    tiers = ["high", "medium", "low"]
    strategies = list(STRATEGY_LABELS.keys())

    for model in models_in(datasets):
        model_data = {s: datasets[(model, s)] for s in strategies if (model, s) in datasets}
        if not model_data:
            continue

        x = np.arange(len(tiers))
        width = 0.15
        n = len(model_data)
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        _, ax = plt.subplots(figsize=(10, 6))

        for i, strategy in enumerate(model_data):
            accs = []
            for tier in tiers:
                subset = [r for r in model_data[strategy]["results"] if r.get("gt_confidence") == tier]
                acc = sum(r["correct"] for r in subset) / len(subset) * 100 if subset else float("nan")
                accs.append(acc)
            bars = ax.bar(x + offsets[i], accs, width,
                          label=STRATEGY_LABELS.get(strategy, strategy),
                          color=STRATEGY_COLORS.get(strategy, None))
            for bar, acc in zip(bars, accs):
                if not np.isnan(acc):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{acc:.1f}%", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tiers])
        ax.set_xlabel("Ground Truth Confidence Tier")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 115)
        ax.set_title(f"Accuracy by GT Confidence Tier — {MODEL_LABELS.get(model, model)}",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower left")
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.tight_layout()

        safe_model = model.replace("/", "-")
        out = os.path.join(PLOTS_DIR, f"02_accuracy_by_gt_confidence_{safe_model}.pdf")
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")


# ── Plot 4: Cost vs Accuracy Pareto — all models combined ─────────────────────

def plot_cost_accuracy_pareto(datasets):
    points = []
    for (model, strategy), d in datasets.items():
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
        points.append((model, strategy, avg_tokens, macro_f1))

    _, ax = plt.subplots(figsize=(10, 6))

    # track which model labels have been added to the legend already
    legend_models = set()

    for model, strategy, tokens, f1 in points:
        color = STRATEGY_COLORS.get(strategy, "gray")
        marker = MODEL_MARKERS.get(model, "^")
        model_label = MODEL_LABELS.get(model, model)

        # one legend entry per model (shape legend)
        legend_entry = model_label if model not in legend_models else None
        legend_models.add(model)

        ax.scatter(tokens, f1, s=130, color=color, marker=marker,
                   label=legend_entry, zorder=5)
        ax.annotate(f"{STRATEGY_LABELS.get(strategy, strategy)}\n({model_label})",
                    (tokens, f1), textcoords="offset points",
                    xytext=(8, 4), fontsize=7.5, color=color)

    # strategy colour legend
    strategy_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=9, label=STRATEGY_LABELS[s])
        for s, c in STRATEGY_COLORS.items()
    ]
    # model shape legend
    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS.get(m, "^"), color="gray",
                   markersize=9, linestyle="none", label=MODEL_LABELS.get(m, m))
        for m in sorted(models_in(datasets))
    ]
    ax.legend(handles=strategy_handles + model_handles, fontsize=8,
              loc="lower right", ncol=2)

    ax.set_xlabel("Avg Tokens / Snapshot", fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Cost vs. Accuracy Pareto — All Models\n(left = cheaper, up = better)",
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
    models = models_in(datasets)
    print(f"Loaded {len(datasets)} result files across models: {models}")

    plot_accuracy_by_gt_confidence(datasets)
    plot_cost_accuracy_pareto(datasets)

    print(f"\nAll plots saved to {PLOTS_DIR}")
