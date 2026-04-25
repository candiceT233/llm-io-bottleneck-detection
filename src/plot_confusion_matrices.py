#!/usr/bin/env python3
"""Plot a confusion matrix for each classification method"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix

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

SHORT_LABELS = [
    "checkpoint",
    "compute",
    "data_skew",
    "io_interf",
    "lock_cont",
    "metadata",
    "network_io",
    "read_bw",
    "serial_io",
    "staging",
    "storage_bw",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "data", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    y_true = [r["ground_truth"] for r in data["results"]]
    y_pred = [r["predicted"] for r in data["results"]]
    return y_true, y_pred


def plot_confusion_matrix(cm: np.ndarray, title: str, ax: plt.Axes) -> None:
    n = len(CLASSES)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums == 0, 0.0, cm / row_sums)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2f")

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=SHORT_LABELS,
        yticklabels=SHORT_LABELS,
        title=title,
        ylabel="Ground truth",
        xlabel="Predicted",
    )
    ax.tick_params(axis="x", rotation=45)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if cm_norm[i, j] > thresh else "black",
            )


def main() -> None:
    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not json_files:
        print("No JSON result files found.")
        return

    for path in json_files:
        y_true, y_pred = load_results(path)
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        with open(path) as f:
            meta = json.load(f)
        strategy = meta["strategy"]
        model = meta["model"]

        fig, ax = plt.subplots(figsize=(8, 7))
        plot_confusion_matrix(cm, f"{model} — {strategy.replace('_', ' ').title()}", ax)
        fig.tight_layout()

        safe_model = model.replace("/", "-")
        out = os.path.join(PLOTS_DIR, f"confusion_matrix_{safe_model}_{strategy}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
