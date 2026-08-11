"""Render presentation-ready chart PNGs from real bench results.

Dev-only tool — not part of the deployed Space, not in requirements.txt.
Needs matplotlib: ``.venv/bin/pip install matplotlib`` (once, locally).

Reads ``bench/results/meralion.json`` (the actual measured BenchResult,
see :mod:`bench.metrics`) and writes two PNGs to ``bench/results/assets/``:

- ``confusion_matrix.png`` — true-label rows x predicted-label columns,
  annotated with row-normalized percentages.
- ``per_class_metrics.png`` — grouped bar chart of precision/recall/F1
  per class.

These replace the hand-typed ASCII bar charts in docs/PRESENTATION.md,
which drifted from the real numbers in bench/results/. Re-run this
after every new bench/run_meralion.py pass so the slides never go
stale again.

Run:

    .venv/bin/python3 bench/make_slide_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(__file__).parent / "results" / "meralion.json"
ASSETS_DIR = Path(__file__).parent / "results" / "assets"


def _load() -> dict:
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def make_confusion_matrix(result: dict, out_path: Path) -> None:
    labels = result["label_set"]
    cm = np.array(result["confusion"], dtype=float)
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{result['model_id']}\nconfusion matrix (n={result['num_samples']}, row %)",
        fontsize=11,
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            pct = row_pct[i, j]
            count = int(cm[i, j])
            color = "white" if pct > 55 else "black"
            ax.text(j, i, f"{pct:.0f}%\n({count})", ha="center", va="center",
                     color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_per_class_metrics(result: dict, out_path: Path) -> None:
    labels = result["label_set"]
    precision = [result["precision"][c] for c in labels]
    recall = [result["recall"][c] for c in labels]
    f1 = [result["f1"][c] for c in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(
        f"{result['model_id']} — per-class metrics (n={result['num_samples']})"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = _load()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    cm_path = ASSETS_DIR / "confusion_matrix.png"
    metrics_path = ASSETS_DIR / "per_class_metrics.png"

    make_confusion_matrix(result, cm_path)
    make_per_class_metrics(result, metrics_path)

    print(f"wrote {cm_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
