"""Pure-Python evaluation metrics for the benchmark harness.

No sklearn / matplotlib / numpy dependencies — these run inside the
HF Space `cpu-basic` runtime where every MB matters. The harness only
needs:

- :func:`accuracy`        — top-1 accuracy over a list of predictions.
- :func:`per_class`       — precision + recall per class label.
- :func:`confusion_matrix` — nested dict ``{true_label: {pred_label: count}}``.
- :func:`to_jsonable`     — convert :class:`BenchResult` → JSON-safe dict.

All inputs are plain Python lists / ints / floats. Confusion matrix
returns a 2-D ``list[list[int]]`` so it's easy to render into either a
markdown table or a PNG later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence


LABEL_NAMES_DEFAULT: List[str] = ["happy", "neutral", "sad", "angry"]


@dataclass
class BenchResult:
    model_id: str
    label_set: List[str]
    num_samples: int
    accuracy: float
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1: Dict[str, float] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    notes: str = ""


def accuracy(preds: Sequence[int], truths: Sequence[int]) -> float:
    if not truths:
        return 0.0
    return sum(1 for p, t in zip(preds, truths) if p == t) / len(truths)


def per_class(
    preds: Sequence[int],
    truths: Sequence[int],
    n_classes: int,
) -> Dict[str, Dict[str, float]]:
    """Return ``{class_name: {"precision": p, "recall": r, "f1": f}}``.

    Undefined (zero support) classes get ``0.0`` rather than NaN.
    """
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    for p, t in zip(preds, truths):
        if p == t:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    out: Dict[str, Dict[str, float]] = {}
    for c in range(n_classes):
        denom_p = tp[c] + fp[c]
        denom_r = tp[c] + fn[c]
        p = tp[c] / denom_p if denom_p else 0.0
        r = tp[c] / denom_r if denom_r else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        name = LABEL_NAMES_DEFAULT[c] if c < len(LABEL_NAMES_DEFAULT) else f"class_{c}"
        out[name] = {"precision": p, "recall": r, "f1": f1}
    return out


def confusion_matrix(
    preds: Sequence[int],
    truths: Sequence[int],
    n_classes: int,
) -> List[List[int]]:
    cm: List[List[int]] = [[0] * n_classes for _ in range(n_classes)]
    for p, t in zip(preds, truths):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    return cm


def evaluate(
    model_id: str,
    preds: Sequence[int],
    truths: Sequence[int],
    n_classes: int = 4,
    label_names: Sequence[str] = LABEL_NAMES_DEFAULT,
    notes: str = "",
) -> BenchResult:
    acc = accuracy(preds, truths)
    pc = per_class(preds, truths, n_classes)
    cm = confusion_matrix(preds, truths, n_classes)
    macro_f1 = sum(d["f1"] for d in pc.values()) / max(n_classes, 1)

    # Weighted F1: weight each class by its support (column sum of CM rows).
    support_per_class: List[int] = [sum(cm[c]) for c in range(n_classes)]
    total = sum(support_per_class)
    if total > 0:
        weighted = sum(pc[label_names[c]]["f1"] * support_per_class[c]
                       for c in range(n_classes)) / total
    else:
        weighted = 0.0

    return BenchResult(
        model_id=model_id,
        label_set=list(label_names),
        num_samples=len(truths),
        accuracy=acc,
        precision={k: round(v["precision"], 4) for k, v in pc.items()},
        recall={k: round(v["recall"], 4) for k, v in pc.items()},
        f1={k: round(v["f1"], 4) for k, v in pc.items()},
        confusion=cm,
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted, 4),
        notes=notes,
    )


def to_jsonable(result: BenchResult) -> Dict:
    """Convert a :class:`BenchResult` to a JSON-safe dict.

    Numbers are rounded to 4 decimals for storage; raw counts (confusion
    matrix) are kept as plain ints.
    """
    return asdict(result)


def render_confusion_table(cm: List[List[int]], label_names: Sequence[str]) -> str:
    """Return a Markdown-formatted confusion matrix for ``docs/``."""
    if not cm:
        return "_(empty)_"
    header = "| true \\ pred | " + " | ".join(label_names) + " |"
    sep = "|" + "---|" * (len(label_names) + 1)
    rows = [header, sep]
    for i, row in enumerate(cm):
        name = label_names[i] if i < len(label_names) else f"cls_{i}"
        rows.append(f"| **{name}** | " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(rows)


__all__ = [
    "BenchResult",
    "accuracy",
    "per_class",
    "confusion_matrix",
    "evaluate",
    "to_jsonable",
    "render_confusion_table",
]
