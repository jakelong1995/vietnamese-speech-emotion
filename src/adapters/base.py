"""Abstract base adapter for emotion-recognition models.

All concrete adapters (MERaLiON-SER-v1 today, future additions) expose
the same interface to :mod:`src.inference`, so the UI never needs to
know which backend is live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelInfo:
    provider: str
    model_id: str
    sample_rate: int
    labels: List[str]
    device: str
    confidence_available: bool
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawPrediction:
    label: str
    confidence: Optional[float]
    class_scores: Optional[Dict[str, float]]
    auxiliary: Dict[str, Any] = field(default_factory=dict)
    full_distribution_available: bool = False


class BaseAdapter:
    """Common interface for an emotion-recognition backend."""

    #: Subclasses MUST set ``model_id`` (HF repo id or local path).
    model_id: str = ""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        )

    # ---- lifecycle ----
    def load(self) -> None:
        """Load weights, allocate buffers, run warm-up if applicable."""

    def unload(self) -> None:
        """Release all heavy state. Default = no-op."""

    # ---- inference ----
    def predict(self, waveform: np.ndarray, sample_rate: int) -> RawPrediction:
        """Run the model and return a ``RawPrediction``.

        Implementations MUST:
          * not raise on out-of-vocabulary labels — return ``unknown`` instead,
          * set ``full_distribution_available`` correctly (True iff
            ``class_scores`` are real softmax outputs from the model).
        """
        raise NotImplementedError

    # ---- introspection ----
    def get_model_info(self) -> ModelInfo:  # pragma: no cover - trivial
        return ModelInfo(
            provider=self.__class__.__name__,
            model_id=self.model_id,
            sample_rate=getattr(self, "sample_rate", 16000),
            labels=list(getattr(self, "labels", [])),
            device="cpu",
            confidence_available=True,
            extra={},
        )


__all__ = ["BaseAdapter", "ModelInfo", "RawPrediction"]
