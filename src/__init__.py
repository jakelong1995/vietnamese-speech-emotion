"""Vietnamese speech emotion recognition — single-model Gradio Space.

Built on :class:`src.adapters.meralion.MeralionAdapter`
(`MERaLiON/MERaLiON-SER-v1`). The public surface is intentionally
small — only the shared types and exception classes are re-exported.

Public surface:
- :class:`src.adapters.base.ModelInfo` / :class:`RawPrediction` — shared types
- :class:`src.exceptions.AppError` (and subclasses) — error surface
- :func:`src.inference.predict` — Gradio-facing entry point
"""

from .adapters.base import ModelInfo, RawPrediction
from .exceptions import (
    AppError,
    AudioTooShortError,
    InvalidAudioError,
    ModelLoadFailedError,
)

__all__ = [
    "ModelInfo",
    "RawPrediction",
    "AppError",
    "AudioTooShortError",
    "InvalidAudioError",
    "ModelLoadFailedError",
]
