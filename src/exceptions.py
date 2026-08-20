"""Domain exceptions for the single-model Streamlit app."""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base application error."""

    code: str = "APP_ERROR"
    recoverable: bool = True

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelLoadFailedError(AppError):
    code = "MODEL_LOAD_FAILED"


class InvalidAudioError(AppError):
    code = "INVALID_AUDIO"


class AudioTooShortError(AppError):
    code = "AUDIO_TOO_SHORT"


class TranscriptionFailedError(AppError):
    """Raised when the ASR model cannot load or cannot decode audio."""

    code = "TRANSCRIPTION_FAILED"