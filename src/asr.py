"""Vietnamese ASR layer — PhoWhisper transcription.

Separate from :mod:`src.inference` on purpose. The emotion model tells
you *how* something was said; this tells you *what* was said. They are
independent models with independent lifecycles, and the UI can use
either one alone.

Model
-----
`vinai/PhoWhisper <https://huggingface.co/vinai/PhoWhisper-large>`_ is
Whisper fine-tuned by VinAI on 844 hours of Vietnamese covering
northern/central/southern accents — state of the art for Vietnamese
ASR. Five sizes ship; we default to ``small`` (~1 GB) because it is the
best accuracy/latency trade-off on an M-series Mac. Override with the
``ASR_MODEL_NAME`` env var, e.g.::

    ASR_MODEL_NAME=vinai/PhoWhisper-medium streamlit run streamlit_app.py

Public API
----------
- ``warmup()`` — force the model load.
- ``transcribe(audio_path)`` — full transcript + timestamped segments.
- ``get_info()`` / ``unload()`` / ``is_loaded()``.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .device import describe as describe_device
from .device import resolve_device, resolve_dtype
from .exceptions import TranscriptionFailedError

log = logging.getLogger("vser.asr")

MODEL_ENV = "ASR_MODEL_NAME"
DEFAULT_MODEL = "vinai/PhoWhisper-small"

#: Group consecutive words into a phrase until the silence between them
#: exceeds this, i.e. the speaker paused. 0.6 s is about the length of a
#: clause boundary in conversational Vietnamese.
PHRASE_GAP_S = 0.6
#: Hard cap so an unbroken monologue still gets split into readable lines.
PHRASE_MAX_S = 6.0

#: Whisper's encoder is fixed at a 30-second receptive field. Longer
#: audio must be fed in overlapping windows; ``stride`` is the overlap
#: the pipeline uses to stitch chunk boundaries without dropping words.
CHUNK_LENGTH_S = 30
STRIDE_LENGTH_S = 5

#: Substrings that mark a load failure as "the network died", not "the
#: model is wrong". transformers reports both as the same ValueError, and
#: the raw text runs to a couple hundred lines of nested tracebacks — far
#: too much to put in front of a user.
_NETWORK_MARKERS = (
    "ConnectionError", "Network error", "Max retries", "timed out",
    "does not appear to have a file named", "Connection reset",
)

_pipe = None
_model_id: str = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
_device: str = "cpu"
_load_seconds: float = 0.0


def model_id() -> str:
    return _model_id


def is_loaded() -> bool:
    return _pipe is not None


def _load_error_message(exc: Exception, model: str) -> str:
    """Turn a model-load exception into one readable line.

    transformers reports a dead connection and a genuinely wrong model id
    as the same ValueError, with a few hundred lines of nested traceback
    in the text. Users need the cause and the way out, not the trace.
    """
    text = str(exc)
    if any(m in text for m in _NETWORK_MARKERS):
        return (
            f"Không tải được model ASR {model!r}: kết nối tới HuggingFace bị "
            "gián đoạn giữa chừng. Thử lại, hoặc tắt CDN Xet bằng biến môi "
            f"trường HF_HUB_DISABLE_XET=1, hoặc trỏ {MODEL_ENV} vào một thư "
            "mục model đã tải sẵn."
        )
    return f"Không tải được model ASR {model!r}: {type(exc).__name__}"


def _load() -> None:
    """Build the ASR pipeline (idempotent)."""
    global _pipe, _device, _load_seconds
    if _pipe is not None:
        return

    device = resolve_device()
    log.info("loading ASR %s onto %s", _model_id, device)
    t0 = time.perf_counter()
    try:
        import torch  # noqa: F401
        from transformers import pipeline

        _pipe = pipeline(
            "automatic-speech-recognition",
            model=_model_id,
            device=device,
            dtype=resolve_dtype(device),
            chunk_length_s=CHUNK_LENGTH_S,
            stride_length_s=STRIDE_LENGTH_S,
        )
    except Exception as exc:
        raise TranscriptionFailedError(
            _load_error_message(exc, _model_id),
            details={"model_id": _model_id, "device": device,
                     "cause": str(exc)[:400]},
        ) from exc
    _device = device
    _load_seconds = time.perf_counter() - t0
    log.info("ASR %s loaded on %s in %.1fs", _model_id, device, _load_seconds)


def warmup() -> Dict[str, Any]:
    """Force the ASR load at startup. Safe to call repeatedly."""
    _load()
    return get_info()


def get_info() -> Dict[str, Any]:
    return {
        "model_id": _model_id,
        "device": _device,
        "accelerator": describe_device(_device),
        "load_seconds": round(_load_seconds, 3),
        "loaded": is_loaded(),
        "language": "vi",
    }


def unload() -> None:
    global _pipe
    _pipe = None
    import gc
    gc.collect()


def _segments_from_chunks(chunks: Optional[List[dict]]) -> List[Dict[str, Any]]:
    """Normalize the pipeline's ``chunks`` into ``{start, end, text}``.

    Whisper emits ``timestamp: (start, end)`` where either side can be
    ``None`` — the final chunk of a stream routinely has an open end.
    We drop entries we cannot place on the timeline rather than
    inventing a boundary for them.
    """
    out: List[Dict[str, Any]] = []
    for ch in chunks or []:
        ts = ch.get("timestamp") or (None, None)
        start, end = (ts + (None, None))[:2] if isinstance(ts, tuple) else (None, None)
        text = (ch.get("text") or "").strip()
        if start is None or not text:
            continue
        out.append({
            "start": float(start),
            "end": float(end) if end is not None else float(start),
            "text": text,
        })
    return out


def _group_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge word-level chunks into phrase-level segments.

    PhoWhisper does not emit sentence timestamps — asking for them yields
    a single chunk spanning the whole clip with an open end, which would
    pin the entire transcript to t=0 and make alignment against the
    emotion timeline meaningless. Word timestamps *are* reliable, so we
    ask for those and rebuild phrases from the pauses between them.
    """
    out: List[Dict[str, Any]] = []
    for w in words:
        if (out
                and w["start"] - out[-1]["end"] <= PHRASE_GAP_S
                and w["end"] - out[-1]["start"] <= PHRASE_MAX_S):
            out[-1]["end"] = w["end"]
            out[-1]["text"] = f"{out[-1]['text']} {w['text']}".strip()
        else:
            out.append(dict(w))
    return out


def transcribe(audio_path: str | Path) -> Dict[str, Any]:
    """Transcribe a Vietnamese audio file.

    Returns ``{text, segments, latency_ms, model_id, device}`` where
    ``segments`` is a list of ``{start, end, text}`` in seconds — the
    anchor the emotion timeline aligns against.
    """
    _load()
    assert _pipe is not None

    t0 = time.perf_counter()
    try:
        # "word", not True — see _group_words for why.
        result = _pipe(str(audio_path), return_timestamps="word")
    except Exception as exc:
        log.exception("ASR failed for %s", audio_path)
        raise TranscriptionFailedError(
            f"Transcription failed: {exc!r}",
            details={"model_id": _model_id, "path": str(audio_path)},
        ) from exc
    latency = time.perf_counter() - t0

    text = (result.get("text") or "").strip() if isinstance(result, dict) else ""
    words = _segments_from_chunks(
        result.get("chunks") if isinstance(result, dict) else None
    )
    segments = _group_words(words)
    return {
        "text": text,
        "segments": segments,
        "words": words,
        "latency_ms": int(round(latency * 1000)),
        "model_id": _model_id,
        "device": _device,
    }


__all__ = ["_load_error_message", "_group_words", "transcribe", "warmup", "get_info", "unload", "is_loaded",
           "model_id", "MODEL_ENV", "DEFAULT_MODEL"]
