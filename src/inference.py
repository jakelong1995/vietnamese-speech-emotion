"""Inference layer — single-model Streamlit app wrapper around MERaLiON-SER-v1.

The Space ships with exactly one emotion model. There is one adapter
loaded in memory at a time, lazy-loaded on first inference call.

Public API
----------
- ``get_adapter()`` — lazy singleton; loads MERaLiON-SER-v1 if not yet loaded.
- ``warmup()`` — force the model load at app startup.
- ``predict(audio_path)`` / ``predict_waveform(wav, sr)`` — inference.
- ``bench_scores()`` — read ``bench/results/scores.json`` (for UI banners).

Set the env var ``SER_MODEL_NAME`` to override the active adapter name
(useful for tests or future model swaps). Defaults to ``meralion-ser-v1``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .adapters import REGISTRY, instantiate
from .adapters.base import BaseAdapter, ModelInfo, RawPrediction
from .audio import load_audio_mono_16k
from .exceptions import AppError

log = logging.getLogger("vser.inference")

_ACTIVE_NAME_ENV = "SER_MODEL_NAME"
_DEFAULT_MODEL = "meralion-ser-v1"

_active: Optional[BaseAdapter] = None
_active_name: str = os.environ.get(_ACTIVE_NAME_ENV, _DEFAULT_MODEL)


def _default_cache_dir() -> Path:
    env = os.environ.get("HF_HOME") or os.environ.get("MODEL_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "huggingface"


def _bench_dir() -> Path:
    """Resolve ``bench/results/`` regardless of cwd."""
    # File lives at src/inference.py → parent → project root → bench/results.
    return Path(__file__).resolve().parents[1] / "bench" / "results"


def list_adapters() -> list[str]:
    return list(REGISTRY.keys())


def bench_scores() -> Dict[str, Any]:
    """Read ``bench/results/scores.json``; return {} if missing."""
    path = _bench_dir() / "scores.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def set_active(name: str) -> Dict[str, Any]:
    """Unload the current adapter, load ``name``. Returns info dict.

    Raises the adapter's own load error (typically
    :class:`~src.exceptions.ModelLoadFailedError`) when the chosen
    adapter cannot be loaded.
    """
    global _active, _active_name
    if name not in REGISTRY:
        raise ValueError(f"unknown adapter: {name!r}")
    if _active_name == name and _active is not None:
        return _as_info_dict(_active.get_model_info())

    if _active is not None:
        log.info("unloading active adapter %s", _active_name)
        try:
            _active.unload()
        except Exception as exc:
            log.warning("unload of %s raised: %s", _active_name, exc)
        _active = None

    log.info("loading adapter %s", name)
    adapter = instantiate(name, cache_dir=_default_cache_dir())
    adapter.load()
    _active = adapter
    _active_name = name
    return _as_info_dict(adapter.get_model_info())


def _as_info_dict(info: ModelInfo) -> Dict[str, Any]:
    return {
        "provider": info.provider,
        "model_id": info.model_id,
        "device": info.device,
        "sample_rate": info.sample_rate,
        "labels": list(info.labels),
        "load_seconds": info.extra.get("model_load_seconds"),
        "extra": dict(info.extra),
    }


def get_adapter() -> BaseAdapter:
    """Lazy-load the active adapter (singleton)."""
    global _active
    if _active is None:
        set_active(_active_name)
    assert _active is not None
    return _active


def warmup() -> Dict[str, Any]:
    """Force adapter load at startup. Safe to call multiple times."""
    adapter = get_adapter()
    return _as_info_dict(adapter.get_model_info())


def predict(audio_path: str | Path) -> Dict[str, Any]:
    """Run inference on an audio file path; returns a UI-shaped dict.

    The dict's ``class_scores`` contains the 7 raw MERaLiON softmax
    buckets. The dict is intentionally backend-agnostic so the UI
    doesn't need to know which model produced it.
    """
    adapter = get_adapter()
    waveform, sr = load_audio_mono_16k(str(audio_path))
    pred = adapter.predict(waveform, sr)
    info = adapter.get_model_info()
    return _format_for_ui(pred, info)


def predict_waveform(waveform: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    adapter = get_adapter()
    if sample_rate != adapter.sample_rate:
        from .audio import resample_linear
        waveform = resample_linear(waveform, sample_rate, adapter.sample_rate)
        sample_rate = adapter.sample_rate
    pred = adapter.predict(waveform, sample_rate)
    info = adapter.get_model_info()
    return _format_for_ui(pred, info)


def _format_for_ui(pred: RawPrediction, info: ModelInfo) -> Dict[str, Any]:
    return {
        "label": pred.label,
        "raw_label": pred.auxiliary.get("raw_label") or pred.label,
        "confidence": pred.confidence,
        "class_scores": pred.class_scores or {},
        "raw_distribution": pred.class_scores or {},
        "latency_ms": int(round((pred.auxiliary.get("inference_seconds") or 0.0) * 1000)),
        "full_distribution_available": pred.full_distribution_available,
        "model_id": info.model_id,
        "provider": info.provider,
        "device": info.device,
        "labels": list(info.labels),
        "role": info.extra.get("role"),
        "language": info.extra.get("language"),
        "vietnamese_verified": info.extra.get("vietnamese_verified"),
        "english_trained": info.extra.get("english_trained"),
        "recommendation": info.extra.get("recommendation"),
        "bench_score": bench_scores().get(info.provider) or {},
    }


def format_error(exc: Exception) -> str:
    if isinstance(exc, AppError):
        return f"[{exc.code}] {exc.message}"
    return f"{type(exc).__name__}: {exc}"
