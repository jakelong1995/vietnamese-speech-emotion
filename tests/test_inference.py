"""Tests for the inference wrapper that the Streamlit UI calls.

These tests use a stubbed adapter so we can verify the public surface
without loading real weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import src.inference as inference


@pytest.fixture(autouse=True)
def _reset_adapter_singleton(monkeypatch):
    """Each test gets a fresh ``_active`` slot."""
    inference._active = None
    inference._active_name = "meralion-ser-v1"
    yield
    inference._active = None


def test_get_adapter_is_singleton(monkeypatch, tmp_path):
    """The wrapper must return the same adapter across calls."""
    captured: dict = {}

    class FakeAdapter:
        def __init__(self, cache_dir=None, temp_audio_dir=None):
            captured["cache_dir"] = cache_dir

        def load(self):
            pass

        def get_model_info(self):
            from src.adapters.base import ModelInfo
            return ModelInfo(
                provider="meralion-ser-v1",
                model_id="MERaLiON/MERaLiON-SER-v1",
                sample_rate=16000,
                labels=["neutral", "happy", "sad", "angry"],
                device="cpu",
                confidence_available=True,
                extra={"role": "vietnamese_partial",
                       "language": "vi"},
            )

    monkeypatch.setitem(inference.REGISTRY, "meralion-ser-v1", FakeAdapter)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    a = inference.get_adapter()
    assert a is inference.get_adapter()  # singleton
    assert captured["cache_dir"] == Path(str(tmp_path))


def test_predict_returns_required_fields(monkeypatch, tmp_path):
    """Smoke check: predict() must return a dict with the keys the UI
    consumes (label, confidence, class_scores, raw_label, latency_ms,
    full_distribution_available, model_id, device)."""
    sample_wav = tmp_path / "tone.wav"
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(sample_wav),
             (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

    class FakeAdapter:
        model_id = "MERaLiON/MERaLiON-SER-v1"
        sample_rate = 16000

        def __init__(self, cache_dir=None, temp_audio_dir=None):
            pass

        def load(self):
            pass

        def predict(self, waveform, sample_rate):
            from src.adapters.base import RawPrediction
            return RawPrediction(
                label="happy",
                confidence=0.83,
                class_scores={"angry": 0.02, "happy": 0.83, "neutral": 0.15},
                auxiliary={
                    "inference_seconds": 0.135,
                },
                full_distribution_available=True,
            )

        def get_model_info(self):
            from src.adapters.base import ModelInfo
            return ModelInfo(
                provider="meralion-ser-v1",
                model_id=self.model_id,
                sample_rate=16000,
                labels=["angry", "happy", "neutral"],
                device="cpu",
                confidence_available=True,
                extra={
                    "role": "vietnamese_partial",
                    "language": "vi",
                },
            )

    monkeypatch.setitem(inference.REGISTRY, "meralion-ser-v1", FakeAdapter)

    result = inference.predict(str(sample_wav))
    for k in (
        "label", "raw_label", "confidence", "class_scores", "raw_distribution",
        "latency_ms", "full_distribution_available", "model_id", "device",
        "role", "language",
    ):
        assert k in result, f"missing key: {k}"
    assert result["label"] == "happy"
    assert result["confidence"] == 0.83
    assert result["latency_ms"] == 135
    assert result["full_distribution_available"] is True
    assert sum(result["class_scores"].values()) == pytest.approx(1.0, abs=1e-3)


def test_format_error_includes_code_for_known_app_errors():
    from src.exceptions import InvalidAudioError
    err = InvalidAudioError("file is corrupt")
    msg = inference.format_error(err)
    assert "[INVALID_AUDIO]" in msg
    assert "file is corrupt" in msg


def test_format_error_includes_class_name_for_unknown():
    class WeirdError(Exception):
        pass
    msg = inference.format_error(WeirdError("boom"))
    assert "WeirdError" in msg
    assert "boom" in msg


def test_list_adapters_returns_meralion():
    names = inference.list_adapters()
    assert names == ["meralion-ser-v1"]


def test_bench_scores_returns_dict(monkeypatch):
    monkeypatch.setattr(inference, "bench_scores",
                        lambda: {"meralion-ser-v1": {"accuracy": 0.4025}})
    s = inference.bench_scores()
    assert "meralion-ser-v1" in s
    assert s["meralion-ser-v1"]["accuracy"] == 0.4025
