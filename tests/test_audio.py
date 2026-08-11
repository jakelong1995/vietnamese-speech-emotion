"""Tests for src.audio (resample helper + load_audio_mono_16k)."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.audio import load_audio_mono_16k, resample_linear


def test_resample_linear_same_sr_returns_input():
    waveform = np.linspace(-1, 1, 16000, dtype=np.float32)
    out = resample_linear(waveform, 16000, 16000)
    assert out is waveform  # no copy on same sr


def test_resample_linear_empty_passthrough():
    out = resample_linear(np.zeros(0, dtype=np.float32), 48000, 16000)
    assert out.size == 0


def test_resample_linear_preserves_amplitude():
    sr = 48000
    duration = 1.0
    waveform = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))).astype(np.float32)
    out = resample_linear(waveform, sr, 16000)
    # Linear interp preserves peaks within a small tolerance
    assert abs(float(np.max(np.abs(out))) - 0.5) < 0.05
    # Target length = 1.0 * 16000 = 16000
    assert out.shape[0] == 16000


def test_load_audio_mono_16k_round_trip(tmp_path):
    sr = 22050
    waveform = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(sr * 2.0)))).astype(np.float32)
    p = tmp_path / "test.wav"
    sf.write(str(p), waveform, sr, subtype="PCM_16")
    out, out_sr = load_audio_mono_16k(str(p))
    assert out_sr == 16000
    assert out.dtype == np.float32
    assert out.shape[0] == 32000  # 2 s at 16 kHz


def test_load_audio_mono_16k_stereo_averaged(tmp_path):
    sr = 16000
    left = np.ones(sr, dtype=np.float32) * 0.5
    right = np.ones(sr, dtype=np.float32) * -0.5
    stereo = np.stack([left, right], axis=1)
    p = tmp_path / "stereo.wav"
    sf.write(str(p), stereo, sr, subtype="PCM_16")
    out, out_sr = load_audio_mono_16k(str(p))
    assert out_sr == 16000
    # Average of +0.5 and -0.5 = 0
    assert abs(float(np.mean(np.abs(out)))) < 0.01