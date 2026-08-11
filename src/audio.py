"""Audio loading and resampling helpers."""
from __future__ import annotations

import numpy as np


def resample_linear(waveform: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resample (no scipy dependency).

    Used by the primary adapter to bring arbitrary sample-rate input to
    the model's expected 16 kHz.
    """
    if src_sr == dst_sr or waveform.size == 0:
        return waveform
    duration = waveform.shape[0] / float(src_sr)
    target_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=waveform.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, waveform).astype(np.float32, copy=False)


def load_audio_mono_16k(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at 16 kHz.

    Returns (waveform, sample_rate). Uses soundfile; falls back to a
    minimal WAV reader if the file extension is unsupported.
    """
    import soundfile as sf

    waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)
    if sr != 16000:
        waveform = resample_linear(waveform, sr, 16000)
        sr = 16000
    return waveform, sr