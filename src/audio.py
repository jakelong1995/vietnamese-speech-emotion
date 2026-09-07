"""Audio loading and resampling helpers."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

TARGET_SR = 16000


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


def _decode_with_ffmpeg(path: str) -> tuple[np.ndarray, int]:
    """Decode any container ffmpeg understands straight to mono 16 kHz f32.

    libsndfile only handles WAV/MP3/OGG/FLAC, but the uploader also offers
    m4a and webm (the latter is what browser MediaRecorder emits), so those
    uploads have no other way in. Asking ffmpeg for ``f32le`` at the target
    rate lets it do the resampling too — better quality than
    :func:`resample_linear` and one less step.
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "Không đọc được file audio này và không tìm thấy ffmpeg để giải mã. "
            "Cài ffmpeg (`brew install ffmpeg`) hoặc đổi sang file WAV/FLAC."
        )

    proc = subprocess.run(
        [exe, "-nostdin", "-v", "error", "-i", path,
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "1", "-ar", str(TARGET_SR), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            "ffmpeg không giải mã được file audio này"
            + (f": {detail[-1]}" if detail else "")
        )

    waveform = np.frombuffer(proc.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise RuntimeError("File audio rỗng hoặc không chứa dữ liệu âm thanh.")
    # frombuffer returns a read-only view over the subprocess buffer; callers
    # downstream write into the array, so hand back an owned copy.
    return waveform.copy(), TARGET_SR


def load_audio_mono_16k(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at 16 kHz.

    Returns (waveform, sample_rate). Tries soundfile first because it is
    in-process and fast, then falls back to ffmpeg for the containers
    libsndfile cannot open (m4a, webm) or refuses to decode.
    """
    import soundfile as sf

    try:
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return _decode_with_ffmpeg(str(path))

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)
    # A header libsndfile accepts can still yield no frames (truncated
    # upload); ffmpeg usually salvages the audio that is actually there.
    if waveform.size == 0:
        return _decode_with_ffmpeg(str(path))
    if sr != TARGET_SR:
        waveform = resample_linear(waveform, sr, TARGET_SR)
        sr = TARGET_SR
    return waveform, sr
