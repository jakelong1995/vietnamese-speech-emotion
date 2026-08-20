"""Emotion-over-time analysis — sliding-window inference.

``src.inference.predict`` collapses a whole clip into a single label.
That is fine for a 3-second sample and useless for a 5-minute phone
call, where the interesting thing is *when* the tone changed.

This module cuts the waveform into overlapping windows, runs the
emotion adapter on each, and returns one row per window so the UI can
plot a curve and line it up against the ASR transcript.

Window sizing
-------------
``WINDOW_SEC = 3.0`` is short enough to localise a mood swing but long
enough for the ECAPA-TDNN pooling layer to have real prosody to work
with — below ~1.5 s the model mostly sees a single syllable and its
output gets noisy.

``HOP_SEC = 1.5`` (50 % overlap) halves the step size without halving
the context, so the resulting curve is smooth instead of blocky. The
cost is 2x the inference calls.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .audio import load_audio_mono_16k
from .exceptions import InvalidAudioError

log = logging.getLogger("vser.timeline")

WINDOW_SEC = 3.0
HOP_SEC = 1.5
#: Windows shorter than this are dropped — a sub-second tail carries
#: too little prosody to score, and keeping it adds a spurious spike
#: at the end of every chart.
MIN_WINDOW_SEC = 1.0


def chunk_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC,
) -> List[Tuple[float, float, np.ndarray]]:
    """Split into overlapping windows → ``[(start_s, end_s, samples), ...]``.

    Clips shorter than one window yield a single window covering the
    whole thing, so short samples still work.
    """
    if waveform.size == 0:
        raise InvalidAudioError("Empty waveform")
    if window_sec <= 0 or hop_sec <= 0:
        raise ValueError("window_sec and hop_sec must be positive")

    total_sec = waveform.shape[0] / float(sample_rate)
    if total_sec <= window_sec:
        return [(0.0, total_sec, waveform)]

    win = int(round(window_sec * sample_rate))
    hop = int(round(hop_sec * sample_rate))
    out: List[Tuple[float, float, np.ndarray]] = []
    for start in range(0, waveform.shape[0], hop):
        seg = waveform[start:start + win]
        if seg.shape[0] / float(sample_rate) < MIN_WINDOW_SEC:
            break
        out.append((start / sample_rate,
                    (start + seg.shape[0]) / sample_rate,
                    seg))
    return out


def _rms(seg: np.ndarray) -> float:
    """Loudness of a window, used by the UI to grey out silence."""
    if seg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(seg.astype(np.float64)))))


def analyze_timeline(
    audio_path: str | Path,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the emotion model across a clip and return a per-window series.

    ``progress`` may be any callable taking ``(done, total)`` — the
    Streamlit UI passes one so a long call doesn't look frozen.

    Returns ``{segments, summary, duration_sec, window_sec, hop_sec,
    total_latency_ms}``.
    """
    from .inference import get_adapter  # local import: avoids a cycle

    adapter = get_adapter()
    waveform, sample_rate = load_audio_mono_16k(str(audio_path))
    windows = chunk_waveform(waveform, sample_rate, window_sec, hop_sec)
    log.info("timeline: %d windows over %.1fs",
             len(windows), waveform.shape[0] / sample_rate)

    t0 = time.perf_counter()
    segments: List[Dict[str, Any]] = []
    for i, (start, end, seg) in enumerate(windows):
        pred = adapter.predict(seg, sample_rate)
        scores = pred.class_scores or {}
        segments.append({
            "index": i,
            "start": round(start, 3),
            "end": round(end, 3),
            "mid": round((start + end) / 2, 3),
            "label": pred.label,
            "raw_label": pred.auxiliary.get("raw_label") or pred.label,
            "confidence": pred.confidence,
            "class_scores": scores,
            "rms": round(_rms(seg), 5),
        })
        if progress is not None:
            progress(i + 1, len(windows))

    return {
        "segments": segments,
        "summary": summarize(segments),
        "duration_sec": round(waveform.shape[0] / sample_rate, 3),
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "total_latency_ms": int(round((time.perf_counter() - t0) * 1000)),
    }


def summarize(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average the per-window distributions into one clip-level verdict.

    Averaging the *probability vectors* rather than voting on per-window
    argmax labels keeps the near-misses: a clip that is weakly angry
    throughout should read as angry, even if no single window happens
    to make angry the top class.
    """
    if not segments:
        return {"label": None, "class_scores": {}, "dominant_share": 0.0}

    totals: Dict[str, float] = {}
    for seg in segments:
        for name, score in (seg.get("class_scores") or {}).items():
            totals[name] = totals.get(name, 0.0) + float(score)
    n = len(segments)
    mean_scores = {k: v / n for k, v in totals.items()}

    label = max(mean_scores, key=mean_scores.get) if mean_scores else None
    # How much of the clip actually argmaxed to the winning label —
    # separates "consistently sad" from "averaged out to sad".
    share = sum(1 for s in segments if s.get("label") == label) / n
    return {
        "label": label,
        "class_scores": mean_scores,
        "dominant_share": round(share, 3),
        "num_segments": n,
    }


def align_transcript(
    segments: List[Dict[str, Any]],
    asr_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach the words spoken during each emotion window.

    Overlap-based, not nearest-neighbour: an ASR segment is attached to
    every emotion window it intersects in time, because a 4-second
    sentence legitimately spans two 3-second windows.
    """
    enriched: List[Dict[str, Any]] = []
    for seg in segments:
        words = [
            a["text"] for a in asr_segments
            if a.get("start") is not None
            and a["start"] < seg["end"] and a.get("end", a["start"]) > seg["start"]
        ]
        enriched.append({**seg, "text": " ".join(words).strip()})
    return enriched


__all__ = ["chunk_waveform", "analyze_timeline", "summarize",
           "align_transcript", "WINDOW_SEC", "HOP_SEC"]
