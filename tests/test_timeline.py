"""Tests for sliding-window emotion analysis.

Chunking and aggregation are pure functions, so they are tested without
loading any weights. The adapter-driven path is covered by the stubbed
adapter in ``test_inference.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import timeline
from src.exceptions import InvalidAudioError

SR = 16000


def test_short_clip_yields_single_window():
    """A clip shorter than one window must still be analysable."""
    wav = np.zeros(int(SR * 1.2), dtype=np.float32)
    chunks = timeline.chunk_waveform(wav, SR)
    assert len(chunks) == 1
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == pytest.approx(1.2)


def test_windows_overlap_by_window_minus_hop():
    total_sec = 10.0
    wav = np.zeros(int(SR * total_sec), dtype=np.float32)
    chunks = timeline.chunk_waveform(wav, SR, window_sec=3.0, hop_sec=1.5)
    starts = [round(c[0], 3) for c in chunks]
    assert starts[:4] == [0.0, 1.5, 3.0, 4.5]
    # Windows are full-length until the waveform runs out; the trailing
    # ones are legitimately short (a 10 s clip cannot supply a full 3 s
    # window starting at 7.5 s or 9.0 s).
    for start, end, _ in chunks:
        expected = min(3.0, total_sec - start)
        assert end - start == pytest.approx(expected)


def test_tail_shorter_than_min_window_is_dropped():
    """A 0.5 s tail must not produce a spurious final spike."""
    wav = np.zeros(int(SR * 4.6), dtype=np.float32)
    chunks = timeline.chunk_waveform(wav, SR, window_sec=3.0, hop_sec=1.5)
    assert all((c[1] - c[0]) >= timeline.MIN_WINDOW_SEC for c in chunks)


def test_empty_waveform_rejected():
    with pytest.raises(InvalidAudioError):
        timeline.chunk_waveform(np.array([], dtype=np.float32), SR)


def test_summarize_averages_probabilities_not_votes():
    """A consistently-weak signal must beat a single loud outlier.

    Two windows lean sad without ever making sad the argmax; a
    vote-based summary would say neutral, an average-based one says sad.
    """
    segs = [
        {"label": "neutral", "class_scores": {"sad": 0.45, "neutral": 0.55}},
        {"label": "neutral", "class_scores": {"sad": 0.49, "neutral": 0.51}},
        {"label": "sad", "class_scores": {"sad": 0.90, "neutral": 0.10}},
    ]
    out = timeline.summarize(segs)
    assert out["label"] == "sad"
    assert out["num_segments"] == 3
    # dominant_share is rounded to 3 dp by summarize().
    assert out["dominant_share"] == pytest.approx(1 / 3, abs=1e-3)


def test_summarize_handles_empty():
    assert timeline.summarize([])["label"] is None


def test_align_transcript_uses_overlap_not_containment():
    """A sentence spanning two windows must appear in both."""
    emo = [{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 6.0}]
    asr = [{"start": 2.5, "end": 3.5, "text": "câu bắc cầu"}]
    out = timeline.align_transcript(emo, asr)
    assert out[0]["text"] == "câu bắc cầu"
    assert out[1]["text"] == "câu bắc cầu"


def test_align_transcript_empty_when_no_overlap():
    emo = [{"start": 0.0, "end": 3.0}]
    asr = [{"start": 5.0, "end": 6.0, "text": "ngoài vùng"}]
    assert timeline.align_transcript(emo, asr)[0]["text"] == ""
