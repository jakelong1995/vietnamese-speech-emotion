"""Smoke tests for the ViSEC loader (``bench/visec.py``).

Marked ``@pytest.mark.slow`` so the default ``pytest -v`` suite in CI
can skip them (they require the 367 MB ViSEC parquet). Run explicitly:

    .venv/bin/python3 -m pytest -v -m slow tests/test_visec_loader.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable so `bench.*` is resolvable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def one_per_class():
    from bench.visec import load_visec
    return load_visec(per_class=1, seed=0)


@pytest.mark.slow
def test_load_visec_returns_one_per_class(one_per_class):
    from bench.visec import LABEL_NAMES
    assert len(one_per_class) == 4
    seen = {s.label for s in one_per_class}
    assert seen == {0, 1, 2, 3}, f"missing some emotion classes: {seen}"
    names = {s.label_name for s in one_per_class}
    assert names == set(LABEL_NAMES)


@pytest.mark.slow
def test_visec_samples_have_audio_and_labels(one_per_class):
    import soundfile as sf
    for s in one_per_class:
        wav, sr = sf.read(s.wav_path)
        assert sr == 16000, f"{s.label_name}: expected 16 kHz, got {sr}"
        assert len(wav) > 0, f"{s.label_name}: empty wav"
        # Each clip is ~1-8 s on ViSEC.
        assert 0.5 <= len(wav) / sr <= 10, (
            f"{s.label_name}: unusual clip length "
            f"{len(wav) / sr:.2f}s"
        )


@pytest.mark.slow
def test_visec_subset_is_cached_after_first_call(one_per_class, tmp_path):
    cache = Path(__file__).resolve().parent.parent / "bench" / "cache"
    jsonl_files = list(cache.glob("visec_subset_p1_s0.jsonl"))
    assert jsonl_files, "subset index should be persisted on disk"
    body = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(body) == 4
