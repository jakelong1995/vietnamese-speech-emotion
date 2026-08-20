"""Tests for the PhoWhisper ASR layer.

Only the pure parts are covered here — building the pipeline downloads
~1 GB of weights, so the model-backed path is exercised manually rather
than in the default suite. The timestamp normalizer is where the real
bugs live: Whisper hands back ``None`` boundaries more often than the
docs suggest.
"""
from __future__ import annotations

from src import asr


def test_chunks_are_normalized_to_start_end_text():
    chunks = [
        {"timestamp": (0.0, 1.5), "text": " Xin chào "},
        {"timestamp": (1.5, 3.0), "text": "bạn khoẻ không"},
    ]
    segs = asr._segments_from_chunks(chunks)
    assert segs == [
        {"start": 0.0, "end": 1.5, "text": "Xin chào"},
        {"start": 1.5, "end": 3.0, "text": "bạn khoẻ không"},
    ]


def test_open_ended_final_chunk_is_kept():
    """Whisper routinely leaves the last chunk's end as None; dropping it
    would silently lose the final sentence of every transcript."""
    segs = asr._segments_from_chunks([{"timestamp": (4.0, None), "text": "cuối"}])
    assert len(segs) == 1
    assert segs[0]["start"] == 4.0
    assert segs[0]["end"] == 4.0


def test_chunk_without_start_is_dropped():
    """A chunk we cannot place on the timeline must not be invented onto it."""
    assert asr._segments_from_chunks([{"timestamp": (None, 2.0), "text": "trôi"}]) == []


def test_empty_text_is_dropped():
    assert asr._segments_from_chunks([{"timestamp": (0.0, 1.0), "text": "   "}]) == []


def test_none_and_empty_inputs():
    assert asr._segments_from_chunks(None) == []
    assert asr._segments_from_chunks([]) == []


def test_get_info_reports_before_load():
    info = asr.get_info()
    assert info["loaded"] is False
    assert info["language"] == "vi"
    assert "PhoWhisper" in info["model_id"]


def test_network_failure_gives_short_actionable_message():
    """A dead connection must not dump the whole nested traceback at the
    user; it must name the cause and the way out."""
    exc = ValueError("ConnectionError: Network error ... " + "x" * 5000)
    msg = asr._load_error_message(exc, "vinai/PhoWhisper-small")
    assert len(msg) < 400, "thông báo lỗi vẫn quá dài"
    assert "HF_HUB_DISABLE_XET" in msg
    assert "PhoWhisper-small" in msg


def test_truncated_download_is_treated_as_a_network_problem():
    """transformers reports a half-downloaded model as a *missing file*.
    Reporting that verbatim sends the user hunting for the wrong bug."""
    exc = OSError("vinai/PhoWhisper-small does not appear to have a file "
                  "named pytorch_model.bin, model.safetensors, ...")
    assert "HF_HUB_DISABLE_XET" in asr._load_error_message(exc, "m")


def test_non_network_failure_names_the_exception_type():
    msg = asr._load_error_message(KeyError("bad config"), "m")
    assert "KeyError" in msg
    assert "HF_HUB_DISABLE_XET" not in msg


def test_words_are_grouped_into_phrases_at_pauses():
    """Consecutive words run together; a real pause starts a new phrase."""
    words = [
        {"start": 0.0, "end": 0.3, "text": "xin"},
        {"start": 0.3, "end": 0.6, "text": "chào"},
        # 1.4 s of silence -> new phrase
        {"start": 2.0, "end": 2.3, "text": "bạn"},
        {"start": 2.3, "end": 2.6, "text": "khoẻ"},
    ]
    out = asr._group_words(words)
    assert [s["text"] for s in out] == ["xin chào", "bạn khoẻ"]
    assert out[0]["start"] == 0.0 and out[0]["end"] == 0.6
    assert out[1]["start"] == 2.0 and out[1]["end"] == 2.6


def test_unbroken_speech_is_still_split_by_the_hard_cap():
    """A monologue with no pauses must not become one giant line."""
    words = [{"start": i * 0.4, "end": i * 0.4 + 0.4, "text": f"w{i}"}
             for i in range(40)]          # 16 s, no gap anywhere
    out = asr._group_words(words)
    assert len(out) > 1
    assert all(s["end"] - s["start"] <= asr.PHRASE_MAX_S + 1e-6 for s in out)


def test_grouping_empty_input():
    assert asr._group_words([]) == []
