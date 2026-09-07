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


def test_repetition_loop_is_flagged():
    """Whisper looping produced 272 words for 8.5 s of audio; that rate is
    physically impossible for speech and must not pass as a transcript."""
    words = [{"start": i * 0.031, "end": i * 0.031 + 0.03, "text": "ha"}
             for i in range(272)]          # 272 words across ~8.5 s
    assert asr.looks_hallucinated(words) is True


def test_normal_speech_rate_is_not_flagged():
    """~4 syllables/second is ordinary Vietnamese and must pass clean."""
    words = [{"start": i * 0.25, "end": i * 0.25 + 0.2, "text": f"w{i}"}
             for i in range(40)]           # 40 words across 10 s
    assert asr.looks_hallucinated(words) is False


def test_hallucination_check_handles_degenerate_spans():
    assert asr.looks_hallucinated([]) is False
    # every word stamped at the same instant -> zero span, no division blowup
    assert asr.looks_hallucinated(
        [{"start": 1.0, "end": 1.0, "text": "a"}] * 5) is False


def test_local_copy_wins_over_hub_id(tmp_path, monkeypatch):
    """A hand-downloaded model must be preferred over the repo id, whose
    HF cache entry may be a half-written file."""
    local = tmp_path / "PhoWhisper-small"
    local.mkdir()
    (local / "config.json").write_text("{}")
    monkeypatch.setattr(asr, "LOCAL_MODEL_DIR", tmp_path)
    assert asr.resolve_model("vinai/PhoWhisper-small") == str(local)


def test_hub_id_passes_through_when_no_local_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "LOCAL_MODEL_DIR", tmp_path)
    assert asr.resolve_model("vinai/PhoWhisper-medium") == "vinai/PhoWhisper-medium"


def test_directory_without_config_is_not_treated_as_a_model(tmp_path, monkeypatch):
    """An empty or half-downloaded directory must not shadow the Hub."""
    (tmp_path / "PhoWhisper-small").mkdir()
    monkeypatch.setattr(asr, "LOCAL_MODEL_DIR", tmp_path)
    assert asr.resolve_model("vinai/PhoWhisper-small") == "vinai/PhoWhisper-small"


def test_explicit_path_is_returned_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "LOCAL_MODEL_DIR", tmp_path / "elsewhere")
    assert asr.resolve_model(str(tmp_path)) == str(tmp_path)
