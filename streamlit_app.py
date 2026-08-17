"""Streamlit front-end for Vietnamese Speech Emotion Recognition.

Talks to the same backend as everything else in this repo
(``src/inference.py`` → the singleton MERaLiON-SER-v1 adapter).

Run:

    .venv/bin/streamlit run streamlit_app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from src import inference

SAMPLE_AUDIO_DIR = Path(__file__).parent / "sample_audio"
SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "ogg", "m4a", "flac", "webm"]
VISEC_CLASSES = ("happy", "neutral", "sad", "angry")

EMOTION_COLORS = {
    "angry": "#dc2626",
    "happy": "#d97706",
    "neutral": "#475569",
    "sad": "#2563eb",
}
DEFAULT_COLOR = "#94a3b8"

st.set_page_config(
    page_title="Vietnamese Speech Emotion Recognition",
    page_icon="🇻🇳",
    layout="centered",
)


@st.cache_resource(show_spinner="Đang tải model MERaLiON-SER-v1 (~10-30s lần đầu)...")
def _warm_model() -> dict:
    return inference.warmup()


def _list_sample_files() -> dict[str, list[Path]]:
    if not SAMPLE_AUDIO_DIR.exists():
        return {}
    out: dict[str, list[Path]] = {}
    for sub in sorted(p for p in SAMPLE_AUDIO_DIR.iterdir() if p.is_dir()):
        files = sorted(sub.glob("*.wav"))
        if files:
            out[sub.name] = files
    return out


def _predict_bytes(audio_bytes: bytes, suffix: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return inference.predict(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _predict_path(path: Path) -> dict:
    return inference.predict(str(path))


def _render_bar(label: str, score: float, color: str) -> str:
    pct = max(0.0, min(1.0, score)) * 100
    return (
        f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
        f"<div style='width:70px;font-size:13px;color:#334155;text-transform:capitalize;'>{label}</div>"
        f"<div style='flex:1;background:#f1f5f9;border-radius:6px;height:14px;overflow:hidden;'>"
        f"<div style='width:{pct:.1f}%;background:{color};height:100%;'></div>"
        f"</div>"
        f"<div style='width:44px;font-size:12px;color:#64748b;text-align:right;'>{pct:.1f}%</div>"
        f"</div>"
    )


def render_result(result: dict) -> None:
    label = result["label"]
    color = EMOTION_COLORS.get(label, DEFAULT_COLOR)
    confidence = result["confidence"]

    st.markdown(
        f"<div style='padding:18px 20px;border-radius:12px;background:{color}14;"
        f"border:1px solid {color}40;margin-bottom:14px;'>"
        f"<div style='font-size:12px;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:0.06em;font-weight:600;'>Cảm xúc dự đoán</div>"
        f"<div style='font-size:38px;font-weight:750;color:{color};"
        f"text-transform:capitalize;line-height:1.1;'>{label}</div>"
        f"<div style='color:#64748b;font-size:13px;margin-top:4px;'>"
        f"confidence {confidence:.0%} &middot; raw: {result['raw_label']} &middot; "
        f"{result['latency_ms']} ms &middot; device: {result['device']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if label not in VISEC_CLASSES:
        st.info(
            f"Nhãn '{result['raw_label']}' nằm ngoài 4 class ViSEC "
            f"(happy/neutral/sad/angry) — model có 7 nhãn thô."
        )

    scores = result["class_scores"]
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    bars = "".join(
        _render_bar(name, score, EMOTION_COLORS.get(name, DEFAULT_COLOR))
        for name, score in ordered
    )
    st.markdown(f"<div style='margin-top:8px;'>{bars}</div>", unsafe_allow_html=True)


def render_bench_banner() -> None:
    scores = inference.bench_scores()
    row = scores.get("meralion-ser-v1")
    if not row:
        return
    acc = row.get("accuracy", 0.0)
    eligible = row.get("eligible_for_deploy", False)
    status = "✅ đạt" if eligible else "❌ chưa đạt"
    st.caption(
        f"Benchmark ViSEC (400 clip, 4-class): accuracy **{acc:.2%}** · "
        f"tiêu chí ≥65% {status} · nguồn: `bench/results/scores.json`"
    )


def main() -> None:
    st.title("🇻🇳 Vietnamese Speech Emotion Recognition")
    st.caption("Streamlit UI — backend `src/inference.py`.")

    try:
        info = _warm_model()
    except Exception as exc:  # noqa: BLE001 - surface any AppError to the page
        st.error(f"Không load được model: {inference.format_error(exc)}")
        st.stop()

    render_bench_banner()
    st.caption(
        f"Model: `{info['model_id']}` · device: `{info['device']}` · "
        f"{len(info['labels'])} class"
    )

    tab_upload, tab_mic, tab_samples = st.tabs(["📁 Upload", "🎙️ Microphone", "🎵 Samples"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Tải lên file audio", type=SUPPORTED_AUDIO_TYPES, key="uploader"
        )
        if uploaded is not None:
            st.audio(uploaded)
            if st.button("Analyze", key="analyze_upload", type="primary"):
                suffix = Path(uploaded.name).suffix or ".wav"
                with st.spinner("Đang phân tích..."):
                    result = _predict_bytes(uploaded.getvalue(), suffix)
                render_result(result)

    with tab_mic:
        recorded = st.audio_input("Thu âm giọng nói (~5-10 giây)", key="mic")
        if recorded is not None:
            if st.button("Analyze", key="analyze_mic", type="primary"):
                with st.spinner("Đang phân tích..."):
                    result = _predict_bytes(recorded.getvalue(), ".wav")
                render_result(result)

    with tab_samples:
        samples = _list_sample_files()
        if not samples:
            st.warning("Không tìm thấy file mẫu trong `sample_audio/`.")
        else:
            emotion = st.selectbox("Chọn class", list(samples.keys()), key="sample_class")
            files = samples[emotion]
            names = [f.name for f in files]
            chosen_name = st.selectbox("Chọn file", names, key="sample_file")
            chosen_path = files[names.index(chosen_name)]
            st.audio(str(chosen_path))
            if st.button("Analyze", key="analyze_sample", type="primary"):
                with st.spinner("Đang phân tích..."):
                    result = _predict_path(chosen_path)
                render_result(result)


if __name__ == "__main__":
    main()
