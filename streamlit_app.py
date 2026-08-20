"""Streamlit front-end for Vietnamese Speech Emotion Recognition.

Three analysis layers, each independently switchable in the sidebar:

1. **Emotion** (always on) — ``src.inference`` → MERaLiON-SER-v1.
2. **Transcript** (opt-in) — ``src.asr`` → PhoWhisper.
3. **Timeline** (opt-in) — ``src.timeline`` → sliding-window emotion.

Layers 2 and 3 are opt-in because each costs a model download or a
multiple of the single-shot latency; the sidebar makes that trade
explicit rather than hiding it.

Run:

    .venv/bin/streamlit run streamlit_app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from src import asr, inference, timeline
from src.device import describe as describe_device
from src.device import resolve_device

SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "ogg", "m4a", "flac", "webm"]
VISEC_CLASSES = ("happy", "neutral", "sad", "angry")

# Chosen to stay legible on both the light and dark Streamlit themes —
# the previous palette used near-white backgrounds that vanished in dark
# mode. These are mid-tone hues that hold contrast either way.
EMOTION_COLORS = {
    "angry": "#ef4444",
    "happy": "#f59e0b",
    "neutral": "#64748b",
    "sad": "#3b82f6",
    "fearful": "#a855f7",
    "disgusted": "#14b8a6",
    "surprised": "#ec4899",
    "other": "#94a3b8",
}
DEFAULT_COLOR = "#94a3b8"

st.set_page_config(
    page_title="Vietnamese Speech Emotion Recognition",
    page_icon="🇻🇳",
    layout="wide",
)


# --------------------------------------------------------------------------
# Model loading (cached across reruns — Streamlit re-executes this file on
# every widget interaction, so an uncached load would refetch the model)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Đang tải model cảm xúc MERaLiON-SER-v1…")
def _warm_emotion() -> dict:
    return inference.warmup()


@st.cache_resource(show_spinner="Đang tải model nhận dạng giọng nói PhoWhisper…")
def _warm_asr(model_name: str) -> dict:
    # ``model_name`` is part of the cache key so switching size in the
    # sidebar rebuilds the pipeline instead of silently reusing the old one.
    import os
    os.environ[asr.MODEL_ENV] = model_name
    asr.unload()
    asr._model_id = model_name
    return asr.warmup()


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def analyze(path: Path, *, do_timeline: bool, do_asr: bool,
            window_sec: float, hop_sec: float) -> dict:
    """Run the enabled layers over one file and return a combined result."""
    out: dict[str, Any] = {"path": str(path)}

    with st.spinner("Đang phân tích cảm xúc…"):
        out["overall"] = inference.predict(str(path))

    if do_timeline:
        bar = st.progress(0.0, text="Đang quét cảm xúc theo thời gian…")
        try:
            out["timeline"] = timeline.analyze_timeline(
                path, window_sec=window_sec, hop_sec=hop_sec,
                progress=lambda done, total: bar.progress(
                    done / total, text=f"Cửa sổ {done}/{total}"),
            )
        finally:
            bar.empty()

    if do_asr:
        with st.spinner("Đang bóc băng bằng PhoWhisper…"):
            try:
                out["asr"] = asr.transcribe(path)
            except Exception as exc:  # noqa: BLE001
                out["asr_error"] = inference.format_error(exc)

    return out


def _predict_bytes_to_tempfile(audio_bytes: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        return Path(tmp.name)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_verdict(result: dict) -> None:
    label = result["label"]
    color = EMOTION_COLORS.get(label, DEFAULT_COLOR)
    st.markdown(
        f"<div style='padding:16px 20px;border-radius:12px;"
        f"border:1px solid {color};border-left:6px solid {color};'>"
        f"<div style='font-size:12px;opacity:.7;text-transform:uppercase;"
        f"letter-spacing:.06em;font-weight:600;'>Cảm xúc dự đoán</div>"
        f"<div style='font-size:38px;font-weight:750;color:{color};"
        f"text-transform:capitalize;line-height:1.15;'>{label}</div>"
        f"<div style='opacity:.7;font-size:13px;'>"
        f"độ tin cậy {result['confidence']:.0%} · nhãn thô "
        f"<code>{result['raw_label']}</code> · {result['latency_ms']} ms · "
        f"{result['device']}</div></div>",
        unsafe_allow_html=True,
    )
    if label not in VISEC_CLASSES:
        st.info(
            f"Nhãn `{result['raw_label']}` nằm ngoài 4 lớp ViSEC "
            "(happy/neutral/sad/angry) — MERaLiON có 7 nhãn thô."
        )


def render_distribution(scores: dict[str, float]) -> None:
    """Horizontal bars via Streamlit's native chart (theme-aware)."""
    df = (pd.DataFrame({"Cảm xúc": list(scores),
                        # ProgressColumn applies `format` to the raw value, so
                        # feed it percentage points rather than a 0-1 fraction.
                        "Xác suất": [v * 100 for v in scores.values()]})
          .sort_values("Xác suất", ascending=False))
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Xác suất": st.column_config.ProgressColumn(
                "Xác suất", format="%.1f%%", min_value=0.0, max_value=100.0),
        },
    )


def render_timeline(tl: dict, asr_segments: Optional[list] = None) -> None:
    """Dominant-label strip stacked above a probability area chart.

    The two are vertically concatenated into one Altair spec so they
    share an x scale — drawn as separate charts they each grew their own
    time axis and drifted out of alignment.
    """
    import altair as alt

    rows = [
        {"Thời gian": seg["mid"], "Cảm xúc": name, "Xác suất": score}
        for seg in tl["segments"]
        for name, score in (seg["class_scores"] or {}).items()
    ]
    if not rows:
        st.warning("Không có dữ liệu timeline.")
        return

    def colour_scale(names: list[str]) -> "alt.Scale":
        return alt.Scale(domain=names,
                         range=[EMOTION_COLORS.get(n, DEFAULT_COLOR) for n in names])

    strip_rows = [{"start": s["start"], "end": s["end"], "Cảm xúc": s["label"],
                   "Độ tin cậy": s["confidence"]} for s in tl["segments"]]
    strip = (
        alt.Chart(pd.DataFrame(strip_rows))
        .mark_rect()
        .encode(
            x=alt.X("start:Q", title=None, axis=None),
            x2="end:Q",
            color=alt.Color("Cảm xúc:N",
                            scale=colour_scale(sorted({r["Cảm xúc"]
                                                       for r in strip_rows})),
                            legend=None),
            tooltip=["start", "end", "Cảm xúc",
                     alt.Tooltip("Độ tin cậy:Q", format=".1%")],
        )
        .properties(height=30, title="Nhãn chủ đạo từng cửa sổ")
    )

    area = (
        alt.Chart(pd.DataFrame(rows))
        .mark_area(interpolate="monotone", opacity=0.85)
        .encode(
            x=alt.X("Thời gian:Q", title="Thời gian (giây)"),
            y=alt.Y("Xác suất:Q", stack="normalize", title="Tỉ lệ",
                    axis=alt.Axis(format="%")),
            color=alt.Color("Cảm xúc:N",
                            scale=colour_scale(sorted({r["Cảm xúc"]
                                                       for r in rows})),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("Thời gian:Q", format=".2f"), "Cảm xúc",
                     alt.Tooltip("Xác suất:Q", format=".1%")],
        )
        .properties(height=260)
    )

    st.altair_chart(
        alt.vconcat(strip, area, spacing=6).resolve_scale(
            x="shared", color="independent"),
        use_container_width=True,
    )

    summary = tl["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cảm xúc chủ đạo", str(summary["label"]).capitalize())
    c2.metric("Độ nhất quán", f"{summary['dominant_share']:.0%}",
              help="Tỉ lệ cửa sổ có cùng nhãn với cảm xúc chủ đạo. "
                   "Thấp = cảm xúc thay đổi nhiều trong bài.")
    c3.metric("Số cửa sổ", summary["num_segments"],
              help=f"{tl['duration_sec']:.1f}s · {tl['total_latency_ms']} ms")


def render_transcript(asr_result: dict, tl: Optional[dict]) -> None:
    st.markdown("#### 📝 Nội dung nói")
    if not asr_result.get("text"):
        st.warning("Không nhận được nội dung nào từ audio.")
        return
    st.markdown(f"> {asr_result['text']}")
    st.caption(
        f"PhoWhisper `{asr_result['model_id']}` · {asr_result['latency_ms']} ms "
        f"· {asr_result['device']}"
    )

    segs = asr_result.get("segments") or []
    if not segs:
        return

    # Colour each spoken line by the emotion window it sits inside.
    def emotion_at(t: float) -> str:
        if not tl:
            return "other"
        for s in tl["segments"]:
            if s["start"] <= t <= s["end"]:
                return s["label"]
        return "other"

    st.markdown("##### Từng câu")
    for s in segs:
        mid = (s["start"] + s["end"]) / 2
        emo = emotion_at(mid)
        color = EMOTION_COLORS.get(emo, DEFAULT_COLOR)
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:4px 0 4px 10px;"
            f"margin:6px 0;'>"
            f"<span style='opacity:.6;font-size:12px;font-variant-numeric:"
            f"tabular-nums;'>{s['start']:6.1f}s</span> "
            f"<span style='color:{color};font-size:12px;font-weight:600;"
            f"text-transform:capitalize;'>{emo}</span><br>{s['text']}</div>",
            unsafe_allow_html=True,
        )


def render_bench_banner() -> None:
    row = inference.bench_scores().get("meralion-ser-v1")
    if not row:
        return
    status = "✅ đạt" if row.get("eligible_for_deploy") else "❌ chưa đạt"
    st.caption(
        f"Benchmark ViSEC (400 clip, 4 lớp): accuracy "
        f"**{row.get('accuracy', 0.0):.2%}** · tiêu chí ≥65% {status} · "
        f"nguồn `bench/results/scores.json`"
    )


def render_results(res: dict) -> None:
    tl = res.get("timeline")
    left, right = st.columns([3, 2])
    with left:
        render_verdict(res["overall"])
    with right:
        render_distribution(res["overall"]["class_scores"])

    if tl:
        st.markdown("#### 📈 Cảm xúc theo thời gian")
        render_timeline(tl)
    if res.get("asr_error"):
        st.error(f"Bóc băng lỗi: {res['asr_error']}")
    elif res.get("asr"):
        render_transcript(res["asr"], tl)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def main() -> None:
    st.title("🇻🇳 Vietnamese Speech Emotion Recognition")

    with st.sidebar:
        st.header("⚙️ Tuỳ chọn")
        device = resolve_device()
        st.success(f"Thiết bị: **{describe_device(device)}** (`{device}`)")

        do_timeline = st.toggle(
            "📈 Cảm xúc theo thời gian", value=True,
            help="Cắt audio thành cửa sổ trượt và chấm cảm xúc từng đoạn.")
        window_sec = hop_sec = 0.0
        if do_timeline:
            window_sec = st.slider("Độ dài cửa sổ (giây)", 1.5, 6.0,
                                   timeline.WINDOW_SEC, 0.5)
            hop_sec = st.slider("Bước nhảy (giây)", 0.5, 3.0,
                                timeline.HOP_SEC, 0.5,
                                help="Nhỏ hơn = đường mượt hơn nhưng chậm hơn.")

        do_asr = st.toggle(
            "📝 Bóc băng (PhoWhisper)", value=False,
            help="Lần đầu bật sẽ tải model (~1 GB).")
        asr_model = asr.DEFAULT_MODEL
        if do_asr:
            asr_model = st.selectbox(
                "Cỡ model ASR",
                ["vinai/PhoWhisper-tiny", "vinai/PhoWhisper-base",
                 "vinai/PhoWhisper-small", "vinai/PhoWhisper-medium"],
                index=2)

    try:
        info = _warm_emotion()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không load được model cảm xúc: {inference.format_error(exc)}")
        st.stop()

    render_bench_banner()
    st.caption(
        f"Model cảm xúc `{info['model_id']}` · {len(info['labels'])} lớp · "
        f"`{info['device']}` ({info['extra'].get('dtype')}) · "
        f"load {info['load_seconds']}s"
    )

    if do_asr:
        try:
            _warm_asr(asr_model)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Không load được PhoWhisper: {inference.format_error(exc)}")
            do_asr = False

    opts = dict(do_timeline=do_timeline, do_asr=do_asr,
                window_sec=window_sec or timeline.WINDOW_SEC,
                hop_sec=hop_sec or timeline.HOP_SEC)

    tab_upload, tab_mic = st.tabs(["📁 Tải lên", "🎙️ Thu âm"])

    with tab_upload:
        uploaded = st.file_uploader("Chọn file audio",
                                    type=SUPPORTED_AUDIO_TYPES, key="uploader")
        if uploaded is not None:
            st.audio(uploaded)
            if st.button("Phân tích", key="go_upload", type="primary"):
                tmp = _predict_bytes_to_tempfile(
                    uploaded.getvalue(), Path(uploaded.name).suffix or ".wav")
                try:
                    render_results(analyze(tmp, **opts))
                finally:
                    tmp.unlink(missing_ok=True)

    with tab_mic:
        recorded = st.audio_input("Thu âm giọng nói (~5–30 giây)", key="mic")
        if recorded is not None and st.button("Phân tích", key="go_mic",
                                              type="primary"):
            tmp = _predict_bytes_to_tempfile(recorded.getvalue(), ".wav")
            try:
                render_results(analyze(tmp, **opts))
            finally:
                tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
