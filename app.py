"""Gradio Space: Vietnamese Speech Emotion Recognition.

Single-model Space — loads MERaLiON-SER-v1 on startup. UI: compact
2-column responsive layout (Input | Result), real `gr.Tabs` with 3 input
sources (Upload / Microphone / Samples), emoji-driven emotion hero card,
metadata pills, full-width class-score bar chart.
"""
from __future__ import annotations

import glob
import html
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from src import inference


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("vser.app")


# ── design tokens ──────────────────────────────────────────────────────────

EMOTION_COLORS: Dict[str, str] = {
    "angry":    "#dc2626",   # red 600
    "disgusted": "#7c3aed",  # violet 600
    "fearful":  "#6366f1",   # indigo 500
    "happy":    "#d97706",   # amber 600
    "neutral":  "#475569",   # slate 600
    "sad":      "#2563eb",   # blue 600
    "surprised":"#0891b2",   # cyan 600
    "other":    "#64748b",
    "unknown":  "#94a3b8",
}

EMOTION_EMOJI: Dict[str, str] = {
    "angry":    "😠",
    "disgusted":"🤢",
    "fearful":  "😨",
    "happy":    "😄",
    "neutral":  "😐",
    "sad":      "😢",
    "surprised":"😲",
    "other":    "❔",
    "unknown":  "❓",
}

SAMPLE_AUDIO_DIR = "sample_audio"
SUPPORTED_AUDIO_TYPES = [".wav", ".mp3", ".ogg", ".m4a", ".flac", ".webm"]


# ── CSS ───────────────────────────────────────────────────────────────────

APP_CSS = r"""
:root {
    --surface:        #ffffff;
    --surface-muted:  #f6f8fb;
    --surface-strong: #eef2f7;
    --ink:            #0f172a;
    --ink-soft:       #334155;
    --muted:          #64748b;
    --line:           #e2e8f0;
    --line-soft:      #eef2f7;
    --primary:        #1d4ed8;
    --primary-soft:   #eff6ff;
    --primary-strong: #1e40af;
    --primary-ring:   rgba(29, 78, 216, 0.18);
    --shadow-sm:      0 1px 2px rgba(15, 23, 42, 0.04);
    --shadow-md:      0 1px 3px rgba(15, 23, 42, 0.05), 0 6px 16px rgba(15, 23, 42, 0.04);
    --radius:         12px;
    --radius-sm:      8px;
    --gap:            20px;
}

/* base */
body, .gradio-container {
    background: #f4f7fb !important;
    color: var(--ink) !important;
    font-family: Inter, "SF Pro Display", ui-sans-serif, system-ui,
                 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-feature-settings: "ss01" on, "cv11" on;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 24px 22px 56px !important;
}
.footer { display: none !important; }

/* ── compact header ────────────────────────────────────────────────── */
.header-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px !important;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    margin-bottom: 20px;
}
.header-mark {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
    color: #ffffff;
    font-size: 22px;
    flex: 0 0 40px;
    box-shadow: 0 4px 12px rgba(29, 78, 216, 0.18);
}
.header-text h1 {
    margin: 0 !important;
    color: var(--ink) !important;
    font-size: 18px !important;
    line-height: 1.2 !important;
    font-weight: 720 !important;
    letter-spacing: -0.01em;
}
.header-text p {
    margin: 2px 0 0 !important;
    color: var(--muted) !important;
    font-size: 13px !important;
    line-height: 1.45 !important;
}
.header-spacer { flex: 1; }
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--primary-soft);
    color: var(--primary);
    font-size: 12px;
    font-weight: 650;
}
.header-badge .dot {
    width: 6px; height: 6px; border-radius: 50%; background: #16a34a;
    box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.18);
}

/* ── shared card ────────────────────────────────────────────────────── */
.panel {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    box-shadow: var(--shadow-md);
}
.panel + .panel { margin-top: 16px; }

.panel-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 14px;
}
.panel-head .title {
    color: var(--ink);
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.005em;
    margin: 0;
}
.panel-head .hint {
    color: var(--muted);
    font-size: 12.5px;
    margin: 0;
}
.panel-rule { height: 1px; background: var(--line-soft); margin: 14px 0; }

/* ── column layout ──────────────────────────────────────────────────── */
.app-grid {
    gap: 18px !important;
    align-items: stretch !important;
}
.col-input  { gap: 14px !important; }
.col-result { gap: 14px !important; }

/* ── Tabs (real gr.Tabs) ────────────────────────────────────────────── */
.source-tabs > .tab-nav {
    border-bottom: 1px solid var(--line) !important;
    gap: 4px !important;
}
.source-tabs > .tab-nav button {
    color: var(--muted) !important;
    font-weight: 650 !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    padding: 8px 14px !important;
    font-size: 13.5px !important;
}
.source-tabs > .tab-nav button.selected {
    color: var(--primary) !important;
    background: var(--primary-soft) !important;
    box-shadow: inset 0 -2px 0 var(--primary);
}

/* ── input surfaces ─────────────────────────────────────────────────── */
#upload-file,
#record-audio {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}
#upload-file { min-height: 132px !important; }
#record-audio { min-height: 116px !important; }
#upload-file *,
#record-audio * { color: var(--ink) !important; }

#upload-file button,
#upload-file [role="button"] {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
    border-radius: var(--radius-sm) !important;
    min-height: 38px !important;
    font-weight: 650 !important;
}
#upload-file button:hover,
#upload-file [role="button"]:hover {
    background: var(--primary-soft) !important;
    border-color: #93c5fd !important;
    color: var(--primary) !important;
}

/* record buttons: subtle recolor */
#record-audio button,
#record-audio [role="button"] {
    background: #f8fafc !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink) !important;
    font-weight: 650 !important;
    min-height: 36px !important;
}
#record-audio button:hover,
#record-audio [role="button"]:hover {
    background: var(--primary-soft) !important;
    border-color: #93c5fd !important;
    color: var(--primary) !important;
}

#record-audio button[aria-label*="Record"] svg,
#record-audio button[title*="Record"] svg {
    color: #dc2626 !important;
}
#record-audio button[aria-label*="Stop"] svg,
#record-audio button[title*="Stop"] svg {
    color: var(--muted) !important;
}

#record-audio select,
#record-audio input,
#record-audio [role="combobox"],
#record-audio [role="listbox"] {
    background: #ffffff !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
    min-height: 36px !important;
    border-radius: var(--radius-sm) !important;
}

#samples-list { gap: 6px !important; }
.samples-head {
    color: var(--muted);
    font-size: 12.5px;
    margin: 0 0 10px !important;
}

/* ── Analyze button ─────────────────────────────────────────────────── */
#analyze-button {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: #ffffff !important;
    border-radius: var(--radius-sm) !important;
    min-height: 46px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    box-shadow: 0 6px 14px var(--primary-ring) !important;
    margin-top: 4px;
}
#analyze-button:hover {
    background: var(--primary-strong) !important;
    border-color: var(--primary-strong) !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 18px var(--primary-ring) !important;
}

/* ── result hero card ───────────────────────────────────────────────── */
.result-empty,
.result-error,
.result-card {
    width: 100%;
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: var(--surface);
}
.result-empty {
    padding: 28px 22px;
    text-align: center;
    color: var(--muted);
    font-size: 13.5px;
}
.result-empty .ico { font-size: 28px; opacity: 0.6; margin-bottom: 6px; }
.result-empty strong { color: var(--ink); font-size: 15px; display: block; margin-bottom: 2px; }

.result-error {
    padding: 18px;
    background: #fef2f2;
    border-color: #fecaca;
    color: #991b1b;
    font-size: 14px;
    line-height: 1.5;
}

/* emotion hero */
.emotion-hero {
    padding: 18px 20px 16px;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 18px;
    align-items: center;
    border-bottom: 1px solid var(--line-soft);
}
.emotion-side { display: flex; align-items: center; gap: 14px; min-width: 0; }
.emotion-emoji {
    width: 56px; height: 56px;
    border-radius: 14px;
    display: grid; place-items: center;
    font-size: 30px;
    background: var(--emotion-bg, #f1f5f9);
    flex: 0 0 56px;
}
.emotion-text { min-width: 0; }
.emotion-kicker {
    color: var(--muted);
    font-size: 11.5px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.emotion-name {
    font-size: clamp(28px, 3.6vw, 44px);
    line-height: 1.05;
    font-weight: 760;
    color: var(--ink);
    margin: 0;
    letter-spacing: -0.02em;
    text-transform: capitalize;
}
.emotion-aux {
    margin-top: 4px;
    color: var(--muted);
    font-size: 12.5px;
    font-weight: 500;
    text-transform: capitalize;
    letter-spacing: 0.02em;
}

/* confidence ring */
.conf-ring {
    width: 84px;
    height: 84px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background:
        conic-gradient(var(--emotion-color, var(--primary)) calc(var(--confidence) * 1%), #e2e8f0 0);
    flex: 0 0 84px;
}
.conf-ring-inner {
    width: 68px;
    height: 68px;
    border-radius: 50%;
    display: grid; place-items: center;
    background: #ffffff;
    text-align: center;
    line-height: 1;
}
.conf-ring-num {
    color: var(--ink);
    font-size: 19px;
    font-weight: 760;
    letter-spacing: -0.01em;
}
.conf-ring-label {
    display: block;
    margin-top: 2px;
    color: var(--muted);
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* meta pills */
.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 14px 20px 0;
}
.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-height: 30px;
    padding: 5px 12px;
    border-radius: 999px;
    background: var(--surface-muted);
    border: 1px solid var(--line);
    color: var(--ink-soft);
    font-size: 12.5px;
    font-weight: 600;
}
.meta-pill .k { color: var(--muted); font-weight: 500; }
.meta-pill .v { color: var(--ink); font-weight: 700; }
.meta-pill strong { color: var(--ink); font-weight: 760; }

/* chart wrapper */
.plot-block {
    padding: 16px 18px 4px;
}
.plot-title {
    color: var(--ink);
    font-size: 13px;
    font-weight: 700;
    margin: 0 0 8px !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* tighten BarPlot internals */
.gradio-container .plot-container,
.gradio-container .plot-container > div {
    border-radius: var(--radius-sm) !important;
}
.panel .block.gradio-box {
    border-color: var(--line) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: none !important;
}

/* ── responsive ─────────────────────────────────────────────────────── */
@media (max-width: 900px) {
    .gradio-container { padding: 16px 12px 28px !important; }
    .header-text h1 { font-size: 16px !important; }
    .panel { padding: 16px !important; }
    .emotion-hero { grid-template-columns: minmax(0, 1fr); }
    .conf-ring { width: 72px; height: 72px; }
    .conf-ring-inner { width: 58px; height: 58px; }
    .conf-ring-num { font-size: 16px; }
    .meta-row { padding: 12px 16px 0; }
    .plot-block { padding: 12px 16px 4px; }
}
"""


# ── helpers ────────────────────────────────────────────────────────────────

def _color_for(label: str) -> str:
    return EMOTION_COLORS.get((label or "").lower(), "#475569")


def _emoji_for(label: str) -> str:
    return EMOTION_EMOJI.get((label or "").lower(), "🎙️")


def _bar_plot_data(class_scores):
    if not class_scores:
        return None
    items = sorted(class_scores.items(), key=lambda kv: -kv[1])
    return pd.DataFrame(
        {
            "emotion": [k for k, _ in items],
            "probability": [round(float(v), 4) for _, v in items],
            "color": [_color_for(k) for k, _ in items],
        }
    )


def _discover_samples(max_per_class: int = 3) -> List[List[Any]]:
    """Build Examples rows: [[label, filepath], ...]."""
    rows: List[List[Any]] = []
    if not os.path.isdir(SAMPLE_AUDIO_DIR):
        return rows
    # emotion-named subdirs → use subdir name as label
    for sub in sorted(os.listdir(SAMPLE_AUDIO_DIR)):
        full = os.path.join(SAMPLE_AUDIO_DIR, sub)
        if not os.path.isdir(full):
            continue
        if sub.startswith(("_", ".")):
            continue
        for path in sorted(glob.glob(os.path.join(full, "*"))[:max_per_class]):
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_AUDIO_TYPES:
                continue
            rows.append([sub, path])
    return rows


def _model_status_html(model_info: Optional[Dict[str, Any]]) -> str:
    ready = bool(model_info and model_info.get("model_id"))
    model_id = (model_info or {}).get("model_id", "meralion-ser-v1")
    provider = (model_info or {}).get("provider", "meralion")
    return f"""
    <div class="header-row">
      <div class="header-mark" aria-hidden="true">🇻🇳</div>
      <div class="header-text">
        <h1>Vietnamese Speech Emotion</h1>
        <p>Phân tích cảm xúc trong giọng nói tiếng Việt — đưa ra nhãn top-1 và phân phối xác suất thật từ model.</p>
      </div>
      <div class="header-spacer"></div>
      <div class="header-badge">
        <span class="dot"></span>
        <span>{html.escape(provider)} · {html.escape(model_id.split('/')[-1])}</span>
      </div>
    </div>
    """ if ready else f"""
    <div class="header-row">
      <div class="header-mark" aria-hidden="true">🇻🇳</div>
      <div class="header-text">
        <h1>Vietnamese Speech Emotion</h1>
        <p>Phân tích cảm xúc trong giọng nói tiếng Việt — đưa ra nhãn top-1 và phân phối xác suất thật từ model.</p>
      </div>
    </div>
    """


def _empty_result_html() -> str:
    return """
    <div class="result-empty">
      <div class="ico">🎧</div>
      <strong>Chưa có kết quả</strong>
      <span>Chọn audio ở panel bên trái rồi nhấn <b>Analyze</b>.</span>
    </div>
    """


def _result_hero_html(label: str, raw_label: str,
                      confidence: float, latency_ms: int,
                      device: str, dtype: str) -> str:
    color = _color_for(label)
    emoji = _emoji_for(label)
    bg = color + "1a"          # ~10% alpha hex suffix
    pct = max(0.0, min(1.0, float(confidence))) * 100
    return f"""
    <div class="result-card">
      <div class="emotion-hero" style="--emotion-color:{color}; --emotion-bg:{bg};">
        <div class="emotion-side">
          <div class="emotion-emoji" aria-hidden="true">{emoji}</div>
          <div class="emotion-text">
            <div class="emotion-kicker">Top emotion</div>
            <div class="emotion-name">{html.escape(label)}</div>
            <div class="emotion-aux">{html.escape(raw_label)}</div>
          </div>
        </div>
        <div class="conf-ring" style="--confidence:{pct:.1f};" aria-label="Confidence {pct:.1f} percent">
          <div class="conf-ring-inner">
            <span class="conf-ring-num">{pct:.0f}<span style="font-size:0.7em">%</span></span>
            <span class="conf-ring-label">conf</span>
          </div>
        </div>
      </div>
      <div class="meta-row">
        <span class="meta-pill"><span class="k">Raw label</span> · <span class="v">{html.escape(raw_label)}</span></span>
        <span class="meta-pill"><span class="k">Latency</span> · <span class="v">{int(latency_ms)} ms</span></span>
        <span class="meta-pill"><span class="k">Device</span> · <span class="v">{html.escape(device.upper())}</span></span>
        <span class="meta-pill"><span class="k">Precision</span> · <span class="v">{html.escape(dtype)}</span></span>
      </div>
    </div>
    """


# ── inference wiring ──────────────────────────────────────────────────────

def _run_inference(audio_path: Optional[str]) -> Tuple[str, Any, Any]:
    if not audio_path:
        return (_empty_result_html(), None, None)

    try:
        result = inference.predict(audio_path)
    except Exception as exc:                          # noqa: BLE001
        msg = inference.format_error(exc)
        log.exception("inference failed")
        err_html = f"""
        <div class="result-error">
          <strong>Lỗi phân tích</strong> · {html.escape(msg)}
        </div>
        """
        return (err_html, None, None)

    label = result["label"]
    raw = result.get("raw_label") or label
    conf = float(result.get("confidence") or 0.0)
    latency = int(result.get("latency_ms") or 0)
    device = result.get("device", "cpu")
    dtype = result.get("dtype", "float32")
    cs = result.get("class_scores") or {}

    hero_html = _result_hero_html(
        label=label, raw_label=raw,
        confidence=conf, latency_ms=latency,
        device=device, dtype=dtype,
    )
    plot = _bar_plot_data(cs)
    label_pct = f"{label} · {conf*100:.1f}%"
    return (hero_html, plot, label_pct)


def _as_filepath(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    path = getattr(value, "path", None) or getattr(value, "name", None)
    return str(path) if path else None


def _resolve_audio(upload_file: Any, record_path: Any, sample_value: Any) -> Optional[str]:
    # .Examples auto-fills the first input it owns; we route by priority:
    if sample_value not in (None, "", []):
        return _as_filepath(sample_value) or (sample_value if isinstance(sample_value, str) else None)
    path = _as_filepath(upload_file)
    if path:
        return path
    return _as_filepath(record_path)


# ── demo builder ──────────────────────────────────────────────────────────

def build_demo(model_info: Optional[Dict[str, Any]] = None) -> gr.Blocks:
    sample_rows = _discover_samples()

    # state used for sample routing (only used when sample chosen)
    sample_router = gr.State(value=None)

    with gr.Blocks(
        title="Vietnamese Speech Emotion",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=APP_CSS,
    ) as demo:
        gr.HTML(_model_status_html(model_info))

        with gr.Row(elem_classes="app-grid"):
            # ── LEFT: input ────────────────────────────────────────────
            with gr.Column(scale=1, min_width=380, elem_classes="col-input"):
                with gr.Column(elem_classes="panel"):
                    gr.HTML(
                        '<div class="panel-head">'
                        '<div class="title">I · Audio input</div>'
                        '<div class="hint">Upload · Mic · Sample</div>'
                        "</div>"
                    )
                    with gr.Tabs(elem_classes="source-tabs") as source_tabs:
                        with gr.TabItem("📁 Upload file", id="tab-upload"):
                            upload_in = gr.File(
                                label="Upload audio file",
                                file_count="single",
                                file_types=SUPPORTED_AUDIO_TYPES,
                                type="filepath",
                                elem_id="upload-file",
                            )
                        with gr.TabItem("🎙️ Microphone", id="tab-mic"):
                            record_in = gr.Audio(
                                label="Record voice",
                                type="filepath",
                                sources=["microphone"],
                                elem_id="record-audio",
                            )
                        with gr.TabItem("🎵 Samples", id="tab-sample"):
                            samples_md = gr.Markdown(
                                "Chọn một clip mẫu — sample sẽ tự động fill vào ô phân tích.",
                                elem_classes="samples-head",
                            )
                            sample_in = gr.Audio(
                                label="Selected sample",
                                type="filepath",
                                visible=False,
                            )
                            sample_label_in = gr.Textbox(
                                label="Sample label",
                                visible=False,
                            )
                            if sample_rows:
                                gr.Examples(
                                    examples=[[r[1]] for r in sample_rows],
                                    inputs=[sample_in],
                                    label=None,
                                )

                with gr.Column(elem_classes="panel"):
                    gr.HTML(
                        '<div class="panel-head">'
                        '<div class="title">II · Run analysis</div>'
                        '<div class="hint">Cần ≥ 5–10 giây audio rõ tiếng</div>'
                        "</div>"
                    )
                    btn = gr.Button(
                        "🚀 Analyze",
                        variant="primary",
                        elem_id="analyze-button",
                    )

            # ── RIGHT: result ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=380, elem_classes="col-result"):
                with gr.Column(elem_classes="panel"):
                    gr.HTML(
                        '<div class="panel-head">'
                        '<div class="title">Result</div>'
                        '<div class="hint">Top-1 + phân phối class</div>'
                        "</div>"
                    )
                    out_top = gr.HTML(value=_empty_result_html())
                    out_plot = gr.BarPlot(
                        x="emotion",
                        y="probability",
                        title=None,
                        x_title="Emotion",
                        y_title="Probability",
                        y_lim=[0, 1],
                        height=300,
                        show_label=False,
                        container=False,
                    )

        # ── events ──────────────────────────────────────────────────────
        btn.click(
            fn=lambda u, r, s: _run_inference(_resolve_audio(u, r, s)),
            inputs=[upload_in, record_in, sample_in],
            outputs=[out_top, out_plot],
        )

        # Auto-fill main inputs when a sample is chosen (so Analyze picks it up).
        # Also auto-jump to result panel by NOT auto-running (user still clicks Analyze).

    return demo


def main() -> None:
    log.info("warming up default adapter...")
    info = inference.warmup()
    log.info(
        "ready: provider=%s model=%s device=%s labels=%d load=%.2fs",
        info["provider"], info["model_id"], info["device"],
        len(info["labels"]), info["load_seconds"] or 0.0,
    )
    demo = build_demo(info)
    demo.queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )


if __name__ == "__main__":
    main()
