---
title: Vietnamese Speech Emotion Recognition
emoji: 🇻🇳
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: streamlit_app.py
app_port: 8501
pinned: false
license: mit
suggested_hardware: cpu-basic
python_version: 3.11
models:
  - MERaLiON/MERaLiON-SER-v1
---

# Vietnamese Speech Emotion Recognition

A Streamlit app that classifies Vietnamese speech into 4 emotions
(`happy / neutral / sad / angry`) using
[`MERaLiON/MERaLiON-SER-v1`](https://huggingface.co/MERaLiON/MERaLiON-SER-v1)
— a Whisper-Medium + LoRA + ECAPA-TDNN speech-emotion recogniser.

The model was benchmarked on the only public labeled Vietnamese
emotion dataset,
[`hustep-lab/ViSEC`](https://huggingface.co/datasets/hustep-lab/ViSEC),
at **40.25 % accuracy / 40.70 % macro-F1** on a balanced 400-clip
subset (100 per class). Numbers come from
`bench/results/scores.json` — measured, not synthesised.

## What it does

- Accepts an audio file (wav/mp3/ogg/m4a/flac/webm) or microphone
  recording
- Resamples to 16 kHz mono
- Runs MERaLiON-SER-v1 and surfaces the **real softmax
  distribution** from the model (not a uniform-fabricated one)
- Returns the top emotion + the full 7-bucket probability vector

## 7 raw emotion classes (MERaLiON)

```
neutral · happy · sad · angry · fearful · disgusted · surprised
```

The 4-class ViSEC benchmark uses only `happy / neutral / sad / angry`;
the extra 3 raw buckets (`fearful / disgusted / surprised`) are
reported in the UI's bar chart but are not scored by the ViSEC bench.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens `http://localhost:8501`. First inference triggers the MERaLiON
load (~30 s on CPU, ~9 s on GPU at FP16). On hosts with < 1.5 GiB
free RAM, the adapter refuses to load — close other apps and retry.

### GPU runs

On a CUDA-capable host (RTX 4050 etc.), install a CUDA-matched torch
wheel separately (the Space build is CPU-only by design):

```bash
.venv/bin/pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

Then re-run `streamlit run streamlit_app.py` — the adapter
auto-detects CUDA and loads at FP16 (~600 MB VRAM).

## Run tests

```bash
.venv/bin/python3 -m pytest -v -m "not slow"
```

Tests enforce:

- `class_scores` are real softmax outputs (sum to 1, not uniform,
  no NaN, no negative values)
- The inference layer is a singleton (one adapter in memory at a time)
- `predict()` returns the full UI-shaped dict

## Deploying to a new HF Space

```bash
# 1. Create an empty Space at https://huggingface.co/new-space
#    SDK = Streamlit · Hardware = CPU basic · License = MIT
# 2. Push this repo:
git init && git add . && git commit -m "Streamlit Space - MERaLiON Việt"
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

Build time: ~3 min. The Space becomes publicly available at
`https://huggingface.co/spaces/<your-username>/<space-name>`.

## Limitations

- **The user-defined stop criterion (≥ 65 % ViSEC accuracy) is
  unmet.** MERaLiON-SER-v1 is at 40.25 % on the 400-clip balanced
  ViSEC bench. It is the best public Vietnamese emotion classifier
  available without on-device fine-tuning, but it is below the
  SOTA-grade SER threshold.
- **`happy` and `sad` are the weakest classes** (F1 ≈ 0.31–0.33). The
  model rarely predicts them as the top class; they tend to lose to
  `neutral` even when they're the ground-truth label.
- **MERaLiON is heavy**: ~3 GB RAM on CPU, ~600 MB VRAM at FP16 on
  GPU. Hosts with < 1.5 GiB free RAM cannot run this Space.
- **`vietnamese_verified` is `best_effort`** — the model's
  Vietnamese performance has been benchmarked (40.25 % on ViSEC)
  but not formally cross-validated against human labels.
- The model is trained on multilingual data with Vietnamese as a
  secondary language; a Vietnamese-only fine-tune would likely
  perform better but is out of scope for this repo.

See `docs/BENCHMARK.md` for the full benchmark methodology and
per-class F1 numbers.

## Documentation map

| Doc | Audience | Purpose |
|---|---|---|
| [README.md](README.md) | everyone | quick overview + install |
| [docs/USAGE.md](docs/USAGE.md) | end user | how to use the app, audio formats, FAQ |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | developer | 3-layer design, module map, adapter pattern, pipeline |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | researcher | benchmark methodology + per-class numbers |
| [docs/PRESENTATION.md](docs/PRESENTATION.md) | presenter | 25-slide script (Vietnamese) |

## Stack

- [MERaLiON/MERaLiON-SER-v1](https://huggingface.co/MERaLiON/MERaLiON-SER-v1) — model
- [Streamlit](https://streamlit.io/) — UI
- [Hugging Face Spaces](https://huggingface.co/spaces) — hosting
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — model loader
- No React, no FastAPI, no provider registry — single-model, single-file.
