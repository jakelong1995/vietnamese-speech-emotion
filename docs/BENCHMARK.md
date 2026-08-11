# Benchmark — `hustep-lab/ViSEC` 4-class

`bench/results/scores.json` is the source of truth for the
single adapter the Gradio UI loads. This document explains the
methodology and what each entry means.

## Other docs

- [README.md](../README.md) — quick overview, install, deploy
- [docs/USAGE.md](USAGE.md) — end-user guide (UI + CLI + FAQ)
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — 3-layer design, module map, adapter pattern, pipeline
- [docs/PRESENTATION.md](PRESENTATION.md) — 25-slide presentation script (Vietnamese)

## Setup

- **Dataset**: [hustep-lab/ViSEC](https://huggingface.co/datasets/hustep-lab/ViSEC)
  — 5,280 utterances, 4 emotions (`happy / neutral / sad / angry`),
  CC-BY-4.0, ~367 MB. The only labeled Vietnamese emotion speech
  dataset publicly downloadable from HF without authentication.
- **Subset**: stratified, balanced, **400 clips (100 per class)** for
  the production benchmark; smaller subsets (1/clip, 25/clip, …)
  supported for fast smoke tests. The seed is fixed (`seed=0`) so
  re-runs hit the exact same clips.
- **Metric**: 4-class accuracy, plus per-class precision / recall /
  F1. Macro-F1 weighs each class equally, so a model that nails
  `neutral` but ignores the other three gets a low macro-F1 even
  with high accuracy-as-argmax.
- **Stop criterion** (user-defined): **≥ 65 % accuracy on ViSEC
  4-class** wins. The current model does not meet it.
- **Hardware used during this measurement**: NVIDIA RTX 4050 Laptop
  GPU (sm_89 / Ada Lovelace, 6 GiB VRAM). FP16 inference.

## Latest results (2026-07-11)

| Model | Acc | Macro-F1 | N | Per-class F1 | Notes |
|---|---|---|---|---|---|
| MERaLiON-SER-v1 | **40.25 %** | **40.70 %** | 400 | happy 0.31 · neutral 0.50 · sad 0.33 · angry 0.50 | Full 400-clip ViSEC bench on RTX 4050 (cuda, float16), 28 s wall time, 0 failures, 69 ms/clip avg latency. No neutral bias; all 4 classes get reasonable F1. Still below the 65 % stop criterion. |

**Limitations surfaced by the bench**:

- `happy` (F1 = 0.31) and `sad` (F1 = 0.33) are the hardest classes.
  The model rarely predicts them as the top class; they tend to lose
  to `neutral` even when they're the ground-truth label.
- `neutral` (F1 = 0.50) and `angry` (F1 = 0.50) are the easiest.
- The 65 % stop criterion is unmet; the model is below the
  SOTA-grade SER threshold. Ship with honest caveats, not as a
  "winner".

## Re-running

```bash
# Smoke test (5 clips/class, ~30 s on GPU, ~5 min on CPU)
.venv/bin/python3 bench/run_meralion.py --per-class 5 --device cuda --dtype float16

# Full 400-clip bench (~5 min on RTX 4050)
.venv/bin/python3 bench/run_meralion.py --per-class 100 --device cuda --dtype float16

# Force CPU (overrides auto-detection; useful for A/B comparison)
.venv/bin/python3 bench/run_meralion.py --per-class 100 --device cpu
```

After each run, `bench/results/meralion.json` is rewritten and
`bench/results/scores.json` is updated. The UI does not display
scores directly (single-model Space — no leaderboard), but the
README quotes the latest number from `scores.json`.

## GPU execution (RTX 4050 / 6 GB VRAM)

The MERaLiON bench supports GPU inference via `--device cuda`. Add a
CUDA-matched torch wheel first, then:

```bash
# Smoke test on GPU (5 clips, ~30 s)
.venv/bin/python3 bench/run_meralion.py --per-class 5 --device cuda --dtype float16

# Full 400-clip bench on GPU (~5 min on RTX 4050)
.venv/bin/python3 bench/run_meralion.py --per-class 100 --device cuda --dtype float16

# Force CPU (overrides auto-detection; useful for A/B comparison)
.venv/bin/python3 bench/run_meralion.py --per-class 100 --device cpu

# VRAM budget override (default 1.0 GiB free VRAM required)
MERALION_MIN_FREE_VRAM_GIB=2.0 .venv/bin/python3 bench/run_meralion.py \
    --per-class 100 --device cuda --dtype float16

# Env-var form (same effect as flags)
MERALION_DEVICE=cuda MERALION_DTYPE=float16 \
    .venv/bin/python3 bench/run_meralion.py --per-class 100
```

**Memory budget on RTX 4050 (6 GB) at FP16**:
- Model weights: ~600 MB
- Activations for 10 s audio: ~150 MB
- Working buffer: ~250 MB
- **Total: ~1 GB → comfortable 6× headroom**

**Why FP16 on GPU**: cuts memory in half and doubles throughput on
Ada Lovelace tensor cores (`RTX 4050` is sm_89). FP32 stays the
default on CPU because consumer CPUs lack native bfloat16 / float16
matmul acceleration.

### Prereq for GPU run

The repo's `requirements.txt` pins `torch>=2.1,<3.0` without a CUDA
marker — the HF Space gets the CPU build. **For local GPU runs**,
install a CUDA-matched wheel separately (the comment in
`requirements.txt` says not to pin torch on the Space):

```bash
# RTX 4050 = sm_89 (Ada Lovelace), needs torch 2.1+ with cu121
.venv/bin/pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

Verify: `.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
should print `True NVIDIA GeForce RTX 4050 Laptop GPU`, and the
bench's `--device cuda` should auto-succeed.

### Safety nets on the GPU path

- **VRAM pre-flight** (`_check_vram`): aborts before any model load if
  free VRAM < 1 GiB (override via `MERALION_MIN_FREE_VRAM_GIB`).
- **RAM guard relaxed** to 0.3 GiB on GPU path (warning only).
- **Per-clip try/except**: CUDA OOM at clip N doesn't crash the
  run — it logs a WARNING, writes `pred=-1`, and continues.
- **CSV streaming with explicit `fh.flush()`**: each row is flushed
  before the next clip. A SIGKILL loses at most one row.
- **Device/dtype recorded in `scores.json`**: every row carries
  `device` and `dtype` fields so future readers can tell under what
  conditions the number was measured.
- **No `torch.cuda.empty_cache()` between clips**: would slow GPU
  inference 5-10× and is unnecessary on a 6 GB GPU running 1 GB
  workloads.

## Safety rails

- `_check_memory()` refuses to load MERaLiON if free RAM is below
  `MERALION_MIN_FREE_GIB` (default 1.5 GiB; override via env).
- All inference is run with `torch.no_grad()` so model parameters
  don't require_grad storage during speech runs.
- Predictions stream to `bench/results/meralion_predictions.csv`
  after each clip — if the host OOM-kills the process, the work done
  up to that point is preserved on disk.
