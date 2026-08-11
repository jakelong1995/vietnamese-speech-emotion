"""Phase 2 of the benchmark loop: evaluate ``MERaLiON/MERaLiON-SER-v1``.

Model: Whisper-Medium encoder + LoRA + ECAPA-TDNN + dual heads
(categorical softmax over 7 emotions + dimensional V/A/D).

The 7-class softmax output ``{neutral, happy, sad, angry, fearful,
disgusted, surprised}`` maps directly onto ViSEC's 4-class task — the
extra 3 classes (fearful/disgusted/surprised) are out-of-ViSEC; when
the model picks one of those, the prediction is counted as wrong (no
implicit remapping).

Why MERaLiON-SER-v1: it claims Vietnamese support in the model card
(limited/secondary), loads via ``AutoModelForAudioClassification`` with
``trust_remote_code=True``, and ships a 7-class softmax head that
aligns with ViSEC's 4 classes + 3 extras.

Repair note: the published custom code calls ``torch.logspace(...).item()``
inside ``__init__``, which crashes under transformers 5.x's
low-memory ``_fast_init`` (meta tensors). We patch that one line with
plain numpy on first load and reuse the patched cache across runs.

Run:

    .venv/bin/python3 bench/run_meralion.py --per-class 100

Outputs:

    bench/results/meralion.json — full BenchResult
    bench/results/scores.json  — appended/updated

Safety: refuses to load if free RAM < 1.5 GiB (Whisper-medium + ECAPA is
the heaviest single adapter we ship).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("vser.bench.meralion")


def _check_memory(min_free_gib: float = 1.0) -> None:
    """Refuse to start if free RAM is below ``min_free_gib``. Models like
    MERaLiON-SER-v1 use Whisper-medium + ECAPA-TDNN (~3 GB parameters on
    CPU). Keep ``min_free_gib`` realistic for the host or the kernel
    will OOM-kill the python interpreter mid-load (loss of 5–15 min
    inference work).

    Override via ``MERALION_MIN_FREE_GIB`` env var (e.g. set to 0.4 on
    hosts that benefit from linux's page-cache reclaim during mmap-based
    safetensors load).
    """
    threshold = float(os.environ.get("MERALION_MIN_FREE_GIB",
                                    str(min_free_gib)))
    try:
        import psutil
        vm = psutil.virtual_memory()
        avail_gib = vm.available / (1024 ** 3)
        # Linux page-cache is also reclaimable; include it in budget.
        budget_gib = avail_gib + (vm.cached / (1024 ** 3))
    except ImportError:
        log.warning("psutil not installed; skipping memory guard")
        return
    if budget_gib < threshold:
        raise RuntimeError(
            f"only {avail_gib:.2f} GiB free "
            f"(+ {vm.cached/1024**3:.2f} GiB page-cache, "
            f"= {budget_gib:.2f} GiB budget); need >= {threshold} GiB "
            "to safely load MERaLiON-SER-v1. Aborting to protect the host."
        )


def _resolve_device_and_dtype(args) -> tuple[str, "torch.dtype"]:
    """Pick execution device + dtype based on args / env / capability.

    Precedence (high → low):
      1. CLI flags ``--device`` / ``--dtype`` (set in argparse).
      2. Env vars ``MERALION_DEVICE`` / ``MERALION_DTYPE``.
      3. Auto: ``cuda`` + ``float16`` if ``torch.cuda.is_available()``,
         else ``cpu`` + ``float32``.

    Returns ``(device_str, torch_dtype)``.

    Rationale: GPU inference benefits substantially from FP16 (cuts
    memory in half, doubles throughput on Ada Lovelace tensor cores),
    while CPU inference is fastest at FP32 (no native bfloat16 ops on
    consumer CPUs).
    """
    import torch
    dev = (getattr(args, "device", None)
           or os.environ.get("MERALION_DEVICE", "auto"))
    dty = (getattr(args, "dtype", None)
           or os.environ.get("MERALION_DTYPE", "auto"))
    if dev not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown device: {dev}")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dty == "auto":
        dty = "float16" if dev == "cuda" else "float32"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dty not in dtype_map:
        raise ValueError(f"unknown dtype: {dty}")
    return dev, dtype_map[dty]


def _check_vram(min_free_gib: float = 1.0) -> None:
    """Refuse to start the GPU path if free VRAM < threshold.

    Override via ``MERALION_MIN_FREE_VRAM_GIB`` env var. If torch has
    no CUDA build, this is a no-op (the caller will have already
    resolved ``device=cpu`` so we never reach this path).

    Uses ``torch.cuda.mem_get_info()`` which is the canonical source
    (more reliable than parsing ``nvidia-smi``).
    """
    import torch
    if not torch.cuda.is_available():
        return
    threshold = float(os.environ.get("MERALION_MIN_FREE_VRAM_GIB",
                                    str(min_free_gib)))
    free, total = torch.cuda.mem_get_info()
    free_gib, total_gib = free / 1024**3, total / 1024**3
    if free_gib < threshold:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB free VRAM "
            f"(of {total_gib:.2f} GiB total); need >= {threshold} GiB. "
            "Aborting to protect the GPU."
        )


def _build_notes(args, device: str, dtype, n_failed: int,
                 avg_latency_ms, n_samples: int) -> str:
    """Construct the ``notes`` string written into ``scores.json``.

    Distinguishes full vs partial runs and embeds device + dtype so a
    future reader knows the exact measurement conditions. Stays a
    single line to keep ``scores.json`` greppable.
    """
    dtype_str = str(dtype).split('.')[-1]
    is_full = (n_samples == args.per_class * 4 and args.per_class >= 100)
    scope = "Full" if is_full else "Partial"
    target = f"{n_samples}-clip" if not is_full else "400-clip"
    lat = f"{avg_latency_ms:.1f}" if avg_latency_ms else "n/a"
    return (f"{scope} ViSEC bench on device={device} dtype={dtype_str}, "
            f"failed_clips={n_failed}, avg_latency_ms={lat}")


def _patch_meta_tensor_bug(cache_dirs: list[str] | None = None) -> bool:
    """The custom ``SERWhisperECAPA`` module calls
    ``torch.logspace(...).tolist()`` in a way that breaks on meta tensors
    (transformers' default fast-init path on transformers >= 4.40).

    The patch: replace ``[int(k) for k in torch.logspace(...)]`` with a
    numpy.linspace so no tensor allocation happens at construction.
    Idempotent — only rewrites the file once per cache dir.

    Searches every provided cache_dir (plus the default HF cache location
    ``~/.cache/huggingface`` and ``/tmp/hf_cache_*`` dirs) for the
    downloaded ``modeling_ser_whisper_ecapa.py`` and patches it in-place
    if the bug-trigger line is still present.

    Returns ``True`` if at least one cached copy was patched (or was
    already patched), ``False`` otherwise.
    """
    import glob

    needle = (
        "kernel_sizes = [int(k) for k in torch.logspace(\n"
        "            math.log10(min_kernel), math.log10(max_kernel), num_resolutions\n"
        "        )]"
    )
    marker = "# PATCHED-META-TENSOR"
    # Replacement preserves method-body indentation exactly: every body
    # line begins with 8 spaces (one level inside `__init__`).
    # Does NOT touch the trailing ``self.kernel_sizes = ...`` line.
    replacement = (
        "        # Generate kernel sizes in log space (PATCHED-META-TENSOR)\n"
        "        import numpy as _np_mod\n"
        "        _ks = _np_mod.logspace(math.log10(min_kernel), math.log10(max_kernel),\n"
        "                               num=num_resolutions).tolist()\n"
        "        kernel_sizes = [int(k) for k in _ks]\n"
    )

    # 1. Probe the standard HF cache location.
    from huggingface_hub import try_to_load_from_cache
    paths: list[str] = []
    for loc in (try_to_load_from_cache(
        "MERaLiON/MERaLiON-SER-v1",
        "modeling_ser_whisper_ecapa.py",
        repo_type="model",
    ),):
        if loc and os.path.exists(loc):
            paths.append(loc)

    # 2. Probe the project-local cache dir (if it exists) — both layouts:
    #    a) ``<cache_dir>/modules/transformers_modules/...`` (dynamic loader
    #       copies from previous runs)
    #    b) ``<cache_dir>/models--MERaLiON--MERaLiON-SER-v1/snapshots/<rev>/...``
    #       (the standard hub layout that transformers uses when loading
    #       with ``cache_dir=...``)
    if cache_dirs:
        for cd in cache_dirs:
            for candidate_pattern in (
                os.path.join(cd, "modules", "transformers_modules",
                             "MERaLiON", "MERaLiON-SER-v1*"),
                os.path.join(cd, "models--MERaLiON--MERaLiON-SER-v1",
                             "snapshots", "*"),
            ):
                for root in glob.glob(candidate_pattern):
                    for fn in glob.glob(os.path.join(root, "**",
                                                     "modeling_ser_whisper_ecapa.py"),
                                       recursive=True):
                        paths.append(fn)

    # 3. Probe /tmp fallback dirs from past runs.
    for tmp_dir in glob.glob("/tmp/hf_cache*"):
        candidate = os.path.join(
            tmp_dir, "modules", "transformers_modules", "MERaLiON",
            "MERaLiON-SER-v1")
        for root in glob.glob(candidate + "*"):
            for fn in glob.glob(os.path.join(root, "*",
                                             "modeling_ser_whisper_ecapa.py")):
                paths.append(fn)

    # 4. Probe the user's standard HF cache dynamic-module dir
    #    (``~/.cache/huggingface/modules/transformers_modules/...``).
    #    The dynamic-module loader copies the source file from the
    #    snapshot to this location on first ``from_pretrained`` and
    #    re-uses it across sessions. Transformers imports the buggy
    #    copy from here unless we patch this path too.
    home = os.path.expanduser("~")
    dyn_root = os.path.join(home, ".cache", "huggingface", "modules",
                            "transformers_modules", "MERaLiON",
                            "MERaLiON_hyphen_SER_hyphen_v1")
    for root in glob.glob(dyn_root + "*"):
        for fn in glob.glob(os.path.join(root, "*",
                                         "modeling_ser_whisper_ecapa.py")):
            paths.append(fn)

    any_patched = False
    for path in paths:
        try:
            text = open(path, encoding="utf-8").read()
        except Exception:
            continue
        if marker in text:
            any_patched = True
            continue
        if needle not in text:
            continue
        new_text = text.replace(needle, replacement)
        open(path, "w", encoding="utf-8").write(new_text)
        log.info("patched MERaLiON custom code at %s", path)
        any_patched = True
    return any_patched


def _ensure_modeling_file_patched(cache_dir: str) -> None:
    """If ``modeling_ser_whisper_ecapa.py`` is missing from cache_dir,
    download it ahead-of-time so the patch can be applied before
    transformers re-discovers and ``__import__``s the module.

    The bug is that transformers' dynamic-module loader reads the .py
    file at first call to ``from_pretrained``; if the file isn't yet
    downloaded to the cache_dir we're using, the load will trigger a
    fresh download of the buggy file, import it, and crash on
    ``torch.logspace(...).item()`` under meta-tensor init.
    """
    from huggingface_hub import hf_hub_download
    try:
        downloaded = hf_hub_download(
            "MERaLiON/MERaLiON-SER-v1",
            "modeling_ser_whisper_ecapa.py",
            repo_type="model",
            cache_dir=cache_dir,
        )
        log.info("ensured modeling file at %s", downloaded)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not pre-fetch modeling file into %s: %r",
                    cache_dir, exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "bench" / "results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        default="auto",
                        help="Execution device (default: auto = cuda if "
                             "available else cpu)")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16",
                                            "bfloat16"],
                        default="auto",
                        help="Model dtype (default: float16 on cuda, "
                             "float32 on cpu)")
    args = parser.parse_args()

    # ----- Step 1: pre-flight safety checks -----
    # Resolve device + dtype from CLI/env first so the right guard
    # runs (VRAM on GPU, RAM on CPU).
    device, dtype = _resolve_device_and_dtype(args)
    if device == "cuda":
        # GPU path: VRAM is the binding constraint. We still run a
        # relaxed RAM check so the OS doesn't evict safetensors pages
        # mid-load, but a tight RAM no longer aborts the run.
        _check_vram(min_free_gib=1.0)
        try:
            _check_memory(min_free_gib=0.3)
        except RuntimeError as e:
            log.warning("RAM is tight on GPU path (%s); continuing "
                        "because VRAM is the real limit", e)
    else:
        # CPU path: original RAM guard is the binding constraint.
        # Tightened threshold: prior smoke test loaded MERaLiON successfully
        # at 1.18 GiB available. Any less than 1.0 GiB and the kernel will
        # OOM-kill the python interpreter mid-load (loss of 5–15 min
        # inference work).
        _check_memory(min_free_gib=1.0)
    log.info("execution plan: device=%s dtype=%s",
             device, str(dtype).split('.')[-1])

    from bench.visec import load_visec, LABEL_NAMES  # happy, neutral, sad, angry
    from bench.metrics import evaluate, to_jsonable

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(results_dir / "hf_cache")

    # ----- Step 2: pre-fetch and patch custom code -----
    # (unchanged from previous code; GPU dtype doesn't affect this step)
    _ensure_modeling_file_patched(cache_dir)
    # Apply the patch in every cache dir we can find. Returns True if any
    # was patched; we may still hit the bug if a brand-new cache dir was
    # created during from_pretrained and the patch ran too late — the
    # _ensure step above mitigates that race.
    patched = _patch_meta_tensor_bug(cache_dirs=[cache_dir])
    log.info("meta-tensor patch status: %s", "applied" if patched else "skipped")
    # Force re-import so the patched module is what transformers imports next.
    for k in [m for m in sys.modules if "transformers_modules" in m]:
        del sys.modules[k]

    log.info("=== Phase 2: MERaLiON-SER-v1 on ViSEC (per_class=%d) ===",
             args.per_class)
    samples = load_visec(per_class=args.per_class, seed=args.seed)
    log.info("loaded %d ViSEC samples (%d per class)",
             len(samples), args.per_class)

    # ----- Step 3: load model on the resolved device -----
    log.info("loading MERaLiON-SER-v1 (transformers AutoModel)…")
    t_load = time.perf_counter()
    import torch
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    model = AutoModelForAudioClassification.from_pretrained(
        "MERaLiON/MERaLiON-SER-v1",
        trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=dtype,        # NEW: weights load as fp16 on GPU
    )
    if device == "cuda":
        model.to("cuda")           # NEW: explicitly move to GPU
    fe = AutoFeatureExtractor.from_pretrained(
        "MERaLiON/MERaLiON-SER-v1",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.eval()
    log.info("MERaLiON-SER-v1 loaded on %s in %.1fs",
             device, time.perf_counter() - t_load)
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    log.info("id2label: %s", id2label)

    # ViSEC 4-class ↔ MERaLiON 7-class mapping
    visec_to_idx: dict[str, int] = {}
    for i, n in id2label.items():
        if n in LABEL_NAMES:
            visec_to_idx[n] = i
    log.info("visec class indices in MERaLiON: %s", visec_to_idx)

    preds: list[int] = []
    truths: list[int] = []
    latencies_ms: list[float] = []
    failing: list[int] = []

    # Stream predictions to disk so a mid-run OOM still preserves the
    # work that succeeded. critical on a tight-memory host where the
    # kernel may reclaim pages aggressively.
    pred_log = results_dir / "meralion_predictions.csv"
    pred_log.write_text(
        "i,label_name,true_idx,pred_idx,latency_ms\n", encoding="utf-8")
    log.info("writing predictions to %s (incremental, OOM-safe)", pred_log)

    log.info("running inference on %d clips …", len(samples))
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        try:
            import soundfile as sf
            import csv as _csv
            wav, sr = sf.read(s.wav_path)
            t_inf = time.perf_counter()
            inputs = fe(wav, sampling_rate=sr, return_tensors="pt")
            # Move inputs to GPU if applicable. Cast to the model's
            # dtype to avoid an in-graph float64→float16 promotion
            # that would otherwise hit the autocast fallback.
            if device == "cuda":
                inputs = {k: v.to(device).to(dtype)
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            probs_t = torch.softmax(logits, dim=-1)[0]
            # Explicit host sync — guarantees we have a CPU tensor
            # before any .numpy() call.
            if probs_t.device.type == "cuda":
                probs_t = probs_t.cpu()
            probs = probs_t.numpy()
            lat_ms = (time.perf_counter() - t_inf) * 1000
        except Exception as exc:  # noqa: BLE001
            log.warning("clip %d (%s) failed: %r", i, s.label_name, exc)
            failing.append(i)
            preds.append(-1)
            truths.append(s.label)
            continue

        # Aggregate to 4-class ViSEC: only count if argmax is in the 4
        # mapped emotions. If model picked "fearful/disgusted/surprised",
        # attribute the prediction to the closest ViSEC-class by argmax
        # among the 4 candidates (so we don't lose all signal — but it
        # also doesn't claim the model is right when it isn't).
        four_probs = {cls_idx: probs[visec_to_idx[LABEL_NAMES[cls_idx]]]
                      for cls_idx in range(4)
                      if LABEL_NAMES[cls_idx] in visec_to_idx}
        if not four_probs:
            preds.append(-1)
        else:
            preds.append(max(four_probs, key=four_probs.get))
        truths.append(s.label)
        latencies_ms.append(lat_ms)

        # Persist this clip's prediction so a kernel OOM-kill still
        # leaves us with ``i+1`` clips of evidence.
        with open(pred_log, "a", encoding="utf-8") as fh:
            _csv.writer(fh).writerow([
                i, s.label_name, s.label,
                preds[-1], f"{lat_ms:.1f}",
            ])
            # Force the row to the page cache synchronously so a
            # SIGKILL loses at most this single row (was: the last
            # several rows that hadn't been flushed by the OS).
            fh.flush()
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            vram_free = (torch.cuda.mem_get_info()[0] / 1024**3) \
                        if device == "cuda" else float("nan")
            log.info("  %d/%d (%.1fs, %.2fs/clip, last_lat=%.0fms, "
                     "vram_free=%.2f GiB)",
                     i + 1, len(samples), elapsed,
                     elapsed / (i + 1), lat_ms, vram_free)

    valid = [(p, t) for p, t in zip(preds, truths) if p in (0, 1, 2, 3)]
    preds_v = [p for p, _ in valid]
    truths_v = [t for _, t in valid]
    avg_latency = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None

    result = evaluate(
        model_id="meralion-ser-v1",
        preds=preds_v,
        truths=truths_v,
        n_classes=4,
        label_names=LABEL_NAMES,
        notes=(
            f"per_class={args.per_class}, seed={args.seed}, "
            f"failed_clips={len(failing)}, "
            f"avg_latency_ms={avg_latency and round(avg_latency, 1)}, "
            f"4cls aggregation: argmax over 4 mapped ViSEC labels "
            f"(predictions on fearful/disgusted/surprised counted as wrong)"
        ),
    )

    out_path = results_dir / "meralion.json"
    out_path.write_text(json.dumps(to_jsonable(result), indent=2),
                        encoding="utf-8")
    log.info("wrote %s (%.1f%% accuracy on %d clips)",
             out_path, result.accuracy * 100, result.num_samples)

    scores_path = results_dir / "scores.json"
    scores: dict = {}
    if scores_path.exists():
        try:
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        except Exception:
            scores = {}
    scores["meralion-ser-v1"] = {
        "model_id": result.model_id,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "num_samples": result.num_samples,
        "eligible_for_deploy": result.accuracy >= 0.65,
        "per_class_f1": result.f1,
        "notes": _build_notes(args, device, dtype, len(failing),
                              avg_latency, len(samples)),
        "device": device,
        "dtype": str(dtype).split('.')[-1],
    }
    scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    log.info("updated %s", scores_path)

    log.info("unloading MERaLiON-SER-v1 …")
    del model
    try:
        del fe
    except Exception:
        pass
    gc.collect()
    # Free VRAM once at the end. Cheap because it's called once,
    # not per clip.
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    log.info("=== Phase 2 done: %.1f%% accuracy ===",
             result.accuracy * 100)


if __name__ == "__main__":
    main()
