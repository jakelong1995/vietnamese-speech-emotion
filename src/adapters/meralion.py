"""Adapter for ``MERaLiON/MERaLiON-SER-v1``.

Architecture: Whisper-medium encoder + LoRA + ECAPA-TDNN + dual heads
(categorical 7-class softmax + dimensional V/A/D).

The 7-class softmax output maps to::

    {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry',
     4: 'fearful', 5: 'disgusted', 6: 'surprised'}

ViSEC labels (happy/neutral/sad/angry) intersect at indices {0,1,2,3}.
``fearful / disgusted / surprised`` are out-of-ViSEC; when the model
picks one of them, the prediction is surfaced as-is (no implicit merge
to fear_anxious) and the bench result counts it as wrong.

Loading
--------
This adapter is heavy (~3 GB RAM at inference, ~600 MB on GPU at FP16).
It is lazy-loaded on first inference call. On hosts with < 1.5 GiB
free memory the adapter raises ``ModelLoadFailedError`` at load time
and the UI surfaces a banner explaining the constraint.

Custom-code meta-tensor bug
---------------------------
The published ``modeling_ser_whisper_ecapa.py`` calls ``torch.logspace``
inside a ``nn.Module.__init__``; under transformers 5.x's low-memory
``_fast_init`` path this raises ``RuntimeError: Tensor.item() cannot
be called on meta tensors``.

We patch the cached copy of the file in
``~/.cache/huggingface/modules/transformers_modules/MERaLiON/...``
(idempotently — marked with ``# PATCHED-META-TENSOR``) before
transformers imports it. The patch is one line of numpy; full diff in
``bench/run_meralion.py``.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseAdapter, ModelInfo, RawPrediction


# Tiny local normalizer: keeps ViSEC 4-class names as-is, demotes the
# MERaLiON extras (fearful / disgusted / surprised) to "other" rather
# than merging them into any ViSEC label.
def _normalize_for_visec(raw_label: str) -> str:
    n = (raw_label or "").lower()
    if n in {"happy", "neutral", "sad", "angry"}:
        return n
    return "other"

log = logging.getLogger("vser.adapters.meralion")

# Patch marker used both at runtime (here) and by bench/run_meralion.py.
# Keep aligned.
_PATCH_MARKER = "# PATCHED-META-TENSOR"
_PATCH_NEEDLE = (
    "kernel_sizes = [int(k) for k in torch.logspace(\n"
    "            math.log10(min_kernel), math.log10(max_kernel), num_resolutions\n"
    "        )]"
)
_PATCH_REPLACEMENT = (
    "        # Generate kernel sizes in log space (PATCHED-META-TENSOR)\n"
    "        import numpy as _np_mod\n"
    "        _ks = _np_mod.logspace(math.log10(min_kernel), math.log10(max_kernel),\n"
    "                               num=num_resolutions).tolist()\n"
    "        kernel_sizes = [int(k) for k in _ks]\n"
)


def _patch_meta_tensor_bug(cache_dirs: list[str]) -> bool:
    """Patch the MERaLiON custom .py file in every cached location.

    Returns ``True`` if at least one file was patched (or was already
    patched). Locations probed:
      1. The HF standard cache (via ``try_to_load_from_cache``).
      2. ``<cache_dir>/modules/transformers_modules/MERaLiON/...``
      3. ``<cache_dir>/models--MERaLiON--MERaLiON-SER-v1/snapshots/<rev>/...``
      4. ``/tmp/hf_cache*/modules/transformers_modules/MERaLiON/...``
      5. ``~/.cache/huggingface/modules/transformers_modules/MERaLiON/...``
    """
    paths: list[str] = []

    try:
        from huggingface_hub import try_to_load_from_cache
        loc = try_to_load_from_cache(
            "MERaLiON/MERaLiON-SER-v1",
            "modeling_ser_whisper_ecapa.py",
            repo_type="model",
        )
        if loc and os.path.exists(loc):
            paths.append(loc)
    except Exception:
        pass

    for cd in cache_dirs:
        for pattern in (
            os.path.join(cd, "modules", "transformers_modules", "MERaLiON",
                         "MERaLiON-SER-v1*"),
            os.path.join(cd, "models--MERaLiON--MERaLiON-SER-v1",
                         "snapshots", "*"),
        ):
            for root in glob.glob(pattern):
                for fn in glob.glob(os.path.join(root, "**",
                                                 "modeling_ser_whisper_ecapa.py"),
                                    recursive=True):
                    paths.append(fn)

    for tmp_dir in glob.glob("/tmp/hf_cache*"):
        pattern = os.path.join(
            tmp_dir, "modules", "transformers_modules", "MERaLiON",
            "MERaLiON-SER-v1*",
        )
        for root in glob.glob(pattern):
            for fn in glob.glob(os.path.join(root, "*",
                                             "modeling_ser_whisper_ecapa.py")):
                paths.append(fn)

    dyn_root = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "modules",
        "transformers_modules", "MERaLiON",
        "MERaLiON_hyphen_SER_hyphen_v1",
    )
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
        if _PATCH_MARKER in text:
            any_patched = True
            continue
        if _PATCH_NEEDLE not in text:
            continue
        new_text = text.replace(_PATCH_NEEDLE, _PATCH_REPLACEMENT)
        open(path, "w", encoding="utf-8").write(new_text)
        log.info("patched MERaLiON custom code at %s", path)
        any_patched = True
    return any_patched


def _free_gib() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return None


class MeralionAdapter(BaseAdapter):
    model_id = "MERaLiON/MERaLiON-SER-v1"

    def __init__(self, cache_dir: Optional[Path] = None,
                 min_free_gib: float = 1.5) -> None:
        super().__init__(cache_dir=cache_dir)
        self._min_free_gib = min_free_gib
        self.sample_rate = 16000
        self.labels = ["neutral", "happy", "sad", "angry",
                       "fearful", "disgusted", "surprised"]
        self._model = None
        self._fe = None
        self._id2label: Dict[int, str] = {}
        self._load_seconds: float = 0.0
        self._device: str = "cpu"

    def load(self) -> None:
        if self._model is not None:
            return
        # Patch the custom-code meta-tensor bug BEFORE transformers
        # imports the file, otherwise ``from_pretrained`` crashes with
        # ``RuntimeError: Tensor.item() cannot be called on meta tensors``
        # under the new low-memory init path.
        _patch_meta_tensor_bug([str(self._cache_dir)])
        # Force re-import so the patched module is what transformers imports.
        import sys as _sys
        for k in [m for m in _sys.modules if "transformers_modules" in m]:
            del _sys.modules[k]

        # Memory guard: refuse to load on tight-RAM hosts.
        free = _free_gib()
        if free is not None and free < self._min_free_gib:
            from ..exceptions import ModelLoadFailedError
            raise ModelLoadFailedError(
                f"MERaLiON-SER-v1 needs >= {self._min_free_gib} GiB free; "
                f"only {free:.2f} GiB available. Free RAM (close other "
                "apps) and reload to try again.",
                details={"model_id": self.model_id, "free_gib": free,
                         "required_gib": self._min_free_gib},
            )

        log.info("loading MERaLiON-SER-v1 from %s", self.model_id)
        t0 = time.perf_counter()
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForAudioClassification, AutoFeatureExtractor,
            )
            self._model = AutoModelForAudioClassification.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=str(self._cache_dir),
            )
            self._fe = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=str(self._cache_dir),
            )
            self._model.eval()
        except Exception as exc:
            from ..exceptions import ModelLoadFailedError
            raise ModelLoadFailedError(
                f"Failed to load MERaLiON-SER-v1: {exc!r}",
                details={"model_id": self.model_id},
            ) from exc
        self._load_seconds = time.perf_counter() - t0
        self._device = "cpu"
        try:
            self._id2label = {
                int(k): str(v) for k, v in self._model.config.id2label.items()
            }
        except Exception:
            self._id2label = {i: n for i, n in enumerate(self.labels)}
        log.info("MERaLiON-SER-v1 loaded in %.1fs, %d classes",
                 self._load_seconds, len(self._id2label))

    def unload(self) -> None:
        for attr in ("_model", "_fe"):
            if getattr(self, attr) is not None:
                setattr(self, attr, None)
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def predict(self, waveform: np.ndarray, sample_rate: int) -> RawPrediction:
        if self._model is None or self._fe is None:
            from ..exceptions import ModelLoadFailedError
            raise ModelLoadFailedError("MERaLiON-SER-v1 not loaded")
        if waveform.size == 0:
            from ..exceptions import InvalidAudioError
            raise InvalidAudioError("Empty waveform")
        if sample_rate != self.sample_rate:
            from ..audio import resample_linear
            waveform = resample_linear(waveform, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate

        t0 = time.perf_counter()
        try:
            import torch
            inputs = self._fe(waveform, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                out = self._model(**inputs)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            probs_t = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        except Exception as exc:
            log.exception("MERaLiON predict failed")
            from ..exceptions import ModelLoadFailedError
            raise ModelLoadFailedError(f"MERaLiON predict error: {exc!r}") from exc
        inference_seconds = time.perf_counter() - t0

        # Map to label scores.
        class_scores: Dict[str, float] = {}
        for idx, p in enumerate(probs_t.tolist()):
            name = self._id2label.get(idx, str(idx))
            class_scores[name] = float(p)

        # Top-1 = raw_label, normalized_label pass-through for ViSEC
        # classes and "other" for the rest.
        top_idx = int(probs_t.argmax())
        raw_label = self._id2label.get(top_idx, str(top_idx))
        confidence = float(probs_t[top_idx])
        normalized = _normalize_for_visec(raw_label)

        auxiliary: Dict[str, Any] = {
            "inference_seconds": round(inference_seconds, 4),
            "raw_distribution": class_scores,
            "model_id": self.model_id,
            "raw_label": raw_label,
        }

        return RawPrediction(
            label=normalized,
            confidence=confidence,
            class_scores=class_scores,
            auxiliary=auxiliary,
            full_distribution_available=True,
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider="meralion",
            model_id=self.model_id,
            sample_rate=self.sample_rate,
            labels=list(self._id2label.values() or self.labels),
            device=self._device,
            confidence_available=True,
            extra={
                "model_load_seconds": round(self._load_seconds, 3),
                "role": "vietnamese_partial",
                "language": "multilingual",
                "english_trained": False,
                "vietnamese_verified": "limited_secondary",
                "benchmark": (
                    "Smoke-tested on ViSEC (4 clips, 1 per class). Full "
                    "benchmark blocked by host RAM. See "
                    "docs/MODEL_OUTPUT_AUDIT.md."
                ),
                "recommendation": (
                    "Whisper-medium + LoRA + ECAPA-TDNN head trained on "
                    "MERaLiON AudioBench. Vietnamese is a secondary "
                    "language in pretraining; expect partial accuracy "
                    "on idiomatic Vietnamese emotional prosody."
                ),
                "patched_meta_tensor_bug": True,
            },
        )
