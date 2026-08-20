"""Execution-device resolution shared by every model in this app.

Both the emotion adapter (:mod:`src.adapters.meralion`) and the ASR
layer (:mod:`src.asr`) need the same answer to "where do I run and in
what precision?", so the policy lives here instead of being duplicated.

Priority (high → low)
---------------------
1. Explicit ``override`` argument.
2. Env var ``SER_DEVICE`` (``auto`` / ``cpu`` / ``mps`` / ``cuda``).
3. Auto-detect: ``mps`` (Apple Silicon) → ``cuda`` → ``cpu``.

Precision policy
----------------
- ``cuda``  → float16. Tensor cores make this a straight win.
- ``mps``   → **float32**. Apple's Metal backend still falls back to
  CPU for several fp16 ops, so half precision there is often *slower*
  and occasionally numerically wrong. Unified memory means we are not
  short on RAM anyway.
- ``cpu``   → float32. No native fp16 path on consumer CPUs.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger("vser.device")

DEVICE_ENV = "SER_DEVICE"
VALID_DEVICES = {"auto", "cpu", "mps", "cuda"}


def resolve_device(override: Optional[str] = None) -> str:
    """Return the torch device string to run on (never ``"auto"``)."""
    choice = (override or os.environ.get(DEVICE_ENV, "auto")).lower()
    if choice not in VALID_DEVICES:
        raise ValueError(
            f"unknown device {choice!r}; expected one of {sorted(VALID_DEVICES)}"
        )
    if choice != "auto":
        return choice

    try:
        import torch
    except ImportError:
        return "cpu"

    # Apple Silicon first: on an M-series Mac this is the only accelerator.
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(device: str) -> Any:
    """Return the torch dtype matching the precision policy for ``device``."""
    import torch
    if device == "cuda":
        return torch.float16
    return torch.float32


def describe(device: str) -> str:
    """Human-readable accelerator name for the UI caption."""
    if device == "mps":
        return "Apple GPU (Metal)"
    if device == "cuda":
        try:
            import torch
            return f"NVIDIA {torch.cuda.get_device_name(0)}"
        except Exception:
            return "NVIDIA GPU"
    return "CPU"


__all__ = ["resolve_device", "resolve_dtype", "describe",
           "DEVICE_ENV", "VALID_DEVICES"]
