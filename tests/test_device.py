"""Tests for the shared device-resolution policy."""
from __future__ import annotations

import pytest
import torch

from src import device as dev


def test_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv(dev.DEVICE_ENV, "cpu")
    assert dev.resolve_device() == "cpu"


def test_explicit_override_beats_env(monkeypatch):
    monkeypatch.setenv(dev.DEVICE_ENV, "cpu")
    assert dev.resolve_device("mps") == "mps"


def test_auto_never_returns_auto(monkeypatch):
    monkeypatch.setenv(dev.DEVICE_ENV, "auto")
    assert dev.resolve_device() in {"cpu", "mps", "cuda"}


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        dev.resolve_device("tpu")


def test_mps_uses_float32_not_float16():
    """Half precision on Metal falls back to CPU for several ops and can
    return NaNs; the policy must keep MPS at fp32."""
    assert dev.resolve_dtype("mps") is torch.float32
    assert dev.resolve_dtype("cpu") is torch.float32
    assert dev.resolve_dtype("cuda") is torch.float16


def test_describe_is_human_readable():
    assert dev.describe("mps") == "Apple GPU (Metal)"
    assert dev.describe("cpu") == "CPU"
