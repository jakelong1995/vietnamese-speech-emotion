"""Adapter registry for the Vietnamese SER Space.

This Space ships with exactly one adapter, MERaLiON-SER-v1. The
registry still exists so ``src.inference`` has a stable import target
and so future contributors can drop in a replacement adapter without
touching the inference layer.

Adding a new adapter
--------------------
1. Subclass :class:`src.adapters.base.BaseAdapter`.
2. Append the new class to ``REGISTRY`` below.
3. Add a benchmark harness in ``bench/run_<name>.py` and a row to
   ``bench/results/scores.json``.

Only one adapter is loaded in memory at a time; the inference layer
swaps adapters lazily on first call after the user picks a new model.
"""
from __future__ import annotations

import logging
from typing import Dict, Type

from .base import BaseAdapter
from .meralion import MeralionAdapter

log = logging.getLogger("vser.adapters")

REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "meralion-ser-v1": MeralionAdapter,
}


def known_models() -> list[str]:
    return list(REGISTRY.keys())


def instantiate(name: str, **kwargs) -> BaseAdapter:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown model {name!r}; "
            f"available: {list(REGISTRY.keys())}")
    return cls(**kwargs)


__all__ = ["BaseAdapter", "REGISTRY", "known_models", "instantiate"]
