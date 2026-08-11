"""Benchmark harness for the Vietnamese Speech Emotion Recognition Space.

Loads :mod:`bench.visec` to pull real labeled Vietnamese emotion samples
from ``hustep-lab/ViSEC`` and runs each candidate adapter, recording:

- per-clip prediction
- 4-class accuracy / per-class precision + recall / confusion matrix
- per-clip latency
- memory footprint snapshot

Output goes to ``bench/results/`` as JSON so the runtime can read it back
(see ``src.inference``) to drive the UI model selector.
"""
