"""ViSEC loader for the benchmark harness.

`hustep-lab/ViSEC <https://huggingface.co/datasets/hustep-lab/ViSEC>`_ is the
only Vietnamese emotional speech dataset publicly downloadable from the
HuggingFace Hub without authentication (CC-BY-4.0, ~367 MB, 5,280 utterances).

Layout (per the dataset card):

- ``"train"`` split, single 5,280-row Parquet at
  ``data/train-00000-of-00001.parquet``.
- Column ``"audio"`` — dict with ``"path"`` (id) and ``"bytes"`` (wav).
- Column ``"text"`` — transcription.
- Column ``"emotion"`` — int label. Per the dataset card:
  - ``0`` → ``happy``
  - ``1`` → ``neutral``
  - ``2`` → ``sad``
  - ``3`` → ``angry``

This module exposes :func:`load_visec` which returns a deterministic
stratified sample (same seed ⇒ same subset across runs) so all candidate
adapters are evaluated on the same clips. The full parquet is downloaded
once to ``bench/cache/visec_train.parquet``; a per-stratified-subset
``.jsonl`` index is cached at ``bench/cache/visec_subset_p*.jsonl`` so
re-runs are sub-second.

Public API
----------
- ``LABEL_NAMES``         — canonical 4-class list, index ⇒ name.
- ``LABEL_NAME_TO_INDEX`` — invert the above for sanity checks.
- ``VisecSample``         — frozen dataclass with ``wav_path``, ``label``,
                            ``label_name``, ``text``.
- ``load_visec(per_class, cache_dir=...)`` — returns a list of
                            ``VisecSample`` of length ``per_class * 4``.
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("vser.bench.visec")


LABEL_NAMES: List[str] = ["happy", "neutral", "sad", "angry"]
LABEL_NAME_TO_INDEX = {n: i for i, n in enumerate(LABEL_NAMES)}


@dataclass
class VisecSample:
    """A single ViSEC clip with its ground-truth label."""

    wav_path: str            # absolute path to a 16-kHz mono wav on disk
    label: int               # 0..3 (see LABEL_NAMES)
    label_name: str          # "happy" / "neutral" / "sad" / "angry"
    text: str = ""           # transcription (kept for downstream text-fusion)


# --------------------------------------------------------------------------
# Parquet source acquisition
# --------------------------------------------------------------------------

def _ensure_parquet(cache_dir: Path) -> Path:
    """Return the path to the cached ViSEC train parquet, downloading if
    necessary via ``huggingface_hub``.

    Tries (in order):
      1. The local file at ``cache_dir/visec_train.parquet`` (pre-placed).
      2. ``huggingface_hub.hf_hub_download`` with the public URL.

    Streaming fallback (used only when the parquet is unreachable) reads
    the dataset via the ``datasets`` library and serializes to a local
    parquet. We prefer the direct approach because ``datasets`` streaming
    has been observed to hang on the first reconnection.
    """
    target = cache_dir / "visec_train.parquet"
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="hustep-lab/ViSEC",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
            cache_dir=str(cache_dir / "hf_hub"),
        )
        # Copy into a stable location so we don't depend on the HF hash layout
        import shutil
        shutil.copy(downloaded, target)
        log.info("downloaded ViSEC parquet → %s (%.1f MB)",
                 target, target.stat().st_size / 1e6)
        return target
    except Exception as exc:
        log.warning("direct parquet download failed: %r; will try datasets", exc)

    # Fallback: materialize via `datasets.load_dataset`.
    try:
        from datasets import load_dataset
        ds = load_dataset("hustep-lab/ViSEC", split="train")
        # Save as parquet
        ds.to_parquet(str(target))
        return target
    except Exception as exc:
        raise RuntimeError(
            "Could not materialize ViSEC. Tried hf_hub_download and "
            f"`datasets.load_dataset`. Last error: {exc!r}"
        ) from exc


def _read_parquet_rows(parquet_path: Path):
    """Yield rows from the ViSEC parquet.

    Uses :mod:`pyarrow` if available (it ships with the ``datasets``
    dep we already require); falls back to ``pandas``. We deliberately
    avoid pulling in ``pyarrow.dataset`` – ``pq.read_table`` then
    ``to_pylist`` is fine for a single 367 MB file on 16 GB RAM.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Need `pyarrow` to read ViSEC parquet. Install with "
            "`pip install pyarrow`."
        ) from exc
    table = pq.read_table(str(parquet_path))
    return table.to_pylist()


# --------------------------------------------------------------------------
# Public sampler
# --------------------------------------------------------------------------

def load_visec(
    per_class: int = 100,
    cache_dir: Optional[Path] = None,
    seed: int = 0,
) -> List[VisecSample]:
    """Return a deterministic stratified sample of ViSEC.

    ``per_class`` clips per emotion (default 100 ⇒ 400 clips total).

    The first call downloads the full parquet (~367 MB); subsequent runs
    re-use the on-disk cache and skip straight to the stratified sample.
    """
    if per_class < 1:
        raise ValueError("per_class must be >= 1")

    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = cache_dir / "wavs"
    samples_dir.mkdir(parents=True, exist_ok=True)

    index_path = cache_dir / f"visec_subset_p{per_class}_s{seed}.jsonl"

    if index_path.exists():
        log.info("loading cached ViSEC index from %s", index_path)
        out: List[VisecSample] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            p = Path(d["wav_path"])
            if not p.is_absolute():
                p = cache_dir / p
            if not p.exists():
                log.warning("cached wav missing (%s); regenerating", p)
                index_path.unlink()
                return load_visec(per_class=per_class, cache_dir=cache_dir,
                                  seed=seed)
            out.append(VisecSample(
                wav_path=str(p),
                label=int(d["label"]),
                label_name=str(d["label_name"]),
                text=str(d.get("text", "")),
            ))
        if len(out) == per_class * 4:
            return out

    parquet = _ensure_parquet(cache_dir)
    log.info("reading ViSEC parquet (%s, %.1f MB)…",
             parquet.name, parquet.stat().st_size / 1e6)
    rows = _read_parquet_rows(parquet)
    log.info("ViSEC rows: %d", len(rows))

    # Bucket by emotion_id (an int 0..3). The ViSEC parquet stores
    # emotion as both a string ("happy" / "neutral" / "sad" / "angry")
    # AND an int ("emotion_id"). Use the int for stability.
    buckets: dict[int, list] = {i: [] for i in range(4)}
    for row in rows:
        # The audio lives under row["path"]["bytes"], not "audio".
        emo_id = row.get("emotion_id")
        if emo_id is None:
            continue
        try:
            emo_id = int(emo_id)
        except (TypeError, ValueError):
            continue
        if emo_id in buckets:
            buckets[emo_id].append(row)

    log.info(
        "ViSEC stratified counts: %s",
        {LABEL_NAMES[k]: len(v) for k, v in buckets.items()},
    )

    rng = random.Random(seed)
    out: List[VisecSample] = []
    index_lines: List[str] = []
    for cls in range(4):
        pool = buckets[cls]
        if len(pool) < per_class:
            raise RuntimeError(
                f"only {len(pool)} '{LABEL_NAMES[cls]}' clips available, "
                f"need {per_class}"
            )
        rng.shuffle(pool)
        chosen = pool[:per_class]
        for raw_idx, row in enumerate(chosen):
            # ViSEC ships the wav bytes under row["path"]["bytes"] (along
            # with a sibling "path" string like "00000.wav"). Pull from
            # there, not from "audio.bytes".
            audio = row.get("path") or {}
            if not isinstance(audio, dict):
                # Defensive: if pyarrow returns the "path" group as a
                # primitive, skip this clip.
                audio = {}
            blob = audio.get("bytes")
            if blob is None:
                log.warning("clip %d/%d (%s) has no audio bytes; skipping",
                            raw_idx, per_class, LABEL_NAMES[cls])
                continue
            text = str(row.get("transcription", "") or row.get("text", "") or "")
            wav_path = samples_dir / f"{LABEL_NAMES[cls]}_{raw_idx:03d}.wav"
            wav_path.write_bytes(blob)
            sample = VisecSample(
                wav_path=str(wav_path),
                label=cls,
                label_name=LABEL_NAMES[cls],
                text=text,
            )
            out.append(sample)
            d = asdict(sample)
            # Store the wav path RELATIVE to the cache root so the cache
            # survives moves of ``bench/``.
            d["wav_path"] = str(Path(wav_path).relative_to(cache_dir))
            index_lines.append(json.dumps(d, ensure_ascii=False))

    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    log.info("cached %d ViSEC samples at %s", len(out), index_path)
    return out


__all__ = [
    "LABEL_NAMES",
    "LABEL_NAME_TO_INDEX",
    "VisecSample",
    "load_visec",
]
