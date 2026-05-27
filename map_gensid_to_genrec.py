#!/usr/bin/env python3
"""
Utilities for converting between GR and LETTER/TIGER data formats.

Pipeline usage: map a LETTER-style SID mapping JSON back into GR .sem_ids format.
The CLI intentionally requires only:
  1. path to SID mapping JSON
  2. path to GR processed directory
  3. dataset/category name

Example:
    python gr_letter_mapper.py \
        --sid_mapping_path ./LETTER_cache/Sports_and_Outdoors.spectral_4lvl_gr.json \
        --gr_path ./cache/AmazonReviews2014/Sports_and_Outdoors/processed \
        --dataset Sports_and_Outdoors
"""

import argparse
import json
import re
import shlex
import sys
import torch
import numpy as np
import pandas as pd
from typing import Any, Iterable, List, Optional, Tuple
from pathlib import Path


DATASET_N_ITEMS = {
    "Beauty": 12101,
    "Sports_and_Outdoors": 18357,
    "Toys_and_Games": 11924,
    "MIND": 20284,
    "Yelp": 94305,
}


def _log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, file=sys.stderr)


def _write_text_file(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _write_shell_env_file(
    env_file: str | Path,
    *,
    sem_ids_path: Path,
    dataset: str,
    codebook_sizes: list[int],
) -> Path:
    codebook_sizes_str = ",".join(map(str, codebook_sizes))
    lines = [
        f"export GR_SEM_IDS_PATH={shlex.quote(str(sem_ids_path))}",
        f"export GR_SEM_IDS_FILE={shlex.quote(str(sem_ids_path))}",
        f"export GR_SEM_IDS_DIR={shlex.quote(str(sem_ids_path.parent))}",
        f"export GR_SEM_IDS_BASENAME={shlex.quote(sem_ids_path.name)}",
        f"export GR_SEM_IDS_CODEBOOK_SIZES={shlex.quote(codebook_sizes_str)}",
        f"export GR_SEM_IDS_DEPTH={len(codebook_sizes)}",
        f"export GR_DATASET={shlex.quote(dataset)}",
        "",
    ]
    return _write_text_file(env_file, "\n".join(lines))


# -----------------------------------------------------------------------------
# LETTER SID mapping -> GR .sem_ids
# -----------------------------------------------------------------------------

def _load_gr_id_mapping(gr_path: Path) -> dict[str, Any]:
    """Load GR id_mapping.json, supporting both normal JSON and the old pandas layout."""
    id_mapping_path = Path(gr_path) / "id_mapping.json"
    if not id_mapping_path.exists():
        raise FileNotFoundError(f"Missing GR id mapping: {id_mapping_path}")

    with open(id_mapping_path, "r") as f:
        raw = f.read().strip()

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "item2id" in payload:
            return payload
    except json.JSONDecodeError:
        pass

    # Keep compatibility with the original script's format assumption:
    # pd.read_json(..., lines=True).T[0]["item2id"]
    payload = pd.read_json(id_mapping_path, lines=True).T[0]
    if "item2id" not in payload:
        raise ValueError(f"Could not find item2id in {id_mapping_path}")
    return payload


def _sid_token_to_int(token: Any) -> int:
    """Convert '<a_123>' / '<z_-1>' / 123 into integer code ids."""
    if isinstance(token, (int, np.integer)):
        return int(token)
    if isinstance(token, str):
        m = re.search(r"-?\d+", token)
        if m is None:
            raise ValueError(f"Cannot parse SID token as integer: {token!r}")
        return int(m.group(0))
    raise TypeError(f"Unsupported SID token type: {type(token).__name__}: {token!r}")


def _normalize_letter_sids(letter_sids: dict[str, Any]) -> dict[int, list[int]]:
    """Convert LETTER item_id -> token list into int item_id -> int SID list."""
    out: dict[int, list[int]] = {}
    for item_id, sid in letter_sids.items():
        if not isinstance(sid, list):
            raise ValueError(f"SID for item {item_id!r} must be a list, got {type(sid).__name__}")
        out[int(item_id)] = [_sid_token_to_int(tok) for tok in sid]

    lengths = {len(v) for v in out.values()}
    if len(lengths) != 1:
        raise ValueError(f"All SIDs must have the same length, got lengths={sorted(lengths)}")
    return out


def _infer_codebook_sizes(sids: Iterable[list[int]]) -> list[int]:
    sid_array = np.asarray(list(sids), dtype=np.int64)
    if sid_array.ndim != 2:
        raise ValueError(f"Expected 2-D SID array, got shape {sid_array.shape}")

    sizes: list[int] = []
    for col in range(sid_array.shape[1]):
        non_negative = sid_array[:, col][sid_array[:, col] >= 0]
        sizes.append(int(non_negative.max()) + 1 if non_negative.size else 0)
    return sizes


def _safe_slug(path: Path, dataset: str) -> str:
    stem = path.stem
    for prefix in (f"{dataset}.index.", f"{dataset}.", f"{dataset}_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    stem = re.sub(r"[^A-Za-z0-9.+-]+", "-", stem).strip("-")
    return stem or "sid"


def _default_output_path(sid_mapping_path: Path, gr_path: Path, dataset: str, codebook_sizes: list[int]) -> Path:
    # Filename shape is friendly to existing eval_sids.py convention:
    # something_<comma-separated-codebook-sizes>_something.sem_ids
    cbs = ",".join(map(str, codebook_sizes))
    slug = _safe_slug(sid_mapping_path, dataset)
    return gr_path / f"sid_{cbs}_{slug}.sem_ids"


def letter_sid_mapping_to_gr_sem_ids(
    sid_mapping_path: str | Path,
    gr_path: str | Path,
    dataset: str,
    output_file: str | Path | None = None,
    output_path_file: str | Path | None = None,
    env_file: str | Path | None = None,
    quiet: bool = False,
) -> Path:
    """
    Convert a LETTER-style SID mapping to GR .sem_ids.

    Input SID mapping format:
        {"0": ["<a_1>", "<b_2>"], "1": ["<a_3>", "<b_4>"], ...}

    Output GR format:
        {"ASIN_RAW_ID": [1, 2], "OTHER_ASIN": [3, 4], ...}
    """
    sid_mapping_path = Path(sid_mapping_path)
    gr_path = Path(gr_path)

    if not sid_mapping_path.exists():
        raise FileNotFoundError(f"Missing SID mapping: {sid_mapping_path}")
    if not gr_path.exists():
        raise FileNotFoundError(f"Missing GR processed path: {gr_path}")

    _log(f"Mapping LETTER SIDs to GR format for dataset={dataset}...", quiet=quiet)
    _log(f"SID mapping: {sid_mapping_path}", quiet=quiet)
    _log(f"GR path:    {gr_path}", quiet=quiet)

    id_mapping = _load_gr_id_mapping(gr_path)
    raw_id2asin = {int(v) - 1: str(k) for k, v in id_mapping["item2id"].items() if int(v) > 0}

    with open(sid_mapping_path, "r") as f:
        letter_sids_raw = json.load(f)
    letter_sids = _normalize_letter_sids(letter_sids_raw)

    missing_items = sorted(set(letter_sids) - set(raw_id2asin))
    if missing_items:
        preview = missing_items[:20]
        raise KeyError(
            f"SID mapping contains {len(missing_items)} item ids absent from GR id_mapping. "
            f"First missing ids: {preview}"
        )

    gr_sids = {raw_id2asin[item_id]: sid for item_id, sid in sorted(letter_sids.items())}
    codebook_sizes = _infer_codebook_sizes(gr_sids.values())

    expected_n = DATASET_N_ITEMS.get(dataset)
    if expected_n is not None and len(gr_sids) != expected_n:
        _log(f"Warning: expected {expected_n} items for {dataset}, converted {len(gr_sids)} items.", quiet=quiet)
    if len(gr_sids) != len(raw_id2asin):
        _log(
            f"Warning: GR id_mapping has {len(raw_id2asin)} items, "
            f"but SID mapping has {len(gr_sids)} items.",
            quiet=quiet,
        )

    out_path = Path(output_file) if output_file is not None else _default_output_path(
        sid_mapping_path=sid_mapping_path,
        gr_path=gr_path,
        dataset=dataset,
        codebook_sizes=codebook_sizes,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(gr_sids, f)

    if output_path_file is not None:
        path_file = _write_text_file(output_path_file, str(out_path) + "\n")
        _log(f"Saved output path file to: {path_file}", quiet=quiet)

    if env_file is not None:
        env_path = _write_shell_env_file(
            env_file,
            sem_ids_path=out_path,
            dataset=dataset,
            codebook_sizes=codebook_sizes,
        )
        _log(f"Saved shell env handoff file to: {env_path}", quiet=quiet)

    _log("Finished mapping LETTER SIDs to GR format!", quiet=quiet)
    _log(f"Saved GR .sem_ids to: {out_path}", quiet=quiet)
    _log(f"Items converted: {len(gr_sids)}", quiet=quiet)
    _log(f"SID depth: {len(codebook_sizes)}", quiet=quiet)
    _log(f"Inferred codebook sizes: {','.join(map(str, codebook_sizes))}", quiet=quiet)

    return out_path


def letter_sids_to_gr(gr_path, letter_cache_path, dataset_src, model_id_srcs):
    """
    Backwards-compatible wrapper for the old manual tuple-based API.

    New pipeline code should call letter_sid_mapping_to_gr_sem_ids(...) or use the CLI.
    """
    gr_path = Path(gr_path)
    letter_cache_path = Path(letter_cache_path)
    for model_id_src, cbs, emb_name, quantizer in model_id_srcs:
        letter_sid_mapping_to_gr_sem_ids(
            sid_mapping_path=letter_cache_path / f"{dataset_src}.{model_id_src}.json",
            gr_path=gr_path,
            dataset=dataset_src,
            output_file=gr_path / f"{emb_name}_{cbs}_{quantizer}.sem_ids",
        )


def gr2letter_main():
    # NOTE! RUN A SINGLE STEP AT A TIME AND ADAPT THE INPUT ARGUMENTS ACCORDINGLY

    # STEP 1: Map GR Dataset and sentence embeddings to LETTER format
    datasets = [("Yelp", None), ("MIND", None)]  # ("AmazonReviews2014", "Books")
    embedders = [("sentence-t5-base", 768), ("Qwen3-Embedding-0.6B", 1024)]
    add_time = False

    for dataset, category in datasets:
        dataset_id = dataset if category is None else category
        letter_path = Path.home() / f"research/graphawaresid/data/{dataset_id}"
        gr2letter(dataset, category, embedders, letter_path, add_time=add_time)

    # STEP 2: Map CF embeddings to LETTER format
    ckpts = [
        ("Yelp", "genrec_default-May-12-2026_15-48-1f95b41c.pth", "SASRec"),
        ("MIND", "genrec_default-May-12-2026_23-48-41eef089.pth", "SASRec")]
    ckpt_path = Path("ckpt")
    letter_ckpt_path = Path.home() / "research/graphawaresid/RQ-VAE/ckpt"
    for dataset_id, ckpt, model in ckpts:
        gr_cf2letter(ckpt_path / ckpt, letter_ckpt_path, dataset_id, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map LETTER SID JSON to GR .sem_ids format.")
    parser.add_argument("--sid_mapping_path", required=True, type=Path, help="Path to LETTER item_id -> SID JSON.")
    parser.add_argument("--gr_path", required=True, type=Path, help="Path to GR processed directory containing id_mapping.json.")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset/category name, e.g. Sports_and_Outdoors.")
    parser.add_argument(
        "--output_file",
        default=None,
        type=Path,
        help="Optional output .sem_ids path. Default: <gr_path>/sid_<inferred_cbs>_<sid_mapping_stem>.sem_ids",
    )
    parser.add_argument(
        "--output_path_file",
        default=None,
        type=Path,
        help="Optional text file where the generated .sem_ids path is written. Useful for shell pipelines.",
    )
    parser.add_argument(
        "--env_file",
        default=None,
        type=Path,
        help="Optional shell file with exports like GR_SEM_IDS_PATH=... for the next pipeline step.",
    )
    parser.add_argument(
        "--print_output_path",
        action="store_true",
        help="Print only the generated .sem_ids path to stdout. Logs still go to stderr unless --quiet is set.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable status logs. Intended for command substitution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = letter_sid_mapping_to_gr_sem_ids(
        sid_mapping_path=args.sid_mapping_path,
        gr_path=args.gr_path,
        dataset=args.dataset,
        output_file=args.output_file,
        output_path_file=args.output_path_file,
        env_file=args.env_file,
        quiet=args.quiet,
    )
    if args.print_output_path:
        print(out_path)


if __name__ == "__main__":
    main()
