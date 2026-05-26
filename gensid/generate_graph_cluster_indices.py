#!/usr/bin/env python3
"""
Generate semantic-ID index mapping directly from hierarchical graph-cluster labels.

This script converts graph clustering output, e.g. from hierarchical_mincut_clustering.py,
into the JSON format used by downstream generative recommendation training:

    {
      "0": ["<a_3>", "<b_17>", "<c_5>"],
      "1": ["<a_3>", "<b_18>", "<c_2>"],
      ...
    }

The expected clustering labels are item-id indexed arrays, usually stored as:

    {
      "coarse_to_fine_item_labels": {
        "level_1": [...],
        "level_2": [...]
      }
    }

For RQ-VAE-style prefix semantics, use the coarse-to-fine group. Each level becomes
one SID token. Optionally, append a uniqueness token to eliminate full-SID collisions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np


DEFAULT_PREFIXES = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>", "<g_{}>", "<h_{}>"]


def _is_label_vector(x: Any) -> bool:
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], int))


def _is_list_of_label_vectors(x: Any) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(isinstance(v, list) for v in x)


def _level_sort_key(key: str):
    m = re.search(r"(\d+)$", str(key))
    if m is not None:
        return int(m.group(1))
    return str(key)


def _normalize_level_name(level: str | int) -> str:
    if isinstance(level, int):
        return f"level_{level}"
    if isinstance(level, str):
        if level.isdigit():
            return f"level_{int(level)}"
        return level
    raise ValueError(f"Invalid level specifier: {level!r}")


def load_cluster_label_levels(
    labels_path: str,
    label_group: str = "coarse_to_fine_item_labels",
    levels: list[str] | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Load one or more item-id indexed cluster-label arrays from a clustering JSON.

    Supports:
      1. grouped dict: {"coarse_to_fine_item_labels": {"level_1": [...], ...}}
      2. flat dict: {"level_1": [...], "level_2": [...]}
      3. list of levels: [[...], [...]]
      4. single vector: [...]
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    with open(labels_path, "r") as f:
        payload = json.load(f)

    if _is_label_vector(payload):
        level_dict = {"level_1": payload}
    elif _is_list_of_label_vectors(payload):
        level_dict = {f"level_{i + 1}": v for i, v in enumerate(payload)}
    elif isinstance(payload, dict):
        if label_group in payload:
            level_dict = payload[label_group]
        elif any(str(k).startswith("level_") for k in payload.keys()):
            level_dict = payload
        else:
            candidate_groups = {
                k: v
                for k, v in payload.items()
                if isinstance(v, dict) and any(str(kk).startswith("level_") for kk in v.keys())
            }
            if len(candidate_groups) == 1:
                level_dict = next(iter(candidate_groups.values()))
            else:
                raise ValueError(
                    f"Could not select label group. Requested {label_group!r}; "
                    f"available candidate groups: {sorted(candidate_groups.keys())}"
                )
    else:
        raise ValueError(f"Unsupported JSON root type: {type(payload).__name__}")

    if not isinstance(level_dict, dict):
        raise ValueError(f"Selected label group must be dict, got {type(level_dict).__name__}")

    available = sorted(level_dict.keys(), key=_level_sort_key)
    selected = available if levels is None else [_normalize_level_name(l) for l in levels]

    out_names: list[str] = []
    out_arrays: list[np.ndarray] = []
    for name in selected:
        if name not in level_dict:
            raise ValueError(f"Requested level {name!r} not found. Available: {available}")
        arr = np.asarray(level_dict[name], dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"Level {name!r} must be 1-D, got shape {arr.shape}")
        out_names.append(name)
        out_arrays.append(arr)

    lengths = {len(a) for a in out_arrays}
    if len(lengths) != 1:
        raise ValueError(f"All levels must have the same length, got {sorted(lengths)}")

    return out_names, out_arrays


def make_contiguous_per_level(labels: list[np.ndarray], keep_negative: bool = True) -> tuple[list[np.ndarray], list[dict[int, int]]]:
    """Remap arbitrary cluster ids to compact 0..K-1 ids per level."""
    remapped = []
    maps = []
    for arr in labels:
        valid = np.unique(arr[arr >= 0] if keep_negative else arr)
        mapping = {int(old): int(new) for new, old in enumerate(sorted(map(int, valid)))}
        new_arr = np.full_like(arr, -1)
        for old, new in mapping.items():
            new_arr[arr == old] = new
        if not keep_negative and np.any(arr < 0):
            # Put negative labels after valid labels if user really asked not to keep them.
            neg_values = sorted(map(int, np.unique(arr[arr < 0])))
            offset = len(mapping)
            for j, old in enumerate(neg_values):
                mapping[old] = offset + j
                new_arr[arr == old] = offset + j
        remapped.append(new_arr)
        maps.append(mapping)
    return remapped, maps


def build_codes(
    labels: list[np.ndarray],
    prefixes: list[str],
    unclustered_strategy: str,
    append_unique_leaf: bool,
    unique_prefix: str,
) -> dict[str, list[str]]:
    """
    Convert level labels into token sequences.

    unclustered_strategy:
      - special: use the same special id -1 at every level.
      - unique: use item id as the code for unclustered values.
      - error: fail if any -1 is present.
    append_unique_leaf:
      If True, append a final per-path rank token to make full SIDs unique.
    """
    if len(labels) == 0:
        raise ValueError("No label levels were loaded")
    if len(prefixes) < len(labels):
        raise ValueError(f"Need at least {len(labels)} token prefixes, got {len(prefixes)}")

    n_items = len(labels[0])
    codes: dict[str, list[str]] = {}

    for item_id in range(n_items):
        item_tokens = []
        for level_idx, arr in enumerate(labels):
            cid = int(arr[item_id])
            if cid < 0:
                if unclustered_strategy == "error":
                    raise ValueError(f"item_id={item_id} has unclustered label {cid} at level {level_idx + 1}")
                if unclustered_strategy == "unique":
                    cid = item_id
                elif unclustered_strategy == "special":
                    cid = -1
                else:
                    raise ValueError(f"Unknown unclustered_strategy={unclustered_strategy!r}")
            item_tokens.append(prefixes[level_idx].format(cid))
        codes[str(item_id)] = item_tokens

    if append_unique_leaf:
        path_to_items: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for item_id, toks in codes.items():
            path_to_items[tuple(toks)].append(item_id)

        for toks, item_ids in path_to_items.items():
            if len(item_ids) == 1:
                # rank 0 still makes all items have same SID length.
                codes[item_ids[0]].append(unique_prefix.format(0))
            else:
                for rank, item_id in enumerate(sorted(item_ids, key=lambda x: int(x))):
                    codes[item_id].append(unique_prefix.format(rank))

    return codes


def collision_stats(codes: dict[str, list[str]]) -> dict[str, Any]:
    paths = [tuple(v) for v in codes.values()]
    c = Counter(paths)
    n = len(paths)
    n_unique = len(c)
    return {
        "num_items": n,
        "num_unique_sids": n_unique,
        "num_colliding_items": int(sum(v for v in c.values() if v > 1)),
        "collision_rate": float((n - n_unique) / n) if n else 0.0,
        "max_collision_group_size": int(max(c.values())) if c else 0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Generate GR SID mapping directly from graph-cluster hierarchy labels.")
    p.add_argument("--labels_path", required=True, help="Path to clustering labels JSON.")
    p.add_argument("--output_file", required=True, help="Where to save item_id -> token-list JSON.")
    p.add_argument("--label_group", default="coarse_to_fine_item_labels", help="Label group in clustering JSON.")
    p.add_argument("--levels", nargs="*", default=None, help="Levels to use, e.g. level_1 level_2. Default: all.")
    p.add_argument("--prefixes", nargs="*", default=None, help="Token format strings, e.g. '<a_{}>' '<b_{}>'.")
    p.add_argument("--reindex_contiguous", action="store_true", help="Remap cluster ids per level to compact 0..K-1 ids.")
    p.add_argument(
        "--unclustered_strategy",
        choices=["special", "unique", "error"],
        default="unique",
        help="How to handle label -1/unclustered items. Default gives each such item unique per-level ids.",
    )
    p.add_argument(
        "--append_unique_leaf",
        action="store_true",
        help="Append final per-path rank token so full SID is unique even if leaf clusters contain multiple items.",
    )
    p.add_argument("--unique_prefix", default="<z_{}>", help="Format for appended uniqueness token.")
    p.add_argument("--stats_file", default=None, help="Optional path to save stats JSON. Default: output_file + '.stats.json'.")
    return p.parse_args()


def main():
    args = parse_args()

    level_names, labels = load_cluster_label_levels(
        labels_path=args.labels_path,
        label_group=args.label_group,
        levels=args.levels,
    )

    remap_maps = None
    if args.reindex_contiguous:
        labels, remap_maps = make_contiguous_per_level(labels, keep_negative=True)

    prefixes = args.prefixes if args.prefixes is not None else DEFAULT_PREFIXES

    codes = build_codes(
        labels=labels,
        prefixes=prefixes,
        unclustered_strategy=args.unclustered_strategy,
        append_unique_leaf=args.append_unique_leaf,
        unique_prefix=args.unique_prefix,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(codes, f)

    stats = collision_stats(codes)
    stats.update(
        {
            "labels_path": args.labels_path,
            "label_group": args.label_group,
            "levels": level_names,
            "num_levels": len(level_names),
            "append_unique_leaf": bool(args.append_unique_leaf),
            "unclustered_strategy": args.unclustered_strategy,
            "level_cluster_counts_excluding_minus1": {
                name: int(len(set(map(int, arr[arr >= 0])))) for name, arr in zip(level_names, labels)
            },
            "level_unclustered_counts": {
                name: int((arr < 0).sum()) for name, arr in zip(level_names, labels)
            },
        }
    )
    if remap_maps is not None:
        stats["reindex_contiguous"] = True
        stats["remap_sizes"] = {name: len(mp) for name, mp in zip(level_names, remap_maps)}
    else:
        stats["reindex_contiguous"] = False

    stats_file = args.stats_file or (args.output_file + ".stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved SID mapping to: {args.output_file}")
    print(f"Saved stats to: {stats_file}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
