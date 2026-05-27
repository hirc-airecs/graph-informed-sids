import logging

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
from torch import Tensor
from torch_cluster import random_walk
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import subgraph
import os
import json
import pandas as pd


def load_train_item_mask(path, num_items, skip_n_last=None):
    with open(os.path.join(path), "r") as f:
        inters = json.load(f)

    mask = np.zeros(num_items, dtype=bool)
    for items in inters.values():
        seq = items if skip_n_last is None else items[:-skip_n_last]
        for item in seq:
            item = int(item)
            if item < 0 or item >= num_items:
                raise ValueError(f"Interaction item_id={item} is outside embeddings range [0, {num_items})")
            mask[item] = True

    return np.flatnonzero(mask).astype(np.int64)


def compute_graph_spectral_vectors(edge_index, edge_weight, num_nodes: int, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((num_nodes, 0), dtype=np.float32)

    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    if edge_weight is None:
        val = np.ones(len(row), dtype=np.float64)
    else:
        val = edge_weight.cpu().numpy().astype(np.float64)

    A = sp.coo_matrix((val, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A = 0.5 * (A + A.T)

    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, 1e-12)
    inv_sqrt_deg = 1.0 / np.sqrt(deg)

    D_inv_sqrt = sp.diags(inv_sqrt_deg)
    L = sp.eye(num_nodes, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

    # ask for one extra eigenvector, then drop the trivial one
    k_eff = min(k + 1, max(num_nodes - 1, 1))

    if num_nodes <= 2:
        return np.zeros((num_nodes, k), dtype=np.float32)

    try:
        evals, evecs = eigsh(L, k=k_eff, which="SM")
        order = np.argsort(evals)
        evecs = evecs[:, order]
    except Exception as e:
        logging.warning(f"eigsh failed, returning zeros for spectral vectors: {e}")
        return np.zeros((num_nodes, k), dtype=np.float32)

    # drop the first trivial eigenvector
    evecs = evecs[:, 1 : 1 + k]

    if evecs.shape[1] < k:
        pad = np.zeros((num_nodes, k - evecs.shape[1]), dtype=np.float32)
        evecs = np.concatenate([evecs.astype(np.float32), pad], axis=1)
    else:
        evecs = evecs.astype(np.float32)

    return evecs


def build_bipartite_rw_item_graph(
    df: pl.DataFrame,
    walk_length: int = 10,
    walks_per_item: int = 10,
    window_size: int = 2,
    self_loops: bool = False,
    dedup_per_walk: bool = False,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Build an item-item graph from plain random walks on a user-item bipartite graph.
    CPU only.
    """
    required = {"user_id", "ts", "item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {missing}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    ui = df.select(["user_id", "item_id"]).unique().sort(["user_id", "item_id"])

    if ui.height == 0:
        return pl.DataFrame(
            {
                "src": pl.Series([], dtype=pl.Int64),
                "dst": pl.Series([], dtype=pl.Int64),
                "weight": pl.Series([], dtype=pl.Float64),
            }
        )

    item_map = ui.select("item_id").unique().sort("item_id").with_row_index("item_node")
    user_map = ui.select("user_id").unique().sort("user_id").with_row_index("user_node")

    num_items = item_map.height
    user_offset = num_items

    user_map = user_map.with_columns((pl.col("user_node") + user_offset).alias("user_node"))

    edges = (
        ui.join(item_map, on="item_id", how="left")
        .join(user_map, on="user_id", how="left")
        .select(["item_id", "item_node", "user_node"])
    )

    item_to_user = edges.select(
        pl.col("item_node").alias("src"),
        pl.col("user_node").alias("dst"),
    )
    user_to_item = edges.select(
        pl.col("user_node").alias("src"),
        pl.col("item_node").alias("dst"),
    )
    edge_df = pl.concat([item_to_user, user_to_item])

    edge_index = torch.tensor(
        edge_df.select(["src", "dst"]).to_numpy().T,
        dtype=torch.long,
    )

    row, col = edge_index[0].contiguous(), edge_index[1].contiguous()

    start_items = torch.arange(num_items, dtype=torch.long)
    start = start_items.repeat_interleave(walks_per_item)

    rw = random_walk(row, col, start=start, walk_length=walk_length)
    item_walks = rw[:, ::2].numpy()

    item_ids = item_map.sort("item_node")["item_id"].to_numpy()

    pair_counts = {}

    for walk in item_walks:
        seen_pairs = set()
        L = len(walk)

        for i in range(L):
            src_node = int(walk[i])
            if src_node >= num_items:
                continue
            src = int(item_ids[src_node])

            lo = max(0, i - window_size)
            hi = min(L, i + window_size + 1)

            for j in range(lo, hi):
                if i == j:
                    continue

                dst_node = int(walk[j])
                if dst_node >= num_items:
                    continue
                dst = int(item_ids[dst_node])

                if not self_loops and src == dst:
                    continue

                pair = (src, dst)

                if dedup_per_walk:
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                pair_counts[pair] = pair_counts.get(pair, 0.0) + 1.0

    if not pair_counts:
        return pl.DataFrame(
            {
                "src": pl.Series([], dtype=pl.Int64),
                "dst": pl.Series([], dtype=pl.Int64),
                "weight": pl.Series([], dtype=pl.Float64),
            }
        )

    return (
        pl.DataFrame(
            {
                "src": [k[0] for k in pair_counts.keys()],
                "dst": [k[1] for k in pair_counts.keys()],
                "weight": [float(v) for v in pair_counts.values()],
            }
        )
        .group_by(["src", "dst"])
        .agg(pl.col("weight").sum().alias("weight"))
        .sort("weight", descending=True)
    )


def build_windowed_cooccurrence_graph(
    df: pl.DataFrame,
    window_size: int = 3,
    bidirectional: bool = False,
    dedup_within_window: bool = False,
    decay: str | None = None,  # None | "inverse_distance"
) -> pl.DataFrame:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    df = df.sort(["user_id", "ts"])

    shifted_frames = []
    for k in range(1, window_size + 1):
        if decay == "inverse_distance":
            w_expr = pl.lit(1.0 / k).alias("pair_weight")
        else:
            w_expr = pl.lit(1.0).alias("pair_weight")

        shifted = df.with_columns(pl.col("item_id").shift(-k).over("user_id").alias("dst")).select(
            [
                pl.col("user_id"),
                pl.col("item_id").alias("src"),
                pl.col("dst"),
                w_expr,
            ]
        )
        shifted_frames.append(shifted)

    pairs = pl.concat(shifted_frames).filter(pl.col("dst").is_not_null())

    if dedup_within_window:
        pairs = pairs.group_by(["user_id", "src", "dst"]).agg(pl.col("pair_weight").max().alias("pair_weight"))

    if bidirectional:
        rev = pairs.select(
            [
                pl.col("user_id"),
                pl.col("dst").alias("src"),
                pl.col("src").alias("dst"),
                pl.col("pair_weight"),
            ]
        )
        pairs = pl.concat([pairs, rev])

    out = (
        pairs.group_by(["src", "dst"]).agg(pl.col("pair_weight").sum().alias("weight")).sort("weight", descending=True)
    )

    return out


def build_adjacent_cooccurrence_graph(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build a directed co-occurrence graph from sequential data using adjacent pairs only.

    Input columns required:
        - user_id
        - ts
        - item_id

    Returns:
        DataFrame with columns:
        - src
        - dst
        - weight
    """
    df = df.sort(["user_id", "ts"])

    out = (
        df.with_columns(pl.col("item_id").shift(-1).over("user_id").alias("next_item"))
        .filter(pl.col("next_item").is_not_null())
        .group_by(["item_id", "next_item"])
        .agg(pl.len().alias("weight"))
        .rename(
            {
                "item_id": "src",
                "next_item": "dst",
            }
        )
        .sort("weight", descending=True)
    )

    return out


def polars_graph_to_pyg(
    graph_df: pl.DataFrame,
    labels_df: pl.DataFrame | None = None,
    make_undirected: bool = True,
    add_identity_features: bool = False,
    num_items: int | None = None,
    all_item_ids: list[int] | None = None,
):
    """
    Convert a Polars edge list into a PyG Data object.

    graph_df columns:
        - src
        - dst
        - weight   (optional but recommended)

    labels_df columns (optional):
        - item_id
        - y

    You must provide one of:
        - num_items: assumes valid item ids are [0, ..., num_items-1]
        - all_item_ids: explicit universe of item ids

    Returns:
        data: torch_geometric.data.Data
        node_map: Polars DataFrame with columns [item_id, node_id]
    """
    required = {"src", "dst"}
    missing = required - set(graph_df.columns)
    if missing:
        raise ValueError(f"graph_df is missing columns: {missing}")

    if "weight" not in graph_df.columns:
        graph_df = graph_df.with_columns(pl.lit(1.0).alias("weight"))

    if (num_items is None) == (all_item_ids is None):
        raise ValueError("Provide exactly one of num_items or all_item_ids")

    if num_items is not None:
        if num_items <= 0:
            raise ValueError("num_items must be positive")

        node_map = (
            pl.DataFrame({"item_id": list(range(num_items))})
            .with_columns(pl.col("item_id").alias("node_id"))
            .select(["item_id", "node_id"])
        )
        num_nodes = int(num_items)

        valid_ids = set(node_map["item_id"].to_list())
        edge_ids = set(graph_df["src"].to_list()) | set(graph_df["dst"].to_list())
        unknown = edge_ids - valid_ids
        if unknown:
            raise ValueError(
                f"graph_df contains item ids not present in provided item universe: {sorted(list(unknown))[:20]}"
            )

        edges = graph_df.select(
            [
                pl.col("src").cast(pl.Int64).alias("src_id"),
                pl.col("dst").cast(pl.Int64).alias("dst_id"),
                pl.col("weight").cast(pl.Float64).alias("weight"),
            ]
        )
    else:
        item_ids_df = pl.DataFrame({"item_id": all_item_ids}).unique().sort("item_id").with_row_index("node_id")
        if item_ids_df.height == 0:
            raise ValueError("all_item_ids is empty")

        ids = item_ids_df["item_id"].to_list()
        if min(ids) < 0:
            raise ValueError("item ids must be non-negative")

        node_map = item_ids_df.select(["item_id", "node_id"])
        num_nodes = int(node_map.height)

        valid_ids = set(node_map["item_id"].to_list())
        edge_ids = set(graph_df["src"].to_list()) | set(graph_df["dst"].to_list())
        unknown = edge_ids - valid_ids
        if unknown:
            raise ValueError(
                f"graph_df contains item ids not present in provided item universe: {sorted(list(unknown))[:20]}"
            )

        src_map = node_map.rename({"item_id": "src", "node_id": "src_id"})
        dst_map = node_map.rename({"item_id": "dst", "node_id": "dst_id"})

        edges = (
            graph_df.join(src_map, on="src", how="left")
            .join(dst_map, on="dst", how="left")
            .select(
                [
                    pl.col("src_id").cast(pl.Int64),
                    pl.col("dst_id").cast(pl.Int64),
                    pl.col("weight").cast(pl.Float64).alias("weight"),
                ]
            )
        )

    if make_undirected:
        rev = edges.select(
            [
                pl.col("dst_id").alias("src_id"),
                pl.col("src_id").alias("dst_id"),
                pl.col("weight"),
            ]
        )
        edges = pl.concat([edges, rev])

    edges = edges.group_by(["src_id", "dst_id"]).agg(pl.col("weight").sum().alias("weight")).sort(["src_id", "dst_id"])

    if edges.height == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(
            edges.select(["src_id", "dst_id"]).to_numpy().T,
            dtype=torch.long,
        )
        edge_weight = torch.tensor(edges["weight"].to_numpy(), dtype=torch.float)

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

    # Labels
    if labels_df is not None:
        lbl = node_map.join(labels_df, on="item_id", how="left").sort("node_id")

        if lbl["y"].null_count() > 0:
            raise ValueError("Some nodes have no label in labels_df")

        y = torch.full((num_nodes,), -1, dtype=torch.long)
        y_idx = torch.tensor(lbl["node_id"].to_numpy(), dtype=torch.long)
        y_val = torch.tensor(lbl["y"].to_numpy(), dtype=torch.long)
        y[y_idx] = y_val
        data.y = y

    # Degree-based fallback feature; isolated nodes get 0
    deg = torch.zeros(num_nodes, dtype=torch.float)
    if edge_weight.numel() > 0:
        deg.scatter_add_(0, data.edge_index[0], data.edge_weight)
    log_deg = torch.log1p(deg).unsqueeze(-1)

    if add_identity_features:
        x = torch.eye(num_nodes, dtype=torch.float)
    else:
        x = log_deg

    data.x = x
    return data, node_map


def load_graph_cluster_labels(
    labels_path: str | None,
    levels=None,
    num_items: int | None = None,
    label_group: str = "coarse_to_fine_item_labels",
):
    """
    Load graph-cluster labels produced by hierarchical clustering scripts.

    This function normalizes several possible clustering-output formats into a
    consistent representation for RQ-VAE graph-cluster supervision.

    Supported JSON formats
    ----------------------
    1. Preferred hierarchical output:

        {
            "coarse_to_fine_item_labels": {
                "level_1": [...],
                "level_2": [...]
            },
            "fine_to_coarse_item_labels": {
                "level_1": [...],
                "level_2": [...]
            },
            ...
        }

    2. Flat level dictionary:

        {
            "level_1": [...],
            "level_2": [...]
        }

    3. Direct list of levels:

        [
            [...],   # level 1
            [...]    # level 2
        ]

    4. Single label vector:

        [...]

    Parameters
    ----------
    labels_path : str | None
        Path to the labels JSON file. If None, returns None.

    levels : str | int | list[str | int] | tuple[str | int] | None
        Which levels to load.

        - None:
            Load all levels in their natural order.
        - str:
            Load one named level, e.g. "level_1".
        - int:
            Load one 1-based level index, e.g. 1 means "level_1".
        - list/tuple:
            Load multiple levels in exactly that order.

        For RQ-VAE prefix supervision, the order matters. For example, if the
        JSON group is fine-to-coarse but you want prefix supervision
        coarse-to-fine, pass the corresponding levels explicitly.

    num_items : int | None
        If provided, every loaded label vector must have this length.

    label_group : str
        Preferred group to load when the JSON contains several groups.
        Usually one of:

            "coarse_to_fine_item_labels"
            "fine_to_coarse_item_labels"
            "coarse_to_fine_graph_labels"

        For RQ-VAE item-level supervision, prefer:

            "coarse_to_fine_item_labels"

    Returns
    -------
    None | np.ndarray | list[np.ndarray]
        - None if labels_path is None.
        - Single np.ndarray[int64] of shape [num_items] if one level is loaded.
        - List[np.ndarray[int64]] if multiple levels are loaded.

    Notes
    -----
    Returned arrays are indexed by global item id:

        labels[item_id] -> graph cluster id

    During training, slice by batch item ids:

        batch_labels = labels[emb_idx]

    For hierarchical prefix supervision:

        batch_labels_per_level = [
            level_labels[emb_idx]
            for level_labels in graph_cluster_labels
        ]

    Items with label -1 are treated as unclustered / ignored by the graph
    cluster contrastive loss.
    """
    import json
    import os
    import re
    import numpy as np

    if labels_path is None:
        return None

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing graph cluster labels file: {labels_path}")

    with open(labels_path, "r") as f:
        payload = json.load(f)

    def _is_label_vector(x):
        return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], (int, np.integer)))

    def _is_list_of_label_vectors(x):
        return isinstance(x, list) and len(x) > 0 and all(isinstance(v, list) for v in x)

    def _level_sort_key(key):
        """
        Sort level_1, level_2, ... numerically.
        Fall back to lexical order for non-standard keys.
        """
        m = re.search(r"(\d+)$", str(key))
        if m is not None:
            return int(m.group(1))
        return str(key)

    def _normalize_level_name(level):
        """
        Accept:
            1 -> "level_1"
            "1" -> "level_1"
            "level_1" -> "level_1"
        """
        if isinstance(level, int):
            return f"level_{level}"

        if isinstance(level, str):
            if level.isdigit():
                return f"level_{int(level)}"
            return level

        raise ValueError(f"Invalid level specifier: {level!r}")

    def _validate_arr(arr, level_name):
        arr = np.asarray(arr, dtype=np.int64)

        if arr.ndim != 1:
            raise ValueError(f"Labels for {level_name!r} must be 1-D, got shape {arr.shape}")

        if num_items is not None and len(arr) != int(num_items):
            raise ValueError(f"Labels for {level_name!r} have length {len(arr)}, but expected num_items={num_items}")

        return arr

    # ------------------------------------------------------------
    # Case 1: payload itself is a single vector: [...]
    # ------------------------------------------------------------
    if _is_label_vector(payload):
        if levels is not None:
            raise ValueError("levels should be None when labels JSON is a single label vector")
        return _validate_arr(payload, "labels")

    # ------------------------------------------------------------
    # Case 2: payload is a list of vectors: [[...], [...]]
    # ------------------------------------------------------------
    if _is_list_of_label_vectors(payload):
        level_dict = {f"level_{i + 1}": v for i, v in enumerate(payload)}

    # ------------------------------------------------------------
    # Case 3: payload is a dict
    # ------------------------------------------------------------
    elif isinstance(payload, dict):
        # Preferred grouped format:
        # {"coarse_to_fine_item_labels": {"level_1": [...], ...}}
        if label_group in payload:
            level_dict = payload[label_group]

        # If requested group does not exist but there is exactly one obvious
        # label group, use it.
        elif any(isinstance(v, dict) and any(str(k).startswith("level_") for k in v.keys()) for v in payload.values()):
            candidate_groups = {
                k: v
                for k, v in payload.items()
                if isinstance(v, dict) and any(str(kk).startswith("level_") for kk in v.keys())
            }

            if len(candidate_groups) == 1:
                level_dict = next(iter(candidate_groups.values()))
            else:
                raise ValueError(
                    f"Requested label_group={label_group!r} not found. "
                    f"Available label groups: {sorted(candidate_groups.keys())}. "
                    f"Pass label_group explicitly."
                )

        # Flat level dict:
        # {"level_1": [...], "level_2": [...]}
        elif any(str(k).startswith("level_") for k in payload.keys()):
            level_dict = payload

        # Single named vector:
        # {"labels": [...]}
        elif "labels" in payload and _is_label_vector(payload["labels"]):
            if levels is not None:
                raise ValueError("levels should be None when labels JSON contains only a single 'labels' vector")
            return _validate_arr(payload["labels"], "labels")

        else:
            raise ValueError(
                "Unrecognized graph cluster labels JSON format. Expected one of: "
                "single vector, list of vectors, flat level dict, or grouped "
                "hierarchical labels dict."
            )

    else:
        raise ValueError(f"Unsupported labels JSON root type: {type(payload).__name__}")

    if not isinstance(level_dict, dict):
        raise ValueError(f"Selected label group must be a dict, got {type(level_dict).__name__}")

    available_levels = sorted(level_dict.keys(), key=_level_sort_key)

    single_level = isinstance(levels, (str, int))

    if levels is None:
        selected_levels = available_levels
    elif single_level:
        selected_levels = [_normalize_level_name(levels)]
    elif isinstance(levels, (list, tuple)):
        selected_levels = [_normalize_level_name(level) for level in levels]
    else:
        raise ValueError("`levels` must be None, str, int, or list/tuple of str/int")

    out = []

    for level in selected_levels:
        if level not in level_dict:
            raise ValueError(f"Requested graph cluster level {level!r} not found. Available levels: {available_levels}")

        out.append(_validate_arr(level_dict[level], level))

    if single_level:
        return out[0]

    return out


def analyze_graph_quick(graph_df: pl.DataFrame, directed: bool = False, weight_col: str = "weight"):
    if weight_col not in graph_df.columns:
        graph_df = graph_df.with_columns(pl.lit(1.0).alias(weight_col))

    if directed:
        edges = graph_df.select(["src", "dst", weight_col])
    else:
        edges = (
            graph_df.with_columns(
                [
                    pl.min_horizontal("src", "dst").alias("u"),
                    pl.max_horizontal("src", "dst").alias("v"),
                ]
            )
            .group_by(["u", "v"])
            .agg(pl.col(weight_col).sum().alias(weight_col))
            .rename({"u": "src", "v": "dst"})
        )

    nodes = pl.concat(
        [
            edges.select(pl.col("src").alias("node")),
            edges.select(pl.col("dst").alias("node")),
        ]
    ).unique()

    num_nodes = nodes.height
    num_edges = edges.height

    if directed:
        out_deg = edges.group_by("src").agg(pl.len().alias("out_degree")).rename({"src": "node"})
        in_deg = edges.group_by("dst").agg(pl.len().alias("in_degree")).rename({"dst": "node"})
        degree_df = (
            nodes.join(out_deg, on="node", how="left")
            .join(in_deg, on="node", how="left")
            .with_columns(
                [
                    pl.col("out_degree").fill_null(0),
                    pl.col("in_degree").fill_null(0),
                ]
            )
            .with_columns((pl.col("out_degree") + pl.col("in_degree")).alias("degree"))
        )
    else:
        degree_df = nodes.join(
            pl.concat(
                [
                    edges.select(pl.col("src").alias("node")),
                    edges.select(pl.col("dst").alias("node")),
                ]
            )
            .group_by("node")
            .agg(pl.len().alias("degree")),
            on="node",
            how="left",
        ).with_columns(pl.col("degree").fill_null(0))

    density = (
        (num_edges / (num_nodes * (num_nodes - 1)) if directed else (2 * num_edges) / (num_nodes * (num_nodes - 1)))
        if num_nodes > 1
        else 0.0
    )

    stats = degree_df.select(
        [
            pl.len().alias("num_nodes"),
            pl.col("degree").mean().alias("avg_degree"),
            pl.col("degree").median().alias("median_degree"),
            pl.col("degree").quantile(0.95).alias("p95_degree"),
            pl.col("degree").max().alias("max_degree"),
            (pl.col("degree") == 0).sum().alias("isolated_nodes"),
        ]
    ).to_dicts()[0]

    stats["num_edges"] = num_edges
    stats["density"] = float(density)
    stats["isolated_ratio"] = float(stats["isolated_nodes"] / num_nodes) if num_nodes else 0.0
    stats["top_degree_nodes"] = degree_df.sort("degree", descending=True).head(10)

    return stats


def get_graph_building_conf(args):
    if args.graph_type == "windowed":
        return {
            "window_size": args.window_size,
            "bidirectional": args.bidirectional,
            "dedup_within_window": args.dedup_within_window,
            "decay": args.decay,
        }

    elif args.graph_type == "rw":
        return {
            "walk_length": args.walk_length,
            "walks_per_item": args.walks_per_item,
            "window_size": args.window_size,
            "self_loops": args.self_loops,
            "dedup_per_walk": args.dedup_per_walk,
            "seed": args.seed,
        }

    elif args.graph_type == "adjacent_cooc":
        return {}

    else:
        raise ValueError(f"Unknown graph_type: {args.graph_type}")


def edge_w_from_subgraph(emb_idx, pyg_data):
    ei, ew = subgraph(
        subset=emb_idx,
        edge_index=pyg_data.edge_index,
        edge_attr=pyg_data.edge_weight,
        relabel_nodes=True,
    )

    B = emb_idx.size(0)
    edge_w = torch.zeros(B, B, device=ei.device)

    if ei.numel() > 0:
        edge_w[ei[0], ei[1]] = ew

    return edge_w


class InteractionGraphBuilder:
    def __init__(
        self,
        interactions_path=None,
        graph_type="adjacent_cooc",
        graph_building_kwargs=None,
        skip_n_last=2,
        make_undirected=True,
        add_identity_features=False,
        load_path=None,
        num_items=None,
        all_item_ids=None,
    ):
        self.interactions_path = interactions_path
        self.graph_type = graph_type
        self.graph_building_kwargs = graph_building_kwargs or {}
        self.skip_n_last = skip_n_last
        self.make_undirected = make_undirected
        self.add_identity_features = add_identity_features

        self.graph_df = None
        self.node_map = None
        self.pyg_data = None
        self.num_items = num_items
        self.all_item_ids = all_item_ids

        if load_path is not None:
            self._load(load_path)
            return

        if interactions_path is None:
            raise ValueError("Either interactions_path or load_path must be provided")

        df = self._load_interactions(interactions_path, skip_n_last=skip_n_last)

        if graph_type == "windowed":
            self.graph_df = build_windowed_cooccurrence_graph(df, **self.graph_building_kwargs)
        elif graph_type == "adjacent_cooc":
            self.graph_df = build_adjacent_cooccurrence_graph(df, **self.graph_building_kwargs)
        elif graph_type == "rw":
            self.graph_df = build_bipartite_rw_item_graph(df, **self.graph_building_kwargs)
        else:
            raise ValueError(f"Unknown graph_type={graph_type}")

        self.pyg_data, self.node_map = polars_graph_to_pyg(
            self.graph_df,
            make_undirected=self.make_undirected,
            add_identity_features=self.add_identity_features,
            num_items=self.num_items,
            all_item_ids=self.all_item_ids,
        )
        self.node_map = self.node_map.sort("node_id")

    def _load_interactions(self, path, skip_n_last=None):
        with open(os.path.join(path), "r") as f:
            inters = json.load(f)

        if skip_n_last is None:
            rows = [(user, item, pos) for user, items in inters.items() for pos, item in enumerate(items)]
        else:
            rows = [
                (user, item, pos) for user, items in inters.items() for pos, item in enumerate(items[:-skip_n_last])
            ]

        df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"]).explode("item_id").reset_index(drop=True)
        return pl.DataFrame(df).select(["user_id", "ts", "item_id"])

    def dump(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        meta = {
            "interactions_path": self.interactions_path,
            "graph_type": self.graph_type,
            "graph_building_kwargs": self.graph_building_kwargs,
            "skip_n_last": self.skip_n_last,
            "make_undirected": self.make_undirected,
            "add_identity_features": self.add_identity_features,
            "num_items": self.num_items,
        }

        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.graph_df.write_parquet(os.path.join(save_dir, "graph_df.parquet"))
        self.node_map.write_parquet(os.path.join(save_dir, "node_map.parquet"))

        pyg_payload = {
            "edge_index": self.pyg_data.edge_index.cpu(),
            "edge_weight": self.pyg_data.edge_weight.cpu()
            if getattr(self.pyg_data, "edge_weight", None) is not None
            else None,
            "x": self.pyg_data.x.cpu() if getattr(self.pyg_data, "x", None) is not None else None,
            "num_nodes": int(self.pyg_data.num_nodes),
        }

        if getattr(self.pyg_data, "y", None) is not None:
            pyg_payload["y"] = self.pyg_data.y.cpu()

        torch.save(pyg_payload, os.path.join(save_dir, "pyg_data.pt"))

    def _load(self, save_dir: str):
        meta_path = os.path.join(save_dir, "meta.json")
        graph_df_path = os.path.join(save_dir, "graph_df.parquet")
        node_map_path = os.path.join(save_dir, "node_map.parquet")
        pyg_data_path = os.path.join(save_dir, "pyg_data.pt")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing {meta_path}")
        if not os.path.exists(graph_df_path):
            raise FileNotFoundError(f"Missing {graph_df_path}")
        if not os.path.exists(node_map_path):
            raise FileNotFoundError(f"Missing {node_map_path}")
        if not os.path.exists(pyg_data_path):
            raise FileNotFoundError(f"Missing {pyg_data_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.interactions_path = meta.get("interactions_path")
        self.graph_type = meta["graph_type"]
        self.graph_building_kwargs = meta.get("graph_building_kwargs", {})
        self.skip_n_last = meta.get("skip_n_last")
        self.make_undirected = meta["make_undirected"]
        self.add_identity_features = meta["add_identity_features"]
        self.num_items = meta.get("num_items")

        self.graph_df = pl.read_parquet(graph_df_path)
        self.node_map = pl.read_parquet(node_map_path).sort("node_id")

        pyg_payload = torch.load(pyg_data_path, map_location="cpu", weights_only=False)

        self.pyg_data = Data(
            edge_index=pyg_payload["edge_index"],
            edge_weight=pyg_payload.get("edge_weight"),
            num_nodes=pyg_payload["num_nodes"],
        )

        if pyg_payload.get("x") is not None:
            self.pyg_data.x = pyg_payload["x"]
        if pyg_payload.get("y") is not None:
            self.pyg_data.y = pyg_payload["y"]

    @classmethod
    def from_dump(cls, load_path: str):
        return cls(load_path=load_path)


class S2GRConv(MessagePassing):
    """
    APPNP-style propagation with modified coefficients (no caching).

    F^(K) = sum_{t=0}^K c_t^(K) * A_hat^t X

    c_t^(K) = alpha * (1-alpha)^(2t) * [2 - alpha - (1-alpha)^(K-t+1)]
    """

    def __init__(self, K: int, alpha: float, add_self_loops: bool = True):
        super().__init__(aggr="add")
        self.K = int(K)
        self.alpha = float(alpha)
        self.add_self_loops = add_self_loops

    def _get_coeffs(self, device, dtype) -> Tensor:
        t = torch.arange(self.K + 1, device=device, dtype=dtype)
        return self.alpha * (1.0 - self.alpha) ** (2 * t) * (2.0 - self.alpha - (1.0 - self.alpha) ** (self.K - t + 1))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        # Normalize adjacency ONCE per forward
        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=x.size(0),
            add_self_loops=self.add_self_loops,
            dtype=x.dtype,
        )

        coeffs = self._get_coeffs(device=x.device, dtype=x.dtype)

        # U^(0)
        u = x
        out = coeffs[0] * u

        # iterative propagation
        for t in range(1, self.K + 1):
            u = self.propagate(edge_index, x=u, edge_weight=edge_weight)
            out = out + coeffs[t] * u

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K}, alpha={self.alpha})"
