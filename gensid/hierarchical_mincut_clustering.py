#!/usr/bin/env python3
"""
Hierarchical graph clustering for graph-aware SID / RQ-VAE supervision.

Graph input is built only through graph_utils.InteractionGraphBuilder.

Methods:
  - stacked_dmon: explicit dense stacked DMoN hierarchy with --cluster_sizes fine->coarse.
  - recursive_spectral: top-down recursive spectral splits.
  - recursive_dmon: top-down recursive local DMoN-style modularity splits.

For recursive methods, hierarchy is controlled by:
  --recursive_factor F
  --recursive_levels L

This creates coarse-to-fine partitions with target cluster counts roughly:
  level_1: F
  level_2: F^2
  ...
  level_L: F^L

Each parent cluster is split into up to F child clusters. Small parents with fewer
than F nodes are split into singleton-like clusters as needed.

Main output:
  <output_dir>/<output_prefix>_labels.json

with labels indexed by global item_id:
  labels_json["coarse_to_fine_item_labels"]["level_1"] -> coarsest labels
  labels_json["coarse_to_fine_item_labels"]["level_2"] -> next finer labels

For RQ-VAE prefix alignment, use coarse_to_fine_item_labels in natural order:
  prefix-1       -> level_1
  prefix-1+2     -> level_2
  ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv

from graph_aware_sid.graph_utils import InteractionGraphBuilder, load_train_item_mask


# -----------------------------------------------------------------------------
# Graph loading / filtering
# -----------------------------------------------------------------------------


def build_graph_from_builder(
    embeddings_path: str,
    interactions_path: str | None,
    graph_builder_dump_path: str | None,
    graph_type: str,
    graph_building_kwargs: Dict[str, Any],
    make_undirected: bool,
    skip_n_last: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, np.ndarray, np.ndarray, Any]:
    raw_embeddings = np.load(embeddings_path)
    if raw_embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [num_items, dim], got {raw_embeddings.shape}")

    num_items, _ = raw_embeddings.shape
    x0 = torch.as_tensor(raw_embeddings, dtype=torch.float32)

    if interactions_path is not None:
        num_items, dim = x0.shape
        train_item_ids = load_train_item_mask(
            interactions_path,
            num_items,
            skip_n_last=skip_n_last,
        )
        builder = InteractionGraphBuilder(
            interactions_path=interactions_path,
            graph_type=graph_type,
            graph_building_kwargs=graph_building_kwargs,
            skip_n_last=skip_n_last,
            make_undirected=make_undirected,
            add_identity_features=False,
            all_item_ids=train_item_ids.tolist(),
        )
    else:
        raise ValueError("One of --interactions_path or --graph_builder_dump_path is required")

    pyg_data = builder.pyg_data
    node_map = builder.node_map.sort("node_id")
    item_ids = node_map["item_id"].to_numpy()

    if len(item_ids) == 0:
        raise ValueError("Graph has no nodes after preprocessing")

    x_graph = torch.as_tensor(raw_embeddings[item_ids], dtype=torch.float32)
    edge_index = pyg_data.edge_index.long()
    edge_weight = getattr(pyg_data, "edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.float()

    return x_graph, edge_index, edge_weight, item_ids, raw_embeddings, builder


def drop_zero_degree_nodes(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    item_ids: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, np.ndarray, np.ndarray]:
    num_nodes = x.size(0)
    deg = torch.zeros(num_nodes, dtype=torch.float32)

    if edge_index.numel() > 0:
        w = torch.ones(edge_index.size(1), dtype=torch.float32) if edge_weight is None else edge_weight.cpu().float()
        deg.scatter_add_(0, edge_index[0].cpu(), w)
        deg.scatter_add_(0, edge_index[1].cpu(), w)

    keep = deg > 0
    kept_old = keep.nonzero(as_tuple=False).view(-1)
    removed_item_ids = item_ids[(~keep).numpy()]

    if kept_old.numel() == num_nodes:
        return x, edge_index, edge_weight, item_ids, removed_item_ids

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
    old_to_new[kept_old] = torch.arange(kept_old.numel(), dtype=torch.long)

    edge_keep = keep[edge_index[0].cpu()] & keep[edge_index[1].cpu()]
    edge_index_f = old_to_new[edge_index[:, edge_keep].cpu()]
    edge_weight_f = None if edge_weight is None else edge_weight.cpu()[edge_keep]

    return x[kept_old], edge_index_f, edge_weight_f, item_ids[kept_old.numpy()], removed_item_ids


def edge_index_to_sparse_adj(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    val = (
        np.ones(row.shape[0], dtype=np.float64) if edge_weight is None else edge_weight.cpu().numpy().astype(np.float64)
    )

    A = sp.coo_matrix((val, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A = 0.5 * (A + A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


# -----------------------------------------------------------------------------
# Shared output helpers
# -----------------------------------------------------------------------------


def project_labels_back_to_item_space(
    graph_labels: Sequence[np.ndarray],
    item_ids: np.ndarray,
    num_items: int,
) -> List[np.ndarray]:
    full = []
    for labels in graph_labels:
        arr = np.full((num_items,), -1, dtype=np.int64)
        arr[item_ids] = labels.astype(np.int64)
        full.append(arr)
    return full


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    out = np.full_like(labels, -1)
    vals = [v for v in np.unique(labels).tolist() if v >= 0]
    for new_id, old_id in enumerate(vals):
        out[labels == old_id] = new_id
    return out


def save_outputs(
    args: argparse.Namespace,
    graph_labels_coarse_to_fine: Sequence[np.ndarray],
    item_ids: np.ndarray,
    raw_embeddings: np.ndarray,
    edge_index: torch.Tensor,
    metadata_extra: Dict[str, Any] | None = None,
    history: List[Dict[str, float]] | None = None,
    artifacts: Dict[str, np.ndarray] | None = None,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    num_items = raw_embeddings.shape[0]
    item_labels_coarse_to_fine = project_labels_back_to_item_space(
        graph_labels=graph_labels_coarse_to_fine,
        item_ids=item_ids,
        num_items=num_items,
    )

    labels_json = {
        "coarse_to_fine_item_labels": {
            f"level_{i}": labels.tolist() for i, labels in enumerate(item_labels_coarse_to_fine, start=1)
        },
        "fine_to_coarse_item_labels": {
            f"level_{i}": labels.tolist() for i, labels in enumerate(reversed(item_labels_coarse_to_fine), start=1)
        },
        "coarse_to_fine_graph_labels": {
            f"level_{i}": labels.astype(np.int64).tolist()
            for i, labels in enumerate(graph_labels_coarse_to_fine, start=1)
        },
    }

    labels_path = output_dir / f"{prefix}_labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_json, f)

    node_map_path = output_dir / f"{prefix}_node_map.csv"
    pd.DataFrame(
        {
            "node_id": np.arange(len(item_ids), dtype=np.int64),
            "item_id": item_ids.astype(np.int64),
        }
    ).to_csv(node_map_path, index=False)

    metadata = {
        "method": args.method,
        "embeddings_path": args.embeddings_path,
        "interactions_path": args.interactions_path,
        "graph_builder_dump_path": args.graph_builder_dump_path,
        "graph_type": args.graph_type,
        "graph_building_kwargs": json.loads(args.graph_building_kwargs),
        "num_items": int(num_items),
        "num_graph_nodes_used": int(len(item_ids)),
        "num_edges_used": int(edge_index.size(1)),
        "make_undirected": bool(not args.directed),
        "skip_n_last": args.skip_n_last,
        "drop_zero_degree_nodes": bool(args.drop_zero_degree_nodes),
        "graph_source": "graph_utils.InteractionGraphBuilder",
    }
    if args.method.startswith("recursive"):
        metadata.update(
            {
                "recursive_factor": int(args.recursive_factor),
                "recursive_levels": int(args.recursive_levels),
                "cluster_counts_coarse_to_fine": [int(len(np.unique(x[x >= 0]))) for x in graph_labels_coarse_to_fine],
            }
        )
    else:
        metadata.update(
            {
                "cluster_sizes_fine_to_coarse": args.cluster_sizes,
                "cluster_sizes_coarse_to_fine": list(reversed(args.cluster_sizes or [])),
            }
        )
    if metadata_extra:
        metadata.update(metadata_extra)

    metadata_path = output_dir / f"{prefix}_meta.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if history is not None:
        history_path = output_dir / f"{prefix}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    else:
        history_path = None

    if artifacts is None:
        artifacts = {}
    artifacts = dict(artifacts)
    artifacts["item_ids_graph_order"] = item_ids.astype(np.int64)
    for i, labels in enumerate(graph_labels_coarse_to_fine, start=1):
        artifacts[f"coarse_to_fine_level{i}_graph_labels"] = labels.astype(np.int64)
    np.savez_compressed(output_dir / f"{prefix}_artifacts.npz", **artifacts)

    print(f"Saved labels to:   {labels_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved node map to: {node_map_path}")
    print(f"Saved artifacts to:{output_dir / f'{prefix}_artifacts.npz'}")
    if history_path is not None:
        print(f"Saved history to:  {history_path}")
    print("\nFor RVQ prefix alignment, consume labels_json['coarse_to_fine_item_labels'].")
    print("  level_1 -> coarsest supervision for prefix-1")
    print("  level_2 -> finer supervision for prefix-2, etc.")


# -----------------------------------------------------------------------------
# Spectral recursive clustering
# -----------------------------------------------------------------------------


def spectral_cluster_subset(A: sp.csr_matrix, subset: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = int(len(subset))
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if k <= 1:
        return np.zeros((n,), dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)

    A_sub = A[subset][:, subset].astype(np.float64).tocsr()
    if A_sub.nnz == 0:
        # No internal structure: deterministic-ish split by semantic order inside subset.
        return np.arange(n, dtype=np.int64) % k

    deg = np.asarray(A_sub.sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(deg)
    L = sp.eye(n, format="csr") - sp.diags(inv_sqrt) @ A_sub @ sp.diags(inv_sqrt)

    eig_k = min(k, n - 1)
    try:
        evals, evecs = eigsh(L, k=eig_k, which="SM")
        order = np.argsort(evals)
        Z = evecs[:, order]
        if Z.shape[1] > 1:
            Z = Z[:, 1:]
        if Z.shape[1] == 0:
            Z = evecs[:, order]
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    except Exception as e:
        print(f"[warn] eigsh failed for subset size={n}, k={k}: {e}. Falling back to adjacency rows.")
        Z = A_sub.toarray()

    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)
    return labels.astype(np.int64)


# -----------------------------------------------------------------------------
# Recursive neural local splits: DMoN
# -----------------------------------------------------------------------------


class LocalAssignNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, k: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k),
        )

    def forward(self, x):
        return self.net(x)


def dense_dmon_local_loss(
    adj: torch.Tensor,
    logits: torch.Tensor,
    balance_weight: float = 1.0,
    entropy_weight: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    s = F.softmax(logits, dim=-1)  # [N, K]
    deg = adj.sum(dim=1)
    m2 = deg.sum().clamp_min(1e-12)
    expected = torch.outer(deg, deg) / m2
    modularity_matrix = adj - expected
    modularity = torch.trace(s.t() @ modularity_matrix @ s) / m2

    # Simple collapse/balance regularizer. Keeps clusters from all collapsing.
    p = s.mean(dim=0)
    k = s.size(1)
    balance = ((p - 1.0 / k) ** 2).sum()

    # Positive entropy term encourages sharper assignments when minimized.
    entropy = -(s * torch.log(s.clamp_min(1e-12))).sum(dim=1).mean()

    loss = -modularity + balance_weight * balance + entropy_weight * entropy
    metrics = {
        "modularity": modularity.detach(),
        "balance": balance.detach(),
        "entropy": entropy.detach(),
    }
    return loss, metrics


def neural_cluster_subset(
    method: str,
    x_all: torch.Tensor,
    A: sp.csr_matrix,
    subset: np.ndarray,
    k: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    n = int(len(subset))
    if n == 0:
        return np.empty((0,), dtype=np.int64), []
    if k <= 1:
        return np.zeros((n,), dtype=np.int64), []
    if n <= k:
        return np.arange(n, dtype=np.int64), []
    if n > args.max_dense_nodes:
        raise ValueError(
            f"Recursive {method} local subset has {n} nodes, above --max_dense_nodes={args.max_dense_nodes}."
        )

    A_sub_np = A[subset][:, subset].astype(np.float32).toarray()
    x_sub = x_all[torch.as_tensor(subset, dtype=torch.long)].to(device)
    adj = torch.as_tensor(A_sub_np, dtype=torch.float32, device=device)

    model = LocalAssignNet(x_sub.size(1), args.assign_hidden_dim, k, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []

    for epoch in range(1, args.recursive_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_sub)
        if method == "recursive_dmon":
            loss, metrics = dense_dmon_local_loss(
                adj=adj,
                logits=logits,
                balance_weight=args.dmon_balance_weight,
                entropy_weight=args.dmon_entropy_weight,
            )
        else:
            raise ValueError(f"Unknown neural recursive method: {method}")

        if torch.isnan(loss):
            raise ValueError(f"NaN loss in {method} local split")
        loss.backward()
        if args.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if epoch == 1 or epoch == args.recursive_epochs:
            row = {"epoch": float(epoch), "loss": float(loss.item()), "subset_size": float(n), "k": float(k)}
            row.update({name: float(value.item()) for name, value in metrics.items()})
            history.append(row)

    model.eval()
    with torch.no_grad():
        labels = model(x_sub).argmax(dim=-1).cpu().numpy().astype(np.int64)
    return labels, history


# -----------------------------------------------------------------------------
# Generic recursive top-down driver
# -----------------------------------------------------------------------------


def recursive_hierarchy(
    method: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    factor: int,
    levels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    if factor < 2:
        raise ValueError("--recursive_factor must be >= 2")
    if levels < 1:
        raise ValueError("--recursive_levels must be >= 1")

    n = x.size(0)
    A = edge_index_to_sparse_adj(edge_index, edge_weight, n)

    parent_labels = np.zeros(n, dtype=np.int64)
    hierarchy: List[np.ndarray] = []
    history: List[Dict[str, float]] = []

    rng_seed = int(args.seed)

    for level in range(1, levels + 1):
        new_labels = np.full(n, -1, dtype=np.int64)
        next_cluster_id = 0
        parents = [p for p in np.unique(parent_labels).tolist() if p >= 0]

        for parent in parents:
            subset = np.where(parent_labels == parent)[0].astype(np.int64)
            k = min(factor, len(subset))

            if method == "recursive_spectral":
                local = spectral_cluster_subset(A, subset, k=k, seed=rng_seed + level * 1009 + int(parent))
                local_history = []
            else:
                local, local_history = neural_cluster_subset(
                    method=method,
                    x_all=x,
                    A=A,
                    subset=subset,
                    k=k,
                    args=args,
                    device=device,
                )

            for local_id in sorted(np.unique(local).tolist()):
                mask = local == local_id
                new_labels[subset[mask]] = next_cluster_id
                next_cluster_id += 1

            for row in local_history:
                row = dict(row)
                row["level"] = float(level)
                row["parent"] = float(parent)
                history.append(row)

        new_labels = relabel_compact(new_labels)
        hierarchy.append(new_labels)
        parent_labels = new_labels
        print(f"recursive level {level}: clusters={len(np.unique(new_labels[new_labels >= 0]))}")

    return hierarchy, history


# -----------------------------------------------------------------------------
# Stacked DMoN baseline
# -----------------------------------------------------------------------------


class SparseEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = float(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm2(x)
        return x


class HierarchicalDMoN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        cluster_sizes: Sequence[int],
        assign_hidden_dim: int,
        dropout: float,
        balance_weight: float,
        entropy_weight: float,
    ):
        super().__init__()
        self.cluster_sizes = list(cluster_sizes)
        self.balance_weight = float(balance_weight)
        self.entropy_weight = float(entropy_weight)
        self.encoder = SparseEncoder(in_dim, hidden_dim, embed_dim, dropout)
        self.assigners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, assign_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(assign_hidden_dim, k),
                )
                for k in self.cluster_sizes
            ]
        )
        self.refiners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in self.cluster_sizes
            ]
        )

    @staticmethod
    def to_dense_adj(edge_index, num_nodes, edge_weight, device):
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        if edge_index.numel() > 0:
            if edge_weight is None:
                adj[edge_index[0], edge_index[1]] = 1.0
            else:
                adj[edge_index[0], edge_index[1]] = edge_weight.float()
        adj = 0.5 * (adj + adj.t())
        adj.fill_diagonal_(0.0)
        return adj

    @staticmethod
    def dmon_pool(
        x: torch.Tensor, adj: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = F.softmax(logits, dim=-1)  # [N, K]
        cluster_mass = s.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)  # [K, 1]
        x_next = (s.t() @ x) / cluster_mass
        adj_next = s.t() @ adj @ s
        adj_next = 0.5 * (adj_next + adj_next.t())
        adj_next.fill_diagonal_(0.0)
        return x_next, adj_next, s

    def forward(self, x, edge_index, edge_weight=None):
        device = x.device
        z0 = self.encoder(x, edge_index, edge_weight)
        x_curr = z0
        adj_curr = self.to_dense_adj(edge_index, x.size(0), edge_weight, device)

        levels = []
        total_loss = x.new_tensor(0.0)

        for i, (assigner, refiner, k) in enumerate(zip(self.assigners, self.refiners, self.cluster_sizes), start=1):
            if k > x_curr.size(0):
                raise ValueError(f"Level {i}: k={k} > nodes={x_curr.size(0)}")

            logits = assigner(x_curr)
            level_loss, metrics = dense_dmon_local_loss(
                adj=adj_curr,
                logits=logits,
                balance_weight=self.balance_weight,
                entropy_weight=self.entropy_weight,
            )
            x_next, adj_next, s_probs = self.dmon_pool(x_curr, adj_curr, logits)
            x_next = refiner(x_next)

            total_loss = total_loss + level_loss
            levels.append(
                {
                    "level": i,
                    "s_logits": logits,
                    "s_probs": s_probs,
                    "loss": level_loss,
                    "modularity": metrics["modularity"],
                    "balance": metrics["balance"],
                    "entropy": metrics["entropy"],
                }
            )

            x_curr = x_next
            adj_curr = adj_next

        return {"encoded_nodes": z0, "levels": levels, "loss": total_loss}


def compose_assignments_to_original(level_probs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    projected = []
    running = None
    for probs in level_probs:
        running = probs if running is None else running @ probs
        projected.append(running)
    return projected


@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    epochs: int
    print_every: int
    grad_clip: float | None


def train_stacked_dmon(model, x, edge_index, edge_weight, cfg: TrainConfig) -> Dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index, edge_weight)
        loss = out["loss"]
        if torch.isnan(loss):
            raise ValueError("NaN loss in stacked_dmon")
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        row = {"epoch": float(epoch), "loss": float(loss.item())}
        for lvl in out["levels"]:
            row[f"level{lvl['level']}_loss"] = float(lvl["loss"].item())
            row[f"level{lvl['level']}_modularity"] = float(lvl["modularity"].item())
            row[f"level{lvl['level']}_balance"] = float(lvl["balance"].item())
            row[f"level{lvl['level']}_entropy"] = float(lvl["entropy"].item())
        history.append(row)

        if epoch == 1 or epoch % cfg.print_every == 0 or epoch == cfg.epochs:
            print(f"epoch={epoch:04d} loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_weight)
    out["history"] = history
    return out


def run_stacked_dmon(args, x_graph, edge_index, edge_weight, device):
    if not args.cluster_sizes:
        raise ValueError("--cluster_sizes is required for --method stacked_dmon")
    if x_graph.size(0) > args.max_dense_nodes:
        raise ValueError(
            f"Graph has {x_graph.size(0)} nodes; stacked_dmon uses dense adjacency. "
            f"Increase --max_dense_nodes if intentional."
        )

    model = HierarchicalDMoN(
        in_dim=x_graph.size(1),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        cluster_sizes=args.cluster_sizes,
        assign_hidden_dim=args.assign_hidden_dim,
        dropout=args.dropout,
        balance_weight=args.dmon_balance_weight,
        entropy_weight=args.dmon_entropy_weight,
    ).to(device)
    out = train_stacked_dmon(
        model=model,
        x=x_graph.to(device),
        edge_index=edge_index.to(device),
        edge_weight=None if edge_weight is None else edge_weight.to(device),
        cfg=TrainConfig(args.lr, args.weight_decay, args.epochs, args.print_every, args.grad_clip),
    )

    level_probs = [lvl["s_probs"].detach().cpu() for lvl in out["levels"]]
    projected = compose_assignments_to_original(level_probs)
    fine_to_coarse = [p.argmax(dim=-1).numpy().astype(np.int64) for p in projected]
    coarse_to_fine = list(reversed(fine_to_coarse))

    artifacts = {"encoded_nodes": out["encoded_nodes"].detach().cpu().numpy()}
    for i, lvl in enumerate(out["levels"], start=1):
        artifacts[f"level{i}_s_probs"] = lvl["s_probs"].detach().cpu().numpy()
        artifacts[f"level{i}_s_logits"] = lvl["s_logits"].detach().cpu().numpy()
    return coarse_to_fine, out["history"], artifacts


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical graph clustering for graph-aware SID")

    parser.add_argument(
        "--method",
        type=str,
        default="stacked_dmon",
        choices=["stacked_dmon", "recursive_spectral", "recursive_dmon"],
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--interactions_path", type=str, default=None)
    parser.add_argument("--graph_builder_dump_path", type=str, default=None)

    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--graph_type", type=str, default="adjacent_cooc", choices=["adjacent_cooc", "windowed", "rw"])
    parser.add_argument("--graph_building_kwargs", type=str, default="{}")
    parser.add_argument("--skip_n_last", type=int, default=2)
    parser.add_argument("--directed", action="store_true", help="Pass make_undirected=False to InteractionGraphBuilder")
    parser.add_argument("--drop_zero_degree_nodes", action="store_true")

    # Stacked DMoN explicit hierarchy.
    parser.add_argument(
        "--cluster_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Only for stacked_dmon. Fine-to-coarse sizes, e.g. --cluster_sizes 128 32",
    )

    # Recursive hierarchy.
    parser.add_argument("--recursive_factor", type=int, default=4)
    parser.add_argument("--recursive_levels", type=int, default=2)
    parser.add_argument("--recursive_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dmon_balance_weight", type=float, default=1.0)
    parser.add_argument("--dmon_entropy_weight", type=float, default=0.01)

    # Neural settings.
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--assign_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_dense_nodes", type=int, default=8000)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="graph_hierarchy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_building_kwargs = json.loads(args.graph_building_kwargs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x_graph, edge_index, edge_weight, item_ids, raw_embeddings, builder = build_graph_from_builder(
        embeddings_path=args.embeddings_path,
        interactions_path=args.interactions_path,
        graph_builder_dump_path=args.graph_builder_dump_path,
        graph_type=args.graph_type,
        graph_building_kwargs=graph_building_kwargs,
        make_undirected=not args.directed,
        skip_n_last=args.skip_n_last,
    )

    print(f"Method: {args.method}")
    print(f"Loaded graph nodes used: {x_graph.size(0)}, edges used: {edge_index.size(1)}, emb dim: {x_graph.size(1)}")

    if args.method == "stacked_dmon":
        labels_c2f, history, artifacts = run_stacked_dmon(args, x_graph, edge_index, edge_weight, device)
    else:
        labels_c2f, history = recursive_hierarchy(
            method=args.method,
            x=x_graph,
            edge_index=edge_index,
            edge_weight=edge_weight,
            factor=args.recursive_factor,
            levels=args.recursive_levels,
            args=args,
            device=device,
        )
        artifacts = {}

    save_outputs(
        args=args,
        graph_labels_coarse_to_fine=labels_c2f,
        item_ids=item_ids,
        raw_embeddings=raw_embeddings,
        edge_index=edge_index,
        history=history,
        artifacts=artifacts,
    )


if __name__ == "__main__":
    main()
