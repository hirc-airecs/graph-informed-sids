#!/usr/bin/env python3
"""
Adaptation of MMGRec-style GNN -> RQ-VAE SID generation for JSON interactions + .npy item embeddings.

Input dataset format:
  interactions_path: JSON dict {user_id: [item_id, item_id, ...]}
  embeddings_path:   .npy float array [num_items, emb_dim], row index == item_id

Pipeline:
  1) Build a user-item bipartite graph from training interactions.
  2) Train a GraphSAGE encoder with BPR loss over (user, positive item, negative item).
  3) Extract item representations from the trained GNN.
  4) Train a residual vector quantizer / tiny RQ-VAE on item representations.
  5) Save item_id -> SID token list JSON, plus numeric SID matrix and stats.

This mirrors the high-level MMGRec tgt_input.py idea, but removes hard-coded user/item counts,
feature files, triple_*.para files, and tgt_mtx.npy-specific assumptions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import SAGEConv

DEFAULT_PREFIXES = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>", "<g_{}>", "<h_{}>"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_interactions(path: str, skip_n_last: int = 0) -> Tuple[Dict[int, List[int]], int, int]:
    with open(path, "r") as f:
        raw = json.load(f)

    user_hist: Dict[int, List[int]] = {}
    max_user = -1
    max_item = -1

    for u_raw, items_raw in raw.items():
        u = int(u_raw)
        items = [int(x) for x in items_raw]
        if skip_n_last > 0:
            items = items[:-skip_n_last]
        if len(items) == 0:
            continue
        user_hist[u] = items
        max_user = max(max_user, u)
        max_item = max(max_item, max(items))

    if not user_hist:
        raise ValueError("No interactions left after applying --skip_n_last")

    return user_hist, max_user + 1, max_item + 1


def build_train_pairs(user_hist: Dict[int, List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    users, items = [], []
    for u, hist in user_hist.items():
        for it in hist:
            users.append(u)
            items.append(it)
    return np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)


class BPRTripletDataset(Dataset):
    def __init__(self, users: np.ndarray, pos_items: np.ndarray, user_pos_sets: Dict[int, set], num_items: int, seed: int = 42):
        self.users = users
        self.pos_items = pos_items
        self.user_pos_sets = user_pos_sets
        self.num_items = int(num_items)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_negative(self, u: int) -> int:
        positives = self.user_pos_sets[u]
        # Simple rejection sampler. For extremely dense users, fallback to candidate list.
        for _ in range(100):
            j = self.rng.randrange(self.num_items)
            if j not in positives:
                return j
        candidates = [j for j in range(self.num_items) if j not in positives]
        if not candidates:
            return self.rng.randrange(self.num_items)
        return self.rng.choice(candidates)

    def __getitem__(self, idx: int):
        u = int(self.users[idx])
        i = int(self.pos_items[idx])
        j = self._sample_negative(u)
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)


class GraphSAGERepr(nn.Module):
    def __init__(self, num_users: int, item_feat_dim: int, gnn_hidden_dim: int, gnn_out_dim: int, dropout: float):
        super().__init__()
        self.num_users = int(num_users)
        self.dropout = float(dropout)

        self.user = nn.Parameter(torch.empty(num_users, gnn_hidden_dim))
        nn.init.xavier_uniform_(self.user)

        self.item_proj = nn.Linear(item_feat_dim, gnn_hidden_dim, bias=False)
        self.conv1 = SAGEConv(gnn_hidden_dim, gnn_hidden_dim)
        self.conv2 = SAGEConv(gnn_hidden_dim, gnn_out_dim)

    def forward(self, item_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x0 = torch.cat([self.user, self.item_proj(item_features)], dim=0)
        x = self.conv1(x0, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class RQVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, codebook_sizes: List[int]):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.codebook_sizes = [int(x) for x in codebook_sizes]

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.codebooks = nn.ModuleList([nn.Embedding(k, self.latent_dim) for k in self.codebook_sizes])
        for cb, k in zip(self.codebooks, self.codebook_sizes):
            cb.weight.data.uniform_(-1.0 / k, 1.0 / k)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    @staticmethod
    def _nearest(residual: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # residual: [B, D], emb: [K, D]
        d = torch.sum(residual ** 2, dim=1, keepdim=True) + torch.sum(emb ** 2, dim=1).view(1, -1) - 2.0 * residual @ emb.t()
        return torch.argmin(d, dim=1)

    def encode_quantize(self, x: torch.Tensor):
        z_e = self.encoder(x)
        residual = z_e
        z_q_sum = torch.zeros_like(z_e)
        all_indices = []
        all_zq = []

        for cb in self.codebooks:
            idx = self._nearest(residual, cb.weight)
            z_q = cb(idx)
            all_indices.append(idx)
            all_zq.append(z_q)
            residual = residual - z_q
            z_q_sum = z_q_sum + z_q

        z_st = z_e + (z_q_sum - z_e).detach()
        indices = torch.stack(all_indices, dim=1)
        return z_e, all_zq, z_st, indices

    def forward(self, x: torch.Tensor):
        z_e, all_zq, z_st, indices = self.encode_quantize(x)
        x_hat = self.decoder(z_st)
        return x_hat, z_e, all_zq, indices


def build_edge_index(users: np.ndarray, items: np.ndarray, num_users: int, device: torch.device) -> torch.Tensor:
    src_u = users
    dst_i = items + num_users
    edges = np.stack([np.concatenate([src_u, dst_i]), np.concatenate([dst_i, src_u])], axis=0)
    return torch.as_tensor(edges, dtype=torch.long, device=device).contiguous()


def train_gnn(args, item_features: np.ndarray, user_hist: Dict[int, List[int]], num_users: int, num_items: int, device: torch.device) -> np.ndarray:
    users, pos_items = build_train_pairs(user_hist)
    user_pos_sets = {u: set(hist) for u, hist in user_hist.items()}
    ds = BPRTripletDataset(users, pos_items, user_pos_sets, num_items, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.gnn_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    edge_index = build_edge_index(users, pos_items, num_users, device)
    x_item = torch.as_tensor(item_features, dtype=torch.float32, device=device)

    model = GraphSAGERepr(
        num_users=num_users,
        item_feat_dim=item_features.shape[1],
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_out_dim=args.gnn_out_dim,
        dropout=args.gnn_dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)

    for epoch in range(1, args.gnn_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch_u, batch_i, batch_j in loader:
            batch_u = batch_u.to(device, non_blocking=True)
            batch_i = batch_i.to(device, non_blocking=True) + num_users
            batch_j = batch_j.to(device, non_blocking=True) + num_users

            out = model(x_item, edge_index)
            emb_u = out[batch_u]
            emb_i = out[batch_i]
            emb_j = out[batch_j]
            score_pos = torch.sum(emb_u * emb_i, dim=1)
            score_neg = torch.sum(emb_u * emb_j, dim=1)
            loss = -torch.mean(F.logsigmoid(score_pos - score_neg))

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = batch_u.size(0)
            total += float(loss.item()) * bs
            n += bs
        print(f"[GNN] epoch={epoch:04d} bpr_loss={total / max(n, 1):.6f}")

    model.eval()
    with torch.no_grad():
        out = model(x_item, edge_index)
        item_repr = out[num_users:num_users + num_items].detach().cpu().numpy()
    return item_repr


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, idx):
        return self.x[idx]


def train_rqvae(args, item_repr: np.ndarray, device: torch.device) -> Tuple[np.ndarray, RQVAE]:
    ds = ArrayDataset(item_repr)
    loader = DataLoader(ds, batch_size=args.rq_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = RQVAE(
        input_dim=item_repr.shape[1],
        hidden_dim=args.rq_hidden_dim,
        latent_dim=args.rq_latent_dim,
        codebook_sizes=args.codebook_sizes,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.rq_lr, weight_decay=args.rq_weight_decay)
    mse = nn.MSELoss()

    for epoch in range(1, args.rq_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            x_hat, z_e, all_zq, _ = model(x)
            recon = mse(x_hat, x)
            embedding_loss = sum(mse(z_e.detach(), zq) for zq in all_zq)
            commitment_loss = sum(mse(z_e, zq.detach()) for zq in all_zq)
            loss = recon + args.embedding_weight * embedding_loss + args.commitment_weight * commitment_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0)
            total += float(loss.item()) * bs
            n += bs
        print(f"[RQ] epoch={epoch:04d} loss={total / max(n, 1):.6f}")

    model.eval()
    ids = []
    with torch.no_grad():
        for x in DataLoader(ds, batch_size=args.rq_eval_batch_size, shuffle=False):
            x = x.to(device)
            _, _, _, indices = model(x)
            ids.append(indices.cpu().numpy())
    return np.concatenate(ids, axis=0).astype(np.int64), model


def append_collision_token(sids: np.ndarray, popularity: np.ndarray | None = None) -> np.ndarray:
    path_to_items = defaultdict(list)
    for item_id, row in enumerate(sids):
        path_to_items[tuple(row.tolist())].append(item_id)

    leaf = np.zeros((sids.shape[0], 1), dtype=np.int64)
    for _, item_ids in path_to_items.items():
        if popularity is not None:
            item_ids = sorted(item_ids, key=lambda i: (-float(popularity[i]), i))
        else:
            item_ids = sorted(item_ids)
        for rank, item_id in enumerate(item_ids):
            leaf[item_id, 0] = rank
    return np.concatenate([sids, leaf], axis=1)


def make_mapping(sids: np.ndarray, prefixes: List[str]) -> Dict[str, List[str]]:
    if len(prefixes) < sids.shape[1]:
        raise ValueError(f"Need at least {sids.shape[1]} prefixes, got {len(prefixes)}")
    out = {}
    for item_id, row in enumerate(sids):
        out[str(item_id)] = [prefixes[level].format(int(cid)) for level, cid in enumerate(row)]
    return out


def compute_popularity(user_hist: Dict[int, List[int]], num_items: int) -> np.ndarray:
    pop = np.zeros(num_items, dtype=np.int64)
    for hist in user_hist.values():
        for i in hist:
            if 0 <= i < num_items:
                pop[i] += 1
    return pop


def save_outputs(args, item_repr: np.ndarray, numeric_sids: np.ndarray, mapping: Dict[str, List[str]], popularity: np.ndarray | None):
    os.makedirs(args.output_dir, exist_ok=True)
    mapping_path = os.path.join(args.output_dir, args.output_name)
    npy_path = os.path.join(args.output_dir, args.output_name.replace(".json", ".npy"))
    repr_path = os.path.join(args.output_dir, "gnn_item_representation.npy")
    stats_path = mapping_path + ".stats.json"

    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    np.save(npy_path, numeric_sids.astype(np.int64))
    np.save(repr_path, item_repr.astype(np.float32))

    paths = [tuple(x) for x in numeric_sids.tolist()]
    c = Counter(paths)
    stats = {
        "num_items": int(numeric_sids.shape[0]),
        "sid_depth": int(numeric_sids.shape[1]),
        "num_unique_sids": int(len(c)),
        "collision_rate": float((numeric_sids.shape[0] - len(c)) / max(numeric_sids.shape[0], 1)),
        "max_collision_group_size": int(max(c.values()) if c else 0),
        "codebook_sizes": [int(x) for x in args.codebook_sizes],
        "append_collision_token": bool(args.append_collision_token),
        "mapping_path": mapping_path,
        "numeric_sid_path": npy_path,
        "representation_path": repr_path,
    }
    if popularity is not None:
        stats["popularity_nonzero_items"] = int((popularity > 0).sum())
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved SID mapping:       {mapping_path}")
    print(f"Saved numeric SID matrix:{npy_path}")
    print(f"Saved GNN item repr:     {repr_path}")
    print(f"Saved stats:             {stats_path}")
    print(json.dumps(stats, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="MMGRec-style GNN + RQ-VAE SID generation for JSON interactions and .npy item embeddings")
    p.add_argument("--interactions_path", required=True, help="JSON {user_id: [item_id, ...]}")
    p.add_argument("--embeddings_path", required=True, help=".npy item features [num_items, dim]")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", default="gnn_sid_mapping.json")
    p.add_argument("--skip_n_last", type=int, default=2, help="Use train prefix only; set 0 to use all interactions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--gnn_epochs", type=int, default=60)
    p.add_argument("--gnn_batch_size", type=int, default=3000)
    p.add_argument("--gnn_hidden_dim", type=int, default=128)
    p.add_argument("--gnn_out_dim", type=int, default=64)
    p.add_argument("--gnn_dropout", type=float, default=0.5)
    p.add_argument("--gnn_lr", type=float, default=1e-3)
    p.add_argument("--gnn_weight_decay", type=float, default=1e-6)

    p.add_argument("--rq_epochs", type=int, default=60)
    p.add_argument("--rq_batch_size", type=int, default=1024)
    p.add_argument("--rq_eval_batch_size", type=int, default=4096)
    p.add_argument("--rq_hidden_dim", type=int, default=16)
    p.add_argument("--rq_latent_dim", type=int, default=4)
    p.add_argument("--rq_lr", type=float, default=1e-3)
    p.add_argument("--rq_weight_decay", type=float, default=1e-4)
    p.add_argument("--codebook_sizes", type=int, nargs="+", default=[128, 128, 128])
    p.add_argument("--embedding_weight", type=float, default=1.0)
    p.add_argument("--commitment_weight", type=float, default=0.25)

    p.add_argument("--append_collision_token", action="store_true", help="Append MMGRec-like popularity rank token to make collisions separable")
    p.add_argument("--collision_sort", choices=["popularity", "item_id"], default="popularity")
    p.add_argument("--prefixes", nargs="*", default=None, help="Token templates, e.g. '<a_{}>' '<b_{}>' '<c_{}>' '<d_{}>'")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    item_features = np.load(args.embeddings_path)
    if item_features.ndim != 2:
        raise ValueError(f"Expected embeddings [num_items, dim], got {item_features.shape}")
    num_items = int(item_features.shape[0])

    user_hist, num_users, max_item_from_inter = load_interactions(args.interactions_path, skip_n_last=args.skip_n_last)
    if max_item_from_inter > num_items:
        raise ValueError(f"Interactions reference item_id up to {max_item_from_inter - 1}, but embeddings have only {num_items} rows")

    print(f"Users: {num_users} | Items: {num_items} | Item feature dim: {item_features.shape[1]}")
    print(f"Train interactions after skip_n_last={args.skip_n_last}: {sum(len(v) for v in user_hist.values())}")

    item_repr = train_gnn(args, item_features, user_hist, num_users, num_items, device)
    numeric_sids, _ = train_rqvae(args, item_repr, device)

    popularity = compute_popularity(user_hist, num_items)
    if args.append_collision_token:
        sort_pop = popularity if args.collision_sort == "popularity" else None
        numeric_sids = append_collision_token(numeric_sids, popularity=sort_pop)

    prefixes = args.prefixes if args.prefixes is not None else DEFAULT_PREFIXES
    mapping = make_mapping(numeric_sids, prefixes)
    save_outputs(args, item_repr, numeric_sids, mapping, popularity)


if __name__ == "__main__":
    main()
