import json
import logging
import os
import random
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.utils.data as data
from gensid.graph_aware_sid.graph_utils import (
    S2GRConv,
    compute_graph_spectral_vectors,
    InteractionGraphBuilder,
    load_graph_cluster_labels,
    load_train_item_mask,
)
from torch.utils.data._utils.collate import default_collate
from torch_geometric.nn import APPNP
from torch_geometric.utils import k_hop_subgraph


from torch_geometric.utils import subgraph


class EdgeWeightCollator:
    def __init__(self, pyg_data, num_anchors=1024):
        # Keep graph on CPU for worker processes.
        self.edge_index = pyg_data.edge_index.cpu()
        self.edge_weight = pyg_data.edge_weight.cpu()
        self.num_nodes = pyg_data.num_nodes
        self.num_anchors = num_anchors

    def __call__(self, batch):
        collated = default_collate(batch)

        # For EmbDataset / APPNPEmbDataset:
        # collated = (data, emb_idx, recon_tgt, maybe spectral)
        emb_idx = collated[1].long()

        ei, ew = subgraph(
            subset=emb_idx,
            edge_index=self.edge_index,
            edge_attr=self.edge_weight,
            relabel_nodes=True,
        )

        B = emb_idx.size(0)
        edge_w = torch.zeros(B, min(B, self.num_anchors), dtype=torch.float32)

        if ei.numel() > 0:
            mask = ei[1] < self.num_anchors
            ei = ei[:, mask]
            ew = ew[mask]
            edge_w[ei[0], ei[1]] = ew.float()

        return (*collated, edge_w)


class EmbDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        interactions_path=None,
        skip_n_last=2,
        eval_mode=True,
    ):
        self.data_path = data_path
        self.eval_mode = eval_mode

        full_embeddings = np.load(data_path)
        self.dim = full_embeddings.shape[-1]
        num_items = full_embeddings.shape[0]

        self.train_item_ids = np.arange(num_items, dtype=np.int64)

        if interactions_path is not None:
            self.train_item_ids = load_train_item_mask(
                interactions_path,
                num_items,
                skip_n_last=skip_n_last,
            )

        if self.eval_mode or interactions_path is None:
            self.embeddings = full_embeddings
        else:
            self.embeddings = full_embeddings[self.train_item_ids]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.as_tensor(emb, dtype=torch.float32)
        return tensor_emb, index, tensor_emb

    def __len__(self):
        return len(self.embeddings)


class APPNPEmbDataset(data.Dataset):
    """
    Offline graph-fused embedding dataset.

    Constructor stays simple:
        - interactions_path: file with columns [user_id, ts, item_id]
        - embeddings_path: .npy file with semantic embeddings [num_items, dim]

    Pipeline:
        1. Build co-occurrence graph with graph_utils, or reuse a provided one
        2. Convert to / use PyG graph
        3. Align embeddings to graph node order
        4. Run APPNP propagation once offline
        5. Scatter fused embeddings back to original item_id order
    Adds optional graph spectral vectors:
        self.graph_spectral_vectors : [num_items, spectral_dim]
    """

    def __init__(
        self,
        interactions_path,
        embeddings_path,
        alpha=0.1,
        num_prop=10,
        fused_cache_path=None,
        use_edge_weight=False,
        skip_n_last=2,
        graph_type: Literal["rw", "windowed", "adjacent_cooc"] = "adjacent_cooc",
        graph_building_kwargs=None,
        convolution_type: Literal["APPNP", "S2GR", "none"] = "APPNP",
        spectral_dim: int = 0,
        graph_builder=None,
        eval_mode: bool = False,
        cluster_labels_path: str | None = None,
    ):
        logging.info("create APPNP dataset")
        self.interactions_path = interactions_path
        self.embeddings_path = embeddings_path
        self.alpha = float(alpha)
        self.num_prop = int(num_prop)
        self.spectral_dim = int(spectral_dim)
        self.eval_mode = bool(eval_mode)
        self.pyg_data = None
        self.node_map = None
        self.graph_df = None
        self.graph_builder = None
        self.graph_cluster_labels = None

        if graph_building_kwargs is None:
            graph_building_kwargs = {}

        full_raw_embeddings = np.load(embeddings_path)
        x0 = torch.as_tensor(full_raw_embeddings, dtype=torch.float32)

        if x0.ndim != 2:
            raise ValueError(f"Expected embeddings of shape [N, D], got {tuple(x0.shape)}")

        self.num_items, self.dim = x0.shape

        self.train_item_ids = load_train_item_mask(
            interactions_path,
            self.num_items,
            skip_n_last=skip_n_last,
        )
        self.num_train_items = len(self.train_item_ids)

        self.graph_spectral_vectors = np.zeros(
            (self.num_train_items, max(self.spectral_dim, 0)),
            dtype=np.float32,
        )

        if fused_cache_path is not None and os.path.exists(fused_cache_path):
            logging.info(f"loading cache path from {fused_cache_path}")
            fused_train = np.load(fused_cache_path)

            expected_shape = (self.num_train_items, self.dim)
            if fused_train.shape != expected_shape:
                raise ValueError(
                    f"Cached fused embeddings shape {fused_train.shape} "
                    f"does not match expected train-only shape {expected_shape}"
                )

            if self.eval_mode:
                restored = full_raw_embeddings.copy()
                restored[self.train_item_ids] = fused_train
                self.embeddings = restored
                self.raw_embeddings = full_raw_embeddings
            else:
                self.embeddings = fused_train
                self.raw_embeddings = full_raw_embeddings[self.train_item_ids]
            return

        if graph_builder is None:
            graph_builder = InteractionGraphBuilder(
                interactions_path=interactions_path,
                graph_type=graph_type,
                graph_building_kwargs=graph_building_kwargs,
                skip_n_last=skip_n_last,
                make_undirected=True,
                add_identity_features=False,
                all_item_ids=self.train_item_ids.tolist(),
            )
        if cluster_labels_path is not None:
            self.graph_cluster_labels = load_graph_cluster_labels(
                cluster_labels_path,
                num_items=self.num_items,
            )
            self.graph_cluster_labels = torch.tensor(self.graph_cluster_labels).T

        self.graph_builder = graph_builder
        self.pyg_data = graph_builder.pyg_data
        self.node_map = graph_builder.node_map
        item_ids = self.node_map["item_id"].to_numpy()
        self.graph_df = graph_builder.graph_df

        if len(item_ids) == 0:
            if self.eval_mode:
                self.embeddings = full_raw_embeddings
                self.raw_embeddings = full_raw_embeddings
            else:
                self.embeddings = full_raw_embeddings[self.train_item_ids]
                self.raw_embeddings = self.embeddings
            return

        x_graph = x0[item_ids]

        edge_index = self.pyg_data.edge_index
        edge_weight = self.pyg_data.edge_weight if use_edge_weight else None

        if convolution_type == "APPNP":
            appnp = APPNP(K=self.num_prop, alpha=self.alpha, cached=False)
            appnp.eval()
        elif convolution_type == "S2GR":
            appnp = S2GRConv(K=self.num_prop, alpha=self.alpha)
            appnp.eval()
        elif convolution_type == "none":
            appnp = lambda x, *_: x
        else:
            raise NotImplementedError(f"{convolution_type} is not implemented")

        with torch.no_grad():
            x_fused_train = appnp(x_graph, edge_index, edge_weight)

        fused_train = x_fused_train.cpu().numpy()

        if self.eval_mode:
            restored = full_raw_embeddings.copy()
            restored[item_ids] = fused_train
            self.embeddings = restored
            self.raw_embeddings = full_raw_embeddings
        else:
            self.embeddings = fused_train
            self.raw_embeddings = full_raw_embeddings[item_ids]

        if self.spectral_dim > 0:
            assert not self.eval_mode
            spectral_train = compute_graph_spectral_vectors(
                edge_index=self.pyg_data.edge_index,
                edge_weight=self.pyg_data.edge_weight,
                num_nodes=self.pyg_data.num_nodes,
                k=self.spectral_dim,
            )

            self.graph_spectral_vectors = spectral_train.astype(np.float32)

        if fused_cache_path is not None:
            fused_dir_name = os.path.dirname(fused_cache_path)
            os.makedirs(fused_dir_name, exist_ok=True)
            np.save(fused_cache_path, fused_train)

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
        df = pl.DataFrame(df)
        return df.select(["user_id", "ts", "item_id"])

    def __getitem__(self, index):
        emb = self.embeddings[index]
        raw_emb = self.raw_embeddings[index]

        if self.spectral_dim > 0:
            spectral_vec = self.graph_spectral_vectors[index]
            return (
                torch.as_tensor(emb, dtype=torch.float32),
                index,
                torch.as_tensor(raw_emb, dtype=torch.float32),
                torch.as_tensor(spectral_vec, dtype=torch.float32),
            )
        elif self.graph_cluster_labels is not None:
            return (
                torch.as_tensor(emb, dtype=torch.float32),
                index,
                torch.as_tensor(raw_emb, dtype=torch.float32),
                self.graph_cluster_labels[index],
            )

        return (
            torch.as_tensor(emb, dtype=torch.float32),
            index,
            torch.as_tensor(raw_emb, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.embeddings)


class MixedFusionDepthDataset(data.Dataset):
    def __init__(
        self,
        interactions_path,
        embeddings_path,
        hops,
        probs=None,
        fused_cache_root=None,
        alpha=0.1,
        use_edge_weight=False,
        skip_n_last=2,
        graph_type="adjacent_cooc",
        graph_building_kwargs=None,
        convolution_type="APPNP",
        return_hop=False,
        seed=None,
    ):
        if not hops:
            raise ValueError("hops must be non-empty")

        if graph_building_kwargs is None:
            graph_building_kwargs = {}

        raw_hops = [int(h) for h in hops]

        # --- probs + dedup ---
        if probs is None:
            raw_probs = [1.0] * len(raw_hops)
        else:
            if len(probs) != len(raw_hops):
                raise ValueError("probs must match hops length")
            raw_probs = [float(p) for p in probs]

        hop_to_prob = {}
        for h, p in zip(raw_hops, raw_probs):
            hop_to_prob[h] = hop_to_prob.get(h, 0.0) + p

        self.hops = sorted(hop_to_prob.keys())
        self.probs = [hop_to_prob[h] for h in self.hops]

        total = sum(self.probs)
        if total <= 0:
            raise ValueError("sum(probs) must be > 0")
        self.probs = [p / total for p in self.probs]

        self.return_hop = return_hop
        self.rng = random.Random(seed)

        def get_cache_path(h):
            if fused_cache_root is None or h == 0:
                return None
            os.makedirs(fused_cache_root, exist_ok=True)
            return os.path.join(fused_cache_root, f"fused_depth{h}.npy")

        self.datasets = []

        # --- find first non-zero hop ---
        first_nonzero = next((h for h in self.hops if h > 0), None)

        shared_graph_builder = None

        # --- build anchor graph dataset ---
        if first_nonzero is not None:
            anchor_ds = APPNPEmbDataset(
                interactions_path=interactions_path,
                embeddings_path=embeddings_path,
                alpha=alpha,
                num_prop=first_nonzero,
                fused_cache_path=get_cache_path(first_nonzero),
                use_edge_weight=use_edge_weight,
                skip_n_last=skip_n_last,
                graph_type=graph_type,
                graph_building_kwargs=graph_building_kwargs,
                convolution_type=convolution_type,
                graph_builder=None,
                eval_mode=False,
            )
            shared_graph_builder = anchor_ds.graph_builder

        # --- build all datasets ---
        for h in self.hops:
            if h == 0:
                ds = EmbDataset(
                    embeddings_path,
                    interactions_path=interactions_path,
                    skip_n_last=skip_n_last,
                    eval_mode=False,
                )
            elif h == first_nonzero:
                ds = anchor_ds
            else:
                ds = APPNPEmbDataset(
                    interactions_path=interactions_path,
                    embeddings_path=embeddings_path,
                    alpha=alpha,
                    num_prop=h,
                    fused_cache_path=get_cache_path(h),
                    use_edge_weight=use_edge_weight,
                    skip_n_last=skip_n_last,
                    graph_type=graph_type,
                    graph_building_kwargs=graph_building_kwargs,
                    convolution_type=convolution_type,
                    graph_builder=shared_graph_builder,
                    eval_mode=False,
                )
            self.datasets.append(ds)

        # --- sanity ---
        lengths = [len(ds) for ds in self.datasets]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent dataset lengths: {lengths}")

        self.dim = self.datasets[0].dim

    def __len__(self):
        return len(self.datasets[0])

    def _sample_dataset_id(self):
        return self.rng.choices(range(len(self.datasets)), weights=self.probs, k=1)[0]

    def __getitem__(self, idx):
        ds_id = self._sample_dataset_id()
        hop = self.hops[ds_id]
        out = self.datasets[ds_id][idx]

        return out


class GNNFusionEmbDataset(EmbDataset):
    """
    PyG-based graph-aware extension of EmbDataset.

    Uses torch_geometric.utils.k_hop_subgraph inside __getitem__,
    so it works with a regular torch DataLoader.

    Returns:
        center_emb, center_item_id, graph_dict
    """

    def __init__(
        self,
        interactions_path: str,
        embeddings_path: str,
        num_hops: int = 1,
        max_neighbors: int | None = None,
        make_undirected: bool = True,
        skip_n_last=2,
        graph_type: Literal["rw", "windowed", "adjacent_cooc"] = "adjacent_cooc",
        graph_building_kwargs=None,
    ):
        super().__init__(embeddings_path)

        self.num_hops = int(num_hops)
        self.max_neighbors = max_neighbors

        graph_builder = InteractionGraphBuilder(
            interactions_path=interactions_path,
            graph_type=graph_type,
            graph_building_kwargs=graph_building_kwargs,
            skip_n_last=skip_n_last,
            make_undirected=make_undirected,
            add_identity_features=False,
            all_item_ids=self.train_item_ids.tolist(),
        )
        pyg_data, node_map = graph_builder.pyg_data, graph_builder.node_map

        self.graph = pyg_data
        self.graph.x = torch.tensor(self.embeddings[node_map["item_id"]], dtype=torch.float)
        self.node_map = node_map.sort("node_id")

        self.item_id_to_node_id = {
            int(row["item_id"]): int(row["node_id"]) for row in self.node_map.iter_rows(named=True)
        }
        self.node_id_to_item_id = {
            int(row["node_id"]): int(row["item_id"]) for row in self.node_map.iter_rows(named=True)
        }

        missing_item_ids = [i for i in range(len(self.embeddings)) if i not in self.item_id_to_node_id]
        if missing_item_ids:
            raise ValueError(
                "Some embedding rows have no graph node. "
                "Either graph does not cover all items or item ids are not aligned "
                "with embedding row indices."
            )

    def _load_interactions(self, path):
        with open(os.path.join(path), "r") as f:
            inters = json.load(f)

        rows = [(user, item, pos) for user, items in inters.items() for pos, item in enumerate(items)]

        df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])
        df = pl.DataFrame(df)
        return df.select(["user_id", "ts", "item_id"])

    def _limit_local_degree(self, subset, edge_index, edge_weight):
        """
        Optional local top-k pruning after k-hop extraction.
        Keeps up to max_neighbors outgoing edges per local node
        by edge weight.
        """
        if self.max_neighbors is None:
            return edge_index, edge_weight

        num_local_nodes = subset.numel()
        src = edge_index[0]
        dst = edge_index[1]

        kept_eids = []

        for u in range(num_local_nodes):
            mask = src == u
            eids = mask.nonzero(as_tuple=False).view(-1)

            if eids.numel() <= self.max_neighbors:
                kept_eids.append(eids)
                continue

            local_w = edge_weight[eids]
            topk = torch.topk(local_w, k=self.max_neighbors, largest=True).indices
            kept_eids.append(eids[topk])

        if len(kept_eids) == 0:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float),
            )

        kept_eids = torch.cat(kept_eids, dim=0)
        kept_eids = kept_eids.unique(sorted=True)

        return edge_index[:, kept_eids], edge_weight[kept_eids]

    def __getitem__(self, index):
        """
        index is assumed to be item_id / embedding row id.
        """
        center_item_id = int(index)

        if center_item_id not in self.item_id_to_node_id:
            raise KeyError(f"item_id {center_item_id} is not present in graph")

        center_node_id = self.item_id_to_node_id[center_item_id]

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=torch.tensor([center_node_id], dtype=torch.long),
            num_hops=self.num_hops,
            edge_index=self.graph.edge_index,
            relabel_nodes=True,
            num_nodes=self.graph.num_nodes,
            directed=True,
        )

        edge_weight = self.graph.edge_weight[edge_mask]

        if self.max_neighbors is not None:
            edge_index, edge_weight = self._limit_local_degree(
                subset=subset,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

        local_item_ids = torch.tensor(
            [self.node_id_to_item_id[int(nid)] for nid in subset.tolist()],
            dtype=torch.long,
        )
        local_semantic_embs = torch.as_tensor(
            self.embeddings[local_item_ids.numpy()],
            dtype=torch.float32,
        )

        center_emb = torch.as_tensor(
            self.embeddings[center_item_id],
            dtype=torch.float32,
        )

        return (
            center_emb,
            center_item_id,
            {
                "center_item_id": torch.tensor(center_item_id, dtype=torch.long),
                "center_node_id": torch.tensor(center_node_id, dtype=torch.long),
                "center_emb": center_emb,
                "node_ids": subset,  # global node ids
                "item_ids": local_item_ids,  # original item ids
                "node_embs": local_semantic_embs,  # semantic features aligned with subset
                "edge_index": edge_index,  # local indexing over subset
                "edge_weight": edge_weight,
                "center_local_idx": mapping[0],  # local idx of center node
            },
        )

    def __len__(self):
        return len(self.embeddings)
