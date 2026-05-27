import argparse
import json
import random
import torch
import logging
import os
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from gensid.datasets_module import EmbDataset, APPNPEmbDataset, MixedFusionDepthDataset, EdgeWeightCollator
from gensid.utils import check_collision, get_collision_item, get_indices_count
from gensid.models.rqvae import RQVAE
from gensid.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="SID Builder")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=20000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
    )
    parser.add_argument("--eval_step", type=int, default=2000, help="eval step")
    parser.add_argument("--learner", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--root_path", type=str, default="./", help="Input data path.")  # Path(__file__).parent
    parser.add_argument("--saved_model_dir", type=str, default=None, help="Directory to save the model.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (Beauty, Yelp, MIND, etc).")
    parser.add_argument("--embedder", type=str, default=None, help="Sentence Embedding model used.")
    parser.add_argument("--output_file", type=str, default=None, help="Output File Name.")

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="l2 regularization weight")
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument(
        "--disable_vq_init", action="store_true", help="run one-pass codebook initialization before training"
    )
    parser.add_argument(
        "--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons"
    )
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--disable_cf_loss", action="store_true")
    parser.add_argument("--disable_sk", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--device", type=str, default="cuda:4", help="gpu or cpu")

    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256, 256], help="emb num of every vq")
    parser.add_argument("--e_dim", type=int, default=32, help="vq codebook embedding size")
    parser.add_argument("--quant_loss_weight", type=float, default=1.0, help="vq quantion loss weight")
    parser.add_argument("--alpha", type=float, default=0.1, help="cf loss weight (encoder-directed)")
    parser.add_argument("--beta", type=float, default=0.1, help="diversity loss weight")
    parser.add_argument("--n_clusters", type=int, default=10, help="n_clusters")
    parser.add_argument("--sample_strategy", type=str, default="all", help="sample_strategy")
    parser.add_argument("--cf_emb", type=str, default=None, help="cf emb; leave unset to disable CF loss")
    parser.add_argument(
        "--cf_loss_use_codebook_rep",
        action="store_true",
        help="compute CF loss on reconstructed codebook representations instead of dense_out",
    )
    parser.add_argument(
        "--cf_loss_last_m_codebooks", type=int, default=0, help="if >0, compute CF loss using only the last M codebooks"
    )

    parser.add_argument(
        "--layers", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64], help="hidden sizes of every layer"
    )

    parser.add_argument("--global_seed", type=int, default=42, help="Random seed")
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument("--graph_signal", action="store_true")
    graph_group.add_argument("--graph_encoder", action="store_true")
    graph_group.add_argument("--graph_augmentation", action="store_true")

    parser.add_argument("--convolution_type", type=str, default="APPNP", help="otherwise time based")

    graph_type = parser.add_subparsers(dest="graph_type")

    p_windowed = graph_type.add_parser("windowed")
    p_windowed.add_argument("--window_size", type=int, default=3, help="Size of the sliding window")
    p_windowed.add_argument("--bidirectional", action="store_true", help="Make graph bidirectional")
    p_windowed.add_argument("--dedup_within_window", action="store_true", help="Deduplicate pairs within window")
    p_windowed.add_argument(
        "--decay", type=str, choices=["inverse_distance"], default=None, help="Edge weighting scheme or None"
    )

    p_rw = graph_type.add_parser("rw")
    p_rw.add_argument("--walk_length", type=int, default=10, help="Length of each random walk")
    p_rw.add_argument("--walks_per_item", type=int, default=10, help="Number of walks per item")
    p_rw.add_argument("--window_size", type=int, default=2, help="Context window size for co-occurrence")
    p_rw.add_argument("--self_loops", action="store_true", help="Allow self-loops in random walks")
    p_rw.add_argument("--dedup_per_walk", action="store_true", help="Deduplicate pairs within each walk")
    p_rw.add_argument("--seed", type=int, default=42, help="Random seed")

    graph_type.add_parser("adjacent_cooc")

    ### GNN param block
    parser.add_argument("--gnn_conv_type", type=str, default="gat", help="Either: gat, gcn, gin, gated, or sage")
    parser.add_argument("--gnn_hidden_dim", type=int, default=None, help="GNN hidden dimension")
    parser.add_argument("--gnn_dropout", type=float, default=0.0, help="GNN dropout")
    parser.add_argument("--gnn_heads", type=int, default=4, help="GNN heads (only used by GAT)")
    parser.add_argument("--gnn_layers", type=int, default=2, help="GNN layers")

    ### APPNP param block
    parser.add_argument("--num_prop", type=int, default=10, help="num propagations")
    parser.add_argument("--appnp_alpha", type=float, default=0.1, help="self-loop prob")
    parser.add_argument("--use_edge_weight", action="store_true", help="whether to use edge weights")

    # APPNP as augmentation block
    parser.add_argument("--hops", type=int, nargs="+", default=[0], help="hops")
    parser.add_argument("--probs", type=int, nargs="+", default=None, help="hops")

    # edge reconstruction block
    parser.add_argument("--plain_reconstruction", action="store_true", help="whether to use edge weights")
    parser.add_argument(
        "--use_edge_reconstruction_loss",
        action="store_true",
        help="whether to use reconstruct edges data on quantizer level",
    )
    parser.add_argument("--num_anchors", type=int, default=1024)

    # spectral clustering block
    parser.add_argument(
        "--spectral_dim", type=int, default=0, help="num spectral clusters, currently either 0 or inner quantizer dim"
    )
    parser.add_argument("--cluster_labels_path", type=str, default=None)
    parser.add_argument("--alpha_graph_cluster", type=float, default=0.0, help="graph_cluster loss coef")
    parser.add_argument("--max_cluster_level", type=int, default=None, help="get cluster up to this level")

    args = parser.parse_args()

    # setup defaults
    args.graph_type = args.graph_type or "adjacent_cooc"

    return args


def get_trained_flag(metrics_path: str) -> bool:
    if not os.path.exists(metrics_path):
        return False

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics.get("fully_trained", False)


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


def setup_train(args, cf_emb):
    # --- Build dataset ---
    if args.graph_augmentation:
        graph_building_kwargs = get_graph_building_conf(args)
        graph_saving_path = os.path.join(args.ckpt_dir, "multihop/")

        dataset = MixedFusionDepthDataset(
            args.interactions_path,
            args.data_path,
            hops=args.hops,
            fused_cache_root=graph_saving_path,
            alpha=args.appnp_alpha,
            use_edge_weight=args.use_edge_weight,
            graph_type=args.graph_type,
            convolution_type=args.convolution_type,
            graph_building_kwargs=graph_building_kwargs,
            probs=args.probs,
        )
    elif args.graph_signal:
        graph_building_kwargs = get_graph_building_conf(args)
        graph_saving_path = os.path.join(args.ckpt_dir, "graph.npy")
        dataset = APPNPEmbDataset(
            args.interactions_path,
            args.data_path,
            fused_cache_path=graph_saving_path,
            num_prop=args.num_prop,
            alpha=args.appnp_alpha,
            use_edge_weight=args.use_edge_weight,
            graph_type=args.graph_type,
            convolution_type=args.convolution_type,
            graph_building_kwargs=graph_building_kwargs,
            spectral_dim=args.spectral_dim,
            eval_mode=False,
            cluster_labels_path=args.cluster_labels_path,
        )
    else:
        dataset = EmbDataset(args.data_path, interactions_path=args.interactions_path, eval_mode=False)

    if args.use_edge_reconstruction_loss:
        collate_fn = EdgeWeightCollator(dataset.pyg_data, num_anchors=args.num_anchors)
    else:
        collate_fn = default_collate

    data_loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=args.persistent_workers,
        collate_fn=collate_fn,
    )

    model = RQVAE(
        in_dim=dataset.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        beta=args.beta,
        alpha=args.alpha,
        n_clusters=args.n_clusters,
        sample_strategy=args.sample_strategy,
        cf_embedding=cf_emb,
        cf_loss_use_codebook_rep=args.cf_loss_use_codebook_rep,
        cf_loss_last_m_codebooks=(args.cf_loss_last_m_codebooks if args.cf_loss_last_m_codebooks > 0 else None),
        use_cf_loss=not args.disable_cf_loss,
        alpha_graph_cluster=args.alpha_graph_cluster,
        num_anchors=args.num_anchors,
    )

    return args, dataset, data_loader, model


def setup_generation(args_setting, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    if args_setting.graph_signal:
        graph_saving_path = os.path.join(args.ckpt_dir, "graph.npy")

        data = APPNPEmbDataset(
            args.interactions_path,
            args.data_path,
            fused_cache_path=graph_saving_path,
            eval_mode=True,
        )
    else:
        data = EmbDataset(args.data_path, eval_mode=True)

    data_loader = DataLoader(data, num_workers=args.num_workers, batch_size=64, shuffle=False, pin_memory=True)

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return args, data, data_loader, model


def generate_indices(args_setting, ckpt_dir, output_file, device):
    _, data, data_loader, model = setup_generation(args_setting, ckpt_dir, device)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>"]

    def constrained_km(data, n_clusters=10):
        from k_means_constrained import KMeansConstrained

        x = data
        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=n_clusters * 6,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        return t_centers, t_labels

    labels = {"0": [], "1": [], "2": [], "3": []}
    embs = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]

    for idx, emb in enumerate(embs):
        _, label = constrained_km(emb)
        labels[str(idx)] = label
    for _data in tqdm(data_loader):
        d, emb_idx = _data[0], _data[1]
        d = d.to(device)

        indices = model.get_indices(d, labels, use_sk=False)

        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))
        # break

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    # model.rq.vq_layers[-1].sk_epsilon = 0.005
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    # model.rq.vq_layers[-1].sk_epsilon = 0.1
    idx = 0
    # There are often duplicate items in the dataset, and we no longer differentiate them
    while True:
        if idx >= 20 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)

        for collision_items in collision_item_groups:
            d = data[collision_items]
            d = d[0].to(device)

            indices = model.get_indices(d, labels, use_sk=True)

            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                all_indices[item] = code
                all_indices_str[item] = str(code)
        idx += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    with open(output_file, "w") as fp:
        json.dump(all_indices_dict, fp)


def train(args, ckpt_dir):
    graph_npy_path = os.path.join(ckpt_dir, "graph.npy")
    if os.path.isfile(graph_npy_path):
        print(f"Removing existing {graph_npy_path}")
        os.remove(graph_npy_path)

    print(args)
    logging.basicConfig(level=logging.DEBUG)

    cf_emb = None
    if args.cf_emb:
        cf_emb = torch.load(args.cf_emb, weights_only=False).squeeze().detach().cpu().numpy()

    args, dataset, data_loader, model = setup_train(args, cf_emb)

    print(model)

    trainer = Trainer(args, model)
    best_loss, best_collision_rate = trainer.fit(dataset, data_loader)

    print("Best Loss:", best_loss)
    print("Best Collision Rate:", best_collision_rate)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    args.data_path = Path(args.root_path) / f"cache/gensid/{args.dataset}/{args.dataset}.emb-{args.embedder}-td.npy"
    args.interactions_path = Path(args.root_path) / f"cache/gensid/{args.dataset}/{args.dataset}.inter.json"
    args.ckpt_dir = Path(args.root_path) / "ckpt/gensid/quantizer"

    init_seed(args.global_seed)

    # args.saved_model_dir contains the alias of the experiment to be able to identify the file
    assert args.saved_model_dir is not None
    metrics_path = os.path.join(args.ckpt_dir, args.saved_model_dir or "", "best_collision_model.json")

    if get_trained_flag(metrics_path) and args.overwrite:
        print("Warning: Training an already trained model according to the --overwrite argument")

    if not (os.path.isfile(metrics_path) and get_trained_flag(metrics_path)) or args.overwrite:
        train(args, args.ckpt_dir)

    ckpt_id = str(args.saved_model_dir).replace("..", "").replace("/", "_")
    output_dir = Path(args.root_path) / f"cache/gensid/{args.dataset}"
    output_file = f"{args.dataset}.index.{ckpt_id}.json"
    output_file = os.path.join(output_dir, output_file)

    ckpt_path = args.ckpt_dir / args.saved_model_dir
    ckpt_path = ckpt_path / "best_collision_model.pth"
    generate_indices(args, ckpt_path, output_file, args.device)


if __name__ == "__main__":
    main()
