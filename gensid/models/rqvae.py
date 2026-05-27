import torch
from torch import nn
from torch.nn import functional as F
from gensid.models.layers import MLPLayers
from gensid.models.graph_encoder import GNNFusionEncoder
from gensid.models.rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim=768,
        num_emb_list=None,
        e_dim=64,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
        alpha=0.1,
        alpha_codebook=0.0,
        alpha_graph_cluster=0.0,
        graph_cluster_tau=0.1,
        beta=0.001,
        n_clusters=10,
        sample_strategy="all",
        cf_embedding=None,
        cf_loss_use_codebook_rep=False,
        cf_loss_last_m_codebooks=None,
        use_cf_loss=True,
        num_anchors=1024,
    ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.alpha = alpha
        self.alpha_codebook = alpha_codebook
        self.alpha_graph_cluster = alpha_graph_cluster
        self.graph_cluster_tau = graph_cluster_tau
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.use_cf_loss = use_cf_loss

        self.cf_loss_use_codebook_rep = cf_loss_use_codebook_rep
        self.cf_loss_last_m_codebooks = cf_loss_last_m_codebooks

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn)

        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout_prob, bn=self.bn)
        self.num_anchors = num_anchors

    def forward(self, x, labels, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices, x_q_codebooks, x_res_list = self.rq(x, labels, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q, x_q_codebooks, x_res_list

    def CF_loss(self, quantized_rep, encoded_rep):
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss

    def graph_loss(self, dense_out, edge_w, tau=0.1, num_anchors=1024):
        z = F.normalize(dense_out, dim=-1)
        edge_w = edge_w[:, :num_anchors]
        sim = z @ z[:num_anchors].t() / tau

        sim.fill_diagonal_(-1e9)  # mask self
        logp = F.log_softmax(sim, dim=1)

        w = edge_w.clone().to(dense_out.device)
        w.fill_diagonal_(0.0)
        # target = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        mask = w.sum(dim=1) > 0  # only rows with positives
        # NOTE: number of elements in average might vary across batches, fix it
        return -(w * logp).sum(dim=1)[mask].mean()

    def graph_cluster_loss(self, dense_out, cluster_labels, tau=None, num_anchors=1024):
        if cluster_labels is None:
            return dense_out.new_tensor(0.0)

        tau = self.graph_cluster_tau if tau is None else tau
        labels = cluster_labels.to(dense_out.device, non_blocking=True).view(-1)
        valid_label_mask = labels >= 0
        if valid_label_mask.sum() <= 1:
            return dense_out.new_tensor(0.0)

        z = F.normalize(dense_out[valid_label_mask], dim=-1)
        labels = labels[valid_label_mask]
        # NOTE: B^2
        sim = z @ z[:num_anchors].t() / tau
        sim.fill_diagonal_(1e-9)

        pos_mask = labels[:, None] == labels[None, :num_anchors]
        pos_mask.fill_diagonal_(False)
        pos_count = pos_mask.sum(dim=1)
        valid_anchor_mask = pos_count > 0
        if not valid_anchor_mask.any():
            return dense_out.new_tensor(0.0)

        logp = F.log_softmax(sim, dim=1)
        per_anchor = -(logp * pos_mask).sum(dim=1) / pos_count.clamp_min(1)
        return per_anchor[valid_anchor_mask].mean()

    def hierarchical_graph_cluster_loss(self, dense_out_codebooks, cluster_labels, num_anchors=1024):
        if cluster_labels is None or dense_out_codebooks is None:
            return dense_out_codebooks.new_tensor(0.0) if dense_out_codebooks is not None else torch.tensor(0.0)
        num_levels = cluster_labels.size(-1)
        if num_levels == 0:
            return dense_out_codebooks.new_tensor(0.0)

        losses = []
        for level_idx in range(0, num_levels):
            losses.append(
                self.graph_cluster_loss(
                    dense_out_codebooks[:, : level_idx + 1, :].sum(dim=1),
                    cluster_labels[:, level_idx],
                    num_anchors=num_anchors,
                )
            )

        if not losses:
            return dense_out_codebooks.new_tensor(0.0)
        return torch.stack(losses).sum()

    def vq_initialization(self, x, use_sk=True):
        self.rq.vq_ini(self.encoder(x), use_sk)

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, _, _ = self.rq(x_e, labels, use_sk=use_sk)
        return indices

    def _aggregate_codebook_representations(self, dense_out_codebook):
        if dense_out_codebook is None:
            raise ValueError("dense_out_codebook is required when CF loss is based on codebooks")
        start_idx = 0
        if self.cf_loss_last_m_codebooks is not None:
            m = min(self.cf_loss_last_m_codebooks, dense_out_codebook.shape[1])
            if m <= 0:
                raise ValueError("cf_loss_last_m_codebooks should be >= 1")
            start_idx = dense_out_codebook.shape[1] - m
        return dense_out_codebook[:, start_idx:, :].sum(dim=1)

    def compute_loss(
        self,
        out,
        quant_loss,
        emb_idx,
        dense_out,
        dense_out_codebooks,
        x_res_list,
        xs=None,
        edges_weights=None,
        spectral_vectors=None,
        cluster_labels=None,
    ):

        if self.loss_type == "mse":
            loss = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")
        loss_recon = loss.detach().clone()

        loss += self.quant_loss_weight * quant_loss

        # CF_Loss (optional)
        cf_input_codebooks = None
        if self.cf_loss_use_codebook_rep or self.cf_loss_last_m_codebooks is not None:
            cf_input_codebooks = self._aggregate_codebook_representations(dense_out_codebooks)

        cf_loss = 0.0
        cf_codebook_loss = 0.0
        # NOTE: repeated code fix
        if self.cf_embedding is not None:
            cf_embedding_in_batch = self.cf_embedding[emb_idx]
            cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(dense_out.device)
            cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)
            if cf_input_codebooks is not None:
                cf_codebook_loss += self.CF_loss(cf_input_codebooks, cf_embedding_in_batch)
        elif edges_weights is not None:
            cf_loss += self.graph_loss(dense_out, edges_weights, num_anchors=self.num_anchors)
            if cf_input_codebooks is not None:
                cf_codebook_loss += self.graph_loss(cf_input_codebooks, edges_weights, num_anchors=self.num_anchors)
        elif spectral_vectors is not None:
            cf_loss += self.CF_loss(dense_out, spectral_vectors)
            if cf_input_codebooks is not None:
                cf_codebook_loss += self.CF_loss(cf_input_codebooks, spectral_vectors)

        graph_cluster_loss = dense_out.new_tensor(0.0)
        if self.alpha_graph_cluster > 0.0:
            graph_cluster_loss_res = self.hierarchical_graph_cluster_loss(
                x_res_list, cluster_labels, num_anchors=self.num_anchors
            )
            graph_cluster_loss_codes = self.hierarchical_graph_cluster_loss(
                dense_out_codebooks, cluster_labels, num_anchors=self.num_anchors
            )
            graph_cluster_loss = graph_cluster_loss_codes + graph_cluster_loss_res
            loss += self.alpha_graph_cluster * graph_cluster_loss

        if self.use_cf_loss:
            loss += self.alpha * (cf_loss + cf_codebook_loss)

        aux_loss = cf_loss + cf_codebook_loss + graph_cluster_loss
        return loss, aux_loss, loss_recon, quant_loss


class GRQVAE(nn.Module):
    def __init__(
        self,
        # -------- GNNFusionEncoder params --------
        semantic_dim: int,
        gnn_hidden_dim: int,
        gnn_conv_type: str = "gat",
        gnn_heads: int = 4,
        gnn_dropout: float = 0.1,
        gnn_layers: int = 2,
        use_residual_fusion: bool = True,
        # -------- RQVAE params: keep same signature --------
        vae_in_dim=768,
        num_emb_list=None,
        e_dim=64,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
        alpha=0.1,
        alpha_codebook=0.0,
        alpha_graph_cluster=0.0,
        graph_cluster_tau=0.1,
        beta=0.001,
        n_clusters=10,
        sample_strategy="all",
        cf_embedding=None,
        cf_loss_use_codebook_rep=False,
        cf_loss_last_m_codebooks=None,
        use_cf_loss=True,
        num_anchors=1024,
    ):
        super().__init__()

        # Graph fusion encoder
        self.graph_encoder = GNNFusionEncoder(
            conv_type=gnn_conv_type,
            semantic_dim=semantic_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=vae_in_dim,
            heads=gnn_heads,
            dropout=gnn_dropout,
            layers=gnn_layers,
            use_residual_fusion=use_residual_fusion,
        )

        # Inner RQVAE
        # By default, its input dim should match graph encoder output dim.
        self.quantizer = RQVAE(
            in_dim=vae_in_dim,
            num_emb_list=num_emb_list,
            e_dim=e_dim,
            layers=layers,
            dropout_prob=dropout_prob,
            bn=bn,
            loss_type=loss_type,
            quant_loss_weight=quant_loss_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
            alpha=alpha,
            alpha_codebook=alpha_codebook,
            alpha_graph_cluster=alpha_graph_cluster,
            graph_cluster_tau=graph_cluster_tau,
            beta=beta,
            n_clusters=n_clusters,
            sample_strategy=sample_strategy,
            cf_embedding=cf_embedding,
            cf_loss_use_codebook_rep=cf_loss_use_codebook_rep,
            cf_loss_last_m_codebooks=cf_loss_last_m_codebooks,
            use_cf_loss=use_cf_loss,
            num_anchors=num_anchors,
        )
        self.rq = self.quantizer.rq

    def forward(self, data, graph_data, labels, use_sk=True):
        fused_data = self.graph_encoder(graph_data)
        return self.quantizer(fused_data, labels, use_sk)

    def CF_loss(self, quantized_rep, encoded_rep):
        return self.quantizer.CF_loss(quantized_rep, encoded_rep)

    def vq_initialization(self, x, use_sk=True):
        return self.quantizer.vq_initialization(x, use_sk=use_sk)

    @torch.no_grad()
    def get_indices(self, xs, graph_data, labels, use_sk=False):
        fused_data = self.graph_encoder(graph_data)
        return self.quantizer.get_indices(fused_data, labels, use_sk=use_sk)

    def compute_loss(
        self,
        out,
        quant_loss,
        emb_idx,
        dense_out,
        dense_out_codebooks,
        x_res_list,
        xs=None,
        edges_weights=None,
        spectral_vectors=None,
        cluster_labels=None,
    ):
        return self.quantizer.compute_loss(
            out=out,
            quant_loss=quant_loss,
            emb_idx=emb_idx,
            dense_out=dense_out,
            xs=xs,
            dense_out_codebooks=dense_out_codebooks,
            x_res_list=x_res_list,
            edges_weights=edges_weights,
            spectral_vectors=spectral_vectors,
            cluster_labels=cluster_labels,
        )
