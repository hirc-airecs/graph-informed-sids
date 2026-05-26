import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    GatedGraphConv,
)


class GNNFusionEncoder(nn.Module):
    """
    GNN encoder that fuses:
      1. semantic node representations
      2. graph neighborhood information from a sampled subgraph

    Supported conv_type:
      - "sage"
      - "gat"
      - "gcn"
      - "gin"
      - "gated_graph"

    Expected inputs:
        graph_data.x          FloatTensor [N_sub, D_sem]
        graph_data.edge_index LongTensor [2, E]
        graph_data.batch_size int

    Returns:
        fused_center: FloatTensor [B, out_dim]
    """

    SUPPORTED_CONVS = {"sage", "gat", "gcn", "gin", "gated_graph"}

    def __init__(
        self,
        semantic_dim: int,
        hidden_dim: int,
        out_dim: int,
        conv_type: str = "gat",
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_residual_fusion: bool = True,
        act: str = "elu",
        gated_aggr: str = "add",
        gin_eps: float = 0.0,
        gin_train_eps: bool = False,
    ):
        super().__init__()

        conv_type = conv_type.lower()
        if conv_type not in self.SUPPORTED_CONVS:
            raise ValueError(f"Unsupported conv_type={conv_type}. Supported: {self.SUPPORTED_CONVS}")
        if layers < 1:
            raise ValueError("layers must be >= 1")

        self.conv_type = conv_type
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.use_residual_fusion = use_residual_fusion
        self.act_name = act.lower()

        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if conv_type == "gated_graph":
            # GatedGraphConv already performs num_layers recurrent propagation internally.
            # It requires input_dim <= out_channels, so projecting to hidden_dim first is important.
            if layers == 1:
                gated_out_dim = out_dim
            else:
                gated_out_dim = hidden_dim

            self.gated_graph_conv = GatedGraphConv(
                out_channels=gated_out_dim,
                num_layers=layers,
                aggr=gated_aggr,
            )
            self.post_gated_proj = None if gated_out_dim == out_dim else nn.Linear(gated_out_dim, out_dim)
            self.final_norm = nn.LayerNorm(out_dim)
        else:
            dims = [hidden_dim]
            if layers == 1:
                dims.append(out_dim)
            else:
                dims.extend([hidden_dim] * (layers - 1))
                dims.append(out_dim)

            for layer_idx in range(layers):
                in_dim = dims[layer_idx]
                target_dim = dims[layer_idx + 1]
                is_last = layer_idx == layers - 1

                conv, norm_dim = self._build_conv_layer(
                    conv_type=conv_type,
                    in_dim=in_dim,
                    out_dim=target_dim,
                    is_last=is_last,
                    heads=heads,
                    gin_eps=gin_eps,
                    gin_train_eps=gin_train_eps,
                    dropout=dropout,
                )
                self.convs.append(conv)
                self.norms.append(nn.LayerNorm(norm_dim))

        if use_residual_fusion:
            self.semantic_skip = nn.Linear(hidden_dim, out_dim)

    def _make_gin_mlp(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _build_conv_layer(
        self,
        conv_type: str,
        in_dim: int,
        out_dim: int,
        is_last: bool,
        heads: int,
        gin_eps: float,
        gin_train_eps: bool,
        dropout: float,
    ):
        if conv_type == "sage":
            conv = SAGEConv(in_dim, out_dim)
            norm_dim = out_dim

        elif conv_type == "gcn":
            conv = GCNConv(in_dim, out_dim)
            norm_dim = out_dim

        elif conv_type == "gin":
            mlp = self._make_gin_mlp(in_dim, out_dim)
            conv = GINConv(mlp, eps=gin_eps, train_eps=gin_train_eps)
            norm_dim = out_dim

        elif conv_type == "gat":
            if is_last:
                conv = GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
                norm_dim = out_dim
            else:
                if out_dim % heads != 0:
                    raise ValueError(
                        f"For GAT hidden layers, out_dim ({out_dim}) must be divisible by heads ({heads}) "
                        f"when concat=True."
                    )
                per_head_dim = out_dim // heads
                conv = GATConv(
                    in_channels=in_dim,
                    out_channels=per_head_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
                norm_dim = out_dim

        else:
            raise ValueError(f"Unsupported conv_type={conv_type}")

        return conv, norm_dim

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name == "relu":
            return F.relu(x)
        if self.act_name == "gelu":
            return F.gelu(x)
        if self.act_name == "elu":
            return F.elu(x)
        raise ValueError(f"Unsupported act={self.act_name}")

    def forward(self, graph_data) -> torch.Tensor:
        semantic_x = graph_data.x
        edge_index = graph_data.edge_index

        x0 = self.semantic_proj(semantic_x)

        if self.conv_type == "gated_graph":
            x = self.gated_graph_conv(x0, edge_index)
            if self.post_gated_proj is not None:
                x = self.post_gated_proj(x)
            x = self.final_norm(x)
        else:
            x = x0
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.norms[i](x)

                is_last = i == len(self.convs) - 1
                if not is_last:
                    x = self._activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_residual_fusion:
            x = x + self.semantic_skip(x0)

        x = F.dropout(x, p=self.dropout, training=self.training)

        return x[: graph_data.batch_size]
