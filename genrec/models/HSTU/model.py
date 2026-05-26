import math
from types import SimpleNamespace
from typing import Optional
import torch
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.models.HSTU.tokenizer import HSTUTokenizer


TIMESTAMPS_KEY = "time"
RATINGS_KEY = "rating"


class HSTU(AbstractModel):
    """
    HSTU model from Wang and McAuley, "Self-Attentive Sequential Recommendation." ICDM 2018.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        gpt2 (GPT2LMHeadModel): The GPT-2 model used for the HSTU model.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: HSTUTokenizer
    ):
        super(HSTU, self).__init__(config, dataset, tokenizer)

        self.num_items = tokenizer.vocab_size
        self.max_seq_len = tokenizer.max_token_seq_len
        self.embed_dim = config['n_embd']
        self.use_temporal_bias = config.get('use_temporal_bias', True)

        # Item embedding (0 is padding)
        self.item_embedding = torch.nn.Embedding(self.num_items + 1, self.embed_dim, padding_idx=0)

        # Embedding dropout
        self.emb_dropout = torch.nn.Dropout(config['embd_pdrop'])

        num_time_buckets: int = config.get("num_time_buckets", 128)
        num_position_buckets: int = config.get("num_position_buckets", 128)
        max_position_distance: int = config.get("max_position_distance", 128)
        
        self.initializer_range: int = config.get("initializer_range", 128)

        loss_type: str = config.get("loss_type", "ce")
        if loss_type == "ce":
            self.forward = self.forward_ce
        elif loss_type == "sampled_softmax":
            self.forward = self.forward_sampled_softmax
            self.num_negatives = config.get("num_negatives", 512)
            self.l2_norm = config.get("l2_norm", True)
            self.temperature = config.get("temperature", 0.05)
        else:
            raise ValueError("Invalid loss function. Possible values: 'ce', 'sampled_softmax'.")

        # HSTU layers
        self.layers = torch.nn.ModuleList([
            HSTULayer(
                embed_dim=self.embed_dim,
                num_heads=config['n_head'],
                dropout=config['resid_pdrop'],
                num_position_buckets=num_position_buckets,
                num_time_buckets=num_time_buckets,
                max_position_distance=max_position_distance,
                use_temporal_bias=self.use_temporal_bias,
            )
            for _ in range(config['n_layer'])
        ])

        # Final layer norm
        self.final_norm = torch.nn.LayerNorm(self.embed_dim)

        self._init_weights()
        self.ignore_idx = tokenizer.ignored_label

        # Causal mask
        self.causal_mask: torch.Tensor
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1).bool(), persistent=False
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=self.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.trunc_normal_(module.weight, std=self.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward_ce(
        self,
        batch: dict
    ) -> SimpleNamespace:
        """
        Forward pass.

        Args:
            input_ids: Item ID sequence [B, L], 0 is padding
            timestamps: Unix timestamps [B, L], optional for temporal bias
            targets: Target item IDs [B, L] for loss computation

        Returns:
            logits: Prediction logits [B, L, num_items+1]
            loss: Cross-entropy loss if targets provided
        """
        input_ids = batch['input_ids']
        padding_mask = batch['attention_mask'] == 0
        time_seq = batch.get(TIMESTAMPS_KEY)
        targets = batch.get('labels')

        _, L = input_ids.shape

        # Padding mask
        # padding_mask = (input_ids == 0)

        # Item embedding
        x = self.item_embedding(input_ids)  # [B, L, D]
        x = self.emb_dropout(x)

        # Apply HSTU layers
        for layer in self.layers:
            x = layer(x, self.causal_mask[:L, :L], padding_mask, time_seq)

        x = self.final_norm(x)

        # Prediction via dot product with item embeddings
        logits = x @ self.item_embedding.weight.T  # [B, L, V]

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_items + 1),
                targets.view(-1),
                ignore_index=self.ignore_idx
            )

        output = SimpleNamespace(logits=logits, loss=loss)

        return output

    def forward_sampled_softmax(
        self,
        batch: dict
    ) -> SimpleNamespace:
        """
        Forward with sampled softmax loss (aligned with Meta's original HSTU).

        Instead of computing logits over ALL items, samples a small set of
        negatives per batch for efficiency and different optimization landscape.
        """
        input_ids = batch['input_ids']
        padding_mask = batch['attention_mask'] == 0
        time_seq = batch.get(TIMESTAMPS_KEY)
        targets = batch.get('labels')

        _, L = input_ids.shape
        device = input_ids.device

        # padding_mask = (input_ids == 0)

        x = self.item_embedding(input_ids)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, self.causal_mask[:L, :L], padding_mask, time_seq)

        x = self.final_norm(x)  # [B, L, D]

        # Full logits for eval (no loss)
        logits = x @ self.item_embedding.weight.T  # [B, L, V]

        loss = None
        if targets is not None:
            # Flatten to [B*L, D] and [B*L]
            x_flat = x.view(-1, self.embed_dim)  # [B*L, D]
            targets_flat = targets.view(-1)  # [B*L]

            # Filter out padding positions
            valid_mask = targets_flat != self.ignore_idx
            x_valid = x_flat[valid_mask]  # [N, D]
            targets_valid = targets_flat[valid_mask]  # [N]

            if x_valid.size(0) > 0:
                # Get positive embeddings
                pos_emb = self.item_embedding(targets_valid)  # [N, D]

                # Sample random negatives (shared across batch for efficiency)
                neg_ids = torch.randint(1, self.num_items + 1, (self.num_negatives,), device=device)
                neg_emb = self.item_embedding(neg_ids)  # [K, D]

                # L2 normalize if enabled (as in Meta's implementation)
                if self.l2_norm:
                    x_valid = F.normalize(x_valid, dim=-1)
                    pos_emb = F.normalize(pos_emb, dim=-1)
                    neg_emb = F.normalize(neg_emb, dim=-1)

                # Compute logits: [N, 1+K]
                pos_logits = (x_valid * pos_emb).sum(dim=-1, keepdim=True)  # [N, 1]
                neg_logits = x_valid @ neg_emb.T  # [N, K]
                all_logits = torch.cat([pos_logits, neg_logits], dim=-1) / self.temperature

                # Target is always index 0 (positive)
                loss_targets = torch.zeros(x_valid.size(0), device=device, dtype=torch.long)
                loss = F.cross_entropy(all_logits, loss_targets)

        output = SimpleNamespace(logits=logits, loss=loss)

        return output

    def gather_index(self, output, index):
        """
        Gather the output at a specific index.

        Args:
            output: The output tensor.
            index: The index tensor.

        Returns:
            torch.Tensor: The gathered output.
        """
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def generate(self, batch, n_return_sequences=1):
        """
        Generate sequences based on the input batch.

        Args:
            batch: The input batch.
            n_return_sequences (int): The number of sequences to generate.

        Returns:
            torch.Tensor: The generated sequences.
        """
        outputs = self.forward(batch | {"labels": None})
        logits = self.gather_index(outputs.logits, batch['seq_lens'] - 1)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return {"preds": preds.unsqueeze(-1)}


class HSTULayer(torch.nn.Module):
    """
    Single HSTU layer.

    Structure:
        1. Pointwise Projection: X -> SiLU(Linear(X)) -> split to U, V, Q, K
        2. Spatial Aggregation: SiLU(QK^T + RAB) @ V
        3. Pointwise Transformation: Norm(Attention) ⊙ U -> FFN
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_position_buckets: int,
        num_time_buckets: int,
        max_position_distance: int,
        use_temporal_bias: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_temporal_bias = use_temporal_bias

        assert embed_dim % num_heads == 0

        # Pointwise projection: projects to 4 * embed_dim (for U, V, Q, K)
        self.projection = torch.nn.Linear(embed_dim, 4 * embed_dim)

        # Relative attention bias (position-based, shared across heads)
        self.position_bias = RelativePositionBias(
            num_buckets=num_position_buckets,
            max_distance=max_position_distance,
            num_heads=num_heads,
        )

        # Temporal bias (optional)
        if use_temporal_bias:
            self.temporal_bias = TemporalBias(
                num_buckets=num_time_buckets,
                num_heads=num_heads,
            )

        # Layer norm for attention output
        self.attn_norm = torch.nn.LayerNorm(embed_dim)

        # FFN (pointwise transformation)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4 * embed_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4 * embed_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

        # Final layer norm
        self.ffn_norm = torch.nn.LayerNorm(embed_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        causal_mask: torch.Tensor,  # [L, L]
        padding_mask: torch.Tensor,  # [B, L]
        timestamps: Optional[torch.Tensor] = None,  # [B, L]
    ) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        # === Pointwise Projection ===
        # Project and apply SiLU, then split into U, V, Q, K
        projected = F.silu(self.projection(x))  # [B, L, 4D]
        U, V, Q, K = projected.chunk(4, dim=-1)  # Each [B, L, D]

        # Reshape for multi-head attention
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # === Spatial Aggregation ===
        # Compute attention scores (without softmax!)
        scores = Q @ K.transpose(-2, -1)  # [B, H, L, L]

        # Add relative position bias
        pos_bias = self.position_bias(L, x.device)  # [H, L, L]
        scores = scores + pos_bias.unsqueeze(0)

        # Add temporal bias if enabled and timestamps provided
        if self.use_temporal_bias and timestamps is not None:
            time_bias = self.temporal_bias(timestamps)  # [B, H, L, L]
            scores = scores + time_bias

        # Apply causal mask (set masked positions to large negative)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Apply padding mask
        scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), -1e9)

        # HSTU key: SiLU instead of softmax!
        # This allows capturing preference intensity
        attn_weights = F.silu(scores)

        # Apply attention to values
        attn_output = attn_weights @ V  # [B, H, L, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]

        # === Pointwise Transformation ===
        # Normalize and gate with U
        attn_output = self.attn_norm(attn_output)
        attn_output = attn_output * U  # Element-wise gating

        # Residual connection
        x = residual + self.dropout(attn_output)

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class RelativePositionBias(torch.nn.Module):
    """
    Relative position bias using logarithmic bucketing (T5-style).

    Buckets relative positions into logarithmically spaced bins,
    allowing the model to generalize to longer sequences.
    """

    def __init__(self, num_buckets: int = 32, max_distance: int = 128, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads

        # Learnable bias for each bucket and head
        self.relative_attention_bias = torch.nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Convert relative position to bucket index using logarithmic bucketing.

        For causal attention, we only care about positions where query >= key,
        so relative_position >= 0.
        """
        # We use half buckets for exact positions, half for log-spaced
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # Clamp to non-negative (causal)
        relative_position = torch.clamp(relative_position, min=0)

        # Half buckets for small distances (exact)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Log-spaced buckets for larger distances
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()

        relative_position_if_large = torch.clamp(relative_position_if_large, max=num_buckets - 1)

        bucket = torch.where(is_small, relative_position, relative_position_if_large)
        return bucket

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias matrix.

        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        # relative_position[i, j] = i - j (query_pos - key_pos)
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]

        # Convert to buckets
        buckets = self._relative_position_bucket(relative_position)  # [L, L]

        # Look up bias values
        bias = self.relative_attention_bias(buckets)  # [L, L, H]
        bias = bias.permute(2, 0, 1)  # [H, L, L]

        return bias


class TemporalBias(torch.nn.Module):
    """
    Temporal attention bias using logarithmic bucketing of time differences.

    Quantizes timestamp differences into log-spaced buckets,
    capturing both recent and long-term temporal patterns.
    """

    def __init__(self, num_buckets: int = 64, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads

        # Learnable bias for each bucket and head
        self.temporal_attention_bias = torch.nn.Embedding(num_buckets, num_heads)

    def _temporal_bucket(self, time_diff: torch.Tensor) -> torch.Tensor:
        """
        Convert time difference to bucket index.

        Uses formula: bucket = floor(log(max(1, |diff|)) / log_base)
        where log_base ≈ 0.301 (log10(2)) as in the paper
        """
        # Take absolute value and ensure minimum of 1
        abs_diff = torch.clamp(torch.abs(time_diff), min=1).float()

        # Log bucketing (using natural log, scaled)
        # Paper uses: floor(log(max(1, |diff|)) / 0.301)
        # We use a similar approach but cap at num_buckets - 1
        buckets = (torch.log(abs_diff) / 0.693).long()  # 0.693 = ln(2)
        buckets = torch.clamp(buckets, min=0, max=self.num_buckets - 1)

        return buckets

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal bias matrix.

        Args:
            timestamps: [B, L] unix timestamps

        Returns:
            bias: [B, num_heads, L, L]
        """
        B, L = timestamps.shape

        # Compute pairwise time differences
        # time_diff[i, j] = timestamps[i] - timestamps[j]
        time_diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [B, L, L]

        # Convert to buckets
        buckets = self._temporal_bucket(time_diff)  # [B, L, L]

        # Look up bias values
        bias = self.temporal_attention_bias(buckets)  # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2)  # [B, H, L, L]

        return bias