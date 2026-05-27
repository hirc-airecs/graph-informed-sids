import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from types import SimpleNamespace


class SampledSoftmaxLoss(torch.nn.Module):
    """
    Sampled Softmax Loss supporting:
      - 'local'    : randomly sample negatives from the full vocabulary
      - 'in_batch' : use other items in the batch as negatives (standard in recsys)
      - 'both'     : combine local negatives + in-batch negatives
    """

    def __init__(
        self,
        emb_layer: torch.nn.Embedding,
        num_sampled: int = 512,
        mode: str = "both",       # 'local' | 'in_batch' | 'both'
        temperature: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        # self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.mode = mode
        self.temperature = temperature
        self.ignore_index = ignore_index

        self.output_layer = emb_layer

    def _sample_negatives(self, targets):
        device = targets.device
        return torch.randint(0, self.output_layer.weight.shape[0], (self.num_sampled,), device=device)

    def _inbatch_ids(self, targets):
        batch_ids = targets.unique()
        return batch_ids[torch.randperm(batch_ids.shape[0])[:self.num_sampled]]

    def forward(self, query, targets):
        # query:   (B, D)  — hidden states from GPT2
        # targets: (B,)    — positive token ids
        # 1. Build negative candidate pool
        neg_ids_list = []
        if self.mode in ("local", "both"):
            neg_ids_list.append(self._sample_negatives(targets))
        if self.mode in ("in_batch", "both"):
            neg_ids_list.append(self._inbatch_ids(targets))

        neg_ids = torch.cat(neg_ids_list).unique()

        valid_mask = (targets != self.ignore_index)
        # 2. Gather embeddings
        pos_emb = self.output_layer(targets[valid_mask])   # (B, D)
        neg_emb = self.output_layer(neg_ids)   # (K, D)

        # 3. Compute logits
        pos_logits = (query[valid_mask] * pos_emb).sum(-1) / self.temperature          # (B,)
        neg_logits = (query[valid_mask] @ neg_emb.T) / self.temperature                # (B, K)
        neg_logits = torch.where(
            targets[valid_mask].unsqueeze(1) == neg_ids.unsqueeze(0),
            -5e4,
            neg_logits,
        )
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)   # (B, 1+K)

        # 4. Positive is always index 0
        loss = -F.log_softmax(logits, dim=1)[:, 0]
        return loss.mean()


class SASRec(AbstractModel):
    """
    SASRec model from Wang and McAuley, "Self-Attentive Sequential Recommendation." ICDM 2018.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        gpt2 (GPT2LMHeadModel): The GPT-2 model used for the SASRec model.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(SASRec, self).__init__(config, dataset, tokenizer)

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )
        self.loss_type = config.get("loss_type", "ce")
        if self.loss_type == "ce":
            self.gpt2 = GPT2LMHeadModel(gpt2config)
            self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
            self.forward = self.forward_ce
            self.generate = self.generate_ce
        elif self.loss_type == "sampled_softmax":
            self.gpt2 = GPT2Model(gpt2config)
            self.loss_fct = SampledSoftmaxLoss(
                emb_layer=self.gpt2.wte,
                num_sampled=config.get("loss_n_negatives", 1024),
                mode=config.get("loss_neg_sampler", "local"),       # 'local' | 'in_batch' | 'both'
                temperature=config.get("loss_temperature", 0.07),
                ignore_index=tokenizer.ignored_label
            )
            self.forward = self.forward_ss
            self.generate = self.generate_ss
        elif self.loss_type == "basic":
            self.gpt2 = GPT2Model(gpt2config)
            self.max_item_token = max(tokenizer.item2tokens.values())
            self.forward = self.forward_basic
            self.generate = self.generate_basic


    @property
    def n_parameters(self) -> str:
        """
        Get the number of parameters in the model.

        Returns:
            str: A string representation of the number of parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward_ce(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the logits and the loss.

        Args:
            batch (dict): The input batch.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        outputs = self.gpt2(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        logits = outputs.logits.view(-1, outputs.logits.shape[-1])
        labels = batch['labels'].view(-1)
        outputs.loss = self.loss_fct(logits, labels)
        return outputs
    
    def forward_basic(self, batch: dict) -> SimpleNamespace:
        """Forward pass for the original SASRec sampled BCE objective."""
        outputs = self.gpt2(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        return SimpleNamespace(
            loss=self._basic_sasrec_loss(outputs.last_hidden_state, batch['labels']),
            logits=None,
        )

    def _sample_negative_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Sample one negative item ID for each target position.

        Original SASRec uses randomly sampled negative items for the basic binary loss.
        Padding and the EOS token are excluded from the sampling range, and sampled items
        are shifted out of the positive-item slot at each valid position.
        """
        if self.max_item_token <= 1:
            return torch.ones_like(labels)

        valid_mask = labels.ne(self.tokenizer.ignored_label)
        positive_labels = labels.clamp(min=1, max=self.max_item_token)
        negative_labels = torch.randint(
            low=1,
            high=self.max_item_token,
            size=labels.shape,
            device=labels.device,
        )
        negative_labels = negative_labels + negative_labels.ge(positive_labels).long()
        return negative_labels.masked_fill(~valid_mask, 1)

    def _basic_sasrec_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the SASRec basic sampled binary cross-entropy loss.

        For every non-padding target, the sequence representation is dotted with the
        positive item embedding and one sampled negative item embedding. The objective is
        ``-log(sigmoid(pos)) - log(1 - sigmoid(neg))``, implemented as BCE with logits.
        """
        valid_mask = labels.ne(self.tokenizer.ignored_label)
        if not valid_mask.any():
            return hidden_states.new_zeros(())

        positive_labels = labels.masked_fill(~valid_mask, 0)
        negative_labels = self._sample_negative_labels(labels)

        item_embedding = self.gpt2.get_input_embeddings().weight
        positive_embeddings = item_embedding[positive_labels]
        negative_embeddings = item_embedding[negative_labels]

        positive_logits = (hidden_states * positive_embeddings).sum(dim=-1)
        negative_logits = (hidden_states * negative_embeddings).sum(dim=-1)

        positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_logits[valid_mask],
            torch.ones_like(positive_logits[valid_mask]),
        )
        negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            negative_logits[valid_mask],
            torch.zeros_like(negative_logits[valid_mask]),
        )
        return positive_loss + negative_loss

    def forward_ss(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the logits and the loss.

        Args:
            batch (dict): The input batch.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        outputs = self.gpt2(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        preds = outputs.last_hidden_state.view(-1, outputs.last_hidden_state.shape[-1])
        labels = batch['labels'].view(-1)
        outputs.loss = self.loss_fct(preds, labels)
        return outputs

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

    def generate_ce(self, batch, n_return_sequences=1):
        """
        Generate sequences based on the input batch.

        Args:
            batch: The input batch.
            n_return_sequences (int): The number of sequences to generate.

        Returns:
            torch.Tensor: The generated sequences.
        """
        outputs = self.gpt2(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = self.gather_index(outputs.logits, batch['seq_lens'] - 1)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return {"preds": preds.unsqueeze(-1)}
    
    def generate_ss(self, batch, n_return_sequences=1):
        """
        Generate sequences based on the input batch.

        Args:
            batch: The input batch.
            n_return_sequences (int): The number of sequences to generate.

        Returns:
            torch.Tensor: The generated sequences.
        """
        outputs = self.gpt2(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        idx = (batch['seq_lens'] - 1).clamp(min=0)          # [B]
        pred = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.size(0)), idx]
        logits = pred @ self.gpt2.wte.weight.T
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return {"preds": preds.unsqueeze(-1)}

    def _topk_generation_output(self, logits: torch.Tensor, n_return_sequences: int):
        """Return top-k item predictions in the sequence shape expected by evaluators."""
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return {"preds":  preds.unsqueeze(-1)} 

    def generate_basic(self, batch, n_return_sequences=1):
        """Generate from the basic-loss GPT2Model backbone using embedding-dot logits."""
        outputs = self.gpt2(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        final_hidden = self.gather_index(outputs.last_hidden_state, batch['seq_lens'] - 1)
        logits = torch.matmul(final_hidden, self.gpt2.get_input_embeddings().weight.t())
        return self._topk_generation_output(logits, n_return_sequences)
