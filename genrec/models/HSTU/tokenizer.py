from collections import defaultdict
from typing import Dict, List
from genrec.dataset import AbstractDataset
from genrec.models.SASRec.tokenizer import SASRecTokenizer


class HSTUTokenizer(SASRecTokenizer):
    """
    Tokenizer for SASRec model.

    An example:
        0: padding
        1-n_items: item tokens
        n_items+1: eos token

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their internal IDs.
        eos_token (int): The end-of-sequence token.
        ignored_label (int): Should be -100. Used to ignore the loss for padding tokens in `transformers`.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(HSTUTokenizer, self).__init__(config, dataset)
        self.input_ids_col = "item"
        self.seq_cols = [self.input_ids_col]
        if "aux_seq_columns" in config and config["aux_seq_columns"] is not None:
            self.seq_cols.extend(config["aux_seq_columns"].split(","))

    def _tokenize_first_n_items(self, seqs: Dict[str, List]) -> tuple:
        """
        Tokenizes the first n items in the given item_seq.
        The losses for the first n items can be computed by only forwarding once.

        Args:
            item_seq (list): The item sequence that contains the first n items.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
        """
        item_seq = seqs[self.input_ids_col]
        seqs = {c: seq[:-1] for c, seq in seqs.items()}
        seqs[self.input_ids_col] = [self.item2tokens[item] for item in seqs[self.input_ids_col]]
        seq_lens = len(seqs[self.input_ids_col])
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        padding = [0] * pad_lens
        for _, seq in seqs.items():
            seq.extend(padding)
        attention_mask.extend(padding)

        labels = [self.item2tokens[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)

        return seqs, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, seqs: Dict[str, List], pad_labels: bool = True) -> tuple:
        """
        Tokenizes the later items in the item sequence.
        Only the last one items are used as the target item.

        Args:
            item_seq (list): The item sequence.

        Returns:
            tuple: A tuple containing the tokenized input IDs, attention mask, labels, and seq_lens.
        """
        item_seq = seqs[self.input_ids_col]
        seqs = {c: seq[:-1] for c, seq in seqs.items()}
        seqs[self.input_ids_col] = [self.item2tokens[item] for item in seqs[self.input_ids_col]]
        seq_lens = len(seqs[self.input_ids_col])
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2tokens[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        padding = [0] * pad_lens
        for _, seq in seqs.items():
            seq.extend(padding)
        attention_mask.extend(padding)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return seqs, attention_mask, labels, seq_lens

    def tokenize_function(self, examples: dict, split: str) -> dict:
        """
        Tokenizes a batch of input examples based on the specified split.
        """
        # Initialize lists to store results for the whole batch
        all_seqs, all_attention_mask, all_labels, all_seq_lens = defaultdict(list), [], [], []
        
        max_item_seq_len = self.config['max_item_seq_len']
        batch_size = len(examples['item_seq'])

        for i in range(batch_size):
            seqs = {c: examples[f'{c}_seq'][i] for c in self.seq_cols}

            if split == 'train':
                # Logic: Generate multiple training samples from one user sequence
                n_return_examples = max(len(seqs[self.input_ids_col]) - max_item_seq_len, 1)

                # 1. Tokenize the first window (start of sequence)
                # Add 1 as the target item is not included in the input sequence
                out_seqs, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                    seqs={c: seq[:min(len(seq), max_item_seq_len + 1)] for c, seq in seqs.items()}
                )
                for c, s in out_seqs.items():
                    all_seqs[c].append(s)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

                # 2. Tokenize the sliding windows (later items)
                for j in range(1, n_return_examples):
                    out_seqs, attention_mask, labels, seq_lens = self._tokenize_later_items(
                        {c: seq[j : j + max_item_seq_len + 1] for c, seq in seqs.items()}
                    )
                    for c, s in out_seqs.items():
                        all_seqs[c].append(s)
                    all_attention_mask.append(attention_mask)
                    all_labels.append(labels)
                    all_seq_lens.append(seq_lens)

            else:
                # Logic: Validation/Test (Only the last window)
                out_seqs, attention_mask, labels, seq_lens = self._tokenize_later_items(
                    seqs={c: seq[-(max_item_seq_len+1):] for c, seq in seqs.items()},
                    pad_labels=False
                )
                for c, s in out_seqs.items():
                    all_seqs[c].append(s)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels[-1:]) # Keep as list for consistency
                all_seq_lens.append(seq_lens)

        all_input_ids = all_seqs.pop(self.input_ids_col)
        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_labels,
            'seq_lens': all_seq_lens,
            **all_seqs
        }