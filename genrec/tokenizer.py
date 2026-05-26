from collections import defaultdict
import json
from logging import getLogger
import os
from typing import Dict, List

import numpy as np

from genrec.dataset import AbstractDataset


def add_dedup_codebook(item2sem_ids: Dict[str, List[int]]) -> Dict[str, List[int]]:
    raise NotImplementedError()


class AbstractTokenizer:
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.config = config
        self.logger = getLogger()
        self.eos_token = None
        self.collate_fn = {'train': None, 'val': None, 'test': None}

    def _init_tokenizer(self, dataset: AbstractDataset):
        raise NotImplementedError('Tokenizer initialization not implemented.')

    def tokenize(self, datasets):
        raise NotImplementedError('Tokenization not implemented.')

    @property
    def vocab_size(self):
        raise NotImplementedError('Vocabulary size not implemented.')

    @property
    def padding_token(self):
        return 0

    @property
    def max_token_seq_len(self):
        raise NotImplementedError('Maximum token sequence length not implemented.')

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)


class BaseSIDTokenizer(AbstractTokenizer):
    """
    Base SID Tokenizer for TIGER-like models.

    An example when "rq_codebook_size == 256, rq_n_codebooks == 3, n_user_tokens == 2000":
        0: padding
        1-256: digit 1
        257-512: digit 2
        513-768: digit 3
        769-1024: digit 4 (used to avoid conflicts)
        1025-3024: user tokens
        3025: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their semantic IDs.
        base_user_id (int): The base user ID.
        n_user_tokens (int): The number of user tokens.
        eos_token (int): The end-of-sequence token.
    """
    CB_SIZE_KEY = "rq_codebook_size"
    N_CB_KEY = "rq_n_codebooks"

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(BaseSIDTokenizer, self).__init__(config, dataset)

        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2id = dataset.item2id
        
        self.dedup_on_creation = config.get("deduplicate_on_creation", False)
        self.dedup_on_load = config.get("deduplicate_on_load", False)
        self.n_extra_codebooks = self.dedup_on_creation or self.dedup_on_load
        self.sem_ids_path = self.config.get('sem_ids_path')
        if self.sem_ids_path is None:
            sem_ids_file = self.config.get('sem_ids_file') or self._get_index_factory()
            self.sem_ids_path = os.path.join(dataset.cache_dir, 'processed', sem_ids_file)
        
        self.share_n_codebooks = max(self.config.get("share_n_codebooks", 1), 1)
        self.item2tokens = self._init_tokenizer(dataset)
        self.n_user_tokens = self.config.get('n_user_tokens', 0)
        if self.n_user_tokens:
            self.base_user_token = sum(self.codebook_sizes[self.share_n_codebooks-1:]) + 1
            self.eos_token = self.base_user_token + self.n_user_tokens
        else:
            self.eos_token = sum(self.codebook_sizes[self.share_n_codebooks-1:]) + 1

        self.input_ids_col = "item"
        self.seq_cols = [self.input_ids_col]
        if "aux_seq_columns" in config and config["aux_seq_columns"] is not None:
            self.seq_cols.extend(config["aux_seq_columns"].split(","))

    def _get_index_factory(self) -> str:
        raise NotImplementedError("Index Factory not implemented for this class")

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _generate_atomic_ids(self, dataset: AbstractDataset) -> dict:
        """
        Generates atomic semantic IDs where each item gets its own unique ID.
        Item with item_id=i gets semantic ID [i].

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to their atomic semantic IDs (single digit).
        """
        item2sem_ids = {}
        for item_id in range(1, dataset.n_items):
            item = dataset.id_mapping['id2item'][item_id]
            # Each item gets its own item_id as the semantic ID
            item2sem_ids[item] = (item_id,)
        self.log(f'[TOKENIZER] Generated atomic semantic IDs for {len(item2sem_ids)} items')
        with open(self.sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)
        return item2sem_ids

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """
        Converts semantic IDs to tokens.

        Args:
            item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """
        sem_id_offsets = [0]
        for digit in range(1, self.n_digit):
            offset = sem_id_offsets[-1]
            if digit >= self.share_n_codebooks:
                offset += self.codebook_sizes[digit - 1]
            sem_id_offsets.append(offset)
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += sem_id_offsets[digit] + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids
    
    def _generate_semantic_ids(self, sem_ids_path: str, dataset: AbstractDataset):
        raise NotImplementedError("Semantic ID generation not implemented for this class!")

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        # Handle atomic IDs case
        if self.use_atomic_ids:
            self.log('[TOKENIZER] Using atomic semantic IDs (naive mode)')
            if not os.path.exists(self.sem_ids_path):
                item2sem_ids = self._generate_atomic_ids(dataset)
            else:
                item2sem_ids = json.load(open(self.sem_ids_path, 'r'))
            item2tokens = self._sem_ids_to_tokens(item2sem_ids)
            return item2tokens

        # Load semantic IDs
        sem_ids_path = self.sem_ids_path

        if not os.path.exists(sem_ids_path):
            self._generate_semantic_ids(sem_ids_path, dataset)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        if self.dedup_on_load:
            item2sem_ids = add_dedup_codebook(item2sem_ids)

        # Verify all SIDs fit in codebooks
        assert ((np.array(list(item2sem_ids.values())) - np.array(self.codebook_sizes)[None, :]) < 0).all()
        
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    @property
    def use_atomic_ids(self):
        """
        Returns True if using atomic IDs (naive semantic IDs where each item gets a unique ID).
        This is enabled when n_codebooks = 1 and codebook_size = -1.
        """
        return (self.config[self.N_CB_KEY] == 1 and 
                self.config[self.CB_SIZE_KEY] == -1)

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        The number of digits is determined by the value of `rq_n_codebooks` in the configuration.
        For atomic IDs, returns 1 (no conflict resolution needed).
        """
        if self.use_atomic_ids:
            return 1
        return self.config[self.N_CB_KEY] + self.n_extra_codebooks

    @property
    def codebook_sizes(self):
        """
        Returns the codebook size for the TIGER tokenizer.

        If `rq_codebook_size` is a list, it returns the list as is.
        If `rq_codebook_size` is -1 (atomic IDs), returns [n_items].
        If `rq_codebook_size` is an integer, it returns a list with `n_digit` elements,
        where each element is equal to `rq_codebook_size`.

        Returns:
            list: The codebook size for the TIGER tokenizer.
        """
        if isinstance(self.config[self.CB_SIZE_KEY], (list, tuple)):
            return self.config[self.CB_SIZE_KEY]
        elif self.config[self.CB_SIZE_KEY] == -1:
            # Atomic IDs: codebook size equals number of items
            n_items = len(self.id2item)
            return [n_items]
        else:
            return [self.config[self.CB_SIZE_KEY]] * self.n_digit

    def _token_single_user(self, user: str) -> int:
        """
        Tokenizes a single user.

        Args:
            user (str): The user to tokenize.

        Returns:
            int: The tokenized user ID.

        """
        user_id = self.user2id[user]
        return self.base_user_token + user_id % self.n_user_tokens

    def tokenize_function(self, examples: dict, split: str) -> dict:
        raise NotImplementedError("tokenizer_function is not implemented in this class!")

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
        """
        BATCH_SIZE = 128
        tokenized_datasets = {}
        for split in datasets:
            # Add idx only for val/test (1:1 mapping, needed for fine-grained eval)
            if split in ['val', 'test']:
                tokenized_datasets[split] = datasets[split].map(
                    lambda t, idx: {**self.tokenize_function(t, split), 'idx': idx},
                    batched=True,
                    batch_size=BATCH_SIZE,
                    with_indices=True,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config['eval_num_proc'],
                    desc=f'Tokenizing {split} set: '
                )
            else:
                tokenized_datasets[split] = datasets[split].map(
                    lambda t: self.tokenize_function(t, split),
                    batched=True,
                    batch_size=BATCH_SIZE,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config['num_proc'],
                    desc=f'Tokenizing {split} set: '
                )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the SID tokenizer.
        """
        return self.eos_token + 1


class EncoderDecoderTokenizerMixin:
    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length for the SID tokenizer.
        """
        # +2 for user token and eos token
        return self.config['max_item_seq_len'] * self.n_digit + 1 + (self.n_user_tokens > 0)
    
    def _token_single_item(self, item: str) -> int:
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return self.item2tokens[item]

    def _tokenize_once(self, example: dict) -> tuple:
        """
        Tokenizes a single example.

        Args:
            example (dict): A dictionary containing the example data.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, and labels.
        """
        max_item_seq_len = self.config['max_item_seq_len']

        # input_ids
        input_ids = []
        if self.n_user_tokens > 0:
            input_ids.append(self._token_single_user(example['user']))
            
        for item in example['item_seq'][:-1][-max_item_seq_len:]:
            input_ids.extend(self._token_single_item(item))
        input_ids.append(self.eos_token)
        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

        # attention_mask
        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)
        attention_mask = [1] * (self.n_digit * item_seq_len + 2)
        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

        # labels
        labels = list(self._token_single_item(example['item_seq'][-1])) + [self.eos_token]

        return input_ids, attention_mask, labels

    def tokenize_function(self, examples: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.
        """
        # Initialize lists to store results for the whole batch
        all_input_ids, all_attention_mask, all_labels = [], [], []
        
        # FIX: Iterate by index over the lists inside the dictionary
        batch_size = len(examples['user'])
        
        for i in range(batch_size):
            user = examples['user'][i]
            item_seq = examples['item_seq'][i]

            if split == 'train':
                # Generate multiple training samples from one sequence
                n_return_examples = len(item_seq) - 1
                for j in range(n_return_examples):
                    cur_example = {
                        'user': user,
                        'item_seq': item_seq[:j+2] # Context + Next Target
                    }
                    input_ids, attention_mask, labels = self._tokenize_once(cur_example)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_labels.append(labels)
            else:
                # Validation/Test: Just one sample per sequence
                cur_example = {
                    'user': user, 
                    'item_seq': item_seq
                }
                input_ids, attention_mask, labels = self._tokenize_once(cur_example)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
               
        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_labels
        }


class DecoderTokenizerMixin:
    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length for the SID tokenizer.
        """
        # +2 for user token and eos token
        return (self.config['max_item_seq_len'] + 1) * self.n_digit - 1 + (self.n_user_tokens > 0)

    def _token_single_item(self, item: str) -> list:
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return list(self.item2tokens[item])

    def _tokenize_first_n_items(self, seqs: Dict[str, List]) -> tuple:
        """
        Tokenizes the first n items in the given item_seq.
        The losses for the first n items can be computed by only forwarding once.

        Args:
            item_seq (list): The item sequence that contains the first n items.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
        """
        nested_sids = [self._token_single_item(item) for item in seqs[self.input_ids_col]]
        for c, seq in seqs.items():
            if c != self.input_ids_col:
                seqs[c] = [v for v, sid in zip(seq, nested_sids) for _ in range(len(sid))]
        sids = sum(nested_sids, start=[])
        seqs[self.input_ids_col] = sids[:]
        seqs = {c: seq[:-1] for c, seq in seqs.items()}
        seq_lens = len(seqs[self.input_ids_col])
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        padding = [0] * pad_lens
        for _, seq in seqs.items():
            seq.extend(padding)
        attention_mask.extend(padding)

        labels = sids[1:]
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
        sids = [self._token_single_item(item) for item in seqs[self.input_ids_col]]
        token_sizes = list(map(len, sids))
        for c, seq in seqs.items():
            if c != self.input_ids_col:
                seqs[c] = [v for v, sz in zip(seq, token_sizes) for _ in range(sz)]
        sids = sum(sids, start=[])
        seqs[self.input_ids_col] = sids[:]
        seqs = {c: seq[:-1] for c, seq in seqs.items()}
        seq_lens = len(seqs[self.input_ids_col])
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * (seq_lens - token_sizes[-1])
        labels += sids[-token_sizes[-1]:]

        pad_lens = self.max_token_seq_len - seq_lens
        padding = [0] * pad_lens
        for _, seq in seqs.items():
            seq.extend(padding)
        attention_mask.extend(padding)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return seqs, attention_mask, labels, seq_lens
    
    def _tokenize_evaluation(self, seqs: Dict[str, List]) -> tuple:
        """
        Tokenizes the later items in the item sequence, except the last one going to the labels fully.

        Args:
            item_seq (list): The item sequence.

        Returns:
            tuple: A tuple containing the tokenized input IDs, attention mask, labels, and seq_lens.
        """
        sids = [self._token_single_item(item) for item in seqs[self.input_ids_col]]
        token_sizes = list(map(len, sids[:-1]))
        seqs[self.input_ids_col] = sum(sids[:-1], start=[])
        for c, seq in seqs.items():
            if c != self.input_ids_col:
                seqs[c] = [v for v, sz in zip(seq[:-1], token_sizes) for _ in range(sz)]
        seq_lens = len(seqs[self.input_ids_col])
        attention_mask = [1] * seq_lens
        # TODO: Check whether I need to provide the full sequence prior to actual labels
        labels = sids[-1]

        # Padding left for evaluation since we will be doing beam search
        pad_lens = self.max_token_seq_len - self.n_digit + 1 - seq_lens
        padding = [0] * pad_lens
        for c in seqs:
            seqs[c] = padding + seqs[c]
        attention_mask = padding + attention_mask

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
                out_seqs, attention_mask, labels, seq_lens = self._tokenize_evaluation(
                    seqs={c: seq[-(max_item_seq_len+1):] for c, seq in seqs.items()},
                )
                for c, s in out_seqs.items():
                    all_seqs[c].append(s)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

        all_input_ids = all_seqs.pop(self.input_ids_col)
        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_labels,
            'seq_lens': all_seq_lens,
            **all_seqs
        }