import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Optional

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


U_COL = "user_id"
I_COL = "news_id"
TIME_COL = "time"
RAT_COL = None  # MIND has no explicit ratings

BEHAVIORS_FILE = "merged_behaviors.tsv"
NEWS_FILE = "merged_news.tsv"

BEHAVIORS_COLS = ["impression_id", U_COL, TIME_COL, I_COL]
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]

PROMPT_TEMPLATE = (
    "News article properties: "
    "category is {category}; subcategory is {subcategory}; "
    "title is {title}; abstract is {abstract}."
)


def _kcore_filter(df: pd.DataFrame, cols: list[str], kcore: int = 5) -> pd.DataFrame:
    mask = df[cols[0]].map(df[cols[0]].value_counts().ge(kcore))
    for col in cols[1:]:
        mask &= df[col].map(df[col].value_counts().ge(kcore))
    return df[mask]


def _filtering_iterative(df: pd.DataFrame, kcore: int = 5) -> pd.DataFrame:
    old_size, new_size = -1, df.shape[0]
    while old_size != new_size:
        df = _kcore_filter(df, cols=[U_COL], kcore=kcore)
        df = _kcore_filter(df, cols=[I_COL], kcore=kcore)
        old_size = new_size
        new_size = df.shape[0]
    return df


class MIND(AbstractDataset):
    """
    A class representing the MIND (Microsoft News Dataset).

    Expects the raw TSV files to be placed manually in the cache directory
    (MIND requires accepting Microsoft's terms of service before download):
        - merged_behaviors.tsv
        - merged_news.tsv

    Args:
        config (dict): Configuration parameters. Expected keys:
            - cache_dir (str): Root cache directory.
            - metadata (str): One of 'none', 'raw', 'sentence'.
            - kcore (int, optional): K-core threshold. Default: 5.

    Attributes:
        cache_dir (str): Directory path for caching the dataset.
        all_seqs (dict): User-item sequences after processing.
        id_mapping (dict): User/item ID mappings.
        item2meta (dict): Item metadata.
    """

    def __init__(self, config: dict):
        super(MIND, self).__init__(config)

        self.kcore = config.get('kcore', 5)
        self.log(f'[DATASET] MIND Dataset (k-core={self.kcore})')

        self.cache_dir = os.path.join(config['cache_dir'], 'MIND')
        self._check_raw_files()
        self._download_and_process_raw()

    def _check_raw_files(self):
        """
        Verifies that the raw MIND TSV files exist in the expected location.

        Raises:
            FileNotFoundError: If any required raw file is missing.
        """
        raw_path = os.path.join(self.cache_dir, 'raw')
        for fname in [BEHAVIORS_FILE, NEWS_FILE]:
            fpath = os.path.join(raw_path, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Required MIND file not found: {fpath}\n"
                    "Please download the MIND dataset from "
                    "https://msnews.github.io and place the merged files in the raw directory."
                )

    def _load_reviews(self, path: str) -> list:
        """
        Loads interactions from merged_behaviors.tsv.

        Each row is a single user-item click interaction. The 'time' column
        uses format 'YYYY-MM-DD HH:MM:SS' and is converted to a Unix timestamp.

        Args:
            path (str): Path to merged_behaviors.tsv.

        Returns:
            list: List of (user_id, news_id, unix_timestamp) tuples.
                  No rating field — MIND encodes implicit feedback (clicks) only.
        """
        self.log('[DATASET] Loading interactions...')
        df = pd.read_csv(path, sep='\t', header=0)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL]).astype('int64') // 10**9
        reviews = list(df[[U_COL, I_COL, TIME_COL]].itertuples(index=False, name=None))
        return reviews

    def _apply_kcore(self, reviews: list) -> list:
        """
        Applies iterative k-core filtering to the interactions.

        Although MIND is typically pre-filtered, this ensures the k-core
        constraint holds for the merged split being used.

        Args:
            reviews (list): List of (user_id, news_id, timestamp) tuples.

        Returns:
            list: Filtered list of tuples.
        """
        self.log(f'[DATASET] Applying iterative {self.kcore}-core filtering...')
        df = pd.DataFrame(reviews, columns=[U_COL, I_COL, TIME_COL])
        df = _filtering_iterative(df, kcore=self.kcore)
        self.log(
            f'[DATASET] After filtering: {df[U_COL].nunique()} users, '
            f'{df[I_COL].nunique()} items, {len(df)} interactions'
        )
        return list(df.itertuples(index=False, name=None))

    def _get_item_seqs(self, reviews: list) -> dict:
        """
        Groups interactions by user and sorts by timestamp.

        Args:
            reviews (list): List of (user_id, news_id, timestamp) tuples.

        Returns:
            dict: {user_id: {'item_seq': [...], 'time_seq': [...]}}
                  No rating_seq — MIND is implicit feedback only.
        """
        self.log('[DATASET] Grouping interactions by user...')
        seqs = {}
        for user, item, time in tqdm(reviews):
            seqs.setdefault(user, []).append((item, time))

        self.log('[DATASET] Sorting user sequences by time...')
        for user, pairs in tqdm(seqs.items()):
            pairs.sort(key=lambda x: x[1])
            seqs[user] = {
                "item_seq": [p[0] for p in pairs],
                "time_seq": [p[1] for p in pairs],
            }
        return seqs

    def _remap_ids(self, item_seqs: dict) -> tuple[dict, dict]:
        """
        Remaps raw user/item string IDs to contiguous integer IDs (1-indexed).
        ID 0 is reserved for padding [PAD].

        Args:
            item_seqs (dict): Output of _get_item_seqs.

        Returns:
            all_seqs (dict): {user: {'item': [...], 'time': [...]}}
            id_mapping (dict): {user2id, item2id, id2user, id2item}
        """
        self.log('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)

            item_seq = items.pop("item_seq")
            for item in item_seq:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)

            self.all_seqs[user] = {
                "item": item_seq,
                "time": items["time_seq"],
            }
        return self.all_seqs, self.id_mapping

    def _process_reviews(self, input_path: str, output_path: str) -> tuple[dict, dict]:
        """
        Full interaction processing pipeline: load → k-core filter → seq grouping → ID remap → save.

        Args:
            input_path (str): Path to merged_behaviors.tsv.
            output_path (str): Directory to save processed files.

        Returns:
            all_seqs (dict), id_mapping (dict)
        """
        seq_file = os.path.join(output_path, 'all_seqs.json')
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        if all(os.path.exists(f) for f in [seq_file, id_mapping_file]):
            self.log('[DATASET] Interactions have been processed...')
            with open(seq_file, 'r') as f:
                all_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_seqs, id_mapping

        self.log('[DATASET] Processing interactions...')
        reviews = self._load_reviews(input_path)
        reviews = self._apply_kcore(reviews)
        seqs = self._get_item_seqs(reviews)
        all_seqs, id_mapping = self._remap_ids(seqs)

        self.log('[DATASET] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_seqs, f)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)
        return all_seqs, id_mapping

    def _load_metadata(self, path: str, item2id: dict) -> dict:
        """
        Loads news metadata from merged_news.tsv, filtered to items in item2id.

        Args:
            path (str): Path to merged_news.tsv.
            item2id (dict): Mapping from news_id to integer ID.

        Returns:
            dict: {news_id: raw_row_dict}
        """
        self.log('[DATASET] Loading metadata...')
        df = pd.read_csv(path, sep='\t', header=0)
        item_ids = set(item2id.keys())
        df = df[df['news_id'].isin(item_ids)]
        return df.set_index('news_id').to_dict(orient='index')

    def _extract_meta_sentences(self, metadata: dict) -> dict:
        """
        Generates a natural-language sentence for each news article using PROMPT_TEMPLATE.
        Uses category, subcategory, title, and abstract only.

        Args:
            metadata (dict): {news_id: row_dict}

        Returns:
            dict: {news_id: meta_sentence_str}
        """
        self.log('[DATASET] Extracting meta sentences...')
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            sentence = PROMPT_TEMPLATE.format(
                category=clean_text(str(meta.get('category', 'N/A'))),
                subcategory=clean_text(str(meta.get('subcategory', 'N/A'))),
                title=clean_text(str(meta.get('title', 'N/A'))),
                abstract=clean_text(str(meta.get('abstract', 'N/A'))),
            )
            item2meta[item] = sentence
        return item2meta

    def _process_meta(self, input_path: str, output_path: str) -> Optional[dict]:
        """
        Processes news metadata according to config['metadata'] mode.

        Args:
            input_path (str): Path to merged_news.tsv.
            output_path (str): Directory to save processed metadata.

        Returns:
            dict or None: {news_id: metadata} or None if mode is 'none'.

        Raises:
            NotImplementedError: If metadata mode is unrecognized.
        """
        process_mode = self.config['metadata']
        meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')
        if os.path.exists(meta_file):
            self.log('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)

        self.log(f'[DATASET] Processing metadata, mode: {process_mode}')

        if process_mode == 'none':
            return None

        item2meta = self._load_metadata(path=input_path, item2id=self.item2id)

        if process_mode == 'raw':
            pass
        elif process_mode == 'sentence':
            item2meta = self._extract_meta_sentences(metadata=item2meta)
        else:
            raise NotImplementedError(f'Metadata processing mode "{process_mode}" not implemented.')

        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        return item2meta

    def _download_and_process_raw(self):
        """
        Orchestrates the full MIND processing pipeline.
        No automatic download — raw files must be present manually.
        """
        raw_data_path = os.path.join(self.cache_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        behaviors_path = os.path.join(raw_data_path, BEHAVIORS_FILE)
        news_path = os.path.join(raw_data_path, NEWS_FILE)

        with self.accelerator.main_process_first():
            self.all_seqs, self.id_mapping = self._process_reviews(
                input_path=behaviors_path,
                output_path=processed_data_path,
            )

        self.item2meta = self._process_meta(
            input_path=news_path,
            output_path=processed_data_path,
        )