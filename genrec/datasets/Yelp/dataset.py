import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Optional

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


# Column constants
U_COL = "user_id"
I_COL = "business_id"
DATE_COL = "date"
RAT_COL = "stars"

# Metadata fields for sentence extraction
META_FIELDS = ["name", "city", "state", "stars", "review_count", "is_open", "categories"]  # , "attributes", "hours"

# Prompt template adapted from SID literature
PROMPT_TEMPLATE = (
    "The point of interest has following attributes: \n"
    "name is {name}; category is {categories}; open status is {is_open}; "
    "review count is {review_count}; city is {city}; state is {state}; "
    "average score is {stars}."
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


class Yelp(AbstractDataset):
    """
    A class representing the Yelp Academic Dataset.

    Expects the raw JSON files to be placed manually in the cache directory
    (Yelp requires accepting terms of service before download):
        - yelp_academic_dataset_review.json
        - yelp_academic_dataset_business.json

    Args:
        config (dict): Configuration parameters. Expected keys:
            - cache_dir (str): Root cache directory.
            - metadata (str): One of 'none', 'raw', 'sentence'.
            - kcore (int, optional): K-core threshold for iterative filtering. Default: 5.

    Attributes:
        cache_dir (str): Directory path for caching the dataset.
        all_seqs (dict): User-item sequences after processing.
        id_mapping (dict): User/item ID mappings.
        item2meta (dict): Item metadata.
    """

    REVIEW_FILE = "yelp_academic_dataset_review.json"
    BUSINESS_FILE = "yelp_academic_dataset_business.json"

    def __init__(self, config: dict):
        super(Yelp, self).__init__(config)

        self.kcore = config.get('kcore', 5)
        self.log(f'[DATASET] Yelp Academic Dataset (k-core={self.kcore})')

        self.cache_dir = os.path.join(config['cache_dir'], 'Yelp')
        self._check_raw_files()
        self._download_and_process_raw()

    def _check_raw_files(self):
        """
        Verifies that the raw Yelp JSON files exist in the expected location.

        Raises:
            FileNotFoundError: If any required raw file is missing.
        """
        raw_path = os.path.join(self.cache_dir, 'raw')
        for fname in [self.REVIEW_FILE, self.BUSINESS_FILE]:
            fpath = os.path.join(raw_path, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Required Yelp file not found: {fpath}\n"
                    "Please download the Yelp Academic Dataset from "
                    "https://www.yelp.com/dataset and place the files in the raw directory."
                )

    def _load_reviews(self, path: str) -> list:
        """
        Load reviews from the Yelp review JSON file.

        Args:
            path (str): Path to yelp_academic_dataset_review.json.

        Returns:
            list: List of (user_id, business_id, timestamp, rating) tuples.
                  Timestamp is a Unix int derived from the 'date' field.
        """
        self.log('[DATASET] Loading reviews...')
        reviews = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                inter = json.loads(line)
                user = inter[U_COL]
                item = inter[I_COL]
                # Convert 'YYYY-MM-DD HH:MM:SS' to Unix timestamp (int)
                time = int(pd.Timestamp(inter[DATE_COL]).timestamp())
                rating = int(inter[RAT_COL])
                reviews.append((user, item, time, rating))
        return reviews

    def _apply_kcore(self, reviews: list) -> list:
        """
        Applies iterative k-core filtering to the reviews.

        Args:
            reviews (list): List of (user, item, time, rating) tuples.

        Returns:
            list: Filtered list of tuples after k-core pruning.
        """
        self.log(f'[DATASET] Applying iterative {self.kcore}-core filtering...')
        df = pd.DataFrame(reviews, columns=[U_COL, I_COL, DATE_COL, RAT_COL])
        self.log(
            f'[DATASET] Before filtering: {df[U_COL].nunique()} users, '
            f'{df[I_COL].nunique()} items, {len(df)} interactions'
        )
        df = _filtering_iterative(df, kcore=self.kcore)
        self.log(
            f'[DATASET] After filtering: {df[U_COL].nunique()} users, '
            f'{df[I_COL].nunique()} items, {len(df)} interactions'
        )
        return list(df.itertuples(index=False, name=None))

    def _get_item_seqs(self, reviews: list) -> dict:
        """
        Groups reviews by user and sorts by timestamp.

        Args:
            reviews (list): List of (user, item, time, rating) tuples.

        Returns:
            dict: {user: {'item_seq': [...], 'time_seq': [...], 'rating_seq': [...]}}
        """
        self.log('[DATASET] Grouping interactions by user...')
        seqs = {}
        for user, item, time, rating in tqdm(reviews):
            seqs.setdefault(user, []).append((item, time, rating))

        self.log('[DATASET] Sorting user sequences by time...')
        for user, triples in tqdm(seqs.items()):
            triples.sort(key=lambda x: x[1])
            seqs[user] = {
                "item_seq": [t[0] for t in triples],
                "time_seq": [t[1] for t in triples],
                "rating_seq": [t[2] for t in triples],
            }
        return seqs

    def _remap_ids(self, item_seqs: dict) -> tuple[dict, dict]:
        """
        Remaps raw user/item string IDs to contiguous integer IDs (1-indexed).
        ID 0 is reserved for padding [PAD].

        Args:
            item_seqs (dict): Output of _get_item_seqs.

        Returns:
            all_seqs (dict): {user: {'item': [...], 'time': [...], 'rating': [...]}}
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
                "rating": items["rating_seq"],
            }
        return self.all_seqs, self.id_mapping

    def _process_reviews(self, input_path: str, output_path: str) -> tuple[dict, dict]:
        """
        Full review processing pipeline: load → k-core filter → seq grouping → ID remap → save.

        Args:
            input_path (str): Path to yelp_academic_dataset_review.json.
            output_path (str): Directory to save processed files.

        Returns:
            all_seqs (dict), id_mapping (dict)
        """
        seq_file = os.path.join(output_path, 'all_seqs.json')
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        if all(os.path.exists(f) for f in [seq_file, id_mapping_file]):
            self.log('[DATASET] Reviews have been processed...')
            with open(seq_file, 'r') as f:
                all_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_seqs, id_mapping

        self.log('[DATASET] Processing reviews...')
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
        Loads business metadata, filtered to items present in item2id.

        Args:
            path (str): Path to yelp_academic_dataset_business.json.
            item2id (dict): Mapping from business_id to integer ID.

        Returns:
            dict: {business_id: raw_meta_dict}
        """
        self.log('[DATASET] Loading metadata...')
        data = {}
        item_ids = set(item2id.keys())
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                info = json.loads(line)
                if info[I_COL] not in item_ids:
                    continue
                data[info[I_COL]] = info
        return data

    def _extract_meta_sentences(self, metadata: dict) -> dict:
        """
        Generates a natural-language sentence for each business using PROMPT_TEMPLATE.

        Args:
            metadata (dict): {business_id: raw_meta_dict}

        Returns:
            dict: {business_id: meta_sentence_str}
        """
        self.log('[DATASET] Extracting meta sentences...')
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            sentence = PROMPT_TEMPLATE.format(
                name=clean_text(meta.get('name', 'N/A')),
                categories=clean_text(str(meta.get('categories', 'N/A'))),
                is_open='open' if meta.get('is_open', 0) == 1 else 'closed',
                review_count=meta.get('review_count', 'N/A'),
                city=clean_text(meta.get('city', 'N/A')),
                state=clean_text(meta.get('state', 'N/A')),
                stars=meta.get('stars', 'N/A'),
            )
            item2meta[item] = sentence
        return item2meta

    def _process_meta(self, input_path: str, output_path: str) -> Optional[dict]:
        """
        Processes business metadata according to config['metadata'] mode.

        Args:
            input_path (str): Path to yelp_academic_dataset_business.json.
            output_path (str): Directory to save processed metadata.

        Returns:
            dict or None: {business_id: metadata} or None if mode is 'none'.

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
        Orchestrates the full Yelp processing pipeline.
        Unlike Amazon, there is no automatic download — raw files must be present manually.
        """
        raw_data_path = os.path.join(self.cache_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        reviews_path = os.path.join(raw_data_path, self.REVIEW_FILE)
        business_path = os.path.join(raw_data_path, self.BUSINESS_FILE)

        with self.accelerator.main_process_first():
            self.all_seqs, self.id_mapping = self._process_reviews(
                input_path=reviews_path,
                output_path=processed_data_path,
            )

        self.item2meta = self._process_meta(
            input_path=business_path,
            output_path=processed_data_path,
        )