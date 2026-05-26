import json
import os
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from main_genrec import parse_args
from genrec.dataset import AbstractDataset
from genrec.utils import get_config, get_dataset, init_device, parse_command_line_args


def encode_sent_emb(dataset: AbstractDataset, output_path: str, config: dict):
    """
    Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

    Args:
        dataset (AbstractDataset): The dataset containing the sentences to encode.
        output_path (str): The path to save the encoded sentence embeddings.

    Returns:
        numpy.ndarray: The encoded sentence embeddings.
    """
    assert config["metadata"] == "sentence", "TIGERTokenizer only supports sentence metadata."

    sent_emb_model = SentenceTransformer(
        config["sent_emb_model"],
    ).to(config["device"])

    if "sent_emb_max_length" in config:
        sent_emb_model.max_seq_length = config["sent_emb_max_length"]

    meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1
    for i in range(1, dataset.n_items):
        meta_sentences.append(dataset.item2meta[dataset.id_mapping["id2item"][i]])
    sent_embs = sent_emb_model.encode(
        meta_sentences,
        convert_to_numpy=True,
        batch_size=config["sent_emb_batch_size"],
        show_progress_bar=True,
        device=config["device"],
    )

    sent_embs.tofile(output_path)
    return sent_embs


def prepare_letter_format(
    dataset: str, category: Optional[str], embedders: List[Tuple[str, int]], letter_path: Path, add_time: bool = False
):
    if category is None:
        gr_path = Path(f"cache/{dataset}/processed")
        dataset_id = dataset
    else:
        gr_path = Path(f"cache/{dataset}/{category}/processed")
        dataset_id = category
    print(f"Mapping {dataset_id} from GR to LETTER...")
    letter_path.mkdir(parents=True, exist_ok=True)

    # Read GR files
    with open(gr_path / "all_seqs.json", "r") as f:
        item_seqs = json.load(f)
    id_mapping = pd.read_json(gr_path / "id_mapping.json", lines=True).T[0]
    # semantic_data = pd.read_json(gr_path / "metadata.sentence.json", lines=True).T[0]

    # Transform to LETTER format:
    #  - <dataset>.index.json (json from raw ID to SID - str: [str (a_i), str (b_j), ...]) -> To be computed with LETTER
    #  - <dataset>.item.json (json from raw ID to Item title and description) -> No needed with semantic embeddings

    # Mappings to allow bidirectional transform GR->LETTER and LETTER->GR
    item_raw2id = {k: v - 1 for k, v in id_mapping["item2id"].items() if v > 0}  # GR reserves the [PAD] token as zero
    user_raw2id = {k: v - 1 for k, v in id_mapping["user2id"].items() if v > 0}  # GR reserves the [PAD] token as zero
    with open(letter_path / "item_raw2id.mapping.json", "w") as f:
        json.dump(item_raw2id, f)

    with open(letter_path / "user_raw2id.mapping.json", "w") as f:
        json.dump(user_raw2id, f)

    #  - <dataset>.inter.json (json from user ID to Item raw sequence)
    if add_time:
        letter_inter = {
            str(user_raw2id[k]): [(item_raw2id[it], ti) for it, ti in zip(v["item"], v["time"])]
            for k, v in item_seqs.items()
        }
        suffix = "-time"
    else:
        letter_inter = {str(user_raw2id[k]): [item_raw2id[vi] for vi in v["item"]] for k, v in item_seqs.items()}
        suffix = ""
    with open(letter_path / f"{dataset_id}{suffix}.inter.json", "w") as f:
        json.dump(letter_inter, f)

    # Item embeddings as numpy array
    for embedder, emb_dim in embedders:
        out_fname = dataset_id + ".emb-" + embedder + "-td" + ".npy"
        sent_embs = np.fromfile(gr_path / f"{embedder}.sent_emb", dtype=np.float32).reshape(-1, emb_dim)
        np.save(letter_path / out_fname, sent_embs)

    print("Finished mapping GR dataset and semantic IDs to LETTER format!")


if __name__ == "__main__":
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    model_name = "TIGER"
    config = get_config(
        model_name=model_name, dataset_name=args.dataset, config_file=args.config_file, config_dict=command_line_configs
    )
    config["device"], _ = init_device()

    dataset = get_dataset(args.dataset)(config)
    split_datasets = dataset.split()

    sent_emb_path = os.path.join(
        dataset.cache_dir, "processed", f"{os.path.basename(config['sent_emb_model'])}.sent_emb"
    )
    if os.path.exists(sent_emb_path):
        print(f"[TOKENIZER] Sentence embeddings already exist at {sent_emb_path}...")
    else:
        print("[TOKENIZER] Encoding sentence embeddings...")
        encode_sent_emb(dataset, sent_emb_path, config)

    # Prepare LETTER-like dataset folder
    dataset_id = dataset if config.get("category") is None else config.get("category")
    prepare_letter_format(
        args.dataset,
        config.get("category"),
        [(os.path.basename(config["sent_emb_model"]), config["sent_emb_dim"])],
        Path(f"cache/gensid/{dataset_id}"),
    )

    print("Finished dataset preparation")
