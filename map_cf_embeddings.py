import argparse
import re
from pathlib import Path

import torch


DATASET_N_ITEMS = {
    "Beauty": 12101,
    "Sports_and_Outdoors": 18357,
    "Toys_and_Games": 11924,
    "MIND": 20284,
    "Yelp": 94305,
}


def gr_cf2letter(gr_ckpt_path: Path, letter_ckpt_path: Path, dataset_id: str, model: str, emb_dim: int = 32):
    print(f"Mapping {dataset_id} - {model} CF embeddings from GR to LETTER...")
    # Save CF embeddings from SASRec
    ckpt_name = f"{dataset_id}-{emb_dim}d-{model}.pt"
    cf_embeddings = torch.load(gr_ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    if "gpt2.transformer.wte.weight" in cf_embeddings:
        cf_embeddings = cf_embeddings["gpt2.transformer.wte.weight"][1:-2]  # Skip PAD, EOS, and extra token
    else:
        cf_embeddings = cf_embeddings["gpt2.wte.weight"][1:-2]
    if dataset_id in DATASET_N_ITEMS:
        assert cf_embeddings.shape[0] == DATASET_N_ITEMS[dataset_id]
    else:
        print("Warning: Num items could not be checked!")

    letter_ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(cf_embeddings, letter_ckpt_path / ckpt_name)
    
    print("Finished mapping CF embeddings from GR to LETTER format!")


def extract_ckpt_from_log(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()
    
    pattern = "Loaded best model checkpoint from ([\w/]+?[\w-]+.pth)"
    for line in lines[::-1]:
        matches = re.findall(pattern, line)
        if matches:
            return matches[0]

    raise Exception("Checkpoint not found in log file!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, default="Beauty", help="Dataset name (folder name)")
    parser.add_argument("--model-ckpt", type=str, default=None, help="Filename of the CF model checkpoint")
    parser.add_argument("--log-path", type=str, default=None, help="Log file including filename of the CF model checkpoint")
    parser.add_argument("--cf-model", type=str, default="SASRec", help="Name of the CF model")
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unparsed_args = parse_args()

    ckpt_path = Path("ckpt")
    sid_ckpt = ckpt_path / "gensid"
    if args.model_ckpt is None and args.log_path is not None:
        ckpt_path = extract_ckpt_from_log(args.log_path)
    elif args.model_ckpt is not None:
        ckpt_path = ckpt_path / args.model_ckpt
    else:
        raise ValueError("Specify either --log-path or --model-ckpt containing the CF embeddings")
    
    gr_cf2letter(ckpt_path, sid_ckpt, args.dataset_id, args.cf_model)

    print("Finished!")