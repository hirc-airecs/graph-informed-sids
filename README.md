# Graph-Informed Semantic IDs (GrIS Framework)

Official implementation of "Neither Black nor White: Balancing Semantic and Collaborative Signals with Graph-Informed Semantic IDs (GrIS)".

## Project Structure

- `gensid`: Based on the [LETTER codebase](https://github.com/HonghuiBao2000/LETTER), extended to generate graph-based semantic IDs via non-parametric message passing.
- `genrec`: Based on [MemGen-GR](https://github.com/Jamesding000/MemGen-GR) and [phonism's](https://github.com/phonism/genrec) codebases, contains the GR and atomic baselines (SASRec, HSTU, and TIGER). Several improvements were added, including fixed evaluation in case of SID collisions, enhanced logging/results saving, a tokenizer abstraction to easily add new models, improved config handling, extended dataset and tokenizer classes to work with extra sequence columns (e.g. time and rating), and support for new datasets ([MIND-small](https://msnews.github.io/#getting-started) and [Yelp](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/download?datasetVersionNumber=1)).

## Installation

```shell
conda create -n gr_env python=3.11 -y
conda activate gr_env
pip install -r requirements.txt
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
```

## Reproducing Experiments

The MIND dataset requires signing Microsoft's license terms, and Yelp 2018 is only available via Kaggle. Download both manually and place them under `cache/{dataset}/raw/`. The dataset classes handle processing the raw files.

1. **Prepare the dataset** — downloads Amazon datasets (if applicable), prepares item text representations, and generates sentence embeddings for SID generation:

```shell
python prepare_dataset.py --category=Beauty \
  --sent_emb_model=sentence-transformers/sentence-t5-base \
  --sent_emb_batch_size=64

# Other Amazon categories:
# python prepare_dataset.py --category=Toys_and_Games
# python prepare_dataset.py --category=Sports_and_Outdoors
# python prepare_dataset.py --category=Books
# python prepare_dataset.py --dataset=Yelp
# python prepare_dataset.py --dataset=MIND
```

2. **Run the experiments:**

```shell
bash scripts/run_sasrec.sh
bash scripts/run_hstu.sh
bash scripts/run_letter.sh                           # Pretrains a CF model for LETTER CF-loss regularization
bash scripts/run_s2gr.sh
bash scripts/run_mmgrec.sh
bash scripts/run_gae_components_ablation.sh          # TIGER, RQ-GAE, RQ-GAE w/o graph-loss, RQ-GAE w/o graph-input
bash scripts/run_recdmon.sh
```

The `scripts/` directory contains all experiments needed to reproduce the results in the paper. Results are saved under `results/`, named by the alias specified in each script.