# Neural Re-ranking on MS MARCO

Fine-tuning and evaluating neural cross-encoder models for passage re-ranking on the [MS MARCO 2019](https://microsoft.github.io/msmarco/) dataset. Explores model fusion strategies and LLM-based query expansion.

## Overview

| Notebook | What it does |
|---|---|
| `01_finetune_crossencoder.ipynb` | Fine-tunes MiniLM, TinyBERT, and DistilRoBERTa cross-encoders on MS MARCO passage ranking |
| `02_evaluate_crossencoder.ipynb` | Evaluates fine-tuned models with NDCG@10, Recall@100, MAP@1000 |
| `03_fusion_reranking.ipynb` | Applies ensemble fusion strategies (RRF, MNZ, Min, Max) across model outputs |
| `04_query_expansion_llm.ipynb` | Query expansion using Zephyr-7B with Chain-of-Thought prompting |

## Results

### Individual model performance (Task 1)

| Model | NDCG@10 | Recall@100 | MAP@1000 |
|---|---|---|---|
| TinyBERT | 69.58 | 50.36 | 45.46 |
| MiniLM | 66.05 | 49.64 | 42.63 |
| DistilRoBERTa | 62.00 | 48.46 | 40.40 |

### Fusion performance (Task 2)

| Method | NDCG@10 | Recall@100 | MAP@1000 |
|---|---|---|---|
| RRF (k=30) | **70.21** | 51.22 | 45.42 |
| RRF (k=60) | 70.02 | **51.32** | **45.55** |
| MNZ | 69.21 | 51.19 | 45.15 |

RRF consistently outperforms individual models, confirming the value of ensemble re-ranking.

## Setup

These notebooks are designed to run on **Google Colab** with GPU. Models and intermediate outputs are stored to Google Drive.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

Download the MS MARCO 2019 passage re-ranking data:
- Queries: `msmarco-test2019-queries.tsv`
- Top-1000 passages: `msmarco-passagetest2019-top1000.tsv`
- Relevance judgements: `2019qrels-pass.txt`

Available from the [MS MARCO passage ranking page](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019).

## Run order

Run the notebooks in order: `01` → `02` → `03` → `04`. Each notebook saves outputs (ranking runs, model checkpoints) to Google Drive for the next step to consume.

## Notes

- All cross-encoders fine-tuned for 1 hour with batch size 32, negative sampling ratio 4:1
- Evaluation uses 200 dev queries with up to 200 negatives each
- Task 4 (query expansion) is partially implemented — expanded queries are generated but the full reranking pipeline is pending
