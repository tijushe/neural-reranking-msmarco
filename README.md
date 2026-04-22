# Bias Detection in News Sentences

Classifying biased sentences in news articles using traditional ML and fine-tuned transformer models. Compares Logistic Regression (TF-IDF, BoW) against BERT and DistilBERT on the [Lim et al. (2020)](https://aclanthology.org/2020.lrec-1.175/) crowdsourced annotation dataset.

## Task

Given a sentence from a news article, predict whether it is **biased** or **not biased**. Labels are derived from crowdsourced ratings (1–5 scale) via majority vote; sentences rated ≥ 3 are treated as biased.

## Models

| Model | Approach |
|---|---|
| Logistic Regression + TF-IDF | Traditional baseline |
| Logistic Regression + BoW | Traditional baseline |
| BERT (`bert-base-uncased`) | Fine-tuned transformer |
| DistilBERT (`distilbert-base-uncased`) | Fine-tuned transformer |

## Dataset

[Biased Sentences in News Articles](https://github.com/skymoonlight/biased-sents-annotation) — loaded directly from GitHub, no manual download needed.

Data is reshaped from wide format (one row per article, 20 sentence columns) to long format (one row per sentence), with majority vote aggregation across annotators.

## Setup

**Requirements:** Python 3.10+, CUDA recommended for transformer fine-tuning.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This will:
1. Download and preprocess the dataset automatically
2. Train and evaluate all four models
3. Save confusion matrices and comparison plots to the working directory

## Output

- `lr_TF-IDF_conf_mat.png` > confusion matrix for LR + TF-IDF
- `lr_BOW_conf_mat.png` > confusion matrix for LR + BoW
- `bert-base-uncased_conf_mat.png` > confusion matrix for BERT
- `distilbert-base-uncased_conf_mat.png` > confusion matrix for DistilBERT
- `f1_comp.png` > F1 score comparison across all models
- `acc_comp.png` > accuracy comparison
- `prec_rec_comp.png` > precision and recall comparison


## Notes

- Transformer models are fine-tuned for 5 epochs with `learning_rate=2e-5`, `batch_size=16`
- Train/val/test split: 60/20/20, stratified by label
- Logistic Regression uses `class_weight='balanced'` to handle label imbalance
- Seeds fixed at 42 for reproducibility
