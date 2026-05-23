# MLOps Assignment 2 — DistilBERT Goodreads Genre Classification

Fine-tuned DistilBERT on UCSD Goodreads reviews to classify books into 7 genres. Built with HuggingFace Transformers, tracked experiments using Weights & Biases, and deployed the model to HuggingFace Hub.

---

## Model Selection Rationale

We chose `distilbert-base-cased` for this assignment. DistilBERT is a distilled version of BERT that is 40% smaller and 60% faster while retaining 97% of BERT language understanding capability. The cased variant was selected because book reviews often contain proper nouns (author names, book titles) where capitalisation carries meaning. For a multi-class text classification task on short-to-medium length reviews, DistilBERT offers an excellent trade-off between speed, memory usage, and accuracy — making it ideal for training on Kaggle free GPU tier within the allotted time.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/0xShashankbtc/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export WANDB_API_KEY=your_wandb_api_key
export HF_TOKEN=your_huggingface_token
```

### 4. Run the notebook
Open `a2-g25ait2100-mlops.ipynb` in Kaggle or Jupyter and run all cells top to bottom.

---

## Training Platform

Trained on **Kaggle Notebooks** using free GPU T4 x2 accelerator.
- Internet enabled for package installation and API access
- API tokens stored securely via Kaggle Secrets (Add-ons → Secrets)
- Kaggle Notebook: [https://www.kaggle.com/g25ait2100/a2-g25ait2100-mlops](https://www.kaggle.com/code/shashankjangid/a2-g25ait2100-mlops)

---

## Results

| Metric     | Score  |
|------------|--------|
| Accuracy   | 0.5931 |
| F1 Score   | 0.5953 |
| Eval Loss  | 2.3951 |

---

## Links

- **Kaggle Notebook:** [https://www.kaggle.com/g25ait2100/a2-g25ait2100-mlops](https://www.kaggle.com/code/shashankjangid/a2-g25ait2100-mlops)
- **Hugging Face Model:** https://huggingface.co/G25AIT2100/distilbert
- **W&B Dashboard:** https://wandb.ai/g25ait2100-iit-jodhpur/mlops-assignment2
