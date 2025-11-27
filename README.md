# Kadaster Dataloader

A utility for loading and processing Kadaster datasets, specifically designed for handling "rechtsfeitcodes" with label encoding.

## Features

- **Dataset Loading**: Efficiently loads JSONL datasets using Hugging Face `datasets`.
- **Label Encoding**: Handles sparse integer codes, mapping them to dense indices.
- **Unknown Handling**: Robustly handles unknown codes by mapping them to a reserved index (0).
- **Logging**: Uses `loguru` for structured logging to both console and file (`logs/dataset.log`).
- **PyTorch Integration**: Provides a custom `Dataset` and `DataLoader` for PyTorch.

## Usage

### 1. Analysis
Generate a label distribution plot to understand the dataset balance.
```bash
kadaster analyze
```
Output: `artifacts/img/label_distribution.png`

### 2. Regex Evaluation
Evaluate the performance of the regex-based model on the training set.
```bash
kadaster evaluate-regex
```
Output: `artifacts/csv/regex_evaluation.csv`

### 3. Training
Train the model. You can choose between the simple BERT classifier or the Hybrid model.

**Regex Only Classifier:**
```bash
kadaster train --model-class RegexOnlyClassifier --epochs 5
```

**Neural Classifier (BERT only):**
```bash
kadaster train --model-class NeuralClassifier --epochs 5 --model-name prajjwal1/bert-mini
```

**Hybrid Classifier (BERT + Regex):**
```bash
kadaster train --model-class HybridClassifier --epochs 5 --model-name prajjwal1/bert-mini
```


## Architecture

The system uses a **Hybrid Architecture** that combines deep learning embeddings with symbolic regex features.

We can view the models as functors mapping from the space of text documents $\mathcal{T}$ to vector spaces $\mathbb{R}^d$.

$$
\begin{aligned}
v_{bert} &= f_{bert}(t) \in \mathbb{R}^{768} \\
v_{regex} &= f_{regex}(t) \in \{0,1\}^{166} \\
v_{combined} &= [v_{bert}, v_{regex}] \in \mathbb{R}^{934} \\
\hat{y} &= \text{softmax}(\text{MLP}(v_{combined}))
\end{aligned}
$$

1.  **Text Vectorizer**: Maps text to a dense semantic space.
    $$ f_{bert}: \mathcal{T} \to \mathbb{R}^{768} $$
2.  **Regex Vectorizer**: Maps text to a sparse symbolic space (binary features).
    $$ f_{regex}: \mathcal{T} \to \{0, 1\}^{166} $$
3.  **Hybrid Combination**: We construct a product space by concatenating the vectors.
    $$ f_{hybrid}(t) = f_{bert}(t) \oplus f_{regex}(t) \in \mathbb{R}^{768+166} $$
4.  **Classification**: A Multi-Layer Perceptron (MLP) maps the combined features to class probabilities.
    $$ f_{mlp}: \mathbb{R}^{934} \to [0, 1]^{NumClasses} $$

## Caching & Versioning

To optimize performance, the system caches vectorized features in `artifacts/vectorcache/`.

*   **Embeddings**: Cached based on the model name (e.g., `prajjwal1_bert-mini_train_embeddings.pt`).
*   **Regex Features**: Cached based on a **hash of the regex patterns** (e.g., `regex_f66bce0c_train.pt`).

This ensures that if you modify the regex logic or the CSV file, the hash changes, invalidating the cache and forcing a recomputation.

Similarly, evaluation results are saved with the hash (e.g., `artifacts/csv/regex_evaluation_f66bce0c.csv`) to allow tracking performance across different regex versions.

## MLFlow Tracking

### Local Setup
By default, experiments are logged to a local SQLite database in `logs/mlflow.db`. To view the dashboard:

```bash
uv run mlflow ui --backend-store-uri sqlite:///logs/mlflow.db
```

### Remote Setup
To log to a remote MLFlow server (e.g., provided by your instructor):

1.  Open `.env` and set the tracking URI:
    ```bash
    MLFLOW_TRACKING_URI=http://<host-ip>:5000
    ```
2.  Run your training commands as usual. The logs will automatically be sent to the server.

curl -sSL https://raw.githubusercontent.com/raoulg/private-data-hosting/refs/heads/main/download_data.sh | bash -s -- -o data/raw