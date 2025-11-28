# Kadaster Dataloader

A utility for loading and processing Kadaster datasets, specifically designed for handling "rechtsfeitcodes" with label encoding.

## 1. Setup
### 1.1. Install the data
First of all, you will need to get the data. To do so, run this command:

```bash
curl -sSL https://raw.githubusercontent.com/raoulg/private-data-hosting/refs/heads/main/download_data.sh | bash -s -- -o assets/
```

This will:
- ask you to accept terms & conditions
- ask you for your email to identify you. Dont worry, we wont spam you :)
- ask you for an API_KEY and an IP. You will get those from your instructor

After these are provided, it downloads & unzips the data to the `assets/` folder.
You will find:
- README.md (please do)
- rechtsfeiten.csv - these are the labels we need to predict, with a short description of the label
- aktes.jsonl - this is the actual data: almost 20.000 anonymized legal documents

### 1.2. Install dependencies
Run `uv sync` to install dependencies.

## 2. Explore the data
To get a basic idea of what we are dealing with, run 

```bash
kadaster analyze
```

This will generate a log in `logs/cli.log`. Check it for some info about the dataset.
You will also find a `artifacts/csv/label_distribution.csv` file, and a plot `artifacts/img/label_distribution.png`

As you can see, the dataset is highly imbalanced. This has consequences for your strategy, so take this into account.

## 3. Models

### 3.1. Regexes
In [regex.py](src/akte_classifier/models/regex.py), you will find a very simple approach:
- `RegexGenerator`: this will read the rechtsfeiten.csv and will automatically generate regexes based on the description. This is in no way finished, so there is a lot to improve here! 
- `RegexVectorizer`: this uses the regexes to create a binary vector for each document: $f\colon \mathcal{T} \to \{0,1\}^{d}$ where $\mathcal{T}$ is the set of documents and $d$ is the number of classes in our training set.
- In the [evaluation.py](src/akte_classifier/utils/evaluation.py), you will find a function to evaluate the performance of the regex vector. This is very simple; if the regex matches, it is a 1, otherwise a 0.

you can run
```bash
kadaster regex
```
The see this in action.

You will find that on the first run, the data is turned into a binary vector, and cached to [artifacts/vectorcache](artifacts/vectorcache/). The `RegexGenerator` creates a hash values based on all the regexes, so if you modify the regexes the hash will change. 

In [artifacts/csv](artifacts/csv) you will find a `regex_evaluation_hash.csv` file. This looks like this:

```csv
code,count,precision,recall,f1,tp,fp,fn,regex
621,2,1.00000,0.50000,0.66667,1,0,1,opheffing\s+ondersplitsing\s+in\s+appartementsrechten
585,1512,0.50774,0.84590,0.63458,1279,1240,233,verklaring\s+van\s+erfrecht
```

For example, the code 621 occurs only 2 times in the data, and this regex has a precision of 1.0, a recall of 0.5, and an F1 score of 0.6... while the code 585 occurs 1512 times, and this regex has a precision of 0.51, a recall of 0.85, and an F1 score of 0.63. 

This should guide you to see how usefull your regexes are, but also helps you invest your time wisely (working on a regex that can only find 2 items in a dataset of about 20k wont have much impact)

### 3.2 TextVectorizers
From huggingface, we can download pre-trained text vectorizers.

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