# Kadaster Dataloader

A utility for loading and processing Kadaster datasets, specifically designed for handling "rechtsfeitcodes" with label encoding.

## 1. Setup
### 1.1. Install dependencies
Run `uv sync` to install dependencies.

### 1.2. Download the data
First of all, you will need to get the data. To do so, go to `http://145.38.195.113` in your browser, provide your email and the api_key (you will get this from your instructor), accept the terms and conditions, and you can download the data.

After unzipping, you will find:
- README.md (please do)
- rechtsfeiten.csv - these are the labels we need to predict, with a short description of the label
- aktes.jsonl - this is the actual data: almost 20.000 anonymized legal documents

### 1.3. Split the data
First, split your dataset into training and testing sets to ensure you have a held-out test set for evaluation.

```bash
kadaster split-data --data-path assets/aktes.jsonl --test-size 0.1
```

This will create `assets/train.jsonl` and `assets/test.jsonl`.
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
- `RegexVectorizer`: this uses the regexes to create a binary vector for each document: $f\colon \mathcal{T} \to \{0,1\}^{C}$ where $\mathcal{T}$ is the set of documents and $C$ is the number of classes in our training set.
- In the [evaluation.py](src/akte_classifier/utils/evaluation.py), you will find a function to evaluate the performance of the regex vector. This is very simple; if the regex matches, it is a 1, otherwise a 0.

you can run
```bash
kadaster regex
```
The see this in action.

You will find that on the first run, the data is turned into a binary vector, and cached to [artifacts/vectorcache](artifacts/vectorcache/). The `RegexGenerator` creates a hash values based on all the regexes, and a regex based on the hash of the filepath used to create the vectors. So if you modify the regexes the hash will change and the vectors will be re-calculated.

In [artifacts/csv](artifacts/csv) you will find a `regex_evaluation_hash.csv` file. This looks like this:

```csv
code,count,precision,recall,f1,tp,fp,fn,regex
621,2,1.00000,0.50000,0.66667,1,0,1,opheffing\s+ondersplitsing\s+in\s+appartementsrechten
585,1512,0.50774,0.84590,0.63458,1279,1240,233,verklaring\s+van\s+erfrecht
```

For example, the code 621 occurs only 2 times in the data, and this regex has a precision of 1.0, a recall of 0.5, and an F1 score of 0.6... while the code 585 occurs 1512 times, and this regex has a precision of 0.51, a recall of 0.85, and an F1 score of 0.63. 

This should guide you to see how usefull your regexes are, but also helps you invest your time wisely (working on a regex that can only find 2 items in a dataset of about 20k wont have much impact)

### 3.2 Regex with NN
A very simple way to improve the regex is to take the binary-regex-vector as input, and compose the vectorizer with a basic neural network that does $g\colon \{0,1\}^{C} \to \mathbb{R}^{C}$ where the outputvector are logits and $C$ is the number of classes in our training set.

If you run
```bash
kadaster train --model-class RegexOnlyClassifier --epochs 5
``` 
this will:

-  load the regex vectorizer which serve as binary features $\{0,1\}^{C}$
-  train the `NeuralClassifier` from [neural.py](src/akte_classifier/models/neural.py) 

### 3.3 Text Vectorizers

The `NeuralClassifier` and `HybridClassifier` use a `TextVectorizer` to convert text into embeddings. From huggingface, we can download pre-trained text vectorizers.
For example, we can use [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)

```bash
kadaster train --model-class NeuralClassifier --model-name prajjwal1/bert-tiny --epochs 10
```
This command will use the NeuralClassifier approach, download the prajjwal1/bert-tiny model as a vectorizer, and train for 10 epochs.

- **Dynamic Max Length**: The vectorizer tries to automatically detect the model's maximum context length (e.g., typically512 for BERT, 8192 for BGE-M3 models). This sometimes fails, so check the logs. You can override this with `--max-length 512` or `--max-length 4000`, for example
- **Automatic Pooling**: The vectorizer automatically detects the optimal pooling strategy (`cls` or `mean`) from the model's configuration. You can override this with `--pooling cls` or `--pooling mean`.

General heuristics:
- BERT-based models (like bert-base-uncased) often work well with Mean Pooling.
- Sentence-BERT models (sbert) are typically trained with Mean Pooling.
- Newer Embedding models (like bge, e5) often use CLS Pooling because it's more efficient for retrieval tasks.

### 3.4 Hybrid Approach

We can combine these two approaches by using a `HybridClassifier`.

- regex vectorizer does $f\colon \mathcal{T} \to \{0,1\}^{d_r}$ with $d_r$ the dimension of the regex vectorizer
- text vectorizer does $g\colon \mathcal{T} \to \mathbb{R}^{d_v}$ with $d_v$ the dimension of the text vectorizer
- we then concatenate the two vectors: $h\colon \mathcal{T} \to \mathbb{R}^{d_r + d_v}$
- finally, we add a neural network $m\colon \mathbb{R}^{d_{r} + d_{v}} \to \mathbb{R}^{C}$ where $C$ is the number of classes in our training set

You can test this with:
```bash
kadaster train --model-class HybridClassifier --model-name prajjwal1/bert-tiny --epochs 10
```

## Evaluate your runs
You have made a train-test split at the beginning, and you are training on the train set where a part is held out for checking if you are overfitting.

So you will get a lot of information just from that, and often there is no need to run an eval because the validation test will tell you enough. 
However, at some point you might want to do an additional test on an unseen set. You can do so by testing your model on the hold-out set, test.jsonl Every training, your model weights are loaded to `artifacts/models/` along with a config and timestamp. If you provide the timestamp, and the eval file path, the model will load the weights for eval, and will run it on the test.jsonl file.

You can test this with:

```bash
kadaster eval \
  --eval-file assets/test.jsonl \
  --timestamp 20250112_090112
```

if `20250112_090112` is the timestamp of your model. You can even do 

```bash
kadaster eval \
  --eval-file assets/test.jsonl \
  --timestamp 090112
```
And this will still guess the timestamp `20250112_090112`, but do doublecheck in the logs.
The first time it will vectorize the test set, and the first eval you will see something like

```bash
SUCCESS  | akte_classifier.utils.tensor:load_or_compute_tensor:38 - Computed tensor and saved to artifacts/vectorcache/regex_f66bce0c_ffc819f4_split100_full.pt
```

in your logs.

## 4. LLM for the long tail

The dataset contains a "long tail" of labels that appear very infrequently. Traditional supervised learning models (like the Neural or Hybrid classifiers above) struggle with these classes because there are not enough examples to learn from.

To address this, we use a **Zero-Shot Learning** approach with Large Language Models (LLMs). Instead of training a model on examples, we provide the LLM with the *description* of the label (from `rechtsfeiten.csv`) and the text of the document, and ask it to determine if the label applies.

### 4.1 Usage

You can use the `llm-classify` command to run this process.

```bash
kadaster llm-classify --threshold 10 --limit 20
```

- `--threshold`: The maximum number of occurrences for a label to be considered "long-tail". Labels with fewer than this many examples in the training set will be targeted.
- `--limit`: (Optional) Limit the number of documents to process (useful for testing).
- `--model-name`: The LLM to use.
- `--max-length`: (Optional) Truncate input text to this many tokens to avoid context length errors.

### 4.2 Available Models

We support several models via the Nebius API. You can switch models using the `--model-name` argument:

- `meta-llama/Meta-Llama-3.1-8B-Instruct-fast` (Default) [128k context]
- `Qwen/Qwen3-32B-fast` [41k context] 
- `Qwen/Qwen3-30B-A3B-Thinking-2507` [262k context]
- `openai/gpt-oss-20b` [132k context]
- `openai/gpt-oss-120b` [131k context]

Example:
```bash
kadaster llm-classify --model-name Qwen/Qwen3-32B-fast --threshold 10 --max-length 30000
```
where you specify a token-limit of 30k tokens.

### 3. Evaluate
Evaluate the trained model on the held-out test set using the timestamp from your training run.

```
Example:
```bash
kadaster eval --eval-file assets/test.jsonl --timestamp 20251130_204405
```
## Your task
Your task is to improve the model. Try to achieve the highest possible F1 micro score.

Make sure we can evaluate the model with the eval command; this will use the saved modelweights in `artifacts/models/`.

## MLFlow Tracking

To log to a remote MLFlow server (IP address and PORT provided by your instructor):

1.  Open `.env` and set the tracking URI:
    ```bash
    MLFLOW_TRACKING_URI=http://<host-ip>:<port>
    ```
2.  **Developer Tracking**:
    Copy `.env.sample` to `.env` and set your `DEV_NAME` to track who ran the experiment.
    ```bash
    cp .env.sample .env
    # Edit .env and set DEV_NAME=your_name
    ```
3.  Run your training commands as usual. The logs will automatically be sent to the server.

## Embedding models from Nebius

You have access to Nebius embedding models. I havent had the time to implement this myself, but this is the demo code;
```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

response = client.embeddings.create(
    model="BAAI/bge-en-icl",
    input="USER_INPUT"
)

print(response.to_json())
```
And you can use these models:
- BAAI/bge-multilingual-gemma2 (8k context)
- BAAI/bge-en-icl (32k context)
- intfloat/e5-mistral-7b-instruct (32k context)
- Qwen/Qwen3-Embedding-8B (32k context)

## Caching & Versioning

To optimize performance, the system caches vectorized features in `artifacts/vectorcache/`.

*   **Embeddings**: Cached based on the model name (e.g., `prajjwal1_bert-tiny_train_embeddings.pt`).
*   **Regex Features**: Cached based on a **hash of the regex patterns** (e.g., `regex_f66bce0c_train.pt`).

This ensures that if you modify the regex logic or the CSV file, the hash changes, invalidating the cache and forcing a recomputation.

Similarly, evaluation results are saved with the hash (e.g., `artifacts/csv/regex_evaluation_f66bce0c.csv`) to allow tracking performance across different regex versions.

