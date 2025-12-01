# Kadaster Legal Document Classification - Complete Project Documentation

**Project:** Dutch Legal Document Classification System
**Repository:** mads-2025-hackathon-serverboys
**Period:** November - December 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Development Timeline](#development-timeline)
4. [Data Preparation](#data-preparation)
5. [Model Development](#model-development)
6. [Cascade Classifier](#cascade-classifier)
7. [Performance Results](#performance-results)
8. [Usage Guide](#usage-guide)
9. [Technical Details](#technical-details)
10. [Future Work](#future-work)

---

## System Architecture

### Final Architecture: 3-Tier Cascade System

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DOCUMENT (Dutch Text)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION                                          │
│  ├─ Text Embeddings: Nebius bge-multilingual-gemma2         │
│  │  (3584 dimensions, multilingual support)                 │
│  └─ Regex Features: 85 legal pattern matchers               │
│     (dates, amounts, specific legal terms)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 1: HYBRID CLASSIFIER                                   │
│  ├─ Text pathway: Linear(3584→256) + ReLU + Dropout         │
│  ├─ Regex pathway: Linear(85→128) + ReLU                    │
│  ├─ Fusion: Concat → Linear(384→85)                         │
│  └─ Output: 85 class probabilities + confidence score       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Confidence  │
              │   ≥ 99%?     │
              └──────┬───────┘
                     │
        ┌────────────┴────────────┐
        │                         │
       YES                       NO
        │                         │
        ▼                         ▼
┌───────────────┐     ┌──────────────────────────┐
│ RETURN HYBRID │     │  TIER 2: LLM FALLBACK    │
│ PREDICTION    │     │  ├─ Model: GPT-OSS 120B  │
│               │     │  ├─ Context: 131k tokens │
│ ✅ <1ms       │     │  └─ Provider: Nebius API │
│ ✅ Free       │     └───────────┬──────────────┘
└───────────────┘                 │
                                  ▼
                          ┌───────────────┐
                          │ RETURN LLM    │
                          │ PREDICTION    │
                          │               │
                          │ ⚠️ 2-3 sec    │
                          │ ⚠️ API cost   │
                          └───────────────┘
```

---

## Development Timeline

### Phase 1: Initial Setup

**Branch:** `liam`

**Tasks:**
1. Environment setup with `uv` package manager
2. MLflow integration for experiment tracking
3. Data splitting (80/20 train/test)

**Commands:**
```bash
# Setup
uv sync
kadaster split-data --data-path assets/aktes.jsonl --test-size 0.2

# Result
# Train: 7,916 samples
# Test: 1,979 samples
```

**Files:**
- Created `.env` with `DEV_NAME`, `NEBIUS_API_KEY`, `MLFLOW_TRACKING_URI`
- Generated `assets/train.jsonl` and `assets/test.jsonl`

---

### Phase 2: Baseline Model 

**Objective:** Establish performance baseline with lightweight model

**Model:** `prajjwal1/bert-tiny`
- Parameters: 4.4M
- Embedding dim: 128
- Training time: ~5 minutes

**Training:**
```bash
kadaster train \
  --model-class NeuralClassifier \
  --model-name prajjwal1/bert-tiny \
  --epochs 10 \
  --batch-size 32
```

**Results:**
- **F1 Micro:** 0.596 (59.6%)
- **F1 Macro:** 0.245 (24.5%)
- **Conclusion:** Too simple, needs better embeddings

**MLflow:**
- Experiment: `rechtsfeit-classification`
- Run ID: `baseline-bert-tiny`
- Metrics logged: loss, F1, precision, recall

---

### Phase 3: Nebius API Integration 

**Objective:** Use high-quality multilingual embeddings

**Solution:** Nebius API for cloud-based embeddings

**Implementation:**
1. Created `NebiusTextVectorizer` class
2. API endpoint: `https://api.studio.nebius.com/v1/embeddings`
3. Model: `BAAI/bge-multilingual-gemma2`
   - Embedding dim: 3584
   - Multilingual support (Dutch optimized)
   - Context window: 8192 tokens

**Code:**
```python
class NebiusTextVectorizer:
    def __init__(self, model_name: str):
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.getenv("NEBIUS_API_KEY")
        )
        self.model_name = model_name

    def vectorize(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return np.array([e.embedding for e in response.data])
```

**Features:**
- Automatic caching (content-based hashing)
- Batch processing
- Progress logging

**Files Modified:**
- `src/akte_classifier/datasets/vectorizers.py`

---

### Phase 4: Regex Feature Engineering 

**Objective:** Add domain-specific pattern matching

**Patterns Implemented (85 total):**

**Dates:**
- `\d{1,2}[-/]\d{1,2}[-/]\d{2,4}` - Dutch date formats
- `\b(januari|februari|maart|...|december)\b` - Month names

**Monetary:**
- `€\s*\d+[.,]\d+` - Euro amounts
- `\d+\s*euro` - Text amounts

**Legal Terms:**
- `\b(koopsom|verkoopsom|koopprijs)\b` - Purchase prices
- `\b(hypotheek|hypothecair)\b` - Mortgage terms
- `\b(eigendom|eigenaar)\b` - Ownership terms
- `\b(erfpacht|erfpachter)\b` - Leasehold terms

**Document Structure:**
- `\b(artikel|lid)\s+\d+` - Article references
- `\bBW\b` - Civil Code references
- `\b(notaris|notarieel)\b` - Notary mentions

**Locations:**
- `\b(gemeente|kadaster)\b` - Municipality/cadastre
- Cadastral identifiers patterns

**Implementation:**
```python
class RegexVectorizer:
    def __init__(self):
        self.patterns = [
            # Monetary patterns
            (r'€\s*\d+[.,]\d+', 'euro_amount'),
            # Date patterns
            (r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', 'date'),
            # Legal terms
            (r'\b(hypotheek|hypothecair)\b', 'mortgage'),
            # ... 82 more patterns
        ]

    def extract_features(self, text: str) -> np.ndarray:
        features = []
        for pattern, _ in self.patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            features.append(1 if matches > 0 else 0)  # Binary
        return np.array(features)
```

**Caching:**
- Hash-based: `regex_827c31ca_735228d6_split80_train.pt`
- Invalidates automatically when patterns change

**Files:**
- `src/akte_classifier/datasets/vectorizers.py` (RegexVectorizer)

---

### Phase 5: Hybrid Model Architecture 

**Objective:** Combine embeddings + regex for better accuracy

**Architecture:**

```python
class HybridClassifier(nn.Module):
    def __init__(
        self,
        input_dim=3584,    # Nebius embedding dimension
        regex_dim=85,      # Number of regex features
        num_classes=85,    # Number of rechtsfeit codes
        hidden_dim=256,    # Hidden layer size
        dropout_rate=0.3   # Dropout for regularization
    ):
        super().__init__()

        # Text embedding pathway
        self.text_pathway = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Regex feature pathway
        self.regex_pathway = nn.Sequential(
            nn.Linear(regex_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Fusion and output
        self.output_layer = nn.Linear(
            hidden_dim + hidden_dim // 2,  # 384 total
            num_classes
        )

    def forward(self, text_emb, regex_feat):
        text_out = self.text_pathway(text_emb)      # [B, 256]
        regex_out = self.regex_pathway(regex_feat)  # [B, 128]
        combined = torch.cat([text_out, regex_out], dim=1)  # [B, 384]
        logits = self.output_layer(combined)        # [B, 85]
        return logits
```

**Design Decisions:**
- **Separate pathways:** Text and regex processed independently before fusion
- **Asymmetric dimensions:** Text gets more capacity (256) than regex (128)
- **Dropout:** Only on text pathway (regex features are sparse)
- **No dropout on output:** Preserve all information at final layer
- **ReLU activation:** Faster than alternatives, works well for classification

**Training Configuration:**
- Optimizer: AdamW
- Learning rate: 1e-4
- Batch size: 32
- Epochs: 10
- Loss: CrossEntropyLoss
- Early stopping: Patience 3 epochs

---

### Phase 6: Hybrid Model Training 

**Training Command:**
```bash
kadaster train \
  --model-class HybridClassifier \
  --model-name BAAI/bge-multilingual-gemma2 \
  --use-nebius \
  --use-regex \
  --epochs 10 \
  --hidden-dim 256 \
  --batch-size 32
```

**Training Process:**
1. **Vectorization phase:**
   - Checks cache for existing embeddings
   - If cache miss: Calls Nebius API
   - Saves to `artifacts/vectorcache/`
   - Hash: `735228d6` (data content), `827c31ca` (regex patterns)

2. **Training phase:**
   - 10 epochs with early stopping
   - Validation on test set each epoch
   - Best model saved based on F1 score
   - MLflow logging at each step

3. **Artifacts saved:**
   - Model weights: `BAAI_bge-multilingual-gemma2_827c31ca_20251201_154538.pt`
   - Label encoder: `*_codes.json`
   - Config: `*_config.json`

**Results:**
- **F1 Micro:** 0.885 (88.5%)
- **F1 Macro:** 0.612 (61.2%)
- **Training time:** ~15 minutes
- **Improvement:** +29% F1 over baseline

**Per-Class Performance:**
- Well-represented classes: F1 > 0.90
- Medium classes: F1 = 0.60-0.80
- Rare classes: F1 < 0.40

**MLflow Metrics:**
```
Experiment: rechtsfeit-classification
Run: hybrid-nebius-regex-20251201_154538
Metrics:
  - train_loss: 0.245
  - val_loss: 0.512
  - f1_micro: 0.885
  - f1_macro: 0.612
  - precision_micro: 0.885
  - recall_micro: 0.885
Parameters:
  - model: BAAI/bge-multilingual-gemma2
  - hidden_dim: 256
  - use_regex: True
  - use_nebius: True
```

---

### Phase 7: Cascade Classifier Implementation 

**Objective:** Add LLM fallback for low-confidence predictions

**Motivation:**
- Hybrid model: F1=0.885, but some classes struggle
- LLM: More accurate but slow (2-3 sec) and expensive
- Solution: Use hybrid for confident cases, LLM for uncertain ones

**Implementation:**

**1. CascadeClassifier Class (`cascade.py`):**
```python
class CascadeClassifier:
    def __init__(
        self,
        hybrid_model: nn.Module,
        llm_classifier: LLMClassifier,
        label_encoder: LabelEncoder,
        confidence_threshold: float = 0.99,
        device: str = "cpu"
    ):
        self.hybrid_model = hybrid_model
        self.llm_classifier = llm_classifier
        self.confidence_threshold = confidence_threshold

        # Statistics tracking
        self.stats = {
            "total_predictions": 0,
            "hybrid_predictions": 0,
            "llm_fallback_predictions": 0,
            "hybrid_confidences": []
        }

    def predict(self, text, text_embedding, regex_features):
        # Tier 1: Hybrid model
        with torch.no_grad():
            logits = self.hybrid_model(text_embedding, regex_features)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        # Decision point
        if confidence >= self.confidence_threshold:
            # High confidence - use hybrid
            self.stats["hybrid_predictions"] += 1
            return {
                "prediction": self.label_encoder.idx2code[pred_idx.item()],
                "confidence": confidence.item(),
                "method": "hybrid"
            }
        else:
            # Low confidence - call LLM
            self.stats["llm_fallback_predictions"] += 1
            llm_codes = self.llm_classifier.classify(text)

            if llm_codes:
                return {
                    "prediction": llm_codes[0],
                    "confidence": 0.99,
                    "method": "llm",
                    "hybrid_confidence": confidence.item()
                }
            else:
                # LLM failed - fallback to hybrid
                return {
                    "prediction": self.label_encoder.idx2code[pred_idx.item()],
                    "confidence": confidence.item(),
                    "method": "llm_failed_hybrid_fallback"
                }
```

**2. LLMClassifier Integration (`llm.py`):**
```python
class LLMClassifier:
    def __init__(self, model_name: str, descriptions: Dict[int, str]):
        self.model_name = model_name
        self.descriptions = descriptions

        # Nebius API client
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.getenv("NEBIUS_API_KEY")
        )

    def classify(self, text: str) -> List[int]:
        prompt = f"""
        You are a legal expert classifier specializing in Dutch property law.
        Classify this document into rechtsfeit codes.

        Available codes:
        {self._format_descriptions()}

        Document:
        {text}

        Return JSON: {{"label_codes": [504, 537, ...]}}
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        data = json.loads(response.choices[0].message.content)
        return data["label_codes"]
```

**3. CLI Command (`cli.py`):**
```python
@app.command()
def eval_cascade(
    eval_file: str,
    timestamp: str,
    confidence_threshold: float = 0.99,
    llm_model: str = "openai/gpt-oss-120b",
    limit: Optional[int] = None
):
    # Load cascade classifier
    cascade = load_cascade_classifier(
        model_timestamp=timestamp,
        confidence_threshold=confidence_threshold,
        llm_model=llm_model
    )

    # Load cached embeddings (optimization!)
    embeddings = torch.load(f"artifacts/vectorcache/BAAI_*_test_embeddings.pt")
    regex_features = torch.load(f"artifacts/vectorcache/regex_*_test.pt")
    labels = torch.load(f"artifacts/vectorcache/BAAI_*_test_labels.pt")

    # Evaluate
    for i in range(num_samples):
        text, _ = test_dataset[i]
        text_emb = embeddings[i].unsqueeze(0)
        regex_feat = regex_features[i].unsqueeze(0)

        result = cascade.predict(text, text_emb, regex_feat)
        all_predictions.append(result["prediction"])

    # Print statistics
    cascade.print_statistics()
```

**4. Prompt Engineering (`prompts.py`):**
```python
DEFAULT_TEMPLATE = """
You are a legal expert classifier specializing in Dutch property law (kadasterrecht).

CRITICAL CLASSIFICATION PRINCIPLE:
The CONTENT of the document is LEADING - the title is SUPPORTING only.

IMPORTANT: BE CAREFUL WITH QUOTED TEXT
The document may contain quoted text from other legal documents.
Focus on what THIS document actually does, not what other documents mentioned in quotes do.

ANALYSIS PROCESS (follow these steps in order):
1. READ AND IDENTIFY: Read through the entire document and identify all legal actions
2. PRIMARY VS SECONDARY: Determine which actions are PRIMARY vs SECONDARY
3. MATCH TO CODES: For each PRIMARY action, find the most specific applicable code
4. VERIFY DISTINCTNESS: Verify that each code represents a separate action
5. TITLE CHECK: Compare with document title (if present)
6. FINAL VALIDATION: Review your selected codes

Available categories (Code: Description):
${descriptions_text}

Text to classify:
\"\"\"
${text}
\"\"\"

Return the applicable category codes as a JSON object with a single key "label_codes"
containing a list of integers.
"""
```

**Files Created/Modified:**
- `src/akte_classifier/models/cascade.py` (new)
- `src/akte_classifier/cli.py` (modified: added eval-cascade command)
- `src/akte_classifier/models/llm.py` (modified: Nebius integration)
- `src/akte_classifier/models/prompts.py` (modified: detailed prompt)

---

## Performance Results

### Summary Table

| Model | F1 Micro | F1 Macro | Speed | Cost |
|-------|----------|----------|-------|------|
| Baseline (bert-tiny) | 0.596 | 0.245 | <1ms | $0 |
| Hybrid (Nebius+Regex) | **0.885** | 0.612 | <1ms | $0 |
| Cascade (80/20 split) | TBD | TBD | ~1s | ~$0.002 |

### Detailed Results

**Baseline Model:**
```
Model: prajjwal1/bert-tiny
Train samples: 7,916
Test samples: 1,979

Metrics:
  F1 Micro: 0.596
  F1 Macro: 0.245
  Precision: 0.613
  Recall: 0.580

Training time: ~5 minutes
Inference: <1ms per document
```

**Hybrid Model:**
```
Model: BAAI/bge-multilingual-gemma2 + Regex
Timestamp: 20251201_154538
Train samples: 7,916
Test samples: 1,979

Metrics:
  F1 Micro: 0.885
  F1 Macro: 0.612
  Precision: 0.885
  Recall: 0.885

Features:
  - Text embedding: 3584-dim
  - Regex features: 85-dim
  - Hidden dim: 256

Training time: ~15 minutes
Inference: <1ms per document
```

**Cascade Classifier (100 samples):**
```
Model: Hybrid + GPT-OSS 120B
Confidence threshold: 0.99
Test samples: 100

Distribution:
  Hybrid predictions: 79 (79.0%)
  LLM fallback: 21 (21.0%)
  Average hybrid confidence: 0.9377

Metrics:
  F1 Micro: 0.220
  F1 Macro: 0.038

Performance:
  Total time: 1m 47s
  Average: 1.07 sec/doc
  Cost: ~$0.21 for 100 docs

Note: Low F1 under investigation
```

---

## Usage Guide

### Setup

**1. Install dependencies:**
```bash
uv sync
```

**2. Configure environment (`.env`):**
```bash
NEBIUS_API_KEY=your_api_key_here
DEV_NAME=YourName
MLFLOW_TRACKING_URI=http://145.38.194.145:5002
```

**3. Activate environment:**
```bash
source .venv/bin/activate
```

### Data Preparation

**Split data:**
```bash
kadaster split-data \
  --data-path assets/aktes.jsonl \
  --test-size 0.2
```

**Analyze distribution:**
```bash
kadaster analyze
```

### Training

**Baseline model:**
```bash
kadaster train \
  --model-class NeuralClassifier \
  --model-name prajjwal1/bert-tiny \
  --epochs 10
```

**Hybrid model:**
```bash
kadaster train \
  --model-class HybridClassifier \
  --model-name BAAI/bge-multilingual-gemma2 \
  --use-nebius \
  --use-regex \
  --epochs 10 \
  --hidden-dim 256 \
  --batch-size 32
```

### Evaluation

**Evaluate hybrid model:**
```bash
kadaster eval \
  --eval-file assets/test.jsonl \
  --timestamp 20251201_154538
```

**Evaluate cascade (small test):**
```bash
kadaster eval-cascade \
  --eval-file assets/test.jsonl \
  --timestamp 20251201_154538 \
  --confidence-threshold 0.99 \
  --limit 100
```

**Evaluate cascade (full test):**
```bash
kadaster eval-cascade \
  --eval-file assets/test.jsonl \
  --timestamp 20251201_154538 \
  --confidence-threshold 0.99
```

**Pure hybrid (no LLM):**
```bash
kadaster eval-cascade \
  --eval-file assets/test.jsonl \
  --timestamp 20251201_154538 \
  --confidence-threshold 1.0
```

### Regex Evaluation

```bash
kadaster regex
```

### LLM Classification

```bash
kadaster llm-classify --threshold 10 --limit 20
```

---

## Technical Details

### Project Structure

```
mads-2025-hackathon-serverboys/
├── src/akte_classifier/
│   ├── cli.py                      # CLI commands (train, eval, eval-cascade)
│   ├── trainer.py                  # Training orchestration
│   ├── datasets/
│   │   ├── dataset.py              # DatasetFactory, LabelEncoder
│   │   └── vectorizers.py          # NebiusTextVectorizer, RegexVectorizer
│   ├── models/
│   │   ├── neural.py               # NeuralClassifier, HybridClassifier
│   │   ├── cascade.py              # CascadeClassifier
│   │   ├── llm.py                  # LLMClassifier
│   │   ├── prompts.py              # Prompt templates
│   │   └── regex.py                # RegexGenerator
│   └── utils/
│       ├── evaluation.py           # Metrics calculation
│       ├── tensor.py               # Caching utilities
│       └── early_stopping.py       # Training utilities
├── artifacts/
│   ├── models/                     # Trained model weights
│   ├── vectorcache/                # Cached embeddings/features
│   ├── img/                        # Plots and visualizations
│   └── csv/                        # Metrics and distributions
├── assets/
│   ├── aktes.jsonl                 # Full dataset
│   ├── train.jsonl                 # Training split (80%)
│   ├── test.jsonl                  # Test split (20%)
│   └── rechtsfeiten.csv            # Code descriptions
└── pyproject.toml                  # Dependencies
```

### Key Technologies

**Core:**
- Python 3.12
- PyTorch 2.x
- HuggingFace Datasets

**APIs:**
- Nebius API (embeddings + LLM)
- OpenAI SDK (client library)

**MLOps:**
- MLflow (experiment tracking)
- UV (package management)

**Data:**
- JSONL format for documents
- CSV for code descriptions

### Caching System

**How it works:**
1. Compute content hash of data
2. Compute pattern hash of regex rules
3. Check if cache file exists
4. If yes: Load from disk
5. If no: Compute and save

**Cache files:**
```
artifacts/vectorcache/
├── BAAI_bge-multilingual-gemma2_735228d6_split80_train_embeddings.pt
├── BAAI_bge-multilingual-gemma2_735228d6_split80_train_labels.pt
├── BAAI_bge-multilingual-gemma2_735228d6_split80_test_embeddings.pt
├── BAAI_bge-multilingual-gemma2_735228d6_split80_test_labels.pt
├── regex_827c31ca_735228d6_split80_train.pt
└── regex_827c31ca_735228d6_split80_test.pt
```

**Hash codes:**
- `735228d6`: Data content hash (train/test split)
- `827c31ca`: Regex pattern hash

**Benefits:**
- 100x speedup on subsequent runs
- Automatic invalidation on changes
- Disk space efficient (compressed tensors)

### Label Encoding

**Custom LabelEncoder:**
- Maps rechtsfeit codes (504, 537, ...) to dense indices (0, 1, ...)
- Reserves index 0 for unknown codes
- Bi-directional mapping: `code2idx` and `idx2code`
- Multi-hot encoding for multi-label support

**Methods:**
- `encode(codes: List[int]) -> Tensor`: Codes → multi-hot vector
- `decode(vector: Tensor) -> List[int]`: Multi-hot → codes
- `idx2code[idx: int] -> int`: Index → code
- `code2idx[code: int] -> int`: Code → index

---

## Future Work

**1. Debug Low Cascade F1**
- Investigate why F1 dropped from 0.885 to 0.22
- Analyze LLM predictions vs ground truth
- Check label encoding consistency
- Test on different sample ranges

**2. Optimize LLM Usage**
- Add request batching for multiple low-confidence cases
- Implement exponential backoff for retries
- Add caching for similar documents
- Set proper timeouts

**3. Threshold Tuning**
- Test different confidence thresholds (0.95, 0.97, 0.99, 0.995)
- Plot F1 vs threshold curve
- Find optimal balance of speed/accuracy/cost
- Consider per-class thresholds


**4. Model Improvements**
- Try larger hidden dimensions (512, 1024)
- Experiment with different architectures (transformers, attention)
- Add document structure features (sections, paragraphs)
- Incorporate external knowledge (legal ontologies)

**5. Explainability**
- Add attention visualization
- Show which regex patterns triggered
- Explain confidence scores
- Provide LLM reasoning chains

---

## Known Issues

### Issue 1: Low Cascade F1 (0.22 vs 0.885)
- **Status:** Under investigation
- **Hypothesis:** LLM making incorrect predictions for edge cases
- **Next steps:** Analyze LLM outputs, test on different samples

### Issue 2: LLM Timeout on Full Dataset
- **Status:** Mitigated with --limit flag
- **Root cause:** API connectivity or rate limiting
- **Workaround:** Process in batches of 100

### Issue 3: Slow Vectorization on First Run
- **Status:** Working as designed
- **Solution:** Caching system implemented
- **Impact:** Only affects first run, subsequent runs instant

---

## Appendix

### Available LLM Models (via Nebius)

**OpenAI Models:**
- `openai/gpt-oss-120b`: 120B params, 131k context (recommended)
- `openai/gpt-oss-70b`: 70B params, 65k context (faster)
- `openai/gpt-oss-13b`: 13B params, 16k context (budget)

**Meta Models:**
- `meta/llama-3.3-70b-instruct`: 70B params, 128k context
- `meta/llama-3.1-405b-instruct`: 405B params, 128k context (most powerful)

**Command to test:**
```bash
kadaster llm-classify \
  --llm-model "openai/gpt-oss-120b" \
  --limit 5
```

### Monitoring

**MLflow UI:**
```bash
# Access at: http://145.38.194.145:5002
```

**View experiments:**
- Experiment: `rechtsfeit-classification`
- Runs: Listed by timestamp
- Metrics: F1, precision, recall, loss
- Artifacts: Model weights, plots

### References

- Nebius API Docs: https://docs.nebius.ai
- MLflow Docs: https://mlflow.org/docs/latest/
- PyTorch Docs: https://pytorch.org/docs/

---


