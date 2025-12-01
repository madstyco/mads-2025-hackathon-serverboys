import sys
from typing import Optional

import typer
from loguru import logger

from akte_classifier.models.regex import RegexGenerator
from akte_classifier.trainer import Trainer, TrainingConfig

app = typer.Typer(no_args_is_help=True)

logger.remove()
logger.add(sys.stderr, level="SUCCESS")
logger.add("logs/cli.log", rotation="1 MB", level="DEBUG")


@app.command()
def llm_classify(
    threshold: int = typer.Option(10, help="Threshold for long-tail labels"),
    limit: Optional[int] = typer.Option(None, help="Number of samples to classify (None for all)"),
    model_name: str = typer.Option("meta-llama/Meta-Llama-3.1-8B-Instruct-fast", help="LLM model name"),
    experiment_name: str = typer.Option("test_experiment", help="MLFlow experiment name"),
    max_length: Optional[int] = typer.Option(None, help="Max token length (None = auto/unlimited)"),
):
    """
    Classify long-tail samples using an LLM.
    """
    from akte_classifier.trainer import LLMRunner

    logger.info("\n\n ======= Starting LLM classification =======")

    runner = LLMRunner(
        threshold=threshold,
        limit=limit,
        model_name=model_name,
        experiment_name=experiment_name,
        max_length=max_length,
    )
    runner.run()
    logger.success("\n\n ======= LLM classification completed ========")


@app.command()
def split_data(
    data_path: str = "assets/aktes.jsonl",
    output_dir: str = "assets",
    test_size: float = typer.Option(0.1, help="Fraction of data to use for testing"),
    seed: int = typer.Option(42, help="Random seed for splitting"),
):
    """
    Split the dataset into train.jsonl and test.jsonl.
    """
    import os

    from datasets import load_dataset

    logger.info("\n\n ======= Starting data split =======")
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.success(f"Loaded dataset from {data_path}")

    logger.info(f"Splitting dataset with test_size={test_size} and seed={seed}...")
    split_ds = dataset.train_test_split(test_size=test_size, seed=seed)

    train_ds = split_ds["train"]
    test_ds = split_ds["test"]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    logger.info(f"Saving train set ({len(train_ds)} samples) to {train_path}...")
    train_ds.to_json(train_path)

    logger.info(f"Saving test set ({len(test_ds)} samples) to {test_path}...")
    test_ds.to_json(test_path)

    logger.success("\n\n ======= Data split completed =======")


@app.command()
def analyze(
    data_path: str = "assets/train.jsonl",
):
    """
    Analyze the dataset and generate a label distribution plot.
    """
    from akte_classifier.datasets.dataset import DatasetFactory
    from akte_classifier.utils.analysis import analyze_label_distribution

    logger.info("\n\n ======= Starting dataset analysis =======")
    factory = DatasetFactory(data_path)
    logger.success(f"Loaded dataset from {data_path}")

    assert factory.train_dataset is not None, "Train dataset is None"
    analyze_label_distribution(factory.train_dataset)
    logger.success("\n\n ======= Dataset analysis completed =======")


@app.command()
def train(
    data_path: str = typer.Option("assets/train.jsonl", help="Path to training data"),
    model_class: str = typer.Option(
        "NeuralClassifier",
        help="Model class to use: NeuralClassifier, HybridClassifier, or RegexOnlyClassifier",
    ),
    model_name: str = typer.Option("prajjwal1/bert-tiny", help="HuggingFace model name for text vectorization"),
    epochs: int = typer.Option(10, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    hidden_dim: int = typer.Option(256, help="Hidden layer dimension"),
    max_length: Optional[int] = typer.Option(None, help="Max token length (None = auto)"),
    pooling: Optional[str] = typer.Option(None, help="Pooling strategy: 'mean', 'cls', or None (auto)"),
    experiment_name: str = typer.Option("kadaster_experiment", help="MLFlow experiment name"),
    device: Optional[str] = None,
    csv_path: str = typer.Option("assets/rechtsfeiten.csv", help="Path to CSV file with regex patterns"),
    patience: int = typer.Option(3, help="Early stopping patience"),
    min_delta: float = typer.Option(0.0, help="Early stopping min delta"),
    use_nebius: bool = typer.Option(False, help="Use Nebius API for embeddings"),
):
    """
    Train a classifier model.
    """
    # set use_regex=True if HybridClassifier or RegexOnlyClassifier is selected
    use_regex = model_class in ["HybridClassifier", "RegexOnlyClassifier"]
    logger.info("\n\n ======= Starting training =======")
    from akte_classifier.trainer import get_default_device

    if device is None:
        device = get_default_device()

    config = TrainingConfig(
        data_path=data_path,
        num_epochs=epochs,
        batch_size=batch_size,
        model_class=model_class,
        model_name=model_name,
        learning_rate=learning_rate,
        use_regex=use_regex,
        hidden_dim=hidden_dim,
        max_length=max_length,
        pooling=pooling,
        experiment_name=experiment_name,
        csv_path=csv_path,
        patience=patience,
        min_delta=min_delta,
        device=device,
        use_nebius=use_nebius,
    )

    trainer = Trainer(config)
    trainer.run()
    logger.success("\n\n ======= Training completed =======")


@app.command()
def regex(
    data_path: str = "assets/train.jsonl",
    csv_path: str = "assets/rechtsfeiten.csv",
    output_path: str = "artifacts/csv/regex_evaluation.csv",
):
    """
    Evaluate the regex model using cached features.
    """
    from pathlib import Path

    from akte_classifier.datasets.dataset import DatasetFactory
    from akte_classifier.models.regex import RegexVectorizer
    from akte_classifier.utils.evaluation import Evaluator

    logger.info("\n\n ======= Starting regex evaluation =======")
    factory = DatasetFactory(data_path)
    logger.success(f"Loaded dataset from {data_path}")

    logger.info("Initializing Regex Model...")
    generator = RegexGenerator(csv_path)
    # Use the encoder from the factory for alignment
    assert factory.encoder is not None, "LabelEncoder is None"
    regex_vectorizer = RegexVectorizer(generator, label_encoder=factory.encoder)

    p = Path(output_path)
    output_path = str(p.with_name(f"{p.stem}_{regex_vectorizer.hash}{p.suffix}"))

    vectorized_data = factory.get_vectorized_dataset(vectorizer=None, regex_vectorizer=regex_vectorizer)
    train_data = vectorized_data["train"]

    logger.info("Evaluating on training set...")
    regex_features = train_data.regex_features  # Shape: (N, NumClasses)
    true_labels = train_data.labels  # Shape: (N, NumClasses)

    if regex_features is None:
        logger.error("Regex features not found in dataset!")
        return

    # Use Evaluator for metrics
    evaluator = Evaluator(num_classes=len(factory.encoder))
    evaluator.evaluate_regex_performance(
        regex_features=regex_features.numpy(),
        true_labels=true_labels.numpy(),
        regex_map=generator.regexes,
        code2idx=factory.encoder.code2idx,
        output_path=output_path,
    )


@app.command()
def eval(
    eval_file: str = typer.Option(..., help="Path to file to evaluate"),
    model_path: Optional[str] = typer.Option(None, help="Path to saved model weights (.pt)"),
    codes_path: Optional[str] = typer.Option(None, help="Path to saved encoder codes (.json)"),
    timestamp: Optional[str] = typer.Option(None, help="Timestamp of the run to evaluate (e.g. 20251130_182204)"),
    model_class: Optional[str] = typer.Option(
        None,
        help="Model class to use: NeuralClassifier, HybridClassifier, or RegexOnlyClassifier (auto-detected from config if not provided)",
    ),
    model_name: str = typer.Option("prajjwal1/bert-tiny", help="HuggingFace model name for text vectorization"),
    batch_size: int = typer.Option(32, help="Batch size"),
    hidden_dim: int = typer.Option(256, help="Hidden layer dimension"),
    max_length: Optional[int] = typer.Option(None, help="Max token length (None = auto)"),
    pooling: Optional[str] = typer.Option(None, help="Pooling strategy: 'mean', 'cls', or None (auto)"),
    csv_path: str = typer.Option("assets/rechtsfeiten.csv", help="Path to label descriptions CSV"),
    device: Optional[str] = None,
):
    """
    Evaluate a trained model on a specific file.
    """
    import glob
    import os

    # Auto-detect model and codes if not provided
    config_path = None

    if timestamp:
        model_dir = "artifacts/models"
        # Search for files matching the timestamp
        pt_files = glob.glob(os.path.join(model_dir, f"*_{timestamp}.pt"))
        json_files = glob.glob(os.path.join(model_dir, f"*_{timestamp}_codes.json"))
        config_files = glob.glob(os.path.join(model_dir, f"*_{timestamp}_config.json"))

        if len(pt_files) == 1:
            model_path = pt_files[0]
            logger.info(f"Resolved model path from timestamp: {model_path}")
        elif len(pt_files) > 1:
            logger.error(f"Ambiguous timestamp. Found {len(pt_files)} models matching {timestamp}.")
            raise typer.Exit(code=1)
        else:
            logger.error(f"No model found for timestamp {timestamp}.")
            raise typer.Exit(code=1)

        if len(json_files) == 1:
            codes_path = json_files[0]
            logger.info(f"Resolved codes path from timestamp: {codes_path}")

        if len(config_files) == 1:
            config_path = config_files[0]
            logger.info(f"Resolved config path from timestamp: {config_path}")

    elif not model_path or not codes_path:
        model_dir = "artifacts/models"
        pt_files = glob.glob(os.path.join(model_dir, "*.pt"))
        json_files = glob.glob(os.path.join(model_dir, "*_codes.json"))
        config_files = glob.glob(os.path.join(model_dir, "*_config.json"))

        if len(pt_files) == 1 and len(json_files) == 1:
            if not model_path:
                model_path = pt_files[0]
                logger.info(f"Auto-detected model path: {model_path}")
            if not codes_path:
                codes_path = json_files[0]
                logger.info(f"Auto-detected codes path: {codes_path}")

            # Try to find matching config
            # Expected: {model_slug}_{timestamp}_config.json
            # We can try to match the timestamp from the model path
            import re

            timestamp_match = re.search(r"(\d{8}_\d{6})", os.path.basename(model_path))
            if timestamp_match:
                ts = timestamp_match.group(1)
                matching_configs = [c for c in config_files if ts in c]
                if len(matching_configs) == 1:
                    config_path = matching_configs[0]
                    logger.info(f"Auto-detected config path: {config_path}")
        else:
            if not model_path:
                logger.error(
                    f"Model path not provided and auto-detection failed. Found {len(pt_files)} .pt files in {model_dir}."
                )
                raise typer.Exit(code=1)
            if not codes_path:
                logger.error(
                    f"Codes path not provided and auto-detection failed. Found {len(json_files)} .json files in {model_dir}."
                )
                raise typer.Exit(code=1)

    # Load config if found
    loaded_config = {}
    if config_path:
        import json

        with open(config_path, "r") as f:
            loaded_config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")

    model_class = loaded_config.get("model_class", model_class)
    if model_class is None:
        model_class = "NeuralClassifier"  # Default fallback
        logger.warning(f"Model class not specified. Using default: {model_class}")

    model_name = loaded_config.get("model_name", model_name)
    hidden_dim = loaded_config.get("hidden_dim", hidden_dim)
    max_length = loaded_config.get("max_length", max_length)
    pooling = loaded_config.get("pooling", pooling)
    use_regex = loaded_config.get("use_regex", model_class in ["HybridClassifier", "RegexOnlyClassifier"])

    logger.info(f"Starting evaluation on {eval_file}...")
    logger.info(f"Model Class: {model_class}, Model Name: {model_name}")

    # We reuse TrainingConfig but only fill relevant parts
    config = TrainingConfig(
        batch_size=batch_size,  # Allow CLI override
        model_class=model_class,
        model_name=model_name,
        use_regex=use_regex,
        hidden_dim=hidden_dim,
        max_length=max_length,
        pooling=pooling,
        # Dummy values for training-specific args
        num_epochs=0,
        learning_rate=0.0,
        experiment_name="eval_run",
    )

    if device:
        config.device = device

    trainer = Trainer(config)
    logger.info(f"starting eval on {eval_file}...")

    assert codes_path is not None, "Codes path must be provided or auto-detected"
    trainer.evaluate_file(eval_file, model_path, codes_path, csv_path=csv_path)
    logger.success("Evaluation completed.")


@app.command()
def eval_cascade(
    eval_file: str = typer.Option(..., help="Path to test data JSONL file"),
    timestamp: str = typer.Option(..., help="Model timestamp to load"),
    confidence_threshold: float = typer.Option(0.99, help="Confidence threshold for LLM fallback (0-1)"),
    llm_provider: str = typer.Option("nebius", help="LLM provider: openai, anthropic, or nebius"),
    llm_model: str = typer.Option("openai/gpt-oss-120b", help="LLM model (default: OpenAI GPT 120B via Nebius)"),
    limit: Optional[int] = typer.Option(None, help="Limit number of predictions (for testing)"),
    device: Optional[str] = None,
):
    """
    Evaluate cascade classifier (hybrid + LLM fallback) on test set.

    Uses hybrid model for high-confidence predictions (≥threshold),
    falls back to LLM for low-confidence cases.
    """

    from akte_classifier.datasets.dataset import DatasetFactory
    from akte_classifier.models.cascade import load_cascade_classifier
    from akte_classifier.utils.evaluation import Evaluator

    logger.info("\n\n ======= Starting cascade evaluation =======")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"LLM provider: {llm_provider}, model: {llm_model}")

    # Load cascade classifier
    logger.info(f"Loading cascade classifier with timestamp {timestamp}...")
    cascade = load_cascade_classifier(
        model_timestamp=timestamp,
        confidence_threshold=confidence_threshold,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # Load test data
    logger.info(f"Loading test data from {eval_file}...")
    factory = DatasetFactory(eval_file)
    test_dataset = factory.train_dataset  # It's actually test data

    if test_dataset is None:
        logger.error("Failed to load test dataset")
        raise typer.Exit(code=1)

    # Load cached embeddings directly to avoid re-vectorization
    import glob

    import torch

    logger.info("Looking for cached embeddings...")

    # Find cached embedding files
    cache_pattern = "artifacts/vectorcache/BAAI_bge-multilingual-gemma2_*_split80_test_embeddings.pt"
    embedding_files = glob.glob(cache_pattern)

    labels_pattern = "artifacts/vectorcache/BAAI_bge-multilingual-gemma2_*_split80_test_labels.pt"
    labels_files = glob.glob(labels_pattern)

    regex_pattern = "artifacts/vectorcache/regex_*_split80_test.pt"
    regex_files = glob.glob(regex_pattern)

    if not embedding_files or not regex_files or not labels_files:
        logger.error("Cached embeddings not found! " "Please run training first to generate cached embeddings.")
        raise typer.Exit(code=1)

    # Load cached data
    embedding_file = embedding_files[0]
    labels_file = labels_files[0]
    regex_file = regex_files[0]

    logger.info(f"Loading cached embeddings from: {embedding_file}")
    logger.info(f"Loading cached labels from: {labels_file}")
    logger.info(f"Loading cached regex features from: {regex_file}")

    embeddings = torch.load(embedding_file, map_location="cpu")
    labels = torch.load(labels_file, map_location="cpu")
    regex_features = torch.load(regex_file, map_location="cpu")

    # Create dataset from cached data
    from akte_classifier.datasets.dataset import VectorizedRechtsfeitDataset

    test_data = VectorizedRechtsfeitDataset(
        embeddings=embeddings,
        labels=labels,
        regex_features=regex_features,
    )

    # Prepare for evaluation
    num_samples = len(test_data)
    if limit:
        num_samples = min(num_samples, limit)

    logger.info(f"Evaluating {num_samples} samples...")

    all_predictions = []
    all_true_labels = []

    # Evaluate each sample
    for i in range(num_samples):
        # Get text from original dataset (returns tuple: (text, label_tensor))
        text, label_tensor = test_dataset[i]

        # Get true label index from multi-hot encoded tensor
        true_label_idx = torch.argmax(label_tensor).item()

        # Get embeddings and regex features
        text_emb = test_data.embeddings[i].unsqueeze(0)  # Add batch dimension
        regex_feat = test_data.regex_features[i].unsqueeze(0)

        # Get cascade prediction
        result = cascade.predict(text, text_emb, regex_feat)

        all_predictions.append(result["prediction"])
        all_true_labels.append(factory.encoder.idx2code.get(true_label_idx, 0))

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{num_samples} samples...")

    # Print cascade statistics
    cascade.print_statistics()

    # Calculate overall metrics
    logger.info("\n\n ======= Overall Metrics =======")
    evaluator = Evaluator(num_classes=len(factory.encoder))

    # Convert predictions to indices
    # Note: LLM might return strings, convert to int first
    pred_indices = [factory.encoder.code2idx.get(int(p), 0) for p in all_predictions]
    true_indices = [factory.encoder.code2idx.get(int(t), 0) for t in all_true_labels]

    # Convert to numpy arrays for sklearn
    import numpy as np

    true_np = np.array(true_indices)
    pred_np = np.array(pred_indices)

    metrics = evaluator.compute_metrics(true_np, pred_np)

    logger.success(f"F1 Micro: {metrics['f1_micro']:.4f}")
    logger.success(f"F1 Macro: {metrics['f1_macro']:.4f}")
    logger.success(f"Precision Micro: {metrics['precision_micro']:.4f}")
    logger.success(f"Recall Micro: {metrics['recall_micro']:.4f}")

    logger.success("\n\n ======= Cascade evaluation completed =======")


if __name__ == "__main__":
    app()
