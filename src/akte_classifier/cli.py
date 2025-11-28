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
def analyze(
    data_path: str = "assets/aktes.jsonl",
):
    """
    Analyze the dataset and generate a label distribution plot.
    """
    from akte_classifier.datasets.dataset import DatasetFactory
    from akte_classifier.utils.analysis import analyze_label_distribution

    factory = DatasetFactory(data_path)

    assert factory.train_dataset is not None, "Train dataset is None"
    analyze_label_distribution(factory.train_dataset)
    logger.success("Analysis Completed.")


@app.command()
def train(
    model_class: str = typer.Option(
        "NeuralClassifier",
        help="Model class to use: NeuralClassifier, HybridClassifier, or RegexOnlyClassifier",
    ),
    model_name: str = typer.Option(
        "prajjwal1/bert-mini", help="HuggingFace model name for text vectorization"
    ),
    epochs: int = typer.Option(10, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    device: Optional[str] = None,
):
    """
    Train the model.
    """
    # set use_regex=True if HybridClassifier or RegexOnlyClassifier is selected
    use_regex = model_class in ["HybridClassifier", "RegexOnlyClassifier"]
    logger.info("Starting training...")
    config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        model_class=model_class,
        model_name=model_name,
        learning_rate=learning_rate,
        use_regex=use_regex,
    )
    if device:
        config.device = device

    trainer = Trainer(config)
    trainer.run()


@app.command()
def regex(
    data_path: str = "assets/aktes.jsonl",
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

    logger.info("Initializing DatasetFactory...")
    factory = DatasetFactory(data_path)

    logger.info("Initializing Regex Model...")
    generator = RegexGenerator(csv_path)
    # Use the encoder from the factory for alignment
    assert factory.encoder is not None, "LabelEncoder is None"
    regex_vectorizer = RegexVectorizer(generator, label_encoder=factory.encoder)

    p = Path(output_path)
    output_path = str(p.with_name(f"{p.stem}_{regex_vectorizer.hash}{p.suffix}"))

    vectorized_data = factory.get_vectorized_dataset(
        vectorizer=None, regex_vectorizer=regex_vectorizer
    )
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


if __name__ == "__main__":
    app()
