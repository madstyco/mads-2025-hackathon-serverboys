import os
from typing import Any, Dict, Protocol

import mlflow
from dotenv import load_dotenv
from loguru import logger

from akte_classifier.utils.env import get_dev_name

load_dotenv()


class ExperimentLogger(Protocol):
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...

    def log_params(self, params: Dict[str, Any]) -> None: ...

    def log_tags(self, tags: Dict[str, str]) -> None: ...

    def log_artifact(self, local_path: str) -> None: ...


class ConsoleLogger:
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} | {log_str}")

    def log_params(self, params: Dict[str, Any]) -> None:
        logger.info(f"Params: {params}")

    def log_tags(self, tags: Dict[str, str]) -> None:
        logger.info(f"Tags: {tags}")

    def log_artifact(self, local_path: str) -> None:
        logger.info(f"Artifact saved at: {local_path}")


class MLFlowLogger:
    def __init__(self, experiment_name: str = "kadaster_experiment"):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            # Ensure logs directory exists
            os.makedirs("logs", exist_ok=True)
            tracking_uri = "sqlite:///logs/mlflow.db"
            logger.info(
                "MLFLOW_TRACKING_URI not set, falling back to local sqlite: sqlite:///logs/mlflow.db"
            )
        else:
            logger.info(f"Using MLFlow tracking URI: {tracking_uri}")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()

        # Log developer name
        dev_name = get_dev_name()
        mlflow.set_tag("dev_name", dev_name)
        logger.info(f"MLflow run started: {self.run.info.run_id} (User: {dev_name})")

    def enable_genai_autolog(self) -> None:
        """
        Enables MLFlow autologging for OpenAI (GenAI).
        """
        try:
            mlflow.openai.autolog()
            logger.info("Enabled MLFlow OpenAI autologging.")
        except Exception as e:
            logger.warning(f"Failed to enable OpenAI autologging: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_tags(self, tags: Dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def __del__(self):
        mlflow.end_run()


class CompositeLogger:
    def __init__(self, loggers: list[ExperimentLogger]):
        self.loggers = loggers

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for exp_logger in self.loggers:
            exp_logger.log_metrics(metrics, step)

    def log_params(self, params: Dict[str, Any]) -> None:
        for exp_logger in self.loggers:
            exp_logger.log_params(params)

    def log_tags(self, tags: Dict[str, str]) -> None:
        for exp_logger in self.loggers:
            exp_logger.log_tags(tags)

    def log_artifact(self, local_path: str) -> None:
        for exp_logger in self.loggers:
            exp_logger.log_artifact(local_path)
