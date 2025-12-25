from pathlib import Path
import random

from clearml import Task
from clearml.model import OutputModel
import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from typer import Typer
import yaml

from epml_da.config import MODELS_DIR, PROCESSED_DATA_DIR

PATH_TO_PARAMS = "params.yaml"
app = Typer()

# ==========================
# Reproducibility
# ==========================


def set_seed(seed: int = 121212):
    random.seed(seed)
    np.random.seed(seed)


# ==========================
# Data & evaluation
# ==========================


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=121212)


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


# ==========================
# Models
# ==========================

MODEL_REGISTRY = {
    "rf": RandomForestClassifier,
    "mlp": MLPClassifier,
    "lr": LogisticRegression,
}


def get_model(model_name: str, params: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](**params)


# ==========================
# Training command
# ==========================


@app.command()
def train(
    data_path: str = str(PROCESSED_DATA_DIR / "processed_data.csv"),
    target: str = "quality_binary",
    model_name: str | None = None,
    output_dir: str = str(MODELS_DIR),
    seed: int = 121212,
):
    """
    Train a model and track everything with ClearML
    """

    # ==========================
    # 1. ClearML Task
    # ==========================
    task = Task.init(
        project_name="Wine Quality Prediction",
        task_name="baseline-model-training",
        task_type=Task.TaskTypes.training,
    )

    task.set_tags(["sklearn", "baseline"])
    logger_clearml = task.get_logger()

    # ==========================
    # 2. Params
    # ==========================
    with open(PATH_TO_PARAMS) as f:
        params = yaml.safe_load(f)

    model_name = model_name or params["train"]["model_name"]

    config = {
        "seed": seed,
        "data_path": data_path,
        "target": target,
        "model_name": model_name,
        "model_params": params["model"][model_name],
    }

    task.connect(config)

    set_seed(seed)

    # ==========================
    # 3. Data
    # ==========================
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, target)

    # ==========================
    # 4. Model
    # ==========================
    model = get_model(model_name, config["model_params"])
    logger.info(f"Training model: {model_name}")
    model.fit(X_train, y_train)

    # ==========================
    # 5. Evaluation
    # ==========================
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    logger.info(f"Metrics: {metrics}")

    for k, v in metrics.items():
        logger_clearml.report_scalar(
            title="metrics",
            series=k,
            value=v,
            iteration=0,
        )

    # ==========================
    # 6. Save artifacts
    # ==========================
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)

    task.upload_artifact(
        name="model_pickle",
        artifact_object=model_path,
    )

    # ==========================
    # 7. Model registry & versioning
    # ==========================
    output_model = OutputModel(
        task=task,
        name=f"wine-quality-{model_name}",
        framework="sklearn",
        tags=[
            "wine-quality",
            "classification",
            model_name,
        ],
    )

    output_model.update_weights(
        weights_filename=str(model_path),
        auto_delete_file=False,
    )

    # ==========================
    # 8. Model metadata (ВАЖНО: по одному ключу)
    # ==========================
    output_model.set_metadata("model_name", model_name)
    output_model.set_metadata("dataset", "wine-quality")
    output_model.set_metadata("task", "classification")
    output_model.set_metadata("target", target)

    for metric_name, metric_value in metrics.items():
        output_model.set_metadata(metric_name, metric_value)

    logger.info("Training finished and tracked with ClearML")


if __name__ == "__main__":
    app()
