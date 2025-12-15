from functools import wraps
import json
from pathlib import Path
import random
from typing import Callable

import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
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
# MLflow utilities
# ==========================


def train_run_name(
    path_to_params_config: str = PATH_TO_PARAMS,
) -> str:
    with open(path_to_params_config) as f:
        params = yaml.safe_load(f)
    model_name = params["train"]["model_name"]
    if model_name is not None:
        return f"{model_name}"


def mlflow_run(
    experiment_name: str,
    run_name_fn: Callable[..., str] | None = None,
):
    """
    MLflow decorator with auto-generated run_name from function arguments.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)
            run_name = (
                f"{func.__name__}-{run_name_fn()}"
                if run_name_fn is not None
                else func.__name__
            )
            with mlflow.start_run(run_name=run_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


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
    ModelClass = MODEL_REGISTRY[model_name]
    return ModelClass(**params)


# ==========================
# Training command
# ==========================


@app.command()
@mlflow_run(
    experiment_name="wine-quality-exp-baseline-models",
    run_name_fn=train_run_name,
)
def train(
    data_path: str = str(PROCESSED_DATA_DIR / "processed_data.csv"),
    target: str = "quality_binary",
    model_name: str | None = None,
    output_dir: str = str(MODELS_DIR),
    seed: int = 121212,
):
    """
    Train a model and log parameters, metrics and artifacts to MLflow.
    """

    # ---- Reproducibility ----
    set_seed(seed)
    mlflow.log_param("seed", seed)

    with open(PATH_TO_PARAMS) as f:
        params = yaml.safe_load(f)

    model_name = model_name or params["train"]["model_name"]

    # ---- General MLflow metadata ----
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("target", target)

    mlflow.set_tag("model_type", model_name)
    mlflow.set_tag("framework", "sklearn")
    mlflow.set_tag("stage", "training")

    model_params = params["model"][model_name]
    mlflow.log_params(model_params)

    # ---- Data ----
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, target)

    # ---- Model ----
    model = get_model(model_name, model_params)
    logger.info(f"Training model: {model_name}")
    model.fit(X_train, y_train)

    # ---- Evaluation ----
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    logger.info(f"Metrics: {metrics}")

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # ---- Save artifacts ----
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = mlflow.active_run().info.run_id

    model_path = output_dir / f"{model_name}_{run_id}.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # ---- Log MLflow model ----
    mlflow.sklearn.log_model(
        model,
        name=f"{model_name}_model",
        registered_model_name=f"wq-demo-{model_name}",
    )

    # ---- Save metrics.json ----
    metrics_path = output_dir / f"metrics_{run_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    mlflow.log_artifact(metrics_path)

    logger.info("Training and logging finished successfully")


if __name__ == "__main__":
    app()
