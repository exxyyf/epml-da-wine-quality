import json
from pathlib import Path

import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from typer import Typer
import yaml

from epml_da.config import MODELS_DIR, PROCESSED_DATA_DIR

app = Typer()


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


@app.command()
def train(
    data_path: str = PROCESSED_DATA_DIR / "dataset.csv",
    target: str = "quality_binary",
    model_name: str | None = None,
    output_dir: str = MODELS_DIR,
):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # if model name not passed, read from config
    model_name = model_name or params["train"]["model_name"]
    # ---- MLflow experiment setup ----
    mlflow.set_experiment("wine-quality-demo")

    with mlflow.start_run(run_name=f"{model_name}_run"):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("target", target)

        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        model_params = params["model"][model_name]

        model = get_model(model_name, model_params)

        mlflow.log_params(model_params)

        df = load_data(data_path)
        X_train, X_test, y_train, y_test = split_data(df, target)

        logger.info(f"Training model: {model_name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        logger.info(f"Metrics: {metrics}")

        # Log metrics to MLflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ---- Save model to local filesystem (for DVC) ----
        model_path = Path(output_dir) / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # ---- Log model to MLflow as artifact ----
        mlflow.sklearn.log_model(
            model,
            name=f"{model_name}_model",
            registered_model_name=f"wq-demo-{model_name}",
        )

        # ---- Save metrics.json for DVC ----
        metrics_path = Path(output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact(str(metrics_path))

        logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    app()
