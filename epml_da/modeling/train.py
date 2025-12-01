import json
from pathlib import Path

import joblib
from loguru import logger
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


def get_model(model_name: str):
    models = {
        "rf": RandomForestClassifier(n_estimators=200, random_state=121212),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), random_state=121212),
        "lr": LogisticRegression(
            random_state=121212,
        ),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name]


@app.command()
def train(
    data_path: str = PROCESSED_DATA_DIR / "dataset.csv",
    target: str = "quality_binary",
    model_name: str = "rf",
    output_dir: str = MODELS_DIR,
):
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, target)

    model = get_model(model_name)

    logger.info(f"Training model: {model_name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    logger.info(f"Metrics: {metrics}")

    model_path = Path(output_dir) / f"{model_name}.pkl"
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")
    metrics_path = Path(output_dir) / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    app()
