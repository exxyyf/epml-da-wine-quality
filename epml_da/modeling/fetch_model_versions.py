from loguru import logger
import mlflow
import pandas as pd
import yaml


def load_model_name(config_path: str) -> str:
    """Loading model_version.yaml"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "model" in config:
        return config["model"]["name"]
    elif "name" in config:
        return config["name"]
    else:
        raise ValueError("model name not found in config file")


def fetch_model_versions(model_name: str) -> pd.DataFrame:
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    records = []
    for v in versions:
        run_id = v.run_id
        run = client.get_run(run_id)

        metrics = run.data.metrics
        params = run.data.params

        records.append(
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": run_id,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                **params,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    model_name = load_model_name("model_version.yaml")
    logger.info(f"Loaded model {model_name}")
    df = fetch_model_versions(model_name)
    logger.info(f"Fetched model info for {model_name}")
    df.to_csv("models/model_versions.csv", index=False)
