import argparse
from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd


def fetch_model_versions(model_name: str) -> pd.DataFrame:
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch versions for model '{model_name}': {e}")

    records = []

    for v in versions:
        run = client.get_run(v.run_id)

        record = {
            "model_name": model_name,
            "version": int(v.version),
            "stage": v.current_stage,
            "run_id": v.run_id,
        }

        record.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
        record.update({f"param_{k}": v for k, v in run.data.params.items()})

        records.append(record)

    df = pd.DataFrame(records)
    return df.sort_values("version")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", required=True, help="Registered MLflow model name"
    )
    parser.add_argument("--output", default="models/model_versions.csv")

    args = parser.parse_args()

    logger.info(f"Fetching versions for model: {args.model_name}")
    df = fetch_model_versions(args.model_name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved model versions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
