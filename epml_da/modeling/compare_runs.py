"""
Utility for comparing MLflow experiment runs.
"""

import argparse
from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd


def get_experiment_id(experiment_name: str) -> str:
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    return exp.experiment_id


def fetch_runs(
    experiment_id: str,
    metric: str,
    top_n: int,
    filter_string: str | None = None,
):
    client = mlflow.tracking.MlflowClient()

    order_by = [f"metrics.{metric} DESC"]
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=order_by,
        max_results=top_n,
    )
    return runs


def runs_to_dataframe(runs) -> pd.DataFrame:
    records = []

    for r in runs:
        record = {
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName"),
        }

        record.update({f"metric_{k}": v for k, v in r.data.metrics.items()})
        record.update({f"param_{k}": v for k, v in r.data.params.items()})

        records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--output", default="models/run_comparison.csv")

    args = parser.parse_args()

    logger.info(f"Comparing runs for experiment: {args.experiment}")

    exp_id = get_experiment_id(args.experiment)
    runs = fetch_runs(exp_id, args.metric, args.top_n, args.filter)

    df = runs_to_dataframe(runs)
    logger.info(f"Fetched {len(df)} runs")

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved comparison to {output_path.resolve()}")
    print(df)


if __name__ == "__main__":
    main()
