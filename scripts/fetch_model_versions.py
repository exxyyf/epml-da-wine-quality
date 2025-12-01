import mlflow
import pandas as pd


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
    df = fetch_model_versions("wq-demo-rf")
    print(df)
    df.to_csv("model_versions.csv", index=False)
