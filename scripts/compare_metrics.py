import pandas as pd


def compare_versions(csv_path="model_versions.csv"):
    df = pd.read_csv(csv_path)

    # Sort by version
    df = df.sort_values("version")

    # Compute deltas vs previous version
    df["delta_accuracy"] = df["accuracy"].diff()
    df["delta_f1"] = df["f1"].diff()
    df["delta_precision"] = df["precision"].diff()
    df["delta_recall"] = df["recall"].diff()

    print(df)
    return df


if __name__ == "__main__":
    compare_versions()
