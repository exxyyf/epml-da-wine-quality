from pathlib import Path
from typing import List

from loguru import logger
import numpy as np
import pandas as pd
import typer

from epml_da.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def load_raw_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    return pd.read_csv(path)


def create_binary_target(
    data: pd.DataFrame,
    target_col: str,
    good_wine_threshold: int = 6,
) -> pd.DataFrame:
    """Creates a binary target column (1/0) based on threshold.

    If value > threshold = 1 (good wine)
    Else = 0 (bad wine).
    """
    data = data.copy()
    binary_target_col = f"{target_col}_binary"

    data[binary_target_col] = np.where(data[target_col] > good_wine_threshold, 1, 0)
    logger.info("Created binary target - good or bad wine.")
    return data


def remove_unused_columns(
    data: pd.DataFrame,
    unused_cols: List[str],
) -> pd.DataFrame:
    """Removes columns from df"""
    data = data.copy()
    data = data.drop(columns=unused_cols).copy()
    logger.info(f"Removed cols {unused_cols}")
    return data


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "WineQT.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- PROCESSING DATA ----
    df = load_raw_data(input_path)
    logger.info("Processing dataset...")
    df = create_binary_target(
        df,
        "quality",
        good_wine_threshold=6,
    )
    df = remove_unused_columns(
        df,
        [
            "quality",
            "Id",
        ],
    )
    df.to_csv(output_path, index=False)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
