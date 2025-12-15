from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

st.title("MLflow Model Version Dashboard")

csv_path = st.text_input(
    "Path to model versions CSV",
    value="models/model_versions.csv",
)

path = Path(csv_path)

if not path.exists():
    st.error(f"File not found: {path}")
    st.stop()

df = pd.read_csv(path)

if df.empty:
    st.warning("No model versions found.")
    st.stop()

df = df.sort_values("version")

st.subheader("Registered Model Versions")
st.dataframe(df, use_container_width=True)

metric_columns = [c for c in df.columns if c.startswith("metric_")]

metric = st.selectbox(
    "Select metric",
    metric_columns,
)

chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x="version:O",
        y=f"{metric}:Q",
        tooltip=["version", "stage", metric],
    )
)

st.altair_chart(chart, use_container_width=True)

st.subheader("Parameters per Version")
param_columns = [c for c in df.columns if c.startswith("param_")]
st.dataframe(df[["version"] + param_columns], use_container_width=True)
