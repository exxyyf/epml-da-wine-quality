import altair as alt
import pandas as pd
import streamlit as st

df = pd.read_csv("models/model_versions.csv")

st.title("ML Model Version Comparison Dashboard")

st.subheader("Model Versions Table")
st.dataframe(df)

metric = st.selectbox(
    "Select metric to visualize", ["accuracy", "f1", "precision", "recall"]
)

chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(x="version:O", y=f"{metric}:Q", tooltip=["version", metric, "stage"])
)
st.altair_chart(chart, use_container_width=True)

st.subheader("Parameters Per Version")
st.write(
    df[
        [
            col
            for col in df.columns
            if col not in ["accuracy", "f1", "precision", "recall"]
        ]
    ]
)
