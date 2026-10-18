import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@st.cache_data
def load():
    df = pd.read_csv(Path(__file__).parent / "test_predictions.csv", parse_dates=["Date"])
    return df

df = load()

st.title("Next-day AQI Forecasting")
st.write("LightGBM forecasts vs actual AQI for 2020 (held-out test year), "
         "compared with a persistence baseline that predicts yesterday's value.")

city = st.selectbox("City", sorted(df["City"].unique()))
d = df[df["City"] == city].set_index("Date")

chart = d[["AQI", "pred_d", "AQI_lag1"]].rename(columns={
    "AQI": "Actual",
    "pred_d": "LightGBM forecast",
    "AQI_lag1": "Baseline (yesterday)",
}).round(1)
show_base = st.checkbox("Show baseline (yesterday's value)")
cols = ["Actual", "LightGBM forecast"]
if show_base:
    cols.append("Baseline (yesterday)")
colors = ["#0D3BE3", "#F87315", "#E9C762"][:len(cols)]
st.line_chart(chart[cols], color=colors)

mae_model = (d["pred_d"] - d["AQI"]).abs().mean()
mae_base = (d["AQI_lag1"] - d["AQI"]).abs().mean()

c1, c2, c3 = st.columns(3)
c1.metric("Model MAE", f"{mae_model:.1f}")
c2.metric("Baseline MAE", f"{mae_base:.1f}")
c3.metric("Improvement", f"{(1 - mae_model / mae_base) * 100:.1f}%")

st.caption("India's national lockdown began on 25 March 2020; note the drop in AQI after it.")