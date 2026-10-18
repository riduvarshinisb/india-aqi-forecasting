# %%
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "city_day.csv", parse_dates=["Date"])

# keep cities with at least 500 days of AQI
aqi_days = df.dropna(subset=["AQI"]).groupby("City").size()
keep = aqi_days[aqi_days >= 500].index
df = df[df["City"].isin(keep)]

# drop columns that are mostly missing or redundant
df = df.drop(columns=["Xylene", "PM10", "NH3", "AQI_Bucket"])

df = df.sort_values(["City", "Date"]).reset_index(drop=True)

print(df.shape)
print(df["City"].nunique(), "cities")
print(df.columns.tolist())

# %%
both = df.dropna(subset=["NO", "CO"])
same = (both["NO"] == both["CO"]).groupby(both["City"]).mean() * 100
print(same[same > 1].round(1).sort_values(ascending=False))

# %%
df.loc[df["City"] == "Ahmedabad", "CO"] = float("nan")

span = df.groupby("City")["Date"].agg(["min", "max", "count"])
span["expected"] = (span["max"] - span["min"]).dt.days + 1
span["missing_dates"] = span["expected"] - span["count"]
print(span[["count", "expected", "missing_dates"]])

# %%
g = df.groupby("City")

# past AQI values
for lag in [1, 2, 3, 7]:
    df[f"AQI_lag{lag}"] = g["AQI"].shift(lag)

# yesterday's pollutant readings
pollutants = ["PM2.5", "NO", "NO2", "NOx", "CO", "SO2", "O3", "Benzene", "Toluene"]
for p in pollutants:
    df[f"{p}_lag1"] = g[p].shift(1)

print(df[df["City"] == "Delhi"][["Date", "AQI", "AQI_lag1", "AQI_lag2"]].head(4))


# %%
# average AQI over the previous 7 and 30 days
for w in [7, 30]:
    df[f"AQI_roll{w}"] = g["AQI"].transform(
        lambda s: s.shift(1).rolling(w, min_periods=w // 2).mean()
    )

# calendar features
df["month"] = df["Date"].dt.month
df["dayofweek"] = df["Date"].dt.dayofweek
df["dayofyear"] = df["Date"].dt.dayofyear

# city as a category so LightGBM treats it as a label, not a number
df["City"] = df["City"].astype("category")

# sanity check: rolling mean on Jan 8 should equal mean of Jan 1-7
d = df[df["City"] == "Delhi"].set_index("Date")
print(d.loc["2015-01-08", "AQI_roll7"])
print(d.loc["2015-01-01":"2015-01-07", "AQI"].mean())

# %%
df = df.dropna(subset=["AQI"]).reset_index(drop=True)

df["split"] = "train"
df.loc[df["Date"].dt.year == 2019, "split"] = "val"
df.loc[df["Date"].dt.year == 2020, "split"] = "test"

print(df["split"].value_counts())

df.to_pickle(ROOT / "data" / "features.pkl")

