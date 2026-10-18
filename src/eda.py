# %%
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "city_day.csv"
df = pd.read_csv(DATA, parse_dates=["Date"])

print(df.shape)
print(df["City"].nunique(), "cities")
print(df["Date"].min(), "to", df["Date"].max())
print(df.head())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

missing = df.isna().mean().sort_values(ascending=False) * 100
print(missing.round(1))

plt.figure(figsize=(9, 5))
missing.plot(kind="barh")
plt.xlabel("% missing")
plt.title("Missing values per column")
plt.tight_layout()
plt.savefig(OUT / "01_missing.png", dpi=120)
plt.show()


# %%
aqi_count = df.dropna(subset=["AQI"]).groupby("City").size().sort_values()
print(aqi_count)

plt.figure(figsize=(9, 7))
aqi_count.plot(kind="barh", color="teal")
plt.xlabel("Days with AQI recorded")
plt.title("Usable AQI days per city")
plt.tight_layout()
plt.savefig(OUT / "02_aqi_days_per_city.png", dpi=120)
plt.show()

# %%
cities = ["Delhi", "Chennai", "Mumbai", "Bengaluru"]
plt.figure(figsize=(12, 5))
for c in cities:
    s = df[df["City"] == c].set_index("Date")["AQI"]
    s.rolling(7).mean().plot(label=c)
plt.ylabel("AQI (7-day avg)")
plt.title("AQI over time")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "03_aqi_trend.png", dpi=120)
plt.show()

# %%
df["Month"] = df["Date"].dt.month
monthly = df.groupby("Month")["AQI"].median()
print(monthly.round(0))

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Month", y="AQI", showfliers=False, color="skyblue")
plt.title("AQI distribution by month (all cities)")
plt.tight_layout()
plt.savefig(OUT / "04_aqi_by_month.png", dpi=120)
plt.show()
