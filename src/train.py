# %%
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_pickle(ROOT / "data" / "features.pkl")

# keep rows where yesterday's AQI exists, so every model is scored on the same days
df = df.dropna(subset=["AQI_lag1"]).reset_index(drop=True)

train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]

def score(y, p):
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    return round(float(mae), 2), round(float(rmse), 2)

print("Baseline val  (MAE, RMSE):", score(val["AQI"], val["AQI_lag1"]))
print("Baseline test (MAE, RMSE):", score(test["AQI"], test["AQI_lag1"]))

# %%
import lightgbm as lgb

features = [c for c in df.columns if "_lag" in c or "_roll" in c]
features += ["month", "dayofweek", "dayofyear", "City"]

model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

model.fit(
    train[features], train["AQI"],
    eval_set=[(val[features], val["AQI"])],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(100, verbose=False)],
)

print("Features used:", len(features))
print("Best iteration:", model.best_iteration_)
print("LightGBM val (MAE, RMSE):", score(val["AQI"], model.predict(val[features])))


# %%
test_pred = model.predict(test[features])

base = score(test["AQI"], test["AQI_lag1"])
lgbm = score(test["AQI"], test_pred)

print("Baseline test (MAE, RMSE):", base)
print("LightGBM test (MAE, RMSE):", lgbm)
print("MAE improvement: ", round((1 - lgbm[0] / base[0]) * 100, 1), "%")
print("RMSE improvement:", round((1 - lgbm[1] / base[1]) * 100, 1), "%")

# %%
test = test.assign(pred=test_pred)
lockdown = pd.Timestamp("2020-03-25")

for name, part in [("Pre-lockdown", test[test["Date"] < lockdown]),
                   ("Lockdown", test[test["Date"] >= lockdown])]:
    print(name, "| rows:", len(part),
          "| baseline:", score(part["AQI"], part["AQI_lag1"]),
          "| lgbm:", score(part["AQI"], part["pred"]),
          "| mean error:", round(float((part["pred"] - part["AQI"]).mean()), 2))


# %%
model_d = lgb.LGBMRegressor(**model.get_params())

model_d.fit(
    train[features], train["AQI"] - train["AQI_lag1"],
    eval_set=[(val[features], val["AQI"] - val["AQI_lag1"])],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(100, verbose=False)],
)

test["pred_d"] = test["AQI_lag1"] + model_d.predict(test[features])

print("Overall      | delta lgbm:", score(test["AQI"], test["pred_d"]))
for name, part in [("Pre-lockdown", test[test["Date"] < lockdown]),
                   ("Lockdown", test[test["Date"] >= lockdown])]:
    print(name, "| delta lgbm:", score(part["AQI"], part["pred_d"]),
          "| mean error:", round(float((part["pred_d"] - part["AQI"]).mean()), 2))


# %%
val_pred_level = model.predict(val[features])
val_pred_delta = val["AQI_lag1"] + model_d.predict(val[features])

print("Val | level model:", score(val["AQI"], val_pred_level))
print("Val | delta model:", score(val["AQI"], val_pred_delta))


# %%
import shap
import matplotlib.pyplot as plt

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

explainer = shap.TreeExplainer(model_d)
shap_values = explainer.shap_values(test[features])

importance = pd.Series(np.abs(shap_values).mean(axis=0), index=features)
print(importance.sort_values(ascending=False).head(8).round(2))

shap.summary_plot(shap_values, test[features], show=False)
plt.tight_layout()
plt.savefig(OUT / "05_shap_summary.png", dpi=120)
plt.show()


# %%
import joblib

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

joblib.dump({"model": model_d, "features": features}, MODELS / "aqi_delta_lgbm.joblib")

test[["City", "Date", "AQI", "AQI_lag1", "pred_d"]].to_csv(ROOT / "app" / "test_predictions.csv", index=False)

print("Saved:", [p.name for p in MODELS.iterdir()])



