import numpy as np
import pandas as pd

from forecasting.models import (
    NaiveModel,
    HoltWinters,
    XGBoost,
    SARIMAModel
)

from forecasting.evaluation import (
    best_model_walk_forward
)


# ----------------------------------
# 1. Create synthetic time series
# ----------------------------------

np.random.seed(42)

dates = pd.date_range(
    start="2025-01-01",
    periods=200,
    freq="D"
)

trend = np.linspace(100, 150, 200)

seasonality = (
    10 * np.sin(
        2 * np.pi * np.arange(200) / 7
    )
)

noise = np.random.normal(
    0,
    2,
    200
)

values = (
    trend
    + seasonality
    + noise
)


series = pd.Series(
    values,
    index=dates,
    name="target"
)


# ----------------------------------
# 2. Split into train/test
# ----------------------------------

split_index = int(
    len(series) * 0.8
)

train = series.iloc[:split_index]

test = series.iloc[split_index:]


print("Training observations:", len(train))
print("Testing observations:", len(test))


# ----------------------------------
# 3. Create candidate models
# ----------------------------------

models = [NaiveModel, HoltWinters, XGBoost, SARIMAModel]


# ----------------------------------
# 4. Evaluate models
# ----------------------------------

best_result, results = (best_model_walk_forward(models, series, 100, 7, 14))


# ----------------------------------
# 5. Print results
# ----------------------------------

print("\nWALK-FORWARD RESULTS")
print("-" * 50)

for result in results:

    print(
        f"{result['model']}: "
        f"MAE = {result['MAE']:.4f}, "
        f"MAPE = {result['MAPE']:.4f}%"
    )


print("\nBEST MODEL")
print("-" * 50)

print(
    best_result["model"]
)

# ----------------------------------
# 7. Train best model on all data
# ----------------------------------

best_model_class = best_result["model_class"]
best_model = best_model_class()
best_model.fit(series)


# ----------------------------------
# 8. Forecast future
# ----------------------------------

horizon = 7

forecast = best_model.predict(horizon)


print("\nFORECAST")
print("-" * 40)

print(forecast)