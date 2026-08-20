import numpy as np
import pandas as pd

from forecasting.models import (
    NaiveModel,
    HoltWinters,
    XGBoost
)

from forecasting.evaluation import (
    select_best_model
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

models = [

    NaiveModel(),

    HoltWinters(
        seasonality=7
    ),

    XGBoost(
        n_lags=7
    )

]


# ----------------------------------
# 4. Evaluate models
# ----------------------------------

best_model, results = (
    select_best_model(
        models,
        train,
        test
    )
)


# ----------------------------------
# 5. Print results
# ----------------------------------

print("\nMODEL RESULTS")
print("-" * 40)

for result in results:

    print(
        f"{result['model']}: "
        f"MAE = {result['MAE']:.4f}, "
        f"MAPE = {result['MAPE']:.4f}%"
    )


# ----------------------------------
# 6. Best model
# ----------------------------------

print("\nBEST MODEL")
print("-" * 40)

print(
    best_model.name
)


# ----------------------------------
# 7. Train best model on all data
# ----------------------------------

best_model.fit(series)


# ----------------------------------
# 8. Forecast future
# ----------------------------------

horizon = 7

forecast = best_model.predict(
    horizon
)


print("\nFORECAST")
print("-" * 40)

print(forecast)