import numpy as np


def mae(actual, predicted):
    """
    Mean Absolute Error.
    """

    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs(actual - predicted))


def mape(actual, predicted):
    """
    Mean Absolute Percentage Error.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])
                   ) * 100


def evaluate_model(model, train, test):
    """
    Train a forecasting model and evaluate it
    on unseen test data.
    """
    model.fit(train)
    predictions = model.predict(len(test))
    model_mae = mae(test, predictions)
    model_mape = mape(test, predictions)

    return {"model": model.name,
            "MAE": model_mae,
            "MAPE": model_mape,
            "predictions": predictions}


def select_best_model(models, train, test):
    """
    Evaluate multiple forecasting models and
    return the model with the lowest MAPE.
    """
    results = []
    for model in models:
        result = evaluate_model(model, train, test)
        results.append(result)

    best_result = min(results, key=lambda x: x["MAPE"])
    return best_result, results