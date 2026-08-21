import numpy as np


def mae(actual, predicted):
    """
    Mean Absolute Error.
    """

    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs(actual - predicted))


# def mape(actual, predicted):
#     """
#     Mean Absolute Percentage Error.
#     """
#     actual = np.array(actual)
#     predicted = np.array(predicted)
#     mask = actual != 0
#     return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])
#                    ) * 100

def mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = actual != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask])
                          / actual[mask])) * 100


# def evaluate_model(model, train, test):
#     """
#     Train a forecasting model and evaluate it
#     on unseen test data.
#     """
#     model.fit(train)
#     predictions = model.predict(len(test))
#     model_mae = mae(test, predictions)
#     model_mape = mape(test, predictions)

#     return {"model": model.name,
#             "MAE": model_mae,
#             "MAPE": model_mape,
#             "predictions": predictions}


# def select_best_model(models, train, test):
#     """
#     Evaluate multiple forecasting models and
#     return the model with the lowest MAPE.
#     """
#     results = []
#     best_model = None
#     best_mape = float("inf")

#     for model in models:
#         result = evaluate_model(model, train, test)
#         results.append(result)
#         if result["MAPE"] < best_mape:
#             best_mape = result["MAPE"]
#             best_model = model

#     # best_result = min(results, key=lambda x: x["MAPE"])
#     # return best_result, results
#     return best_model, results


def walk_forward_validation(model_class, series, train_size, horizon, step):
    """
    Alternative to the evaluation() method, it conducts walk-forward
    validation on the data to ensure the model is the best fit
    """
    results = []
    train_end = train_size
    while train_end + horizon <= len(series):
        train = series.iloc[:train_end]
        test = series.iloc[train_end:(train_end + horizon)]
        model = model_class()
        model.fit(train)
        predictions = model.predict(horizon)

        mod_mae = mae(test, predictions)
        mod_mape = mape(test, predictions)
        results.append({"MAE": mod_mae,
                        "MAPE": mod_mape})

        train_end += step

    return results


def evaluate_walk_forward(model_class, series, train_size, horizon, step):
    """
    This function evaluates the results of the walk_forward validation
    and averages them
    """
    fold_results = walk_forward_validation(model_class, series, train_size,
                                           horizon, step)
    average_mae = np.mean([results["MAE"] for results in fold_results])
    average_mape = np.mean([results["MAPE"] for results in fold_results])

    return {"model": model_class.name,
            "MAE": average_mae,
            "MAPE": average_mape,
            "folds": fold_results}


def best_model_walk_forward(model_classes, series, train_size, horizon, step):
    """
    Selecting the best model based off the results of the walk-forward method
    """
    results = []
    for model_class in model_classes:
        result = evaluate_walk_forward(model_class, series, train_size,
                                       horizon, step)
        result["model_class"] = model_class
        results.append(result)

    best = min(results, key=lambda x: x["MAPE"])
    return best, results
