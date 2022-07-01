from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
)

# ---------------------------------------------------------------------------------------------------


def mean_average_percentage_error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / y_true.mean()


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def get_evaluation_metrics(y_true, y_pred, prefix):
    y_true_non_zero = y_true[y_true != 0]
    y_true_is_zero = y_true == 0

    y_pred_non_zero = y_pred[y_true != 0]
    y_pred_is_zero = y_pred == 0

    return {
        prefix + "_mse": mean_squared_error(y_true, y_pred),
        prefix + "_rmse": root_mean_squared_error(y_true, y_pred),
        prefix + "_mae": mean_absolute_error(y_true, y_pred),
        prefix
        + "_non_zero_mape": mean_average_percentage_error(
            y_true_non_zero, y_pred_non_zero
        ),
        prefix + "_zero_accuracy": accuracy_score(y_true_is_zero, y_pred_is_zero),
    }
