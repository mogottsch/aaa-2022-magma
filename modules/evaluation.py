from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

# ---------------------------------------------------------------------------------------------------

def mean_average_percentage_error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / y_true.mean()


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def get_evaluation_metrics(y_true, y_pred, prefix):
    return {
        prefix+'_mse': mean_squared_error(y_true, y_pred),
        prefix+'_mae': mean_absolute_error(y_true, y_pred),
        prefix+'_mape': mean_average_percentage_error(y_true, y_pred),
        prefix+'_rmse': root_mean_squared_error(y_true, y_pred),
    }