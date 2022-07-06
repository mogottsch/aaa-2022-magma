from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
)
import numpy as np
import pandas as pd

from modules.config import FINAL_MODEL_RESULTS_PATH

# ---------------------------------------------------------------------------------------------------


def mean_average_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred) / y_true.mean()


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def get_evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str
) -> float:
    y_true_non_zero = y_true[y_true != 0]
    y_true_is_zero = y_true == 0

    y_pred_non_zero = y_pred[y_true != 0]
    y_pred_is_zero = np.round(y_pred) == 0

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


def save_final_result(
    model: str,
    outcome: str,
    h3_res: int,
    time_interval_length: int,
    train_duration: float,
    test_mse: float,
    test_rmse: float,
    test_mae: float,
    test_non_zero_mape: float,
    test_zero_accuracy: float,
) -> None:
    results_df = pd.read_parquet(FINAL_MODEL_RESULTS_PATH)
    if model in results_df["model"].values:
        results_df = results_df.drop(results_df[results_df["model"] == model].index)

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                {
                    "model": [model],
                    "outcome": [outcome],
                    "h3_res": [h3_res],
                    "time_interval_length": [time_interval_length],
                    "train_duration": [train_duration],
                    "test_mse": [test_mse],
                    "test_rmse": [test_rmse],
                    "test_mae": [test_mae],
                    "test_non_zero_mape": [test_non_zero_mape],
                    "test_zero_accuracy": [test_zero_accuracy],
                }
            ),
        ],
    )
    results_df.to_parquet(FINAL_MODEL_RESULTS_PATH)
