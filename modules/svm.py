from fsspec import Callback
from numpy import NaN
import pandas as pd
from itertools import product
from tqdm.notebook import tqdm
from typing import Tuple

# for support vector machines
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from modules.evaluation import get_evaluation_metrics
from modules.storage import (
    get_results_df,
    store_results,
)
from modules.config import *
from modules.svm import *


def check_if_model_result_empty(
    results: pd.DataFrame, h3_res: int, time_interval_length: int, meta: list
) -> bool:
    return results[
        (results["h3_res"] == h3_res)
        & (results["time_interval_length"] == time_interval_length)
        & (results["param_kernel"] == meta[0])
        & (results["param_C"] == meta[1])
        & ((results["param_gamma"] == meta[2]) | (pd.isnull(results["param_gamma"])))
        & ((results["param_degree"] == meta[3]) | (pd.isnull(results["param_degree"])))
    ]["mean_test_score"].empty


def get_param_grid(model_meta: dict) -> dict:
    param_grid = {
        "kernel": [model_meta[0]],
        "C": [model_meta[1]],
        "max_iter": [model_meta[4]],
    }
    if model_meta[2] > 0:
        param_grid = {**param_grid, "gamma": [model_meta[2]]}
    if model_meta[3] > 0:
        param_grid = {**param_grid, "degree": [model_meta[3]]}
    return param_grid


def get_model_meta_function_for_stage(stage: str) -> Callback:
    return (
        get_availabe_models_metas_first_stage
        if (stage == "first_stage")
        else get_availabe_models_metas_second_stage
    )


def get_availabe_models_metas_first_stage(
    h3_res: int,
    time_interval_length: int,
    all_possible_metas: dict,
    first_stage_path: str,
    _: str,
) -> list:
    results = get_results_df(first_stage_path)

    # the following code will create all possible combinations of parameters for all models
    metas = [list(product(*meta.values())) for meta in all_possible_metas]
    metas = [item for sublist in metas for item in sublist]
    available_metas = metas
    if not results.empty:
        available_metas = [
            meta
            for meta in metas
            if check_if_model_result_empty(results, h3_res, time_interval_length, meta)
        ]

    # group_by h3 and time, put other params in param grid
    metas_grouped = []
    for kernel in ["linear", "rbf", "poly"]:
        param_grid = [
            get_param_grid(meta) for meta in available_metas if (meta[0] == kernel)
        ]
        if len(param_grid) == 0:
            continue
        metas_grouped.append(param_grid)

    return metas_grouped


def get_availabe_models_metas_second_stage(
    h3_res: int,
    time_interval_length: int,
    _: dict,
    first_stage_path: str,
    second_stage_path: str,
) -> list:
    results_first_stage = get_results_df(first_stage_path)
    best_model = results_first_stage.sort_values(
        by=["mean_test_score"], ascending=False
    )
    meta = [
        h3_res,
        time_interval_length,
        best_model["param_kernel"].iloc[0],
        best_model["param_C"].iloc[0],
        best_model["param_gamma"].iloc[0],
        best_model["param_degree"].iloc[0],
    ]

    results_second_stage = get_results_df(second_stage_path)
    if (results_second_stage.empty) or (
        check_if_model_result_empty(
            results_second_stage, h3_res, time_interval_length, meta
        )
    ):
        kernel = best_model["param_kernel"].iloc[0]
        params = {
            "kernel": [kernel],
            "C": [best_model["param_C"].iloc[0]],
            "max_iter": [best_model["param_max_iter"].iloc[0]],
        }
        if kernel == "poly":
            params = {**params, "degree": [best_model["param_degree"].iloc[0]]}
        if kernel == "rbf":
            params = {**params, "gamma": [best_model["param_gamma"].iloc[0]]}

        return [[params]]

    return []


def split_and_scale_data(
    model_data_train: pd.DataFrame, model_data_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y_train = model_data_train["outcome"]
    X_train = model_data_train.drop(columns=["outcome"])

    y_test = model_data_test["outcome"]
    X_test = model_data_test.drop(columns=["outcome"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_model(
    param_grid: list, X_train: pd.DataFrame, y_train: pd.Series
) -> HalvingGridSearchCV:
    svr = SVR(cache_size=2000)
    if param_grid[0]["kernel"] == "linear":
        svr = LinearSVR()

    models = HalvingGridSearchCV(
        svr, param_grid, n_jobs=-1, scoring="neg_mean_squared_error", random_state=42
    )
    models.fit(X_train, y_train)
    return models


def check_and_append_missing_columns(
    results: pd.DataFrame, missing_columns: list
) -> pd.DataFrame:
    for column in missing_columns:
        if column not in results.columns:
            results[column] = NaN
    return results


def get_results(
    models: HalvingGridSearchCV,
    h3_res: int,
    time_interval_length: int,
    stage: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    results = pd.DataFrame(models.cv_results_)
    results["n_iter"] = 0
    results.loc[0, "n_iter"] = models.best_estimator_.n_iter_
    results["h3_res"] = h3_res
    results["time_interval_length"] = time_interval_length

    if stage == "second_stage":
        y_pred = models.best_estimator_.predict(X_test)
        evaluation_metrics = get_evaluation_metrics(y_test, y_pred, "test")
        results["test_mse"] = evaluation_metrics["test_mse"]
        results["test_rmse"] = evaluation_metrics["test_rmse"]
        results["test_mae"] = evaluation_metrics["test_mae"]
        results["test_non_zero_mape"] = evaluation_metrics["test_non_zero_mape"]
        results["test_zero_accuracy"] = evaluation_metrics["test_zero_accuracy"]

        check_and_append_missing_columns(results, ["param_gamma", "param_degree"])

    return results


def execute_stage(
    stage: str,
    first_stage_path: str,
    second_stage_path: str,
    h3_res: int,
    time_interval_length: int,
    all_possible_metas: dict,
    model_data_getter: Callback,
):
    model_metas_getter = get_model_meta_function_for_stage(stage)
    metas = model_metas_getter(
        h3_res,
        time_interval_length,
        all_possible_metas,
        first_stage_path,
        second_stage_path,
    )

    iterator = tqdm(metas) if (stage == "first_stage") else metas
    for param_grid in iterator:
        if stage == "first_stage":
            feedback = (
                f"h3: {h3_res} | t:{time_interval_length} | - "
                + param_grid[0]["kernel"][0]
            )
            tqdm.write(feedback, end="\r")

        model_data_train, model_data_test = model_data_getter(
            h3_res, time_interval_length
        )
        if len(model_data_train) > SVM_MAX_TRAIN_SET_SIZE:
            model_data_train = model_data_train.sample(SVM_MAX_TRAIN_SET_SIZE)

        X_train, X_test, y_train, y_test = split_and_scale_data(
            model_data_train, model_data_test
        )
        models = train_model(param_grid, X_train, y_train)
        results = get_results(
            models, h3_res, time_interval_length, stage, X_test, y_test
        )

        results_path = (
            first_stage_path if (stage == "first_stage") else second_stage_path
        )
        store_results(results, results_path)

        if stage == "first_stage":
            tqdm.write(feedback + " ✓")
