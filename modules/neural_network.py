from typing import Tuple
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from itertools import product
import warnings
import time
from datetime import datetime

# for neural networks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from modules.storage import get_results_df
from modules.config import *
from modules.storage import get_demand_model_data, store_results
from modules.evaluation import get_evaluation_metrics
from modules.neural_network import *

# ---------------------------------------------------------------------------------------------------


def check_if_model_result_exists(
    results_path: str, h3_res: int, time_interval_length: int, model_params: dict
) -> bool:
    results = get_results_df(results_path)
    if results.empty:
        return False
    return not results[
        (results["h3_res"] == h3_res)
        & (results["time_interval_length"] == time_interval_length)
        & (results["batch_size"] == model_params["batch_size"])
        & (results["nodes_per_feature"] == model_params["nodes_per_feature"])
        & (results["n_layers"] == model_params["n_layers"])
        & (results["activation"] == model_params["activation"])
        & (results["dropout"] == model_params["dropout"])
    ].empty


def get_model_meta_as_dict(model_meta: dict) -> dict:
    return {
        "batch_size": model_meta[0],
        "nodes_per_feature": model_meta[1],
        "n_layers": model_meta[2],
        "activation": model_meta[3],
        "dropout": model_meta[4],
    }


def get_first_stage_hyperparameters() -> list:
    metas = {
        "batch_size": [8, 16, 32, 64, 128, 256, 512],
        "nodes_per_feature": [1],
        "n_layers": [1],
        "activation": ["relu"],
        "dropout": [-1],
    }
    metas_list = list(product(*metas.values()))
    models_metas = [get_model_meta_as_dict(model_meta) for model_meta in metas_list]
    return models_metas


def get_second_stage_hyperparameters(best_batch_size: int) -> list:
    metas = {
        "batch_size": [best_batch_size],
        "nodes_per_feature": [0.5, 1, 1.5],
        "n_layers": [1, 2, 3],
        "activation": ["relu", "tanh"],
        "dropout": [-1],
    }
    metas_list = list(product(*metas.values()))
    models_metas = [get_model_meta_as_dict(model_meta) for model_meta in metas_list]
    return models_metas


def get_third_stage_hyperparameters(
    best_batch_size: int,
    best_nodes_per_feature: float,
    best_n_layers: int,
    best_activation: str,
) -> list:
    metas = {
        "batch_size": [best_batch_size],
        "nodes_per_feature": [best_nodes_per_feature],
        "n_layers": [best_n_layers],
        "activation": [best_activation],
        "dropout": [0, 0.05, 0.1, 0.2, 0.5],
    }
    metas_list = list(product(*metas.values()))
    models_metas = [get_model_meta_as_dict(model_meta) for model_meta in metas_list]
    return models_metas


def save_model(model: Sequential, h3_res: int, time_interval_length: int):
    model.save(
        os.path.join(KERAS_MODELS_DIR_PATH, f"{h3_res}_{time_interval_length}.h5")
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_params: dict,
) -> Sequential:
    model = Sequential()
    model.add(
        Dense(
            model_params["nodes_per_feature"],
            activation=model_params["activation"],
            input_shape=(X_train.shape[1],),
        )
    )
    n_features = X_train.shape[1]
    for _ in range(model_params["n_layers"]):

        model.add(
            Dense(
                model_params["nodes_per_feature"] * n_features,
                activation=model_params["activation"],
            )
        )
        if model_params["dropout"] >= 0:
            model.add(Dropout(model_params["dropout"]))

    model.add(Dense(1, activation="relu"))  # our outcomes are always positive

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    # the number of epochs does not matter for us, as long as it
    # is high enough so that we can stop the training early
    n_epochs = 100
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=n_epochs,
        batch_size=model_params["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )
    n_trained_epochs = len(history.history["loss"])
    if n_trained_epochs == n_epochs:
        warnings.warn("Model was stopped while validation loss was still improving")
    return model


def evaluate_k_fold(
    X_train: pd.DataFrame, y_train: pd.Series, model_params: dict, prefix: str
) -> dict:
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    evaluation_metrics = []
    for train, valid in kfold.split(X_train):
        evaluation_metrics.append(
            evaluate(
                model_params,
                X_train.iloc[train],
                y_train.iloc[train],
                X_train.iloc[valid],
                y_train.iloc[valid],
                prefix=prefix,
            )
        )

    return {
        key: np.mean([metric[key] for metric in evaluation_metrics])
        for key in evaluation_metrics[0].keys()
    }


def evaluate(
    model_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    prefix: str,
    save: bool = False,
    h3_res: int = None,
    time_interval_length: int = None,
) -> dict:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    model = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        model_params,
    )
    if save:
        save_model(model, h3_res, time_interval_length)
    y_pred = model.predict(X_valid)
    return get_evaluation_metrics(y_valid, y_pred, prefix)


def execute_stage(
    model_data_getter,
    results_path,
    get_hyperparameters,
    h3_res,
    time_interval_length,
    test_phase=False,
    silent=False,
):
    model_data_train, model_data_test = model_data_getter(h3_res, time_interval_length)

    X_train, y_train = (
        model_data_train.drop(columns=["outcome"]),
        model_data_train.outcome,
    )

    X_test, y_test = (
        model_data_test.drop(columns=["outcome"]),
        model_data_test.outcome,
    )
    # if test_phase:
    #     X_train = np.concatenate([X_train, X_valid])
    #     y_train = np.concatenate([y_train, y_valid])

    #     X_valid = X_test
    #     y_valid = y_test

    iterator = tqdm(get_hyperparameters()) if not silent else get_hyperparameters()
    for model_params in iterator:
        start_time = time.time()
        if not silent:
            console_out = (
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                + f"batch_size: {model_params['batch_size']} - "
                + f"nodes_per_feature: {model_params['nodes_per_feature']} - "
                + f"n_layers: {model_params['n_layers']} - "
                + f"activation: {model_params['activation']} - "
                + f"dropout: {model_params['dropout']}"
            )
            tqdm.write(console_out, end="\r")

        if check_if_model_result_exists(
            results_path, h3_res, time_interval_length, model_params
        ):
            if not silent:
                tqdm.write(console_out + " # already trained")
            continue

        train_start = time.time()

        results = (
            evaluate(
                model_params,
                X_train,
                y_train,
                X_test,
                y_test,
                prefix="test",
                save=True,
                h3_res=h3_res,
                time_interval_length=time_interval_length,
            )
            if test_phase
            else evaluate_k_fold(X_train, y_train, model_params, prefix="val")
        )
        train_duration = time.time() - train_start

        if not silent:
            tqdm.write(console_out + " # trained", end="\r")

        results = {
            "h3_res": h3_res,
            "time_interval_length": time_interval_length,
            "batch_size": model_params["batch_size"],
            "nodes_per_feature": model_params["nodes_per_feature"],
            "n_layers": model_params["n_layers"],
            "activation": model_params["activation"],
            "dropout": model_params["dropout"],
            "train_duration": train_duration,
            **results,
        }
        store_results(pd.DataFrame(data=results, index=[0]), results_path)
        duration = time.time() - start_time

        if not silent:
            tqdm.write(console_out + " # evaluated" + f" ({duration:.2f}s)")
