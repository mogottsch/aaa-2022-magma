import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from itertools import product
import warnings

# for neural networks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)
from modules.evaluation import (
    mean_average_percentage_error,
    root_mean_squared_error
)
from modules.storage import get_results_df
from modules.config import *

# ---------------------------------------------------------------------------------------------------

def check_if_model_result_empty(results_path, h3_res, time_interval_length, model_params):
    results = get_results_df(results_path)
    return not results[
        (results["h3_res"] == h3_res)
        & (results["time_interval_length"] == time_interval_length)
        & (results["batch_size"] == model_params["batch_size"])
        & (results["nodes_per_feature"] == model_params["nodes_per_feature"])
        & (results["n_layers"] == model_params["n_layers"])
        & (results["activation"] == model_params["activation"])
        & (results["dropout"] == model_params["dropout"])
    ]["val_mape"].empty


def get_model_meta_as_dict(model_meta):
    return {
        "batch_size": model_meta[0],
        "nodes_per_feature": model_meta[1],
        "n_layers": model_meta[2],
        "activation": model_meta[3],
        "dropout": model_meta[4],
    }


def get_first_stage_hyperparameters():
    metas = {
        "batch_size": [32, 64, 128, 256],
        "nodes_per_feature": [1],
        "n_layers": [1],
        "activation": ["relu"],
        "dropout": [-1],
    }
    metas_list = list(product(*metas.values()))
    models_metas = [get_model_meta_as_dict(model_meta) for model_meta in metas_list]
    return models_metas


def get_second_stage_hyperparameters(best_batch_size):
    metas = {
        "batch_size": [best_batch_size],
        "nodes_per_feature": [0.5, 1, 1.5],
        "n_layers": [1, 2, 3],
        "activation": ["relu", "sigmoid", "tanh"],
        "dropout": [-1],
    }
    metas_list = list(product(*metas.values()))
    models_metas = [get_model_meta_as_dict(model_meta) for model_meta in metas_list]
    return models_metas


def get_third_stage_hyperparameters(
    best_batch_size, best_nodes_per_feature, best_n_layers, best_activation
):
    metas = {
        "batch_size": [best_batch_size],
        "nodes_per_feature": [best_nodes_per_feature],
        "n_layers": [best_n_layers],
        "activation": [best_activation],
        "dropout": [0, 0.05, 0.1, 0.2],
    }
    metas_list = list(product(*metas.values()))
    models_metas = [get_model_meta_as_dict(model_meta) for model_meta in metas_list]
    return models_metas


def split_and_scale_data(model_data, predicted_variable):
    y = model_data[predicted_variable]
    X = model_data.drop(columns=[predicted_variable])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, train_size=0.7, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
