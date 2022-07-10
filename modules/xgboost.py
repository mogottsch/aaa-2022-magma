from sklearn.model_selection import HalvingRandomSearchCV
from xgboost import XGBRegressor
from pandas import DataFrame
from modules.evaluation import get_evaluation_metrics
from modules.storage import get_results_df


def perform_randomized_grid_search(
    model_data_getter,
    hyperparameters,
    h3_res,
    time_interval_length,
):
    model_data_train, _ = model_data_getter(h3_res, time_interval_length)

    X_train, y_train = (
        model_data_train.drop(columns=["outcome"]),
        model_data_train.outcome,
    )

    model = XGBRegressor()
    search = HalvingRandomSearchCV(
        model,
        hyperparameters,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_candidates=1000,
        min_resources="exhaust",
        verbose=1,
    )

    search.fit(X_train, y_train)

    return DataFrame(search.cv_results_)


def get_best_hyperparameters(result):
    param_cols = [col for col in result.columns if col.startswith("param_")]
    best_hyperparameters = result.loc[result.rank_test_score == 1, param_cols].to_dict(
        "list"
    )
    return {k.replace("param_", ""): v[0] for k, v in best_hyperparameters.items()}


def train_model(model_data_train, hyperparameters):
    X_train, y_train = (
        model_data_train.drop(columns=["outcome"]),
        model_data_train.outcome,
    )

    model = XGBRegressor(**hyperparameters)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model_data_test, model):

    X_test, y_test = (
        model_data_test.drop(columns=["outcome"]),
        model_data_test.outcome,
    )

    y_pred = model.predict(X_test)

    return get_evaluation_metrics(y_test, y_pred, prefix="test")


def evaluate(model_data_getter, hyperparameters, h3_res, time_interval_length):
    model_data_train, model_data_test = model_data_getter(h3_res, time_interval_length)

    model = train_model(model_data_train, hyperparameters)
    return evaluate_model(model_data_test, model)


def model_already_evaluated(results_path, h3_res, time_interval_length):
    res = get_results_df(results_path)
    if res.empty:
        return False
    return not res.loc[
        (res.h3_res == h3_res) & (res.time_interval_length == time_interval_length)
    ].empty
