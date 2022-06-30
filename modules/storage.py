import pandas as pd
from modules.config import *
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------------------------------


def get_demand_model_data(h3_res, time_interval_length) -> pd.DataFrame:
    return get_model_data(h3_res, time_interval_length, "demand", "demand")


def get_demand_orig_dest_model_data() -> pd.DataFrame:
    model_data = pd.read_feather(
        os.path.join(
            MODEL_DATA_DIR_PATH,
            f"demand_{ORIGIN_DESTINATION_H3_RESOLUTION}_{ORIGIN_DESTINATION_TIME_INTERVAL_LENGTH}.feather",
        )
    )
    model_data = model_data.rename(columns={"demand": "outcome"})

    model_data_train, model_data_test = train_test_split(
        model_data, train_size=0.5, random_state=42
    )
    return model_data_train, model_data_test


def get_availability_model_data(h3_res, time_interval_length) -> pd.DataFrame:
    return get_model_data(h3_res, time_interval_length, "availability", "n_bikes")


def get_model_data(h3_res, time_interval_length, predicted_variable, outcome_column):
    model_data = pd.read_feather(
        os.path.join(
            MODEL_DATA_DIR_PATH,
            f"{predicted_variable}_{h3_res}_{time_interval_length}.feather",
        )
    )
    model_data = model_data.rename(columns={outcome_column: "outcome"})
    model_data_train, model_data_test = train_test_split(
        model_data, train_size=0.5, random_state=42
    )
    return model_data_train, model_data_test


def get_results_df(path) -> pd.DataFrame:
    if os.path.isfile(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def store_results(new_results, path):
    if os.path.isfile(path):
        results = pd.read_parquet(path)
        results = pd.concat([results, new_results], ignore_index=True)
        results.to_parquet(path)
    else:
        new_results.to_parquet(path)
