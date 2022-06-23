import pandas as pd
from modules.config import *

# ---------------------------------------------------------------------------------------------------

def get_demand_model_data(h3_res, time_interval_length):
    model_data = pd.read_feather(os.path.join(MODEL_DATA_DIR_PATH, f"demand_{h3_res}_{time_interval_length}.feather"))
    return model_data


def get_availability_model_data(h3_res, time_interval_length):
    model_data = pd.read_feather(os.path.join(MODEL_DATA_DIR_PATH, f"availability_{h3_res}_{time_interval_length}.feather"))
    return model_data


def get_results_df(path):
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

