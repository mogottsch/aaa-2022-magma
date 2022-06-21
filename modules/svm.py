import pandas as pd
from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    train_test_split,
    HalvingGridSearchCV,
)

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

def check_if_model_result_empty(meta, results, h3_res, time_interval_length):
    return results[
        (results['h3_res'] == h3_res) &
        (results['time_interval_length'] == time_interval_length) &
        (results['param_kernel'] == meta[0]) &
        (results['param_C'] == meta[1]) &
        ((results['param_gamma'] == meta[2]) | (pd.isnull(results['param_gamma']))) &
        ((results['param_degree'] == meta[3]) | (pd.isnull(results['param_degree']))) 
    ]['mean_test_score'].empty


def get_param_grid(model_meta):
    param_grid = {
        'kernel': [model_meta[0]],
        'C': [model_meta[1]],
        'max_iter': [model_meta[4]]
    }
    if model_meta[2] > 0: param_grid = {**param_grid, 'gamma': [model_meta[2]]}
    if model_meta[3] > 0: param_grid = {**param_grid, 'degree': [model_meta[3]]}
    return param_grid


def get_availabe_models_metas_first_stage(h3_res, time_interval_length, all_possible_metas):
    results = get_results_df(SVM_FIRST_STAGE_RESULTS_PATH)    

    # the following code will create all possible combinations of parameters for all models
    metas = [list(product(*meta.values())) for meta in all_possible_metas]
    metas = [item for sublist in metas for item in sublist]
    available_metas = metas
    if not results.empty:
        available_metas = [meta for meta in metas if check_if_model_result_empty(meta, results, h3_res, time_interval_length)]

    # group_by h3 and time, put other params in param grid
    metas_grouped = []
    for kernel in ['linear', 'rbf', 'poly']:
        param_grid = [get_param_grid(meta) for meta in available_metas if (meta[0] == kernel)]
        if len(param_grid) == 0: continue
        metas_grouped.append(param_grid)

    return metas_grouped

def get_availabe_models_metas_second_stage(h3_res, time_interval_length, all_possible_metas):
    results = get_results_df(SVM_FIRST_STAGE_RESULTS_PATH)    
    best_model = results.sort_values(by=['mean_train_score'], ascending=False)
    meta = [
        h3_res,
        time_interval_length,
        best_model['param_kernel'].iloc[0],
        best_model['param_C'].iloc[0],
        best_model['param_gamma'].iloc[0],
        best_model['param_degree'].iloc[0]
    ]
    
    if ((not results.empty) & (check_if_model_result_empty(meta, results, h3_res, time_interval_length))):
        kernel = best_model['param_kernel'].iloc[0]
        params = {
            'kernel': [kernel],
            'C': [best_model['param_C'].iloc[0]],            
            'max_iter': [best_model['param_max_iter'].iloc[0]]
        }
        if kernel == 'poly':
            params = {**params, 'degree': [best_model['param_degree'].iloc[0]]}
        if kernel == 'rbf':
            params = {**params, 'gamma': [best_model['param_gamma'].iloc[0]]}

        return [[params]]

    return []


def split_and_scale_data(model_data, predicted_variable):
    y = model_data[predicted_variable]
    X = model_data.drop(columns=[predicted_variable])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_model(param_grid, X_train, y_train):
    svr = SVR()
    models = HalvingGridSearchCV(svr, param_grid, n_jobs=-1, scoring="neg_mean_squared_error", random_state=42)
    models.fit(X_train, y_train)
    return models


def get_results(models, h3_res, time_interval_length, do_evaluate_model, X_test, y_test):
    results = pd.DataFrame(models.cv_results_)
    results['n_iter'] = 0
    results.loc[0, 'n_iter'] = models.best_estimator_.n_iter_
    results['h3_res'] = h3_res
    results['time_interval_length'] = time_interval_length

    if do_evaluate_model:
        y_pred = models.best_estimator_.predict(X_test)
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['mape'] = mean_average_percentage_error(y_test, y_pred)
        results['rmse'] = root_mean_squared_error(y_test, y_pred)
        
    return results

