import os
import warnings

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from models.ann_mlflow import run_mlp
from models.dt_mlflow import run_dt
from models.elasticnet_mlflow import run_elasticnet
from models.knn_mflow import run_knn
from models.lightgbm_mlflow import run_lgbm
from models.xgb_mlflow import run_xgb
from preprocess.utils import get_current_time, load_cleaned_data, split_data
from utils.functions import test_best_model, print_test_errors, eval_metrics, save_best_params, plot_frecuencies


def train_model(experiment_name, train_data, test_data, params=None, verbose=False):
    try:
        experiment = client.create_experiment(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name).experiment_id

    data = None
    params_stats = None

    if experiment_name == "ElasticNet":
        run_elasticnet(experiment_id=experiment,
                       dataset=train_data,
                       params=params,
                       verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        data = evaluate_model(experiment_id=experiment,
                              name=experiment_name,
                              test=test_data)
    elif experiment_name == "KNN":
        run_knn(experiment_id=experiment,
                dataset=train_data,
                params=params,
                verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        data = evaluate_model(experiment_id=experiment,
                              name=experiment_name,
                              test=test_data)
    elif experiment_name == "LGBM":
        run_lgbm(experiment_id=experiment,
                 dataset=train_data,
                 params=params,
                 verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        data = evaluate_model(experiment_id=experiment,
                              name=experiment_name,
                              test=test_data)
    elif experiment_name == "DT":
        run_dt(experiment_id=experiment,
               dataset=train_data,
               params=params,
               verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        data = evaluate_model(experiment_id=experiment,
                              name=experiment_name,
                              test=test_data)
    elif experiment_name == "XGB":
        run_xgb(experiment_id=experiment,
                dataset=train_data,
                params=params,
                verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        data = evaluate_model(experiment_id=experiment,
                              name=experiment_name,
                              test=test_data)
    elif experiment_name == "MLP":
        run_mlp(experiment_id=experiment,
                dataset=train_data,
                verbose=verbose,
                params=params)
        params_stats = save_best_params(experiment_id=experiment)
        data = evaluate_model(experiment_id=experiment,
                              name=experiment_name,
                              test=test_data)
    return data, params_stats


def evaluate_model(experiment_id, name, test):
    real, predictions = test_best_model(experiment_id, test)
    data = pd.DataFrame(data={
        'Date': test['Date'],
        'Real': real,
        'Pred': predictions
    })
    print_test_errors(data,
                      method=name)
    (rmse, _, _) = eval_metrics(real, predictions)

    print(get_current_time(), "- Score RMSE", name, "Test -", rmse)
    print(get_current_time(), "- Saved results to CSV")
    data.to_csv('predictions/15mins/' + name + '_2017.csv')

    return data


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(2021)

    trainmodels = True

    data = load_cleaned_data()
    train, test = split_data(data)
    client = MlflowClient()

    models = ['ElasticNet']
    if trainmodels:
        for model in models:
            train_model(experiment_name=model,
                        train_data=train,
                        test_data=test,
                        verbose=True)

    models = os.listdir('predictions/15mins')
    name_models = [name.split('_')[0] for name in models]

    print(get_current_time(), '- Plotting predictions..')
    for model in models:
        path = 'predictions/15mins/' + model
        data = pd.read_csv(path)
        data = data.drop(columns=['Date.1'])
        plot_frecuencies(data=data,
                         method=name_models[models.index(model)])
