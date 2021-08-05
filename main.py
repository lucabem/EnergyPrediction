import os
import warnings

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from models.ann_mlflow import run_mlp
from models.dt_mlflow import run_dt
from models.elasticnet_mlflow import run_elasticnet
from models.ensemble_mlflow import run_ensemble
from models.knn_mflow import run_knn
from models.lightgbm_mlflow import run_lgbm
from models.xgb_mlflow import run_xgb
from preprocess.utils import get_current_time, load_cleaned_data, split_data
from utils.functions import plot_frecuencies, save_best_params, test_best_model, eval_metrics, \
    print_test_errors


def modify_model_info(filename=None):
    if filename is not None:
        dataset = pd.read_csv('modelInfo/' + filename + '.csv')
        dataset = dataset.drop(columns=['Unnamed: 0', 'run_id', 'experiment_id', 'status',
                                        'artifact_uri', 'start_time', 'end_time',
                                        'tags.mlflow.source.type', 'tags.mlflow.user',
                                        'tags.train', 'tags.mlflow.source.git.commit',
                                        'tags.type_model', 'tags.mlflow.log-model.history',
                                        'tags.mlflow.source.name'])
        dataset.to_csv('modelInfo/' + filename + '.csv',
                       index=False)

    else:
        modelfiles = os.listdir('modelInfo')
        for modelfile in modelfiles:
            dataset = pd.read_csv('modelInfo/' + modelfile)
            dataset = dataset.drop(columns=['Unnamed: 0', 'run_id', 'experiment_id', 'status',
                                            'artifact_uri', 'start_time', 'end_time',
                                            'tags.mlflow.source.type', 'tags.mlflow.user',
                                            'tags.train', 'tags.mlflow.source.git.commit',
                                            'tags.type_model', 'tags.mlflow.log-model.history',
                                            'tags.mlflow.source.name'])
            dataset.to_csv('modelInfo/' + modelfile,
                           index=False)


def train_model(experiment_name, train_data, test_data, params=None, verbose=False):
    client = MlflowClient()
    try:
        experiment = client.create_experiment(experiment_name)
        print(get_current_time(), '- Experiment with name ' + experiment_name + ' has been created')
    except:
        print(get_current_time(), '- Experiment with name ' + experiment_name + ' already exists. Importing it...')
        experiment = client.get_experiment_by_name(experiment_name).experiment_id

    test_results = None
    params_stats = None

    if experiment_name == "ElasticNet":
        run_elasticnet(experiment_id=experiment,
                       dataset=train_data,
                       params=params,
                       verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        test_results = evaluate_model(experiment_id=experiment,
                                      name=experiment_name,
                                      test_data=test_data)
    elif experiment_name == "KNN":
        run_knn(experiment_id=experiment,
                dataset=train_data,
                params=params,
                verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        test_results = evaluate_model(experiment_id=experiment,
                                      name=experiment_name,
                                      test_data=test_data)
    elif experiment_name == "LGBM":
        run_lgbm(experiment_id=experiment,
                 dataset=train_data,
                 params=params,
                 verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        test_results = evaluate_model(experiment_id=experiment,
                                      name=experiment_name,
                                      test_data=test_data)
    elif experiment_name == "DT":
        run_dt(experiment_id=experiment,
               dataset=train_data,
               params=params,
               verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        test_results = evaluate_model(experiment_id=experiment,
                                      name=experiment_name,
                                      test_data=test_data)
    elif experiment_name == "XGB":
        run_xgb(experiment_id=experiment,
                dataset=train_data,
                params=params,
                verbose=verbose)
        params_stats = save_best_params(experiment_id=experiment)
        test_results = evaluate_model(experiment_id=experiment,
                                      name=experiment_name,
                                      test_data=test_data)
    elif experiment_name == "MLP":
        run_mlp(experiment_id=experiment,
                dataset=train_data,
                verbose=verbose,
                params=params)
        params_stats = save_best_params(experiment_id=experiment)
        test_results = evaluate_model(experiment_id=experiment,
                                      name=experiment_name,
                                      test_data=test_data)
    elif experiment_name == 'ENSEMBLE':
        run_ensemble(experiment_id=experiment,
                     dataset=train_data,
                     verbose=verbose)
    else:
        print(get_current_time(), '- No model named ' + experiment_name + '. Skipping...')
    return test_results, params_stats


def evaluate_model(experiment_id, name, test_data):
    real, predictions = test_best_model(experiment_id, test_data)

    data_pred = pd.DataFrame(data={
        'Date': test_data['Date'],
        'Real': real,
        'Pred': predictions
    })
    print_test_errors(data_pred,
                      method=name)

    (rmse_test, _, _) = eval_metrics(real, predictions)

    print(get_current_time(), "- Score 15mins RMSE", name, "Test -", rmse_test)
    print(get_current_time(), "- Saved results to CSV")

    data_pred.to_csv('predictions/15mins/' + name + '_2017.csv')

    return data_pred


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(2021)

    trainmodels = True

    data = load_cleaned_data()
    train, test = split_data(data)

    models = []

    if trainmodels:
        print(get_current_time(), '- Training models -', models)
        for model in models:
            train_model(experiment_name=model,
                        train_data=train,
                        test_data=test,
                        params=None,
                        verbose=False)
            modify_model_info(filename=model)
    else:
        print(get_current_time(), '- Not training models -', models)

    models = os.listdir('predictions/15mins')
    name_models = [name.split('_')[0] for name in models]

    print(get_current_time(), '- Plotting predictions..')

    for model in models:
        path = 'predictions/15mins/' + model
        data = pd.read_csv(path)
        data = data.drop(columns=['Date.1'])
        plot_frecuencies(data=data,
                         method=name_models[models.index(model)])

    print(get_current_time(), '- RMSE on Different Models')

    dict_data_rmse = {'Model': ['EN', 'KNN', 'DT', 'LGBM', 'XGB']}
    dict_data_r2 = {'Model': ['EN', 'KNN', 'DT', 'LGBM', 'XGB']}

    directories = os.listdir('predictions')
    for direct in directories:
        dict_data_rmse[direct] = []
        dict_data_r2[direct] = []
        files = os.listdir('predictions/' + direct)
        for file in files:
            name = file.split('_')[0]
            data = pd.read_csv('predictions/' + direct + '/' + file)
            (rmse, _, r2) = eval_metrics(data['Real'], data['Pred'])
            dict_data_rmse[direct].append(rmse / 1000.0)
            dict_data_r2[direct].append(r2)

    data = pd.DataFrame(dict_data_rmse)
    cols_sorted = ['Model', '15mins', 'daily', 'weekly', 'monthly']
    data = data[cols_sorted]
    data.to_csv('scores/final_scores_rmse.csv',
                index=False)
    print(data)

    print(get_current_time(), '- R2 on Different Models')
    data = pd.DataFrame(dict_data_r2)
    cols_sorted = ['Model', '15mins', 'daily', 'weekly', 'monthly']
    data = data[cols_sorted]
    data.to_csv('scores/final_scores_r2.csv',
                index=False)
    print(data)
