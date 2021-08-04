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


def modify_model_info(fileName=None):
    if fileName is not None:
        data = pd.read_csv('modelInfo/' + fileName + '.csv')
        data = data.drop(columns=['Unnamed: 0', 'run_id', 'experiment_id', 'status',
                                  'artifact_uri', 'start_time', 'end_time',
                                  'tags.mlflow.source.type', 'tags.mlflow.user',
                                  'tags.train', 'tags.mlflow.source.git.commit',
                                  'tags.type_model', 'tags.mlflow.log-model.history', 'tags.mlflow.source.name'])
        data.to_csv('modelInfo/' + fileName + '.csv',
                    index=False)

    else:
        files = os.listdir('modelInfo')
        for file in files:
            data = pd.read_csv('modelInfo/' + file)
            data = data.drop(columns=['Unnamed: 0', 'run_id', 'experiment_id', 'status',
                                      'artifact_uri', 'start_time', 'end_time',
                                      'tags.mlflow.source.type', 'tags.mlflow.user',
                                      'tags.train', 'tags.mlflow.source.git.commit',
                                      'tags.type_model', 'tags.mlflow.log-model.history', 'tags.mlflow.source.name'])
            data.to_csv('modelInfo/' + file + '.csv',
                        index=False)


def train_model(experiment_name, train_data, test_data, params=None, verbose=False):
    client = MlflowClient()
    try:
        experiment = client.create_experiment(experiment_name)
        print(get_current_time(), '- Experiment with name ' + experiment_name + ' has been created')
    except:
        print(get_current_time(), '- Experiment with name ' + experiment_name + ' already exists. Importing it...')
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
    elif experiment_name == 'ENSEMBLE':
        run_ensemble(experiment_id=experiment,
                     dataset=train_data,
                     verbose=verbose,
                     params=params)
    else:
        print(get_current_time(), '- No model named ' + experiment_name + '. Skipping...')
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

    print(get_current_time(), "- Score 15mins RMSE", name, "Test -", rmse)
    print(get_current_time(), "- Saved results to CSV")

    data.to_csv('predictions/15mins/' + name + '_2017.csv')

    return data


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(2021)

    trainmodels = True

    data = load_cleaned_data()
    train, test = split_data(data)

    models = ['MLP']

    if trainmodels:
        print(get_current_time(), '- Training models -', models)
        for model in models:
            train_model(experiment_name=model,
                        train_data=train,
                        test_data=test,
                        params=None,
                        verbose=False)
            modify_model_info(fileName=model)
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

    dict_data = {'Model': ['EN', 'KNN', 'DT', 'LGBM', 'XGB']}

    directories = os.listdir('predictions')
    for direct in directories:
        dict_data[direct] = []
        files = os.listdir('predictions/' + direct)
        for file in files:
            name = file.split('_')[0]
            data = pd.read_csv('predictions/' + direct + '/' + file)
            (rmse, _, _) = eval_metrics(data['Real'], data['Pred'])
            dict_data[direct].append(rmse / 1000.0)

    data = pd.DataFrame(dict_data)
    cols_sorted = ['Model', '15mins', 'daily', 'weekly', 'monthly']
    data = data[cols_sorted]
    data.to_csv('scores/final_scores.csv',
                index=False)
    print(data)
