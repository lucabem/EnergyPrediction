import datetime
import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from models.ann_mlflow import run_mlp
from models.dt_mlflow import run_dt
from models.elasticnet_mlflow import run_elasticnet
from models.knn_mflow import run_knn
from models.lightgbm_mlflow import run_lgbm
from models.sarima_mlflow import run_sarima
from models.timeseries_mlflow import run_ts, test_ts
from models.xgb_mlflow import run_xgb
from preprocess.utils import get_current_time, load_cleaned_data, split_data, scale_data, remove_zeros_rows
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


def train_model(experiment_name, train_data, test_data, params=None, verbose=False, scaler=None):
    client = MlflowClient()
    try:
        experiment = client.create_experiment(experiment_name)
        print(get_current_time(), '- Experiment with name ' + experiment_name + ' has been created')
    except:
        print(get_current_time(), '- Experiment with name ' + experiment_name + ' already exists. Importing it...')
        experiment = client.get_experiment_by_name(experiment_name).experiment_id

    if experiment_name == "EN":
        run_elasticnet(experiment_id=experiment,
                       dataset=train_data,
                       params=params,
                       verbose=verbose)
    elif experiment_name == "KNN":
        run_knn(experiment_id=experiment,
                dataset=train_data,
                params=params,
                verbose=verbose)
    elif experiment_name == "LGBM":
        run_lgbm(experiment_id=experiment,
                 dataset=train_data,
                 params=params,
                 verbose=verbose)
    elif experiment_name == "DT":
        run_dt(experiment_id=experiment,
               dataset=train_data,
               params=params,
               verbose=verbose)
    elif experiment_name == "XGB":
        run_xgb(experiment_id=experiment,
                dataset=train_data,
                params=params,
                verbose=verbose)
    elif experiment_name == "MLP":
        run_mlp(experiment_id=experiment,
                dataset=train_data,
                verbose=verbose,
                params=params)
    elif experiment_name == 'TS':
        run_ts(experiment_id=experiment,
               dataset=train_data,
               test_data=test_data,
               verbose=verbose)
    else:
        print(get_current_time(), '- No model named ' + experiment_name + '. Skipping...')

    params_stats = save_best_params(experiment_id=experiment)
    test_results = evaluate_model(experiment_id=experiment,
                                  name=experiment_name,
                                  test_data=test_data,
                                  scaler=scaler)
    return test_results, params_stats


def evaluate_model(experiment_id, name, test_data, train_data=None, scaler=None):
    if name == 'TS':
        real, predictions = test_ts(experiment_id, train_data, test_data)
    else:
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

    remove_outliers = data['VoltageAC_L1'] < 450
    data = data[remove_outliers]
    remove_outliers = data['VoltageAC_L1L2'] < 800
    data = data[remove_outliers]
    remove_outliers = data['VoltageAC_L2'] < 450
    data = data[remove_outliers]
    remove_outliers = data['VoltageAC_L3'] < 450
    data = data[remove_outliers]

    data = data.set_index(data['Date'])
    data = data.sort_index()
    data = data.resample('h').sum()
    data.reset_index(inplace=True)

    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day
    data['hour'] = data['Date'].dt.hour

    data['month_sin'] = np.sin(2 * np.pi * data['month'])
    data['month_cos'] = np.cos(2 * np.pi * data['month'])

    data['day_sin'] = np.sin(2 * np.pi * data['day'])
    data['day_cos'] = np.cos(2 * np.pi * data['day'])

    data['hour_sin'] = np.sin(2 * np.pi * data['hour'])
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'])

    data_eda = data.drop(columns=['Date', 'month', 'day', 'hour'])

    scaled_data = scale_data(data_eda,
                             vars=data_eda.columns)
    scaled_features_df = pd.DataFrame(scaled_data,
                                      columns=data_eda.columns)

    scaled_features_df['Date'] = data['Date']

    data = scaled_features_df[['Date',
                               'month_sin', 'month_cos',
                               'day_sin', 'day_cos',
                               'hour_sin', 'hour_cos',
                               'PV_Production']]

    data.to_csv("full_data.csv")

    for i in range(1, 25):
        data['t-' + str(i)] = data['PV_Production'].shift(i)
    data['t'] = data['PV_Production']

    data.dropna(inplace=True)

    cols = ['Date',
            'month_sin', 'month_cos',
            'day_sin', 'day_cos',
            'hour_sin', 'hour_cos']

    for i in range(3, 0, -1):
        cols.append('t-' + str(i))

    cols.append('t')

    data = data[cols]

    data['Date'] = pd.to_datetime(data['Date'])
    train, test = split_data(data)

    train.to_csv("train_clean_h.csv")
    test.to_csv("test_clean_h.csv")

    models = []

    if trainmodels:
        print(get_current_time(), '- Training models -', models)
        for model in models:
            train_model(experiment_name=model,
                        train_data=train,
                        test_data=test,
                        params=None,
                        verbose=True)
            if model != 'TS':
                modify_model_info(filename=model)
    else:
        print(get_current_time(), '- Not training models -', models)

    models = os.listdir('predictions/15mins')
    name_models = [name.split('_')[0] for name in models]

    print(get_current_time(), '- Plotting predictions..')

    for model in models:
        path = 'predictions/15mins/' + model
        data = pd.read_csv(path)
        plot_frecuencies(data=data,
                         method=name_models[models.index(model)])

    print(get_current_time(), '- RMSE on Different Models')

    dict_data_rmse = {
        'Model': ['DT', 'EN', 'KNN', 'LGBM', 'MLP', 'XGB']
    }
    dict_data_prmse = {
        'Model': ['DT', 'EN', 'KNN', 'LGBM', 'MLP', 'XGB']
    }
    dict_data_r2 = {
        'Model': ['DT', 'EN', 'KNN', 'LGBM', 'MLP', 'XGB']
    }

    directories = os.listdir('predictions')
    for direct in directories:
        dict_data_rmse[direct] = []
        dict_data_prmse[direct] = []
        dict_data_r2[direct] = []
        files = os.listdir('predictions/' + direct)
        for file in files:
            name = file.split('_')[0]
            data = pd.read_csv('predictions/' + direct + '/' + file)
            (rmse, prmse, r2) = eval_metrics(data['Real'], data['Pred'])
            dict_data_rmse[direct].append(rmse)
            dict_data_prmse[direct].append(prmse)
            dict_data_r2[direct].append(r2)

    data = pd.DataFrame(dict_data_rmse)
    cols_sorted = ['Model', '15mins', 'daily', 'weekly', 'monthly']
    data = data[cols_sorted]
    data.to_csv('scores/final_scores_rmse.csv',
                index=False)

    print(get_current_time(), '- R2 on Different Models')
    data = pd.DataFrame(dict_data_r2)
    cols_sorted = ['Model', '15mins', 'daily', 'weekly', 'monthly']
    data = data[cols_sorted]
    data.to_csv('scores/final_scores_r2.csv',
                index=False)

    print(get_current_time(), '- MAPE on Different Models')
    data = pd.DataFrame(dict_data_prmse)
    cols_sorted = ['Model', '15mins', 'daily', 'weekly', 'monthly']
    data = data[cols_sorted]
    data.to_csv('scores/final_scores_mape.csv',
                index=False)
