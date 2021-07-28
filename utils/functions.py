import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold

from models.ann_mlflow import run_mlp
from models.dt_mlflow import run_dt
from models.elasticnet_mlflow import run_elasticnet
from models.knn_mflow import run_knn
from models.lightgbm_mlflow import run_lgbm
from models.xgb_mlflow import run_xgb
from preprocess.utils import scale_data, get_current_time
from utils.constants import X


def plot_frecuencies(data, method):
    data['Date'] = pd.to_datetime(data['Date'])

    daily_mean = data.groupby(pd.Grouper(key='Date',
                                         freq='1D')).mean()
    print_test_errors(data=daily_mean,
                      method=method,
                      frecuency='daily')

    weekly_mean = data.groupby(pd.Grouper(key='Date',
                                          freq='7D')).mean()
    print_test_errors(data=weekly_mean,
                      method=method,
                      frecuency='weekly')

    monthly_mean = data.groupby(pd.Grouper(key='Date',
                                           freq='1M')).mean()
    print_test_errors(data=monthly_mean,
                      method=method,
                      frecuency='monthly')


def print_test_errors(data, method, frecuency='15mins'):
    plt.plot(data['Real'],
             color='red',
             label='Real PV Production')
    plt.plot(data['Pred'],
             color='blue',
             label='Pred PV Production')

    plt.title('PV Prediction over 2017 with ' + method)
    plt.xlabel('Time')
    plt.ylabel('PV Production ' + frecuency)
    plt.legend()
    plt.savefig('graphs/' + frecuency + '/prediction_2017_' + method)
    plt.close()


def test_best_model(experiment_id, test_data, label_column='PV_Production'):
    test_data = test_data.drop(columns=['Energy', 'Date'])

    df = mlflow.search_runs(experiment_ids=[experiment_id],
                            filter_string="metrics.rmse < 400")

    run_id = df.loc[df['metrics.rmse'].idxmin()]['run_id']
    model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")

    test_x = test_data[X]
    test_y = test_data[label_column]

    test_data_scaled = scale_data(test_x)

    print(get_current_time(), "- Making predictions for test data...")

    return test_y, model.predict(test_data_scaled)


def train_test(dataset, label_column='PV_Production', train_percentage=0.75):
    dataset = dataset.drop(columns=['Energy', 'Date'])
    train, test = train_test_split(dataset,
                                   train_size=train_percentage)

    train_values = train.drop(columns=[label_column])
    test_values = test.drop(columns=[label_column])
    train_labels = train[[label_column]]
    test_labels = test[label_column]

    return train_values, test_values, train_labels, test_labels


def eval_metrics(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred)), \
           mean_absolute_error(actual, pred), \
           r2_score(actual, pred)


# Función que obtiene el MAE, RMSE y R2 de VC con repetición
def train_cv(model, dataset, label_column='PV_Production', num_folds=10, num_bags=10):
    np.random.seed(2021)
    X_tot = dataset.copy()
    y_tot = dataset[label_column].values.reshape(-1)

    X_tot = scale_data(X_tot, vars=X)

    # Creamos arrays para las predicciones
    preds_val = np.empty((len(X_tot), num_bags))
    preds_val[:] = np.nan

    # Entrena y extrae la predicciones con validación cruzada repetida
    folds = RepeatedKFold(n_splits=num_folds, n_repeats=num_bags, random_state=2021)

    for niter, (train_index, val_index) in enumerate(folds.split(X_tot, y_tot)):
        nbag = niter // num_folds  # Extrae el número de repetición (bag)
        X_train, X_val = X_tot[train_index], X_tot[val_index]
        y_train, y_val = y_tot[train_index], y_tot[val_index]
        model.fit(X_train, y_train)
        preds_val[val_index, nbag] = model.predict(X_val)

    # Promedia las predicciones
    preds_val_mean = preds_val.mean(axis=1)

    (rmse, mae, r2) = eval_metrics(y_tot, preds_val_mean)

    return rmse, mae, r2


def save_best_params(experiment_id, nrows=20):
    client = MlflowClient()
    data = mlflow.search_runs(experiment_ids=[experiment_id],
                              max_results=nrows,
                              order_by=['metric.rmse ASC'])
    name = client.get_experiment(experiment_id).name
    df = pd.DataFrame(data=data)
    df.to_csv("modelInfo/" + name + '.csv')
    return data


def train_model(experiment_name, train_data, test_data, params=None, verbose=False):
    client = MlflowClient()
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
