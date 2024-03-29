import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

from preprocess.utils import scale_data, get_current_time, split_data
from utils.constants import X


def plot_frecuencies(data, method):
    data['Date'] = pd.to_datetime(data['Date'])

    print_test_errors(data=data,
                      method=method)

    daily_sum = data.groupby(pd.Grouper(key='Date',
                                        freq='1D')).sum()
    daily_sum.to_csv('predictions/daily/' + method + '_2017.csv')

    print_test_errors(data=daily_sum,
                      method=method,
                      frecuency='daily')

    weekly_sum = data.groupby(pd.Grouper(key='Date',
                                         freq='7D')).sum()
    weekly_sum.to_csv('predictions/weekly/' + method + '_2017.csv')

    print_test_errors(data=weekly_sum,
                      method=method,
                      frecuency='weekly')

    monthly_sum = data.groupby(pd.Grouper(key='Date',
                                          freq='1M')).sum()
    monthly_sum.to_csv('predictions/monthly/' + method + '_2017.csv')

    print_test_errors(data=monthly_sum,
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


def test_best_model(experiment_id, test_data, label_column='t', scaler=None):
    test_data = test_data.drop(columns=['Date'])

    df = mlflow.search_runs(experiment_ids=[experiment_id])

    run_id = df.loc[df['metrics.rmse'].idxmin()]['run_id']
    model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")

    test_x = test_data[['month_sin', 'month_cos',
                        'day_sin', 'day_cos',
                        'hour_sin', 'hour_cos',
                        't-3', 't-2', 't-1']]

    test_y = test_data[label_column]

    print(get_current_time(), "- Making predictions for test data...")

    yhat = model.predict(test_x).reshape(-1, 1).reshape(-1)
    y = test_y.values.reshape(-1, 1).reshape(-1)

    return y, yhat


def train_test(dataset, label_column='t'):
    train, test = split_data(dataset, year=2016)

    train = train.drop(columns=['Date'])
    test = test.drop(columns=['Date'])

    train_values = train.drop(columns=[label_column])
    test_values = test.drop(columns=[label_column])
    train_labels = train[[label_column]]
    test_labels = test[label_column]

    return train_values, test_values, train_labels, test_labels


def eval_metrics(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred)), \
           np.mean(np.abs((actual - pred))), \
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
                              order_by=['metric.r2 DESC'])
    name = client.get_experiment(experiment_id).name
    df = pd.DataFrame(data=data)
    df.to_csv("modelInfo/" + name + '.csv')
    return data
