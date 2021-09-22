import itertools

import mlflow

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout

from preprocess.utils import get_current_time
from utils.functions import train_test, eval_metrics

import numpy as np

def test_ts(experiment_id, traindata, testdata):

    print(traindata)

    print(testdata)
    train_x, test_x, train_y, test_y = train_test(traindata)

    train_x = np.reshape(train_x.values, (train_x.shape[0], 1, train_x.shape[1]))

    params = pd.read_csv('modelInfo/TS.csv')
    k1 = params['params.k1'][0]
    k2 = params['params.k2'][0]

    regressor = Sequential()
    # Añadimos una capa de entrada y la capa LSTM
    regressor.add(LSTM(
        units=k1,
        activation='sigmoid',
        return_sequences=True
    ))

    # Añadimos la capa de salida con una única neurona
    regressor.add(Dense(units=1))
    # Compilamos la RNN
    # usamos el error cuadrático medio
    # MSE para regresión
    regressor.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Ajustamos la RNN al conjunto de entrenamiento
    regressor.fit(train_x,
                  train_y,
                  batch_size=32,
                  epochs=5,
                  shuffle=False,
                  verbose=0)

    # test_set.reshape(-1), predicted_pv.reshape(-1)

    return 0


def run_ts(experiment_id, dataset, test_data, params=None, verbose=False):
    print("@@@@ TESTING @@@@")

    train_x, test_x, train_y, test_y = train_test(dataset)

    train_x = np.reshape(train_x.values, (train_x.shape[0], 1, train_x.shape[1]))

    # Inicializando la RNN
    # utilizaremos un modelo continuo, modelo de regresión
    for k1 in [50, 100, 150, 200, 250, 300, 350, 400, 500]:
            with mlflow.start_run(experiment_id=experiment_id,
                                  tags={'type_model': 'TS',
                                        'train': 'holdout'}):
                regressor = Sequential()

                # Añadimos una capa de entrada y la capa LSTM
                regressor.add(LSTM(
                    units=k1,
                    input_shape=(1, 9),
                    activation='relu'
                ))
                regressor.add(Dropout(0.20))

                # Añadimos la capa de salida con una única neurona
                regressor.add(Dense(units=1))

                # Compilamos la RNN
                # usamos el error cuadrático medio
                # MSE para regresión
                regressor.compile(optimizer='adam',
                                  loss='mean_squared_error',
                                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

                # Ajustamos la RNN al conjunto de entrenamiento
                regressor.fit(train_x,
                              train_y,
                              batch_size=1,
                              epochs=200,
                              validation_split=0.2)

                test_x = np.reshape(test_x.values, (test_x.shape[0], 1, test_x.shape[1]))
                testPredict = regressor.predict(test_x)

                (rmse, rmspe, r2) = eval_metrics(test_y.reshape(-1), testPredict.reshape(-1))

                mlflow.log_param('k1', k1)
                mlflow.log_param('k2', 0)
                mlflow.log_param('batch_size', 500)
                mlflow.log_param('epochs', 100)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", rmspe)

                if True:
                    print(get_current_time(), "- [k1={}, k2={}] - [mae={:.3f}, rmse={:.3f}, r2={:.3f}]".format(k1, 0,
                                                                                                                 rmspe,
                                                                                                                 rmse,
                                                                                                                 r2))


