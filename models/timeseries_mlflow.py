import mlflow
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

from preprocess.utils import get_current_time
from utils.functions import train_test, eval_metrics


def run_ts(experiment_id, dataset, params=None, verbose=False):
    print("@@@@ TESTING @@@@")
    train_x, test_x, train_y, test_y = train_test(dataset)

    data_train = train_x.iloc[:, 6:7].values
    data_test = test_x.iloc[:, 6:7].values

    sc = StandardScaler()
    training_set = sc.fit_transform(data_train)

    X_train = training_set[0:len(training_set) - 1]
    y_train = training_set[1:len(training_set)]
    X_train = np.reshape(X_train, (len(training_set) - 1, 1, 1))

    # Inicializando la RNN
    # utilizaremos un modelo continuo, modelo de regresión
    with mlflow.start_run(experiment_id=experiment_id,
                          tags={'type_model': 'TS',
                                'train': 'holdout'}):
        regressor = Sequential()

        # Añadimos una capa de entrada y la capa LSTM
        regressor.add(LSTM(
            units=10,
            activation='sigmoid',
            input_shape=(None, 1)
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
        regressor.fit(X_train,
                      y_train,
                      batch_size=32,
                      epochs=5)

        test_set = data_test
        inputs = test_set
        inputs = sc.transform(inputs)
        inputs = np.reshape(inputs, (17867, 1, 1))

        predicted_pv = regressor.predict(inputs)
        predicted_pv = sc.inverse_transform(predicted_pv)

        (rmse, mae, r2) = eval_metrics(test_set.reshape(-1), predicted_pv.reshape(-1))

        if verbose:
            print(get_current_time(), "- [[mae={:.3f}, rmse={:.3f}, r2={:.3f}]".format(mae, rmse, r2))
