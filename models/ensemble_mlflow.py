import random

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from preprocess.utils import get_current_time, scale_data
from utils.constants import X
from utils.functions import train_test, eval_metrics

from scipy.optimize import minimize


def combina_predicciones(weigths, test_y, predicted_1, predicted_2, predicted_3):
    preds_combi = ((3.0 - weigths[0] - weigths[1]) * predicted_1 +
                   weigths[0] * predicted_2 +
                   weigths[1] * predicted_3) / 3.0
    (rmse, mae, r2) = eval_metrics(test_y, preds_combi)
    return rmse


def run_ensemble(experiment_id, dataset, params=None, verbose=False):
    print(get_current_time(), "- Starting Ensemble Model...")

    train_x, test_x, train_y, test_y = train_test(dataset)

    train_x = scale_data(train_x, vars=X)
    test_x = scale_data(test_x, vars=X)

    best_params_xgb = pd.read_csv('modelInfo/XGB.csv')
    best_params_mlp = pd.read_csv('modelInfo/MLP.csv')
    best_params_lgbm = pd.read_csv('modelInfo/LGBM.csv')

    mlp1 = MLPRegressor(hidden_layer_sizes=eval(best_params_mlp['params.hidden_layer_sizes'][0]),
                        solver=best_params_mlp['params.solver'][0],
                        activation=best_params_mlp['params.activation'][0],
                        alpha=best_params_mlp['params.alpha'][0],
                        random_state=2021,
                        learning_rate_init=best_params_mlp['params.learning_rate_init'][0])
    mlp1.fit(train_x, train_y)
    predicted_mlp1 = mlp1.predict(test_x)

    mlp2 = MLPRegressor(hidden_layer_sizes=eval(best_params_mlp['params.hidden_layer_sizes'][1]),
                        solver=best_params_mlp['params.solver'][1],
                        activation=best_params_mlp['params.activation'][1],
                        alpha=best_params_mlp['params.alpha'][1],
                        random_state=2021,
                        learning_rate_init=best_params_mlp['params.learning_rate_init'][1])
    mlp2.fit(train_x, train_y)
    predicted_mlp2 = mlp2.predict(test_x)

    mlp3 = MLPRegressor(hidden_layer_sizes=eval(best_params_mlp['params.hidden_layer_sizes'][2]),
                        solver=best_params_mlp['params.solver'][2],
                        activation=best_params_mlp['params.activation'][2],
                        alpha=best_params_mlp['params.alpha'][2],
                        random_state=2021,
                        learning_rate_init=best_params_mlp['params.learning_rate_init'][2])
    mlp3.fit(train_x, train_y)
    predicted_mlp3 = mlp3.predict(test_x)

    min_rmse = 70
    print(get_current_time(), '- Starting mixing predictions...')
    for iter in range(10000):
        with mlflow.start_run(experiment_id=experiment_id,
                              tags={'type_model': 'Ensemble',
                                    'train': 'holdout'}):
            mlpw2 = random.random()
            mlpw3 = random.random()
            mlpw1 = (3.0 - mlpw2 - mlpw3)
            res = minimize(combina_predicciones,
                           [mlpw2, mlpw3],
                           args=(test_y, predicted_mlp1, predicted_mlp2, predicted_mlp3),
                           method='nelder-mead',
                           options={'xatol': 1e-8,
                                    'disp': False}
                           )

            mlflow.log_param("mlpw1", mlpw1)
            mlflow.log_param("mlpw2", mlpw2)
            mlflow.log_param("mlpw3", mlpw3)

            mlflow.log_metric("rmse", res.fun)

            if verbose:
                if res.fun < min_rmse:
                    min_rmse = res.fun
                    print(get_current_time(), "- [mlp_peso={:.2f}, lgb_peso={:.2f}, xgb_peso={:.2f}] - "
                                              "[rmse={:.3f}]".format(mlpw1, mlpw2, mlpw3, res.fun))
