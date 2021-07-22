import mlflow
import numpy as np

from utils.constants import X
from lightgbm import LGBMRegressor

from preprocess.utils import get_current_time, scale_data
from utils.functions import train_test, eval_metrics, train_cv


def run_lgbm(experiment_id, dataset, params=None):
    print(get_current_time(), "- Starting LGBM Model...")

    train_x, test_x, train_y, test_y = train_test(dataset)

    train_x = scale_data(train_x, columns=X)
    test_x = scale_data(test_x, columns=X)

    if params is None:
        num_leaves = [leave for leave in range(10, 51, 10)]
        max_depth = [depth for depth in range(1, 11)]
        learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
        n_estimators = [trees for trees in range(100, 1001, 100)]
        n_jobs = 10
    else:
        num_leaves = params['num_leaves']
        max_depth = params['max_depth']
        learning_rate = params['learning_rate']
        n_estimators = params['n_estimators']
        n_jobs = 8

    for leaves in num_leaves:
        for depth in max_depth:
            for lr in learning_rate:
                for trees in n_estimators:
                    with mlflow.start_run(experiment_id=experiment_id,
                                          tags={'type_model': 'LGBM',
                                                'train': 'holdout'}):

                        model = LGBMRegressor(num_leaves=leaves,
                                              max_depth=depth,
                                              learning_rate=lr,
                                              n_estimators=trees,
                                              n_jobs=n_jobs)

                        model.fit(train_x, train_y)
                        predicted_qualities = model.predict(test_x)

                        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                        mlflow.log_param("num_leaves", leaves)
                        mlflow.log_param("max_depth", depth)
                        mlflow.log_param("learning_rate", lr)
                        mlflow.log_param("n_estimators", trees)

                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("r2", r2)
                        mlflow.log_metric("mae", mae)

                        mlflow.sklearn.log_model(model, "model")

    print(get_current_time(), "- Ended LGBM Model...")
