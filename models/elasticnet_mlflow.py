import mlflow
import numpy as np
from sklearn.linear_model import ElasticNet

from preprocess.utils import get_current_time, scale_data
from utils.constants import X
from utils.functions import train_test, eval_metrics, train_cv


def run_elasticnet(experiment_id, dataset, params=None, verbose=False):
    print(get_current_time(), "- Starting ElasticNet Model...")

    train_x, test_x, train_y, test_y = train_test(dataset)

    train_x = scale_data(train_x, columns=X)
    test_x = scale_data(test_x, columns=X)

    if params is None:
        alphas = [i for i in np.arange(0, 1.25, 0.25)]
        l1_ratios = [i for i in np.arange(0, 1.25, 0.25)]
    else:
        alphas = params['alphas']
        l1_ratios = params['l1_ratios']

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            name = '_'.join(['elastic', 'alpha', str(alpha), 'l1_ratio', str(l1_ratio)])
            with mlflow.start_run(experiment_id=experiment_id,
                                  run_name=name,
                                  tags={'type_model': 'ElasticNet',
                                        'train': 'holdout'}):
                lr = ElasticNet(alpha=alpha,
                                l1_ratio=l1_ratio)
                lr.fit(train_x, train_y)

                predicted_qualities = lr.predict(test_x)

                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                mlflow.sklearn.log_model(lr, "model")

                if verbose:
                    print(get_current_time(), "- [alpha={:.2f}, l1_ratio={:.2f}] - [mae={:.3f}, rmse={:.3f},"
                                              " r2={:.3f}]".format(alpha, l1_ratio, mae, rmse, r2))

    print(get_current_time(), "- Ended ElasticNet Model...")


def run_elasticnet_cv(experiment_id, dataset, params=None, verbose=False):
    print(get_current_time(), "- Starting ElasticNet CV Model...")

    if params is None:
        alphas = [i for i in np.arange(0, 1.25, 0.25)]
        l1_ratios = [i for i in np.arange(0, 1.25, 0.25)]
    else:
        alphas = params['alphas']
        l1_ratios = params['l1_ratios']

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            name = '_'.join(['elastic', 'alpha', str(alpha), 'l1_ratio', str(l1_ratio)])
            with mlflow.start_run(experiment_id=experiment_id,
                                  run_name=name,
                                  tags={'type_model': 'ElasticNet',
                                        'train': 'cv'}):
                lr = ElasticNet(alpha=alpha,
                                l1_ratio=l1_ratio)

                rmse, mae, r2 = train_cv(lr, dataset)

                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                if verbose:
                    print(get_current_time(), "- [alpha={:.2f} , l1_ratio={:.2f}] - [mae={:.3f}, rmse={:.3f},"
                                              " r2={:.3f}]".format(alpha, l1_ratio, mae, rmse, r2))

    print(get_current_time(), "- Ended ElasticNet CV Model...")
