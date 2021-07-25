import mlflow
from xgboost import XGBRegressor

from utils.constants import X

from preprocess.utils import get_current_time, scale_data
from utils.functions import train_test, eval_metrics, train_cv


def run_xgb(experiment_id, dataset, params=None, verbose=False):
    print(get_current_time(), "- Starting XGB Model...")

    train_x, test_x, train_y, test_y = train_test(dataset)

    train_x = scale_data(train_x, columns=X)
    test_x = scale_data(test_x, columns=X)

    if params is None:
        max_depth = [depth for depth in range(1, 11)]
        eta = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        n_estimators = [trees for trees in range(100, 1251, 200)]
        subsample = [0.2, 0.4, 0.8, 1]
        colsample_bytree = [0.2, 0.4, 0.8, 1]
        n_jobs = 10
    else:
        max_depth = params['max_depth']
        eta = params['eta']
        n_estimators = params['n_estimators']
        subsample = params['subsample']
        colsample_bytree = params['colsample_bytree']
        n_jobs = 8

    for e in eta:
        for depth in max_depth:
            for ss in subsample:
                for trees in n_estimators:
                    for csby in colsample_bytree:
                        with mlflow.start_run(experiment_id=experiment_id,
                                              tags={'type_model': 'XGB',
                                                    'train': 'holdout'}):

                            model = XGBRegressor(eta=e,
                                                 max_depth=depth,
                                                 subsample=ss,
                                                 n_estimators=trees,
                                                 colsample_bytree=csby,
                                                 n_jobs=n_jobs)

                            model.fit(train_x, train_y)
                            predicted_qualities = model.predict(test_x)

                            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                            mlflow.log_param("eta", e)
                            mlflow.log_param("max_depth", depth)
                            mlflow.log_param("subsample", ss)
                            mlflow.log_param("n_estimators", trees)
                            mlflow.log_param('colsample_bytree', csby)

                            mlflow.log_metric("rmse", rmse)
                            mlflow.log_metric("r2", r2)
                            mlflow.log_metric("mae", mae)

                            mlflow.sklearn.log_model(model, "model")

                            if verbose:
                                print(get_current_time(), "- [eta={}, max_depth={}, subsample={}, "
                                                          "n_estimators={}, colsample_bytree={}] - [mae={:.3f}, "
                                                          "rmse={:.3f}, "
                                                          "r2={:.3f}]".format(e, depth, ss, trees,
                                                                              csby, mae, rmse, r2))

    print(get_current_time(), "- Ended XGB Model...")
