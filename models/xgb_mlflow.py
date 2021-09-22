import mlflow
from xgboost import XGBRegressor

from preprocess.utils import get_current_time, scale_data
from utils.constants import X
from utils.functions import train_test, eval_metrics


def run_xgb(experiment_id, dataset, params=None, verbose=False):
    print(get_current_time(), "- Starting XGB Model...")

    train_x, test_x, train_y, test_y = train_test(dataset)

    if params is None:
        max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
        eta = [0.01, 0.05, 0.1, 0.15, 0.3]
        n_estimators = [100, 300, 500, 700, 900, 1000]
        subsample = [0.7, 0.8, 1]
        colsample_bytree = [0.8, 1]
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
                            mlflow.log_metric("rmspe", mae)

                            mlflow.sklearn.log_model(model, "model")

                            if verbose:
                                print(get_current_time(), "- [eta={}, max_depth={}, subsample={}, "
                                                          "n_estimators={}, colsample_bytree={}] - [rmspe={:.3f}, "
                                                          "rmse={:.3f}, "
                                                          "r2={:.3f}]".format(e, depth, ss, trees,
                                                                              csby, mae, rmse, r2))

    print(get_current_time(), "- Ended XGB Model...")
