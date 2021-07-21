import mlflow
from sklearn.tree import DecisionTreeRegressor

from preprocess.utils import get_current_time, scale_data
from utils.constants import X
from utils.functions import train_test, eval_metrics


def run_dt(experiment_id, dataset, params=None):
    print(get_current_time(), "- Starting DT Model...")

    train_x, test_x, train_y, test_y = train_test(dataset)

    train_x = scale_data(train_x, columns=X)
    test_x = scale_data(test_x, columns=X)

    if params is None:
        criterion = ['mse', 'friedman_mse']
        splitter = ['best']
        max_depth = [i for i in range(1, 11)]
        min_samples_split = [2, 4, 8, 16, 32]
        min_samples_leaf = [1, 2, 4, 8, 16, 32]
    else:
        criterion = params['criterion']
        splitter = params['splitter']
        max_depth = params['max_depth']
        min_samples_split = params['min_Samples_split']
        min_samples_leaf = params['min_samples_leaf']

    cont = 0
    for cr in criterion:
        for spl in splitter:
            for depth in max_depth:
                for mss in min_samples_split:
                    for msl in min_samples_leaf:
                        name = '_'.join(['decisionTree', str(cont)])
                        with mlflow.start_run(experiment_id=experiment_id,
                                              run_name=name,
                                              tags={'type_model': 'DecisionTree',
                                                    'train': 'holdout'}):
                            model = DecisionTreeRegressor(
                                criterion=cr,
                                splitter=spl,
                                max_depth=depth,
                                min_samples_leaf=msl,
                                min_samples_split=mss
                            )

                            model.fit(train_x, train_y)

                            predicted_qualities = model.predict(test_x)

                            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                            mlflow.log_params({
                                'criterion': cr,
                                'splitter': spl,
                                'max_depth': depth,
                                'min_samples_leaf': msl,
                                'min_samples_split': mss
                            })

                            mlflow.log_metrics({
                                "rmse": rmse,
                                "r2": r2,
                                "mae": mae
                            })

                            cont += 1

    print(get_current_time(), "- Ended DT Model...")
