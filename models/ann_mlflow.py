import itertools

import mlflow
from sklearn.neural_network import MLPRegressor

from preprocess.utils import get_current_time, scale_data
from utils.constants import X
from utils.functions import train_test, eval_metrics


def run_mlp(experiment_id, dataset, params=None, verbose=False):
    print(get_current_time(), "- Starting MLP Model...")

    train_x, test_x, train_y, test_y = train_test(dataset)

    train_x = scale_data(train_x, vars=X)
    test_x = scale_data(test_x, vars=X)

    if params is None:
        num_units_per_layer = [4, 8, 16, 32, 64, 128]
        num_layers = [1, 2, 3]
        solver = ['adam']
        activation = ['relu']
        alpha = [0.00001, 0.0001, 0.001]
        learning_rate_init = [0.0001, 0.001, 0.01, 0.1]
        random_state = 2021
    else:
        num_units_per_layer = params['num_units_per_layer']
        num_layers = params['num_layers']
        solver = params['solver']
        activation = params['activation']
        alpha = params['alpha']
        learning_rate_init = params['learning_rate_init']
        random_state = 2021

    layers = []
    for size in num_layers:
        layers += list(itertools.combinations(num_units_per_layer, size))

    for layer in layers:
        for s in solver:
            for a in activation:
                for al in alpha:
                    for lri in learning_rate_init:
                        with mlflow.start_run(experiment_id=experiment_id,
                                              tags={'type_model': 'MLPR',
                                                    'train': 'holdout'}):
                            model = MLPRegressor(hidden_layer_sizes=layer,
                                                 solver=s,
                                                 activation=a,
                                                 alpha=al,
                                                 learning_rate_init=lri,
                                                 random_state=random_state)
                            model.fit(train_x, train_y)
                            predicted_qualities = model.predict(test_x)

                            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

                            mlflow.log_param("hidden_layer_sizes", layer)
                            mlflow.log_param("solver", s)
                            mlflow.log_param("activation", a)
                            mlflow.log_param("alpha", al)
                            mlflow.log_param("learning_rate_init", lri)

                            mlflow.log_metric("rmse", rmse)
                            mlflow.log_metric("r2", r2)
                            mlflow.log_metric("mae", mae)

                            if verbose:
                                print(get_current_time(), "- [hidden_layer_sizes={} , solver={}, activation={}, "
                                                          "alpha={}, learning_rate_init={}] - [mae={:.3f}, rmse={:.3f},"
                                                          " r2={:.3f}]".format(layer, s, a, al,
                                                                               lri, mae, rmse, r2))

                            mlflow.sklearn.log_model(model, "model")

    print(get_current_time(), "- Ended MLP Model...")
