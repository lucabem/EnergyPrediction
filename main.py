import logging
import warnings

import numpy as np
from mlflow.tracking import MlflowClient

from models.dt_mlflow import run_dt
from models.elasticnet_mlflow import run_elasticnet
from models.knn_mflow import run_knn
from models.lightgbm_mlflow import run_lgbm
from preprocess.utils import *
from utils.functions import test_best_model, print_test_errors, eval_metrics, save_best_params, plot_frecuencies

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(2021)

    trainmodels = False
    if trainmodels:
        data = load_cleaned_data()

        train, test = split_data(data)

        client = MlflowClient()

        try:
            experiment_elasticnet = client.create_experiment("ElasticNet")
        except:
            experiment_elasticnet = client.get_experiment_by_name("ElasticNet").experiment_id

        params = {
            'alphas': [i for i in np.arange(0, 1.1, 0.1)],
            'l1_ratios': [i for i in np.arange(0, 1.1, 0.1)]
        }

        run_elasticnet(experiment_id=experiment_elasticnet,
                       dataset=train,
                       params=params)

        params_stats = save_best_params(experiment_id=experiment_elasticnet)

        real, predictions = test_best_model(experiment_elasticnet, test)

        data = pd.DataFrame(data={
            'Date': test['Date'],
            'Real': real,
            'Pred': predictions
        })

        print_test_errors(data,
                          method='ElasticNet')
        (rmse, mae, r2) = eval_metrics(real, predictions)

        print(get_current_time(), "- Metrica RMSE Test", rmse)
        data.to_csv('predictions/elasticNet_2017.csv')
        print(get_current_time(), "- Saved results of ElasticNet to CSV")

        try:
            experiment_knn = client.create_experiment("KNN")
        except:
            experiment_knn = client.get_experiment_by_name("KNN").experiment_id
        run_knn(experiment_id=experiment_knn,
                dataset=train)

        real, predictions = test_best_model(experiment_knn, test)

        predictions = predictions.reshape(-1)

        data = pd.DataFrame(data={
            'Date': test['Date'],
            'Real': real,
            'Pred': predictions
        })

        print_test_errors(data,
                          method='KNN')
        (rmse, mae, r2) = eval_metrics(real, predictions)

        print(get_current_time(), "- Metrica RMSE KNN Test", rmse)
        data.to_csv('predictions/KNN_2017.csv')
        print(get_current_time(), "- Saved results of KNN to CSV")

        try:
            experiment_lgbm = client.create_experiment("LGBM")
        except:
            experiment_lgbm = client.get_experiment_by_name("LGBM").experiment_id
        run_lgbm(experiment_id=experiment_lgbm,
                 dataset=train)

        real, predictions = test_best_model(experiment_lgbm, test)

        data = pd.DataFrame(data={
            'Date': test['Date'],
            'Real': real,
            'Pred': predictions
        })

        print_test_errors(data,
                          method='LGBM')
        (rmse, mae, r2) = eval_metrics(real, predictions)

        print(get_current_time(), "- Metrica RMSE LGBM Test", rmse)
        data.to_csv('predictions/LGBM_2017.csv')
        print(get_current_time(), "- Saved results of LGBM to CSV")

        try:
            experiment_dt = client.create_experiment("DT")
        except:
            experiment_dt = client.get_experiment_by_name("DT").experiment_id
        run_dt(experiment_id=experiment_dt,
               dataset=train)
        real, predictions = test_best_model(experiment_dt, test)

        data = pd.DataFrame(data={
            'Date': test['Date'],
            'Real': real,
            'Pred': predictions
        })

        print_test_errors(data,
                          method='DT')
        (rmse, mae, r2) = eval_metrics(real, predictions)

        print(get_current_time(), "- Metrica RMSE DT Test", rmse)
        print(get_current_time(), "- Saved results of DT to CSV")
        data.to_csv('predictions/dt_2017.csv')
    else:
        print(get_current_time(), '- No Models Trained. Using CSV predictions...')

        models = os.listdir('predictions/15mins')
        name_models = [name.split('_')[0] for name in models]

        for model in models:
            path = 'predictions/15mins/' + model
            data = pd.read_csv(path)
            data = data.drop(columns=['Date.1'])
            plot_frecuencies(data=data,
                             method=name_models[models.index(model)])
