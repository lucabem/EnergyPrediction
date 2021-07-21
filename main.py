import logging
import warnings

import numpy as np
from mlflow.tracking import MlflowClient

from models.dt_mlflow import run_dt
from models.elasticnet_mlflow import run_elasticnet
from models.knn_mflow import run_knn
from models.lightgbm_mlflow import run_lgbm
from preprocess.utils import *

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(2021)

    data = load_cleaned_data()

    train, test = split_data(data)

    client = MlflowClient()

    if not os.path.isdir('mlruns/1'):
        try:
            experiment_elasticnet = client.create_experiment("ElasticNet")
        except:
            experiment_elasticnet = client.get_experiment_by_name("ElasticNet").experiment_id

        params = {
            'alphas': [i for i in np.arange(0, 1.25, 0.25)],
            'l1_ratios': [i for i in np.arange(0, 1.25, 0.25)]
        }

        run_elasticnet(experiment_id=experiment_elasticnet,
                       dataset=train,
                       params=params)
    else:
        print("Already")

    if not os.path.isdir('mlruns/2'):
        try:
            experiment_knn = client.create_experiment("KNN")
        except:
            experiment_knn = client.get_experiment_by_name("KNN").experiment_id
        run_knn(experiment_id=experiment_knn,
                dataset=train)
    else:
        print("Already exists")

    if not os.path.isdir('mlruns/3'):
        try:
            experiment_lgbm = client.create_experiment("LGBM")
        except:
            experiment_lgbm = client.get_experiment_by_name("LGBM").experiment_id
        run_lgbm(experiment_id=experiment_lgbm,
                 dataset=train)
    else:
        print("Hey")

    if not os.path.isdir('mlruns/4'):
        try:
            experiment_dt = client.create_experiment("DT")
        except:
            experiment_dt = client.get_experiment_by_name("DT").experiment_id
        run_dt(experiment_id=experiment_dt,
               dataset=train)
    else:
        print("Hey")
