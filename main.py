import os
import warnings

import numpy as np
import pandas as pd

from preprocess.utils import get_current_time, load_cleaned_data, split_data
from utils.functions import plot_frecuencies, train_model

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(2021)

    trainmodels = True

    data = load_cleaned_data()
    train, test = split_data(data)

    models = ['ElasticNet', 'KNN', 'LGBM', 'DT', 'XGB', 'MLP']
    if trainmodels:
        for model in models:
            train_model(experiment_name=model,
                        train_data=train,
                        test_data=test,
                        params=None,
                        verbose=False)

    models = os.listdir('predictions/15mins')
    name_models = [name.split('_')[0] for name in models]

    print(get_current_time(), '- Plotting predictions..')
    for model in models:
        path = 'predictions/15mins/' + model
        data = pd.read_csv(path)
        data = data.drop(columns=['Date.1'])
        plot_frecuencies(data=data,
                         method=name_models[models.index(model)])
