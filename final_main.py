import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from preprocess.utils import scale_data
from utils.functions import eval_metrics

if __name__ == "__main__":
    forecast = True
    preds = []

    final_models = []

    train_data = pd.read_csv("train_clean_h.csv")
    ttest_data = pd.read_csv("test_clean_h.csv")

    ttest_data['Date'] = pd.to_datetime(ttest_data['Date'])
    ttest_data.drop(columns=['Date',
                             'Date.1'], inplace=True)


    DT = DecisionTreeRegressor(splitter='best',
                               min_samples_split=32,
                               max_depth=11,
                               criterion='friedman_mse',
                               min_samples_leaf=32)
    final_models.append(DT)

    LGBM = LGBMRegressor(max_depth=10,
                         num_leaves=20,
                         n_estimators=100,
                         learning_rate=0.1,
                         n_jobs=10)
    final_models.append(LGBM)

    model_names = ['DT', 'LGBM']

    tt = train_data.drop(columns=['Date', 'Date.1', 't'])

    test_y = ttest_data['t'].values
    test_x = ttest_data.drop(columns=['t'])
    test_x.reset_index(drop=True, inplace=True)

    dfuture = 1
    errors = []

    if forecast:
        for model in final_models:
            model.fit(tt, train_data['t'])
            for dfuture in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                preds = []
                for i in range(0, test_x.shape[0] - dfuture):
                    lags = [test_x['t-3'].values[i], test_x['t-2'].values[i], test_x['t-1'].values[i]]
                    for j in range(0, dfuture + 1):
                        vars = [test_x.iloc[i].month_sin, test_x.iloc[i].month_cos,
                                test_x.iloc[i].day_sin, test_x.iloc[i].day_cos,
                                test_x.iloc[i].hour_sin, test_x.iloc[i].hour_cos]

                        for value in lags:
                            vars.append(value)

                        obs = np.array([vars])

                        lag0 = model.predict(obs.reshape(1, -1))
                        lags = lags[1:] + [lag0[0]]
                    preds.append(lag0[0])

                rmse, ma, r2 = eval_metrics(test_y[dfuture:], preds)
                print("LAG", dfuture, "-", rmse)
                errors.append(rmse)

            df = pd.DataFrame(errors)
            df.to_csv('future/errors_' + model_names[final_models.index(model)] + '.csv')
