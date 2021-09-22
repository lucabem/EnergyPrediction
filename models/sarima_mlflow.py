import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def test_ts(experiment_id, traindata, testdata):
    return 0


def run_sarima(experiment_id, train_data, test_data, params=None, verbose=False):

    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data.set_index('Date', inplace=True)
    ts = train_data['PV_Production']
    decomposition = seasonal_decompose(ts)

    return 0