from pandas import concat

import pandas as pd
import os
from datetime import datetime
from utils.constants import X
from sklearn.preprocessing import StandardScaler, MinMaxScaler

columns = ["Date",
           "ApparentPower",
           "CurrentAC_L1", "CurrentAC_L2", "CurrentAC_L3",
           "CurrentDC_MPP1",
           "Energy",
           "VoltageAC_L1", "VoltageAC_L1L2", "VoltageAC_L2", "VoltageAC_L3", "VoltageDC_MPP1",
           "Consumed_Directly", "Consumption",
           "Energy_From_Battery", "Energy_From_Grid",
           "Energy_To_Battery", "Energy_To_Grid",
           "PV_Production"]


def load_datasets_15mnts():
    data_files = os.listdir("CononsythFarm/Raw_data/North_Mains_Solar_15minutes")
    list_data = []

    for file in data_files:
        path = "CononsythFarm/Raw_data/North_Mains_Solar_15minutes/" + file
        dataset = pd.read_excel(io=path,
                                skiprows=2,
                                names=columns)
        list_data.append(dataset)

    full_data = pd.DataFrame(columns=columns)

    for d in list_data:
        full_data = full_data.append(d)

    full_data.to_excel('CononsythFarm/Raw_data/15minutes_full/full_raw_data.xlsx',
                       index=False)
    return full_data


def get_current_time():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return f'[{dt_string}]'


def remove_empty_columns(dataset):
    columns_aux = columns[1:]
    columns_to_drop = []

    for column in columns_aux:
        uniques_values = dataset[column].unique()
        if len(uniques_values) < 3:
            columns_to_drop.append(column)

    dataset = dataset.drop(columns=columns_to_drop)
    dataset.to_excel('CononsythFarm/Raw_data/15minutes_Full/raw_data_without_empty_columns.xlsx',
                     index=False)
    return dataset


def remove_empty_rows(dataset):
    new_dataset = dataset.dropna(how='all',
                                 subset=dataset.columns[1:])
    new_dataset.to_excel('CononsythFarm/Raw_data/15minutes_Full/raw_data_without_nan_rows.xlsx',
                         index=False)

    return new_dataset


def remove_nan_rows(dataset):
    return dataset.dropna(how='any')


def remove_zeros_rows(dataset):
    return dataset.loc[(dataset[dataset.columns[1:]] != 0).all(axis=1)]


def load_cleaned_data():
    path_nan_rows = "CononsythFarm/Raw_data/15minutes_Full/raw_data_without_nan_rows.xlsx"
    if not os.path.exists(path_nan_rows):
        path_empty_columns = "CononsythFarm/Raw_data/15minutes_Full/raw_data_without_empty_columns.xlsx"
        if not os.path.exists(path_empty_columns):
            path = "CononsythFarm/Raw_data/15minutes_Full/full_raw_data.xlsx"
            if not os.path.exists(path):
                print(get_current_time(), "- Creating full raw dataset...")
                dataset = load_datasets_15mnts()
                print(get_current_time(), "- Done!")
            else:
                print(get_current_time(), "- Loading raw dataset exists on disk...\nLoading...")
                dataset = pd.read_excel(io=path,
                                        names=columns)
                print(get_current_time(), "- Loaded!")

            print(get_current_time(), "- Removing columns with all values NaN...")
            dataset = remove_empty_columns(dataset)
            print(get_current_time(), "- Done!")
        else:
            print(get_current_time(), "- Dataset without empty columns exists on disk...\nLoading...")
            dataset = pd.read_excel(io=path_empty_columns)
            print(get_current_time(), "- Loaded!")
        print(get_current_time(), "- Removing rows with all values NaN...")
        dataset = remove_empty_rows(dataset)
        print(get_current_time(), "- Done!")
    else:
        print(get_current_time(), "- Loading dataset without empty rows exists on disk...")
        dataset = pd.read_excel(io=path_nan_rows)
        print(get_current_time(), "- Loaded!")

    return remove_zeros_rows(remove_nan_rows(dataset))


def scale_data(dataset, vars=["month", "day", "hour", "t-3", "t-2", "t-1"]):
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataset[vars].values)


def split_data(dataset, year=2017):
    data_datetime = dataset.set_index(dataset['Date'])
    data_datetime = data_datetime.sort_index()
    start = str(year) + '-01-01'
    ended = str(year) + '-12-31'
    return data_datetime.loc[data_datetime["Date"] < start], data_datetime.loc[start:ended]


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
