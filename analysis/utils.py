
import matplotlib as plt
import numpy as np

def print_hist(dataset, column_name):
    bin_values = np.linspace(start=dataset[column_name].min(),
                             stop=dataset[column_name].max(),
                             num=25)

    dataset[column_name].hist(bins=bin_values,
                              figsize=[14, 6])


def print_bloxplot(dataset, column_name):
    return 0


def check_normality(dataset, column_name, alpha=0.05):
    return 0


