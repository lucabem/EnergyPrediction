import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold

from preprocess.utils import scale_data
from utils.constants import X


def train_test(dataset, label_column='PV_Production', train_percentage=0.75):
    dataset = dataset.drop(columns=['Energy', 'Date'])
    train, test = train_test_split(dataset,
                                   train_size=train_percentage)

    train_values = train.drop(columns=[label_column])
    test_values = test.drop(columns=[label_column])
    train_labels = train[[label_column]]
    test_labels = test[label_column]

    return train_values, test_values, train_labels, test_labels


def eval_metrics(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred)), \
           mean_absolute_error(actual, pred), \
           r2_score(actual, pred)


# Función que obtiene el MAE, RMSE y R2 de VC con repetición
def train_cv(model, dataset, label_column='PV_Production', num_folds=10, num_bags=10):
    np.random.seed(2021)
    X_tot = dataset.copy()
    y_tot = dataset[label_column].values.reshape(-1)

    X_tot = scale_data(X_tot, columns=X)

    # Creamos arrays para las predicciones
    preds_val = np.empty((len(X_tot), num_bags))
    preds_val[:] = np.nan

    # Entrena y extrae la predicciones con validación cruzada repetida
    folds = RepeatedKFold(n_splits=num_folds, n_repeats=num_bags, random_state=2021)

    for niter, (train_index, val_index) in enumerate(folds.split(X_tot, y_tot)):
        nbag = niter // num_folds  # Extrae el número de repetición (bag)
        X_train, X_val = X_tot[train_index], X_tot[val_index]
        y_train, y_val = y_tot[train_index], y_tot[val_index]
        model.fit(X_train, y_train)
        preds_val[val_index, nbag] = model.predict(X_val)

    # Promedia las predicciones
    preds_val_mean = preds_val.mean(axis=1)

    (rmse, mae, r2) = eval_metrics(y_tot, preds_val_mean)

    return rmse, mae, r2
