from typing import Tuple, List

import numpy as np
from sklearn.datasets import fetch_california_housing


def get_california_housing_dataset() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    data = fetch_california_housing()
    X, y = data.data, data.target

    N = 6
    latitudes = X[:, -2]  # Assuming latitude is the second last feature
    bins = np.linspace(latitudes.min(), latitudes.max(), N + 1)
    group_indices = np.digitize(latitudes, bins) - 1

    X_splits = [X[group_indices == i] for i in range(N)]
    y_splits = [y[group_indices == i] for i in range(N)]
    X_splits_train, X_splits_test = [data[:-100] for data in X_splits], [data[-100:] for data in X_splits]
    y_splits_train, y_splits_test = [data[:-100] for data in y_splits], [data[-100:] for data in y_splits]
    return X_splits_train, X_splits_test, y_splits_train, y_splits_test