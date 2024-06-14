from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def get_features(data_fn):
    data = np.loadtxt(data_fn, dtype='str', delimiter=',')
    df = pd.DataFrame(data)
    df = df.replace(' ?', np.nan)
    df.dropna(how='any', inplace=True)


    df.iloc[:, -1] = df.iloc[:, -1].map({' <=50K': 0, ' >50K': 1}).astype(int).values

    df.iloc[:, -2] = df.iloc[:, -2].apply(lambda x: 1 if x == " United-States" else 0)
    df.iloc[:, -4] = df.iloc[:, -2].apply(lambda x: 0 if x == " 0" else 1)
    df.iloc[:, -5] = df.iloc[:, -2].apply(lambda x: 0 if x == " 0" else 1)


    cols = [1, 3, 5, 6, 7, 9]
    for col in cols:
        df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes

    X, y = [], []

    for group, val in df.groupby(df.columns[8]):
        val.drop(df.columns[[8]], axis=1, inplace=True)
        val = val.sample(frac=1, random_state=42).reset_index(drop=True)
        y.append(val.iloc[:, -1].to_numpy().astype(int))
        val.drop(df.columns[[-1]], axis=1, inplace=True)
        val = val.astype(int)
        X.append(val.to_numpy())
    return X, y

def get_adult_dataset() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    X_train, y_train = get_features('data/adult.data')
    X_test, y_test = get_features('data/adult_w.test')
    return X_train, X_test, y_train, y_test