from typing import Tuple, List

import numpy as np
import pandas as pd


def get_penguins_dataset() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    penguins = pd.read_csv("data/datasets/penguins/penguins.csv")
    penguins.dropna(how="any", inplace=True)
    penguins = penguins.sample(frac=1, random_state=42).reset_index(drop=True)

    cols = ["island", "species", "sex"]
    for col in cols:
        penguins[col] = pd.Categorical(penguins[col], categories=penguins[col].unique()).codes

    X_splits_train, X_splits_test = [], []
    y_splits_train, y_splits_test = [], []

    for group, val in penguins.groupby("year"):
        val.drop(["year"], axis=1, inplace=True)
        val = val.sample(frac=1, random_state=42).reset_index(drop=True)
        y_splits_train.append(val.species.iloc[:-10].to_numpy())
        y_splits_test.append(val.species.iloc[-10:].to_numpy())
        val.drop(["species"], axis=1, inplace=True)
        X_splits_train.append(val.iloc[:-10, :].to_numpy())
        X_splits_test.append(val.iloc[-10:, :].to_numpy())

    return X_splits_train, X_splits_test, y_splits_train, y_splits_test
