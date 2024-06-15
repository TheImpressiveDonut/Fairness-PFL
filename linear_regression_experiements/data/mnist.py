import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.datasets import fetch_openml

root = Path(__file__).parent.joinpath('data/mnist/root/')

NUM_CLASSES = 10
DIM = (28, 28)


def get_mnist() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if not os.path.exists(root):
        os.makedirs(root)
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.uint8)
    X = X / 255.0

    Xs = []
    ys = []

    for i in range(NUM_CLASSES):
        Xs.append(X[y == i])
        ys.append(y[y == i])

    N = 5
    labels = np.array([0, 1, 7, 8, 9]) # np.arange(10)
    num_samples = np.random.randint(300, 3000, size=N)
    N = num_samples.shape[0]
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for labels in range(N):

        np.random.shuffle(samples)
        train_samples = samples[:int(samples.shape[0] * 0.75)]
        test_samples = samples[int(samples.shape[0] * 0.75):]
        X_train.append(X[train_samples, :])
        X_test.append(X[test_samples, :])
        y_train.append(y[train_samples])
        y_test.append(y[test_samples])

    return X_train, X_test, y_train, y_test
