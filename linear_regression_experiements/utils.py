from typing import Tuple, List

import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score


def linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pseudo_inv_X = np.linalg.pinv(X)
    w = pseudo_inv_X.dot(y)
    trace = np.trace(pseudo_inv_X.dot(pseudo_inv_X.T))
    return w, trace


def get_weights(expected_betas: np.ndarray, betas_opt: np.ndarray, std_squared: np.ndarray) -> np.ndarray:
    C = expected_betas.dot(expected_betas.T)
    K = betas_opt.dot(betas_opt.T)
    return scipy.linalg.solve(C + np.diag(std_squared), K.T, assume_a='sym').T


def train(Xs: List[np.ndarray], ys: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    std_squared = []
    betas = []
    sizes = []

    for X_small, y_small in zip(Xs, ys):
        w, trace = linear_regression(X_small, y_small)
        betas.append(w)
        m = X_small.shape[0]
        sizes.append(m)

        r = sum((y_small - X_small.dot(w)) ** 2)
        std_squared.append(r / m * trace)

    betas = np.array(betas)
    std_squared = np.array(std_squared)
    sizes = np.array(sizes)
    return betas, std_squared, sizes


def plot_betas(betas: np.ndarray, sizes: np.ndarray) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=betas.shape[0], sharex=True, sharey=True, figsize=(20, 10))
    for idx, b in enumerate(betas):
        axs[idx].plot(b)
        axs[idx].set_title(f"Beta n°{idx} of size {sizes[idx]}")
    plt.show()


def OLS_iterative_fusion(betas: np.ndarray, std_squared: np.ndarray, num_iter: int = 2) -> List[np.ndarray]:
    new_betas = betas
    Ws = []
    for idx in range(num_iter):
        W = get_weights(new_betas, new_betas, std_squared)
        new_betas = W.dot(new_betas)
        Ws.append(W)

    return Ws


def get_fusion_betas(betas: np.ndarray, Ws: List[np.ndarray]) -> np.ndarray:
    new_betas = betas
    for W in Ws:
        new_betas = W.dot(new_betas)
    return new_betas


def get_fair_fusion_betas(betas: np.ndarray, Ws: List[np.ndarray], std_squared: np.ndarray,
                          sizes: np.ndarray) -> np.ndarray:
    new_betas = betas
    for W in Ws:
        new_betas = W.dot(new_betas)

    C = (Ws[-1].T - Ws[-1]).sum(axis=1)
    C -= C.min()
    C = C / C.max()
    C = (1 - C) * (std_squared * np.random.normal(loc=0, scale=1, size=betas.shape[0]) / sizes ** 2)
    print(C)
    return new_betas + C[:, np.newaxis]


def get_mse_local_fusion(Xs: List[np.ndarray], ys: List[np.ndarray],
                         betas_local: np.ndarray, betas_fusion: np.ndarray) -> np.ndarray:
    mse = np.zeros((2, len(Xs)))
    for idx, (X_small, y_small) in enumerate(zip(Xs, ys)):
        mse[0, idx] = ((y_small - X_small.dot(betas_local[idx, :])) ** 2).mean()
        mse[1, idx] = ((y_small - X_small.dot(betas_fusion[idx, :])) ** 2).mean()
    return mse


def get_auc_local_fusion(Xs: List[np.ndarray], ys: List[np.ndarray],
                         betas_local: np.ndarray, betas_fusion: np.ndarray) -> np.ndarray:
    auc_res = np.zeros((2, len(Xs)))
    for idx, (X_small, y_small) in enumerate(zip(Xs, ys)):
        auc_res[0, idx] = roc_auc_score(y_small, X_small.dot(betas_local[idx, :]))
        auc_res[1, idx] = roc_auc_score(y_small, X_small.dot(betas_fusion[idx, :]))
    return auc_res


def plot_score(score: np.ndarray, std_squared: np.ndarray, sizes: np.ndarray, order: str = "normal") -> None:
    indices = np.argsort(score[0, :] - score[1, :])
    if order == "reverse":
        indices = indices[::-1]
    score = score[:, indices]
    print(f"scores: {score}")
    std_squared = std_squared[indices]
    sizes = sizes[indices]

    fig, ax1 = plt.subplots()
    labels = {0: "local", 1: "fusion"}
    for idx in range(score.shape[0]):
        ax1.scatter(range(score.shape[1]), score[idx, :], label=labels[idx])
    ax2 = ax1.twinx()
    ax2.scatter(range(std_squared.shape[0]), std_squared, color="r", label="Difficulty")
    ax2.set_ylabel('Task difficulty', color='r')
    ax3 = ax1.twinx()
    ax3.scatter(range(sizes.shape[0]), sizes, color="black", label="Sizes")
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Sizes', color='black')

    handles, labels = [], []
    for ax in [ax1, ax2, ax3]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.5, 0.5))

    plt.show()
