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


def plot_betas(betas_local: np.ndarray, betas_fusion: np.ndarray, betas_fair_fusion: np.ndarray,
               std_squared: np.ndarray, sizes: np.ndarray) -> None:
    betas_local, std_squared, sizes = sort_betas_by_task_difficulty(betas_local, std_squared, sizes)
    betas_fusion, _, _ = sort_betas_by_task_difficulty(betas_fusion, std_squared, sizes)
    betas_fair_fusion, _, _ = sort_betas_by_task_difficulty(betas_fair_fusion, std_squared, sizes)
    fig, axs = plt.subplots(nrows=1, ncols=betas_local.shape[0], sharex=True, sharey=True, figsize=(20, 10))
    for idx, (b1, b2, b3) in enumerate(zip(betas_local, betas_fusion, betas_fair_fusion)):
        axs[idx].plot(b1)
        axs[idx].plot(b2)
        axs[idx].plot(b3)
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


def sort_betas_by_task_difficulty(betas: np.ndarray, std_squared: np.ndarray, sizes: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.argsort(std_squared / sizes)
    return betas[indices, :], std_squared[indices], sizes[indices]


def get_fusion_betas(betas: np.ndarray, Ws: List[np.ndarray]) -> np.ndarray:
    new_betas = betas
    for W in Ws:
        new_betas = W.dot(new_betas)
    return new_betas


def get_fair_fusion_betas(betas: np.ndarray, Ws: List[np.ndarray], std_squared: np.ndarray,
                          sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_betas = betas
    for W in Ws:
        new_betas = W.dot(new_betas)

    C = (Ws[-1].T - Ws[-1]).sum(axis=1)
    C -= C.min()
    C = C / C.max()
    normalized_C = C.copy()
    C = (1 - C) * std_squared / (betas.shape[1] * sizes)
    C /= C.sum()
    C = np.var(betas, axis=0) * 10 * C[:, np.newaxis] * np.random.normal(loc=0, scale=1, size=betas.shape)
    return new_betas + C, normalized_C


def get_mse(Xs: List[np.ndarray], ys: List[np.ndarray], betas: np.ndarray) -> np.ndarray:
    mse = np.zeros(len(Xs))
    for idx, (X_small, y_small) in enumerate(zip(Xs, ys)):
        mse[idx] = ((y_small - X_small.dot(betas[idx, :])) ** 2).mean()
    return mse


def get_expected_fair_mse(betas_local: np.ndarray, Ws: List[np.ndarray], std_squared: np.ndarray, sizes: np.ndarray,
                          Xs: List[np.ndarray], ys: List[np.ndarray], num_iter: int = 1000) -> Tuple[
    np.ndarray, np.ndarray]:
    e_mse = np.zeros((num_iter, betas_local.shape[0]))

    for idx in range(num_iter):
        betas_fair_fusion, _ = get_fair_fusion_betas(betas_local, Ws, std_squared, sizes)
        e_mse[idx] = get_mse(Xs, ys, betas_fair_fusion)

    return np.mean(e_mse, axis=0), np.var(e_mse, axis=0)


def get_auc_local_fusion(Xs: List[np.ndarray], ys: List[np.ndarray],
                         betas_local: np.ndarray, betas_fusion: np.ndarray) -> np.ndarray:
    auc_res = np.zeros((2, len(Xs)))
    for idx, (X_small, y_small) in enumerate(zip(Xs, ys)):
        auc_res[0, idx] = roc_auc_score(y_small, X_small.dot(betas_local[idx, :]))
        auc_res[1, idx] = roc_auc_score(y_small, X_small.dot(betas_fusion[idx, :]))
    return auc_res


def is_fair_fusion_bias_higher(mse_local_fusion: np.ndarray):
    local_bias = mse_local_fusion[1, :] - mse_local_fusion[0, :]
    local_bias
    N = 100
    # sum = np.zeros((N, betas.shape[0]))
    # for i in range(N):
    # betas_fair_fusion = get_fair_fusion_betas(betas, Ws, std_squared, sizes)
    # sum[i] = (get_mse_local_fusion(Xs_test, ys_test, betas_local, betas_fair_fusion)[1, :] - mse[0, :])

    # local_fair_bias = np.mean(sum, axis=0)
    tolerance = 1e-5

    # result = (local_fair_bias > local_bias - tolerance) | (np.isclose(local_fair_bias, local_bias, atol=tolerance))
    # result
    pass


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


def plot_mse_local_fusion_fair(mse_local: np.ndarray, mse_fusion: np.ndarray, mse_fair_fusion: np.ndarray,
                               std_squared: np.ndarray, sizes: np.ndarray) -> None:
    indices = np.argsort(std_squared / sizes)
    mse_local = mse_local[indices]
    mse_fusion = mse_fusion[indices]
    mse_fair_fusion = mse_fair_fusion[indices]
    task_difficulty = (std_squared / sizes)[indices]

    num_clients = mse_local.shape[0]
    clients = [f"Client {i + 1} - {task_difficulty[i]:.10f}" for i in range(num_clients)]
    bar_width = 0.25
    r1 = np.arange(num_clients)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    plt.figure(figsize=(15, 6))

    plt.bar(r1, mse_local, color='blue', width=bar_width, edgecolor='grey', label='MSE Local')
    plt.bar(r2, mse_fusion, color='green', width=bar_width, edgecolor='grey', label='Fusion MSE Gain')
    plt.bar(r3, mse_fair_fusion, color='red', width=bar_width, edgecolor='grey', label='Fair Fusion MSE Gain')

    plt.xlabel('Clients', fontweight='bold')
    plt.ylabel('MSE', fontweight='bold')
    plt.yscale('log')
    plt.title('None')
    plt.xticks([r + bar_width for r in range(num_clients)], clients)
    plt.legend()
    plt.show()
