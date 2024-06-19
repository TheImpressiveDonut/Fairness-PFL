from typing import Tuple, List

import numpy as np
import scipy
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
                          sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_betas = betas
    for W in Ws:
        new_betas = W.dot(new_betas)

    contribution = (np.abs(Ws[-1].T) - np.abs(Ws[-1])).sum(axis=1)
    C_vec = -contribution
    C_vec -= C_vec.min()
    C_vec /= C_vec.max()

    normalized_C = C_vec.copy()
    C = (1 - C_vec) * std_squared / (2 * betas.shape[1] * sizes)
    C /= C.sum()
    C = np.var(betas, axis=0) * 10 * C[:, np.newaxis]
    # 10 for adult, 2 betas for
    noise = C * np.random.normal(loc=0, scale=1, size=betas.shape)
    return new_betas + noise, normalized_C, C


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
        betas_fair_fusion, _, _ = get_fair_fusion_betas(betas_local, Ws, std_squared, sizes)
        e_mse[idx] = get_mse(Xs, ys, betas_fair_fusion)

    return np.mean(e_mse, axis=0), np.var(e_mse, axis=0)


def get_auc_local_fusion(Xs: List[np.ndarray], ys: List[np.ndarray],
                         betas_local: np.ndarray, betas_fusion: np.ndarray) -> np.ndarray:
    auc_res = np.zeros((2, len(Xs)))
    for idx, (X_small, y_small) in enumerate(zip(Xs, ys)):
        auc_res[0, idx] = roc_auc_score(y_small, X_small.dot(betas_local[idx, :]))
        auc_res[1, idx] = roc_auc_score(y_small, X_small.dot(betas_fusion[idx, :]))
    return auc_res
