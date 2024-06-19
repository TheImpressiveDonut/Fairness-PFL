import numpy as np


def get_random_data(N: int):
    mus = np.random.uniform(-50., 50., size=N)
    sigmas = np.random.uniform(0.001, 2, size=N)
    num_samples = np.random.randint(100, 1000, size=N)
    return mus, sigmas, num_samples


def get_normal_data(N: int):
    mus = np.random.normal(50, 1, size=N)
    sigmas = np.random.uniform(0.001, 2, size=N)
    num_samples = np.random.randint(100, 1000, size=N)
    return mus, sigmas, num_samples


def get_local_mse_fmse_C(mus, sigmas, num_samples):
    N = mus.shape[0]
    V = np.eye(N) * sigmas
    C = mus.reshape((N, 1)) @ mus.reshape(N, 1).T
    W = C @ np.linalg.inv(C + V)

    contribution = (np.abs(W.T) - np.abs(W)).sum(axis=1)
    C_vec = -contribution
    C_vec -= C_vec.min()
    C_vec /= C_vec.max()

    local = sigmas ** 2 / num_samples
    mse = mus ** 2 / (1 + (mus ** 2 / local).sum())

    fmse = mse + C_vec * local

    return local, mse, fmse, (contribution - contribution.min()) / (contribution.max() - contribution.min())