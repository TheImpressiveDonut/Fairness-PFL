import math
from itertools import combinations
from math import factorial
from typing import Tuple

import numpy as np
from tqdm.notebook import tqdm


def val(indices: Tuple, idx: int, mus, sigmas, num_samples, utility, E_cache, u_E_cache) -> float:
    np_indices = np.array(list(indices))
    if indices in E_cache:
        E = E_cache[indices]
    else:
        E = (mus[np_indices] ** 2 * num_samples[np_indices] / sigmas[np_indices] ** 2).sum()
        E_cache[indices] = E

    if indices in u_E_cache:
        u_E = u_E_cache[indices]
    else:
        u_E = (utility(mus[np_indices] ** 2 / (1 + E))).sum()
        u_E_cache[indices] = u_E

    indices_i = indices + (idx,)
    if indices_i in E_cache:
        E_i = E_cache[indices_i]
    else:
        E_i = E + (mus[idx] ** 2 * num_samples[idx] / sigmas[idx] ** 2)
        E_cache[indices_i] = E_i

    if len(np_indices) == 1:
        return (utility(mus[np_indices] ** 2 / (1 + E_i))).sum() - utility(
            (sigmas[np_indices] ** 2 / num_samples[np_indices]).item())
    else:
        return (utility(mus[np_indices] ** 2 / (1 + E_i))).sum() - u_E


def factorial_sp(n: int, fac_cache) -> int:
    if n in fac_cache:
        return fac_cache[n]
    else:
        val = factorial(n)
        fac_cache[n] = val
        return val


def shapley_value(idx: int, N: int, pbar: tqdm, mus, sigmas, num_samples, utility, E_cache, u_E_cache,
                  fac_cache) -> float:
    res = 0
    for i in range(1, N - 1):
        pbar.set_description(f'{idx}: {i}', refresh=True)
        for comb in combinations([j for j in range(N) if j != idx], i):
            res += (factorial_sp(len(comb), fac_cache) * factorial_sp(N - len(comb) - 1, fac_cache) *
                    val(comb, idx, mus, sigmas, num_samples, utility, E_cache, u_E_cache))
            pbar.update(1)
    return res / factorial(N)


def shapley_values(N: int, mus, sigmas, num_samples, utility) -> np.ndarray:
    E_cache = {}
    u_E_cache = {}
    fac_cache = {}

    total_iterations = N * sum([math.comb(N - 1, i) for i in range(1, N - 1)])
    pbar = tqdm(total=total_iterations, desc='Shapley value computation Progress', leave=False)

    shap_val = np.array(list(
        map(lambda x: shapley_value(x, N, pbar, mus, sigmas, num_samples, utility, E_cache, u_E_cache, fac_cache),
            range(N))))

    pbar.close()

    return shap_val
