from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def sort_betas_by_task_difficulty(betas: np.ndarray, std_squared: np.ndarray, sizes: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.argsort(std_squared / sizes)
    return betas[indices, :], std_squared[indices], sizes[indices]


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
