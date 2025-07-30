import json

import matplotlib.pyplot as plt
import os
import numpy as np


def plot_comparison(
        original: np.ndarray,
        degraded: np.ndarray,
        restored_inv: np.ndarray,
        restored_wiener: np.ndarray,
        save_path: str = None
) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["Original", "Degraded", "Inverse Filter", "Wiener Filter"]
    images = [original, degraded, restored_inv, restored_wiener]

    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)


def plot_metrics(json_path,save_path: str = None) -> None:

    with open(json_path, 'r') as f:
        data = json.load(f)

    metrics = ['PSNR', 'SSIM', 'MSE']
    methods = ['degraded', 'inverse', 'wiener']

    for metric in metrics:
        values = [data[f"{metric}_{method}"] for method in methods]

        plt.figure(figsize=(6, 4))
        plt.bar(methods, values, color=['gray', 'orange', 'green'])
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xlabel('Method')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        plt.savefig(save_path, dpi=150)



