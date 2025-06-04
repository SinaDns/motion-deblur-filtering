# src/visualization.py

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
    """
    Create a 2×2 figure showing:
      [Original]    [Degraded]
      [Inverse-restored] [Wiener-restored]
    If save_path is not None, saves the figure to disk.
    """
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
    plt.show()
