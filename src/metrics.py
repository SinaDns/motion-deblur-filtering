from typing import Any

import numpy as np
from numpy import floating
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """
    Compute PSNR (in dB) between original and restored images.
    """
    return peak_signal_noise_ratio(original, restored, data_range=1.0)

def compute_ssim(original: np.ndarray, restored: np.ndarray) -> float:
    """
    Compute SSIM between original and restored images.
    """
    # skimage's structural_similarity returns (ssim, diff), so take first output
    ssim_val, _ = structural_similarity(
        original, restored, data_range=1.0, full=True
    )
    return ssim_val

def compute_mse(original: np.ndarray, restored: np.ndarray) -> floating[Any]:
    """
    Compute Mean Squared Error (MSE).
    """
    return np.mean((original - restored) ** 2)

def evaluate_metrics(
        original: np.ndarray,
        degraded: np.ndarray,
        restored_inv: np.ndarray,
        restored_wiener: np.ndarray
) -> dict:
    """
    Compute and return a dictionary of metrics for:
      - degraded vs. original
      - inverse-restored vs. original
      - wiener-restored vs. original
    """
    metrics = {"PSNR_degraded": compute_psnr(original, degraded), "PSNR_inverse": compute_psnr(original, restored_inv),
               "PSNR_wiener": compute_psnr(original, restored_wiener),
               "SSIM_degraded": compute_ssim(original, degraded), "SSIM_inverse": compute_ssim(original, restored_inv),
               "SSIM_wiener": compute_ssim(original, restored_wiener), "MSE_degraded": compute_mse(original, degraded),
               "MSE_inverse": compute_mse(original, restored_inv), "MSE_wiener": compute_mse(original, restored_wiener)}

    return metrics
