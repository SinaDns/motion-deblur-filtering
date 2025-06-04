# src/utils.py

import cv2
import numpy as np
from scipy.signal import convolve2d

def img_as_float(img_uint8: np.ndarray) -> np.ndarray:
    # Assumes img_uint8 is dtype=np.uint8, range [0,255]
    return img_uint8.astype(np.float64) / 255.0

def img_as_ubyte(img_float: np.ndarray) -> np.ndarray:
    # Assumes img_float is float in [0,1]
    return (np.clip(img_float, 0.0, 1.0) * 255.0).round().astype(np.uint8)
def load_image(filepath: str) -> np.ndarray:
    """
    Read an image from disk and convert to float64 range [0,1].
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load {filepath}")
    return img_as_float(img)

def save_image(filepath: str, image: np.ndarray) -> None:
    """
    Convert to uint8 and write to disk.
    """
    img_uint8 = img_as_ubyte(np.clip(image, 0, 1))
    cv2.imwrite(filepath, img_uint8)

def create_psf(psf_type: str, size: int, param: float) -> np.ndarray:
    """
    Generate a point-spread function (PSF) kernel.
    - psf_type: "motion" or "gaussian"
    - size: kernel size (odd integer)
    - param: angle (if motion) or sigma (if gaussian)
    Returns a 2D float64 array of shape (size, size).
    """
    if psf_type.lower() == "motion":
        # Example: simple horizontal motion blur
        psf = np.zeros((size, size), dtype=np.float64)
        center = size // 2
        # Fill a tilted line at angle 'param' degrees (optional implementation detail)
        # ... (implement motion line generation here)
        # Normalize so sum(psf) = 1
        psf /= psf.sum()
        return psf

    elif psf_type.lower() == "gaussian":
        # generate a Gaussian kernel
        ax = np.arange(-size // 2 + 1, size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        sigma = param
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel

    else:
        raise ValueError(f"Unsupported PSF type: {psf_type}")

def degrade_image(
        original: np.ndarray,
        psf: np.ndarray,
        noise_mean: float,
        noise_var: float
) -> np.ndarray:
    """
    Convolve `original` with PSF and add Gaussian noise.
    Returns degraded image as float in [0,1].
    """
    # 1) Blur via 2D convolution (use 'same' mode)
    blurred = convolve2d(original, psf, mode="same", boundary="wrap")
    # 2) Additive Gaussian noise
    noise = np.random.normal(loc=noise_mean, scale=np.sqrt(noise_var), size=original.shape)
    degraded = blurred + noise
    # Clip to [0,1]
    degraded = np.clip(degraded, 0, 1)
    return degraded
