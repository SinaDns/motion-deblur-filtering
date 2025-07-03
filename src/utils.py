import cv2
import numpy as np
from scipy.signal import convolve2d

def load_image(filepath: str) -> np.ndarray:
    """
    Read an image from disk and convert to float64 in [0,1].
    - Works for uint8, uint16, float32/float64, or color images.
    - If color, it first converts to grayscale.
    """
    # Let cv2 load whatever the format is (uint8, uint16, or float)
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load {filepath}")

    # If this is a color image, convert to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float64 in [0,1]
    dtype = img.dtype
    if np.issubdtype(dtype, np.integer):
        # Integer types: scale by their max (e.g. 255 for uint8, 65535 for uint16)
        info = np.iinfo(dtype)
        max_val = info.max if info.max > 0 else 1
        img_float = img.astype(np.float64) / float(max_val)
    else:
        # Float types: linearly rescale min→0 and max→1
        img_f = img.astype(np.float64)
        min_val = np.nanmin(img_f)
        max_val = np.nanmax(img_f)
        diff = max_val - min_val
        if diff < 1e-8:
            # If nearly constant, subtract min and leave as is
            img_float = img_f - min_val
        else:
            img_float = (img_f - min_val) / diff

    # Final clip & replace any NaN/inf
    img_float = np.nan_to_num(img_float, nan=0.0, posinf=1.0, neginf=0.0)
    img_float = np.clip(img_float, 0.0, 1.0)
    return img_float

def save_image(filepath: str, image: np.ndarray) -> None:
    """
    Convert a float64 image in [0,1] to uint8 and save it.
    Clips + rounds to avoid invalid casts.
    """
    img_f = image.astype(np.float64)
    img_f = np.nan_to_num(img_f, nan=0.0, posinf=1.0, neginf=0.0)
    img_clipped = np.clip(img_f, 0.0, 1.0)
    img_scaled = img_clipped * 255.0
    img_u8 = np.round(img_scaled).astype(np.uint8)
    cv2.imwrite(filepath, img_u8)

def create_psf(psf_type: str, size: int, param: float) -> np.ndarray:
    """
    Generate a normalized PSF kernel (float64), sum = 1, with no chance of zero-sum.
    - psf_type: "motion" (simple horizontal line) or "gaussian"
    - size: odd integer ≥ 1
    - param: for "gaussian", sigma; for "motion", ignored
    """
    if size < 1 or (size % 2 == 0):
        raise ValueError("PSF size must be a positive odd integer.")

    if psf_type.lower() == "motion":
        # Simple horizontal motion blur
        psf = np.zeros((size, size), dtype=np.float64)
        center = size // 2
        psf[center, :] = 1.0
        s = psf.sum()
        if s <= 0:
            # Should never happen if we set one entire row to 1, but just in case:
            raise RuntimeError("Motion PSF sum is zero, check 'size' parameter.")
        psf /= s
        return psf

    elif psf_type.lower() == "gaussian":
        # 2D Gaussian with sigma = param
        sigma = float(param)
        if sigma <= 0:
            raise ValueError("Gaussian sigma must be positive.")
        ax = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float64)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        s = kernel.sum()
        if s <= 0:
            raise RuntimeError("Gaussian PSF sum is zero, check 'size' or 'sigma'.")
        kernel /= s
        return kernel

    else:
        raise ValueError(f"Unsupported PSF type: '{psf_type}'")

def degrade_image(
        original: np.ndarray,
        psf: np.ndarray,
        noise_mean: float,
        noise_var: float
) -> np.ndarray:
    """
    Convolve `original` (float64 [0,1]) with PSF and add Gaussian noise.
    Returns a float64 image in [0,1]. Any NaN/inf get clipped.
    """
    # 1) Convolution (blur)
    blurred = convolve2d(original, psf, mode="same", boundary="wrap")

    # 2) Add Gaussian noise (variance no less than 0)
    std = np.sqrt(max(0.0, float(noise_var)))
    noise = np.random.normal(loc=float(noise_mean), scale=std, size=original.shape)
    degraded = blurred #+ noise

    # 3) Replace any NaN/inf then clip to [0,1]
    degraded = np.nan_to_num(degraded, nan=0.0, posinf=1.0, neginf=0.0)
    degraded = np.clip(degraded, 0.0, 1.0)
    return degraded
