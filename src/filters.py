import numpy as np
from numpy.fft import fft2, ifft2


def inverse_filter(
        degraded: np.ndarray,
        psf: np.ndarray,
        eps: float = 1e-6
) -> np.ndarray:
    """
    Apply inverse filtering in the frequency domain.
    - degraded: degraded image (float), shape (M,N)
    - psf: same-size (or smaller) PSF kernel
    - eps: small constant to avoid division by zero
    Returns restored image (float).
    """
    # 1) Zero-pad PSF to degraded image size
    M, N = degraded.shape
    pad_psf = np.zeros_like(degraded)
    psf_h, psf_w = psf.shape
    pad_psf[:psf_h, :psf_w] = psf
    # Optionally, center the PSF
    pad_psf = np.roll(pad_psf, -psf_h // 2, axis=0)
    pad_psf = np.roll(pad_psf, -psf_w // 2, axis=1)

    # 2) Compute transforms
    G = fft2(degraded)
    H = fft2(pad_psf)

    # 3) Avoid division by very small |H|
    H_mag = np.abs(H)
    H_safe = H.copy()
    H_safe[H_mag < eps] = eps

    # 4) Inverse filter: F_est = G / H_safe
    F_est = G / H_safe

    # 5) Inverse FFT and return real part
    f_restored = np.real(ifft2(F_est))
    # Clip to [0,1]
    f_restored = np.clip(f_restored, 0, 1)
    return f_restored


def wiener_filter(degraded: np.ndarray, psf: np.ndarray, K: float) -> np.ndarray:
    """
    Apply Wiener filtering in the frequency domain.

    Parameters:
    -----------
    degraded : np.ndarray
        2D array (float in [0,1]) of the degraded (blurred + noisy) image.
    psf : np.ndarray
        2D point-spread function (kernel) array (float), smaller than or equal to degraded shape.
    K : float
        Noise-to-signal power ratio (can be a scalar or a 2D array of same shape as degraded).

    Returns:
    --------
    np.ndarray
        Restored image (float), clipped to [0,1].
    """

    # 1) Determine sizes
    M, N = degraded.shape
    psf_h, psf_w = psf.shape

    # 2) Zero-pad PSF to the same size as degraded image
    pad_psf = np.zeros_like(degraded, dtype=np.float64)
    pad_psf[:psf_h, :psf_w] = psf

    # 3) Center the PSF (so its anchor is at the (0,0) frequency)
    pad_psf = np.roll(pad_psf, -psf_h // 2, axis=0)
    pad_psf = np.roll(pad_psf, -psf_w // 2, axis=1)

    # 4) Compute FFTs
    G = fft2(degraded)        # FFT of degraded image
    H = fft2(pad_psf)         # FFT of padded, centered PSF
    H_conj = np.conj(H)
    H_mag2 = np.abs(H) ** 2

    # 5) Build Wiener filter transfer function:
    #    W(u,v) = H*(u,v) / (|H(u,v)|^2 + K)
    #    If K is scalar, broadcasting will apply automatically.
    W = H_conj / (H_mag2 + K)

    # 6) Apply filter in frequency domain and inverse FFT
    F_est = W * G
    f_restored = np.real(ifft2(F_est))

    # 7) Clip result to [0,1] and return
    f_restored = np.clip(f_restored, 0.0, 1.0)
    return f_restored