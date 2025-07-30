import numpy as np
from numpy.fft import fft2, ifft2


def inverse_filter(
        degraded: np.ndarray,
        psf: np.ndarray,
        eps: float = 1e-6
) -> np.ndarray:

    M, N = degraded.shape
    pad_psf = np.zeros_like(degraded)
    psf_h, psf_w = psf.shape
    pad_psf[:psf_h, :psf_w] = psf
    pad_psf = np.roll(pad_psf, -psf_h // 2, axis=0)
    pad_psf = np.roll(pad_psf, -psf_w // 2, axis=1)

    G = fft2(degraded)
    H = fft2(pad_psf)

    H_mag = np.abs(H)
    H_safe = H.copy()
    H_safe[H_mag < eps] = eps

    F_est = G / H_safe

    f_restored = np.real(ifft2(F_est))
    f_restored = np.clip(f_restored, 0, 1)
    return f_restored


def wiener_filter(degraded: np.ndarray, psf: np.ndarray, K: float) -> np.ndarray:


    M, N = degraded.shape
    psf_h, psf_w = psf.shape

    pad_psf = np.zeros_like(degraded, dtype=np.float64)
    pad_psf[:psf_h, :psf_w] = psf

    pad_psf = np.roll(pad_psf, -psf_h // 2, axis=0)
    pad_psf = np.roll(pad_psf, -psf_w // 2, axis=1)

    G = fft2(degraded)
    H = fft2(pad_psf)
    H_conj = np.conj(H)
    H_mag2 = np.abs(H) ** 2

    W = H_conj / (H_mag2 + K)

    F_est = W * G
    f_restored = np.real(ifft2(F_est))

    f_restored = np.clip(f_restored, 0.0, 1.0)
    return f_restored