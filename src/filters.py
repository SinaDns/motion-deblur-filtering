import numpy as np

def inverse_filter(blurred, kernel, eps=1e-3): ##### TRY larger eps, like eps=1e-1 if it was unstable
    """
    Restore a blurred image using inverse filtering in the frequency domain.
    """

    kh, kw = kernel.shape
    ih, iw = blurred.shape
    pad = ((0, ih - kh), (0, iw - kw))
    kernel_padded = np.pad(kernel, pad, 'constant')

    dft_blurred = np.fft.fft2(blurred)
    dft_kernel = np.fft.fft2(kernel_padded)

    dft_kernel[np.abs(dft_kernel) < eps] = eps

    dft_restored = dft_blurred / dft_kernel
    restored = np.fft.ifft2(dft_restored)
    restored = np.abs(restored)

    restored = np.clip(restored, 0, 255).astype('uint8')
    return restored


def wiener_filter():
   #TODO