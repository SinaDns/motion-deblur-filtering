import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, 'data/original_images')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'data/degraded_images')
RESULTS_DIR = os.path.join(BASE_DIR, 'reports/figures')


# Degradation parameters
PSF_SIZE = 21            # e.g. kernel size for blur
PSF_TYPE = "motion"      # options: "motion", "gaussian", etc.
BLUR_PARAM = 0.5         # e.g. motion angle or gaussian sigma

NOISE_MEAN = 0.0
NOISE_VARIANCE = 0.001   # variance of additive Gaussian noise


WIENER_K = 0.01

# Metric computation flags
COMPUTE_PSNR = True
COMPUTE_SSIM = True
