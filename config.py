# config.py

# Paths
SOURCE_DIR = "data/original_images"
OUTPUT_DIR = "data/degraded_images"
RESULTS_DIR = "reports/figures"

# Degradation parameters
PSF_SIZE = 21            # e.g. kernel size for blur
PSF_TYPE = "motion"      # options: "motion", "gaussian", etc.
BLUR_PARAM = 0.1         # e.g. motion angle or gaussian sigma

NOISE_MEAN = 0.0
NOISE_VARIANCE = 0.001   # variance of additive Gaussian noise

# Wiener filter parameters
# If you want to assume a constant noise/signal power ratio, define K:
WIENER_K = 0.01

# Metric computation flags
COMPUTE_PSNR = True
COMPUTE_SSIM = True
