import json
import os

from config import (
    BASE_DIR,
    SOURCE_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    PSF_TYPE,
    PSF_SIZE,
    BLUR_PARAM,
    NOISE_MEAN,
    NOISE_VARIANCE,
    WIENER_K,
)
from src.visualization import plot_metrics
from utils import load_image, create_psf, degrade_image, save_image
from filters import inverse_filter, wiener_filter
from metrics import evaluate_metrics
from visualization import plot_comparison
def process_image(filename: str) -> None:
    """
    For a single image:
      1) Load original.
      2) Generate PSF.
      3) Create degraded image (blur + noise).
      4) Apply Inverse filter.
      5) Apply Wiener filter.
      6) Compute metrics.
      7) Save results and metrics.
      8) Plot comparisons.
    """

    # 1) Load original
    orig_path = os.path.join(SOURCE_DIR, filename)
    original = load_image(orig_path)

    # 2) Create PSF
    psf = create_psf(PSF_TYPE, PSF_SIZE, BLUR_PARAM)

    # 3) Degrade
    degraded = degrade_image(original, psf, NOISE_MEAN, NOISE_VARIANCE)

    # Save degraded image
    degraded_filename = os.path.join(OUTPUT_DIR, f"degraded_{filename}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_image(degraded_filename, degraded)

    # 4) Inverse filtering
    restored_inv = inverse_filter(degraded, psf)

    # 5) Wiener filtering
    restored_wiener = wiener_filter(degraded, psf, WIENER_K)

    # 6) Compute metrics
    results = evaluate_metrics(original, degraded, restored_inv, restored_wiener)
    print(f"Metrics for {filename}:")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")

        DIR = '/Users/raya/Desktop/motion-deblur-filtering/reports/metrics'
        output_path = os.path.join(DIR, f"results_{filename}.json")

        with open(output_path, 'w') as f:
            json.dump(results, f,indent=4)




    # 7) Save comparison plot
    figure_path = os.path.join(RESULTS_DIR, f"comparison_{os.path.splitext(filename)[0]}.png")
    plot_comparison(original, degraded, restored_inv, restored_wiener, save_path=figure_path)

def main():
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Process each image in SOURCE_DIR
    for fname in os.listdir(SOURCE_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Processing {fname} ...")
            process_image(fname)



if __name__ == "__main__":
    main()
