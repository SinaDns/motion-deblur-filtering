import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

json_path = Path(__file__).parent.parent / "reports" / "metrics" / "results_0.jpg.json"

with open(json_path, 'r') as f:
    data = json.load(f)


methods = ['Degraded', 'Inverse', 'Wiener']
psnr = [data[f'PSNR_{m.lower()}'] for m in methods]
ssim = [data[f'SSIM_{m.lower()}'] for m in methods]
mse  = [data[f'MSE_{m.lower()}']  for m in methods]

# --- Plot ---
plt.figure(figsize=(7,5))

plt.plot(methods, psnr, marker='o', label='PSNR')
plt.plot(methods, ssim, marker='s', label='SSIM')
plt.plot(methods, mse, marker='^', label='MSE')

plt.title("Metrics Comparison")
plt.xlabel("Methods")
plt.ylabel("Metric Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()