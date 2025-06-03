import kagglehub
import shutil
import os

dataset_path = kagglehub.dataset_download("orvile/brain-cancer-mri-dataset")

target_dir = "data"

os.makedirs(target_dir, exist_ok=True)

shutil.copytree(dataset_path, target_dir, dirs_exist_ok=True)

print(f"Dataset downloaded and copied to {target_dir}")