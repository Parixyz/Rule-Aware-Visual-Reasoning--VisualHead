import os
import cv2
import numpy as np

# === Paths ===
mask_dir = "C://VLNLP//Test//D7K//binary_masks_kmeans"
actual_dir = "C://VLNLP//Test//D7K"

# === Parameters ===
threshold_ratio = 0.5  # Max allowed white pixel ratio

# === Function to apply adaptive thresholding ===
def fix_mask_from_actual(actual_img):
    for t in range(128, 256, 10):
        _, binary = cv2.threshold(actual_img, t, 255, cv2.THRESH_BINARY)
        white_ratio = np.mean(binary == 255)
        if white_ratio <= threshold_ratio:
            return binary
    return binary  # Return the most aggressive one if all fail

# === Processing ===
fixed_count = 0

for fname in os.listdir(mask_dir):
    if "_mask" in fname and fname.endswith("_binary.png"):
        mask_path = os.path.join(mask_dir, fname)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Corresponding actual image path (without "_binary")
        actual_fname = fname.replace("_binary", "")
        actual_path = os.path.join(actual_dir, actual_fname)
        actual_img = cv2.imread(actual_path, cv2.IMREAD_GRAYSCALE)

        if actual_img is None:
            print(f" Actual image not found for: {fname}")
            continue

        # Check mask for faults
        unique_vals = np.unique(mask_img)
        white_ratio = np.mean(mask_img == 255)

        if not set(unique_vals).issubset({0, 255}) or white_ratio > threshold_ratio:
            print(f" Fixing: {fname} | White ratio: {white_ratio:.2f} | Unique: {unique_vals}")
            fixed_mask = fix_mask_from_actual(actual_img)
            cv2.imwrite(mask_path, fixed_mask)  # Overwrite the original binary mask
            fixed_count += 1

print(f" Replaced {fixed_count} faulty binary masks in-place.")
