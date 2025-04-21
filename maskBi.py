import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# === Configuration ===
image_dir = "C://VLNLP//Test//D7K"
save_dir = os.path.join(image_dir, "binary_masks_kmeans")
#os.makedirs(save_dir, exist_ok=True)
start_file = "scene_16981_mask_0_angle_0.png"

# List all relevant files
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") and "_mask_" in f]

# Sort the filenames (important for ordering)
image_files_sorted = sorted(image_files)
start_index = image_files_sorted.index(start_file)
image_files = image_files[start_index:]
# === Process and save binary masks ===
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    img = np.array(Image.open(img_path).convert("L"))
    pixels = img.reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_.reshape(img.shape)

    cluster_centers = kmeans.cluster_centers_.flatten()
    foreground_cluster = np.argmax(cluster_centers)

    binary_mask = np.uint8((labels == foreground_cluster) * 255)
    binary_mask = cv2.medianBlur(binary_mask, 3)

    save_path = os.path.join(save_dir, img_file.replace(".png", "_binary.png"))
    cv2.imwrite(save_path, binary_mask)

# Return total number of masks saved
len(os.listdir(save_dir))
