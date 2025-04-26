

import os
import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
import shutil

#Convert To coco....

csv_path = "C:/VLNLP/Test/D7K/scene_objects_balanced_7k.csv"
mask_dir = "C:/VLNLP/Test/D7K/binary_masks_kmeans"
image_dir = "C:/VLNLP/Test/D7K"
output_json_path = "C:/VLNLP/Test/D7K/COCO/annotations/instances_train.json"
output_image_dir = "C:/VLNLP/Test/D7K/COCO/images"
output_mask_dir = "C:/VLNLP/Test/D7K/COCO/masks"

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# === MAP SHAPE TO CLASS INDEX ===
shape_to_class = {
    "sphere": 1,
    "cone": 2,
    "cylinder": 3,
    "torus": 4,
    "cube": 5  # 0 is background
}


coco_output = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "sphere"},
        {"id": 2, "name": "cone"},
        {"id": 3, "name": "cylinder"},
        {"id": 4, "name": "torus"},
        {"id": 5, "name": "cube"}
    ]
}

# === LOAD CSV ===
df = pd.read_csv(csv_path)
grouped = df.groupby("scene_id")
annotation_id = 1
image_id = 1



for scene_id, group in tqdm(grouped):
    scene_image_name = f"{scene_id}_angle_0.png"
    scene_image_path = os.path.join(image_dir, scene_image_name)
    if not os.path.exists(scene_image_path):
        continue

    img = np.array(Image.open(scene_image_path))
    height, width = img.shape[:2]

    shutil.copy(scene_image_path, os.path.join(output_image_dir, scene_image_name))

    coco_output['images'].append({
        "id": image_id,
        "file_name": scene_image_name,
        "height": height,
        "width": width
    })

    for _, row in group.iterrows():
        shape = str(row['shape']).strip().lower()
        class_id = shape_to_class.get(shape, 0)
        object_id = row['object_id']

        mask_name = f"{scene_id}_mask_{object_id}_angle_0_binary.png"
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue

        shutil.copy(mask_path, os.path.join(output_mask_dir, mask_name))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        bin_mask = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.shape[0] < 3:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                print(f" Invalid box in {mask_name}: {[x, y, w, h]}")
                continue

            segmentation = contour.flatten().tolist()
            if len(segmentation) < 6:
                continue  # Must be at least one triangle

            area = float(cv2.contourArea(contour))
            coco_output['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x, y, x+w, y+h],
                "area": area,
                "segmentation": [segmentation],
                "iscrowd": 0
            })
            annotation_id += 1

    image_id += 1


with open(output_json_path, 'w') as f:
    json.dump(coco_output, f)


