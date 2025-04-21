import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image
from collections import defaultdict
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.models.detection import maskrcnn_resnet50_fpn

# === CONFIG ===
data_root = "C:/VLNLP/Test/D7K/COCO"
image_dir = os.path.join(data_root, "images")
ann_path = os.path.join(data_root, "annotations/instances_train.json")
mask_model_path = os.path.join(data_root, "mask_rcnn_model.pth")
mat_model_path = os.path.join(data_root, "material_classifier.pth")
IMG_SIZE = 64
NUM_CLASSES = 6
MATERIAL_CLASSES = ["opaque", "transparent", "transparent_blue", "mirror", "gold"]
SHAPE_CLASSES = {
    1: "sphere",
    2: "cone",
    3: "cylinder",
    4: "torus",
    5: "cube"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load COCO-style annotations ===
with open(ann_path, 'r') as f:
    coco = json.load(f)
image_map = {img["id"]: img["file_name"] for img in coco["images"]}
scene_ids = list(image_map.keys())[:10]

# === Load Mask R-CNN ===
mask_model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
mask_model.load_state_dict(torch.load(mask_model_path))
mask_model.to(device).eval()

# === Material Classifier ===
class MaterialNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * IMG_SIZE * IMG_SIZE, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(MATERIAL_CLASSES))
        )
    def forward(self, x):
        return self.net(x)

mat_model = MaterialNet().to(device)
mat_model.load_state_dict(torch.load(mat_model_path))
mat_model.eval()

transform = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])

# === Run Visualization ===
for image_id in scene_ids:
    file_name = image_map[image_id]
    img_path = os.path.join(image_dir, file_name)
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = mask_model(img_tensor)[0]

    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    masks = output['masks'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    keep = scores > 0.5

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(np.array(img_pil))

    # For summary
    summary = defaultdict(int)

    for i, box in enumerate(boxes[keep]):
        label_id = labels[keep][i]
        shape_label = SHAPE_CLASSES.get(label_id, "unknown")
        x1, y1, x2, y2 = box.astype(int)

        # Crop + Predict material
        crop = img_pil.crop((x1, y1, x2, y2))
        crop_tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mat = mat_model(crop_tensor)
            mat_idx = pred_mat.argmax(dim=1).item()
            material_label = MATERIAL_CLASSES[mat_idx]

        # Combine shape + material
        tag = f"{material_label} {shape_label}"
        summary[tag] += 1

        # Draw box + label
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='lime', facecolor='none', lw=2))
        ax.text(x1, y1 - 5, tag, color='black', fontsize=8, backgroundcolor='white')

        # Draw contour
        mask_bin = (masks[keep][i, 0] > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = cnt.squeeze(1)
            ax.plot(cnt[:, 0], cnt[:, 1], color='red', linewidth=1.5)

    # Add summary list below the figure
    lines = [f"{v} × {k}" for k, v in sorted(summary.items())]
    fig.suptitle(f"Scene: {file_name}\n" + "\n".join(lines), fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(data_root, f"{file_name.replace('.png', '_scene_summary.png')}"))
    plt.show()
