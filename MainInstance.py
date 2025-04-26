

import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2

# === CONFIG ===
data_root = "C:/VLNLP/Test/D7K/COCO"
image_dir = os.path.join(data_root, "images")
ann_path = os.path.join(data_root, "annotations/instances_train.json")
save_path = os.path.join(data_root, "mask_rcnn_model.pth")
NUM_CLASSES = 6  # 5 shapes + background
BATCH_SIZE = 2
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORM ===
def get_transform():
    return T.Compose([
        T.ToTensor()
    ])

# === DATASET ===
class CocoInstanceDataset(Dataset):
    def __init__(self, image_dir, ann_path, transforms=None):
        with open(ann_path, 'r') as f:
            self.coco = json.load(f)
        self.image_dir = image_dir
        self.transforms = transforms
        self.imgs = self.coco['images']
        self.anns = self.coco['annotations']
        self.img_id_to_anns = {}
        for ann in self.anns:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        ann_list = self.img_id_to_anns.get(img_info['id'], [])
        masks, boxes, labels = [], [], []

        for ann in ann_list:
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for seg in ann['segmentation']:
                contour = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [contour], color=1)
            masks.append(mask)
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(np.stack(masks), dtype=torch.uint8),
            'image_id': torch.tensor([img_info['id']])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

# === MODEL ===
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(256, 256, NUM_CLASSES)
model = model.to(device)

# === TRAINING ===
dataset = CocoInstanceDataset(image_dir, ann_path, transforms=get_transform())
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_loss = float('inf')
early_stopping_counter = 0
losses_per_epoch = []

print(" Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, targets in loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    avg_loss = total_loss / len(loader)
    losses_per_epoch.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # === Save best checkpoint ===
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f" Model saved at epoch {epoch+1} with loss {avg_loss:.4f}")
    else:
        early_stopping_counter += 1
        print(f" No improvement. Early stop counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(" Early stopping triggered.")
            break

    # === Show example after each epoch ===
    model.eval()
    with torch.no_grad():
        img, target = dataset[0]
        pred = model([img.to(device)])[0]
        img_np = img.permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np)
        for i in range(len(pred['masks'])):
            mask = pred['masks'][i, 0].cpu().numpy()
            if pred['scores'][i] > 0.5:
                ax.contour(mask, colors='r', linewidths=2)
        ax.set_title(f"Epoch {epoch+1} Prediction Example")
        plt.axis('off')
        plt.show()

# === LOSS CURVE ===
plt.figure(figsize=(8, 4))
plt.plot(losses_per_epoch, marker='o', label="Train Loss")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


model.load_state_dict(torch.load(save_path))
model.eval()
with torch.no_grad():
    img, target = dataset[0]
    pred = model([img.to(device)])[0]
    img_np = img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)
    for i in range(len(pred['masks'])):
        mask = pred['masks'][i, 0].cpu().numpy()
        if pred['scores'][i] > 0.5:
            ax.contour(mask, colors='r', linewidths=2)
    ax.set_title("Mask R-CNN Final Predictions")
    plt.axis('off')
    plt.show()
