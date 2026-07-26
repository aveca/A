"""
Fine-tuning YOLOv8x sur le dataset collecté
Utilise la RTX 4060 Ti pour un entraînement optimisé
"""
import os
import sys
import json
import time
import shutil
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')
TRAIN_DIR = os.path.join(PROJECT_DIR, 'train')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'training_results')
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  Fine-tuning YOLOv8x — RTX 4060 Ti")
print("=" * 60)

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

classes = []
class_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
if not class_dirs:
    print("\n❌ Aucun dossier d'images trouvé dans dataset/")
    print("   Lance d'abord: python collect_data.py nom_objet")
    sys.exit(1)

print(f"\nClasses trouvées: {len(class_dirs)}")
for cls in sorted(class_dirs):
    cls_path = os.path.join(DATASET_DIR, cls)
    imgs = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"  - {cls}: {len(imgs)} images")
    classes.append(cls)

yaml_content = f"""path: {os.path.abspath(PROJECT_DIR)}
train: train/images
val: train/images

nc: {len(classes)}
names: {classes}
"""

yaml_path = os.path.join(PROJECT_DIR, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
print(f"\nFichier YAML créé: {yaml_path}")

train_images = os.path.join(TRAIN_DIR, 'images')
train_labels = os.path.join(TRAIN_DIR, 'labels')
os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)

print("\nPréparation du dataset pour YOLO...")
image_count = 0
for cls_idx, cls_name in enumerate(classes):
    cls_dir = os.path.join(DATASET_DIR, cls_name)
    imgs = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for img_name in imgs:
        src_img = os.path.join(cls_dir, img_name)
        dst_img = os.path.join(train_images, f"{cls_name}_{img_name}")
        shutil.copy2(src_img, dst_img)

        img = cv2.imread(src_img)
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = os.path.join(train_labels, f"{cls_name}_{os.path.splitext(img_name)[0]}.txt")
        with open(label_path, 'w') as lf:
            lf.write(f"{cls_idx} 0.5 0.5 0.9 0.9\n")
        image_count += 1

import cv2
print(f"Images préparées: {image_count}")

from ultralytics import YOLO

print("\n" + "=" * 60)
print("  DÉMARRAGE DU TRAINING")
print("  Model: YOLOv8x (extra large)")
print("  GPU: RTX 4060 Ti")
print("  Epochs: 50")
print("=" * 60)

model = YOLO('yolov8x.pt')
model.to('cuda:0')

results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    device='cuda:0',
    project=RESULTS_DIR,
    name=f"finetune_{datetime.now().strftime('%Y%m%d_%H%M')}",
    patience=15,
    save=True,
    save_period=10,
    plots=True,
    verbose=True,
)

print("\n" + "=" * 60)
print("  TRAINING TERMINÉ!")
print("=" * 60)

best_model = os.path.join(RESULTS_DIR, results.save_dir, 'weights', 'best.pt')
final_model = os.path.join(PROJECT_DIR, 'best_model.pt')

if os.path.exists(best_model):
    shutil.copy2(best_model, final_model)
    print(f"\nMeilleur modèle sauvegardé: {final_model}")
    print("Utilise ce modèle dans app.py en remplaçant 'yolov8m.pt' par 'best_model.pt'")

print("\nMétriques finales:")
if hasattr(results, 'box'):
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")

print(f"\nRésultats complets dans: {RESULTS_DIR}")
print("Pour utiliser le modèle amélioré, modifie app.py:")
print(f"  yolo = YOLO('{final_model}')")
