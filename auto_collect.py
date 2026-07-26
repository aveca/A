"""
Batch Data Collector — Capture automatique de frames uniques
Utilise la caméra pour collecter des données d'entraînement
"""
import cv2
import os
import sys
import time
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

def augment_image(img):
    augmented = []
    h, w = img.shape[:2]
    augmented.append(cv2.flip(img, 1))
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augmented.append(cv2.rotate(img, cv2.ROTATE_180))
    M = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
    augmented.append(cv2.warpAffine(img, M, (w, h)))
    brightness = np.clip(img.astype(float) * 1.3, 0, 255).astype(np.uint8)
    augmented.append(brightness)
    darkness = np.clip(img.astype(float) * 0.7, 0, 255).astype(np.uint8)
    augmented.append(darkness)
    kernel = np.ones((3, 3), np.uint8)
    augmented.append(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel))
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)
    return augmented


def auto_collect(num_frames=100, interval=0.5):
    print("Capture automatique de données d'entraînement")
    print(f"Nombre de frames: {num_frames}")
    print(f"Intervalle: {interval}s")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Caméra non disponible")
        return

    detector = YOLO('yolov8m.pt')
    seen_classes = {}
    total_captured = 0

    print("Détecte les objets et capture automatiquement...")
    print("Ctrl+C pour arrêter\n")

    try:
        frame_count = 0
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector(frame, conf=0.4, verbose=False, device='cuda:0')

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    cls_name = r.names[cls_id]
                    conf = float(box.conf)
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]

                    if cls_name not in seen_classes:
                        seen_classes[cls_name] = 0
                    seen_classes[cls_name] += 1

                    if seen_classes[cls_name] % 5 == 0:
                        crop = frame[max(0,y1):min(frame.shape[0],y2),
                                     max(0,x1):min(frame.shape[1],x2)]
                        if crop.size < 100:
                            continue

                        label_dir = os.path.join(DATASET_DIR, cls_name.replace(' ', '_'))
                        os.makedirs(label_dir, exist_ok=True)
                        count = len([f for f in os.listdir(label_dir) if f.endswith('.jpg')]) + 1

                        cv2.imwrite(os.path.join(label_dir, f"{count:04d}.jpg"), crop)
                        augmented = augment_image(crop)
                        for i, aug in enumerate(augmented):
                            cv2.imwrite(os.path.join(label_dir, f"{count:04d}_aug{i}.jpg"), aug)
                        total_captured += 1 + len(augmented)

            cv2.putText(frame, f"Frames: {frame_count}/{num_frames} | Captures: {total_captured}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Classes: {seen_classes}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Auto Collector', frame)

            if cv2.waitKey(int(interval * 1000)) & 0xFF == ord('q'):
                break
            frame_count += 1

    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*40}")
    print(f"RÉSUMÉ:")
    print(f"  Frames analysées: {frame_count}")
    print(f"  Total captures: {total_captured}")
    print(f"  Classes détectées: {seen_classes}")
    print(f"  Dataset: {DATASET_DIR}")

    for cls, count in seen_classes.items():
        print(f"\n  {cls}: {count} captures sauvegardées")


if __name__ == '__main__':
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    auto_collect(num_frames=num)
