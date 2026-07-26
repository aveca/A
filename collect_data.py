"""
Data Collector — Capture des frames via caméra pour enrichir le dataset
Lance ce script, et le système capture automatiquement les frames uniques.
"""
import cv2
import os
import time
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

def augment_image(img):
    """Génère 8 variations d'une image"""
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


def collect_from_webcam(label=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Caméra non disponible")
        return

    if label is None:
        label = input("Nom de l'objet à capturer: ").strip()
    if not label:
        print("Label invalide")
        return

    label_dir = os.path.join(DATASET_DIR, label.replace(' ', '_'))
    os.makedirs(label_dir, exist_ok=True)
    count = len([f for f in os.listdir(label_dir) if f.endswith('.jpg')])

    print(f"Capture '{label}' — Appuie sur ESPACE pour capturer, Q pour quitter")
    print(f"Images existantes: {count}")

    last_capture_time = 0
    MIN_INTERVAL = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Objet: {label} | Captures: {count}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "ESPACE=capturer | Q=quitter", (10, 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow('Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            now = time.time()
            if now - last_capture_time < MIN_INTERVAL:
                print("  Attends 1s entre les captures")
                continue
            last_capture_time = now

            count += 1
            filepath = os.path.join(label_dir, f"{count:04d}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"  Capturé: {filepath}")

            augmented = augment_image(frame)
            for i, aug in enumerate(augmented):
                count += 1
                aug_path = os.path.join(label_dir, f"{count:04d}_aug{i}.jpg")
                cv2.imwrite(aug_path, aug)
            print(f"  + {len(augmented)} augmentations ({count} total)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal: {count} images pour '{label}'")


if __name__ == '__main__':
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else None
    collect_from_webcam(label)
