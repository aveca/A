import os
import cv2
import numpy as np
import base64
import json
import time
import threading
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

print("=" * 60)
print("  Object Camera Recognition — Full GPU Pipeline")
print("  YOLOv8m + ByteTrack + Florence-2 VLM + TensorRT")
print("=" * 60)

print("\n[1/3] Chargement YOLOv8m + tracking...")
yolo = YOLO('yolov8m.pt')
print(f"  YOLO prêt. Device: {yolo.device}")

print("\n[2/3] Chargement Florence-2 VLM...")
vlm = None
vlm_processor = None
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    vlm_name = "microsoft/Florence-2-base"
    vlm_processor = AutoProcessor.from_pretrained(vlm_name, trust_remote_code=True)
    vlm = AutoModelForCausalLM.from_pretrained(
        vlm_name, trust_remote_code=True, torch_dtype="auto"
    ).to("cuda:0")
    print("  Florence-2 prêt sur GPU")
except Exception as e:
    print(f"  ⚠ Florence-2 non disponible: {e}")
    print("  → Descriptions basiques uniquement")

print("\n[3/3] Vérification TensorRT...")
TRT_AVAILABLE = False
try:
    import tensorrt
    TRT_AVAILABLE = True
    print(f"  TensorRT {tensorrt.__version__} disponible")
except:
    print("  TensorRT non importable (non critique)")

COCO_CLASSES_FR = {
    0: ('personne', 'humain'), 1: ('vélo', 'véhicule à deux roues'),
    2: ('voiture', 'véhicule motorisé'), 3: ('moto', 'véhicule motorisé'),
    4: ('avion', 'aéronef'), 5: ('bus', 'transport en commun'),
    6: ('train', 'transport ferroviaire'), 7: ('camion', 'véhicule de transport'),
    8: ('bateau', 'navire'), 9: ('feu tricolore', 'signal routier'),
    10: ("bouche d'incendie", 'sécurité'), 11: ('panneau stop', 'signalisation'),
    12: ('parcmètre', 'stationnement'), 13: ('banc', 'siège public'),
    14: ('oiseau', 'animal'), 15: ('chat', 'animal'),
    16: ('chien', 'animal'), 17: ('cheval', 'animal'),
    18: ('mouton', 'animal'), 19: ('vache', 'animal'),
    20: ('éléphant', 'animal'), 21: ('ours', 'animal'),
    22: ('zèbre', 'animal'), 23: ('girafe', 'animal'),
    24: ('sac à dos', 'accessoire'), 25: ('parapluie', 'accessoire'),
    26: ('sac à main', 'accessoire'), 27: ('cravate', 'vêtement'),
    28: ('valise', 'bagage'), 29: ('frisbee', 'sport'),
    30: ('skis', 'sport'), 31: ('snowboard', 'sport'),
    32: ('ballon de sport', 'sport'), 33: ('cerf-volant', 'jouet'),
    34: ('batte de baseball', 'sport'), 35: ('gant de baseball', 'sport'),
    36: ('skateboard', 'sport'), 37: ('surfboard', 'sport'),
    38: ('raquette de tennis', 'sport'), 39: ('bouteille', 'cuisine'),
    40: ('verre à vin', 'cuisine'), 41: ('tasse', 'cuisine'),
    42: ('fourchette', 'cuisine'), 43: ('couteau', 'cuisine'),
    44: ('cuillère', 'cuisine'), 45: ('bol', 'cuisine'),
    46: ('banane', 'aliment'), 47: ('pomme', 'aliment'),
    48: ('sandwich', 'aliment'), 49: ('orange', 'aliment'),
    50: ('brocoli', 'aliment'), 51: ('carotte', 'aliment'),
    52: ('hot-dog', 'aliment'), 53: ('pizza', 'aliment'),
    54: ('donut', 'aliment'), 55: ('gâteau', 'aliment'),
    56: ('chaise', 'meuble'), 57: ('canapé', 'meuble'),
    58: ('plante en pot', 'végétal'), 59: ('lit', 'meuble'),
    60: ('table à manger', 'meuble'), 61: ('toilettes', 'sanitaire'),
    62: ('télévision', 'électronique'), 63: ('ordinateur portable', 'électronique'),
    64: ('souris', 'électronique'), 65: ('télécommande', 'électronique'),
    66: ('clavier', 'électronique'), 67: ('téléphone portable', 'électronique'),
    68: ('micro-ondes', 'électroménager'), 69: ('four', 'électroménager'),
    70: ('grille-pain', 'électroménager'), 71: ('évier', 'électroménager'),
    72: ('réfrigérateur', 'électroménager'), 73: ('livre', 'culture'),
    74: ('horloge', 'décoration'), 75: ('vase', 'décoration'),
    76: ('ciseaux', 'outil'), 77: ('ours en peluche', 'jouet'),
    78: ('sèche-cheveux', 'électroménager'), 79: ('brosse à dents', 'hygiène'),
}

corrections = []
corrections_lock = threading.Lock()

custom_model = None


def get_dominant_color(crop):
    if crop is None or crop.size == 0:
        return "de couleur inconnue"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = s > 30
    if mask.sum() < 10:
        return "de couleur claire" if v.mean() > 180 else "de couleur sombre" if v.mean() < 80 else "neutre"
    h_mean = h[mask].mean()
    if h_mean < 10 or h_mean > 160: return "rouge"
    elif h_mean < 25: return "orange"
    elif h_mean < 35: return "jaune"
    elif h_mean < 75: return "vert"
    elif h_mean < 100: return "bleu"
    elif h_mean < 130: return "violet"
    elif h_mean < 160: return "rose"
    return "vif"


def get_vlm_description(crop):
    if vlm is None or vlm_processor is None:
        return None
    try:
        from PIL import Image
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        prompt = "<CAPTION>"
        inputs = vlm_processor(text=prompt, images=pil_img, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            generated = vlm.generate(
                **inputs, max_new_tokens=50, num_beams=3,
                repetition_penalty=1.1, do_sample=False
            )
        caption = vlm_processor.decode(generated[0], skip_special_tokens=True)
        caption = caption.replace(prompt, "").strip()
        return caption if len(caption) > 5 else None
    except Exception as e:
        return None


import torch


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    img_b64 = data.get('image', '')
    use_tracking = data.get('tracking', True)
    use_vlm = data.get('vlm', True)

    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception:
        return jsonify({'error': 'Image invalide'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Impossible de décoder l\'image'}), 400

    global custom_model
    active_model = custom_model if custom_model else yolo

    if use_tracking:
        results = active_model.track(
            frame, conf=0.35, verbose=False, device='cuda:0',
            tracker="bytetrack.yaml", persist=True
        )
    else:
        results = active_model(frame, conf=0.35, verbose=False, device='cuda:0')

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            crop = frame[y1:y2, x1:x2]

            fr_name, category = COCO_CLASSES_FR.get(cls_id, (r.names[cls_id], 'objet'))

            with corrections_lock:
                for c in corrections:
                    if c['from'] == fr_name or c.get('from_en') == r.names[cls_id]:
                        fr_name = c['to']
                        break

            color_desc = get_dominant_color(crop)
            vlm_desc = get_vlm_description(crop) if use_vlm and vlm else None

            track_id = None
            if use_tracking and box.id is not None:
                track_id = int(box.id)

            desc = f"{fr_name} {color_desc}"
            if vlm_desc:
                desc += f" — {vlm_desc}"

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': fr_name,
                'class_en': r.names[cls_id],
                'category': category,
                'confidence': round(conf, 3),
                'description': desc,
                'color': color_desc,
                'vlm_description': vlm_desc,
                'track_id': track_id,
            })

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return jsonify({
        'detections': detections,
        'count': len(detections),
        'vlm_active': vlm is not None,
        'tracking_active': use_tracking,
    })


@app.route('/corrections', methods=['GET'])
def get_corrections():
    with corrections_lock:
        return jsonify(corrections)


@app.route('/corrections', methods=['POST'])
def add_correction():
    data = request.json
    with corrections_lock:
        for c in corrections:
            if c['from'] == data['from'] and c['to'] == data['to']:
                return jsonify({'status': 'exists'})
        corrections.append({
            'from': data['from'],
            'to': data['to'],
            'from_en': data.get('from_en', ''),
            'timestamp': time.time(),
        })
    return jsonify({'status': 'ok'})


@app.route('/corrections', methods=['DELETE'])
def clear_corrections():
    with corrections_lock:
        corrections.clear()
    return jsonify({'status': 'ok'})


@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    data = request.json
    label = data.get('label', '').strip()
    img_b64 = data.get('image', '')
    if not label or not img_b64:
        return jsonify({'error': 'label et image requis'}), 400

    label_dir = os.path.join(DATASET_DIR, label.replace(' ', '_'))
    os.makedirs(label_dir, exist_ok=True)

    count = len([f for f in os.listdir(label_dir) if f.endswith('.jpg')])
    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]
    img_bytes = base64.b64decode(img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    filepath = os.path.join(label_dir, f"{count+1:04d}.jpg")
    cv2.imwrite(filepath, frame)

    return jsonify({'status': 'ok', 'saved': filepath, 'count': count + 1})


@app.route('/stats')
def stats():
    return jsonify({
        'gpu': 'RTX 4060 Ti',
        'model': 'YOLOv8m',
        'vlm': 'Florence-2' if vlm else 'non disponible',
        'tensorrt': TRT_AVAILABLE,
        'corrections': len(corrections),
        'dataset_samples': sum(
            len([f for f in os.listdir(os.path.join(DATASET_DIR, d)) if f.endswith('.jpg')])
            for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ) if os.path.exists(DATASET_DIR) else 0,
    })


if __name__ == '__main__':
    print("\n✅ Serveur prêt: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
