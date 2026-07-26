import os
import cv2
import numpy as np
import base64
import json
import time
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

print("Chargement du modèle YOLOv8m sur GPU...")
yolo = YOLO('yolov8m.pt')
print(f"Modèle chargé. Device: {yolo.device}")

COCO_CLASSES_FR = {
    0: ('personne', 'humain'),
    1: ('vélo', 'véhicule à deux roues'),
    2: ('voiture', 'véhicule motorisé'),
    3: ('moto', 'véhicule à deux roues motorisé'),
    4: ('avion', 'aéronef'),
    5: ('bus', 'transport en commun'),
    6: ('train', 'transport ferroviaire'),
    7: ('camion', 'véhicule de transport'),
    8: ('bateau', 'navire'),
    9: ('feu tricolore', 'signal routier lumineux'),
    10: ('bouche d\'incendie', 'équipement de sécurité'),
    11: ('panneau stop', 'signalisation routière'),
    12: ('parcmètre', 'stationnement'),
    13: ('banc', 'siège public'),
    14: ('oiseau', 'animal volatile'),
    15: ('chat', 'animal domestique félin'),
    16: ('chien', 'animal domestique canin'),
    17: ('cheval', 'équidé'),
    18: ('mouton', 'ovin'),
    19: ('vache', 'bovin'),
    20: ('éléphant', 'grand mammifère'),
    21: ('ours', 'mammifère carnivore'),
    22: ('zèbre', 'équidé rayé'),
    23: ('girafe', 'mammifère à long cou'),
    24: ('sac à dos', 'accessoire de transport'),
    25: ('parapluie', 'protection contre la pluie'),
    26: ('sac à main', 'accessoire de mode'),
    27: ('cravate', 'accessoire vestimentaire'),
    28: ('valise', 'bagage de voyage'),
    29: ('frisbee', 'jouet de sport'),
    30: ('skis', 'équipement de glisse'),
    31: ('snowboard', 'planche à neige'),
    32: ('ballon de sport', 'équipement sportif'),
    33: ('cerf-volant', 'jouet aérien'),
    34: ('batte de baseball', 'équipement de baseball'),
    35: ('gant de baseball', 'équipement de baseball'),
    36: ('skateboard', 'planche à roulettes'),
    37: ('surfboard', 'planche de surf'),
    38: ('raquette de tennis', 'équipement de tennis'),
    39: ('bouteille', 'conteneur de boisson'),
    40: ('verre à vin', 'récipient à vin'),
    41: ('tasse', 'récipient pour boisson chaude'),
    42: ('fourchette', 'ustensile de cuisine'),
    43: ('couteau', 'ustensile de cuisine tranchant'),
    44: ('cuillère', 'ustensile de cuisine'),
    45: ('bol', 'récipient de cuisine'),
    46: ('banane', 'fruit tropical'),
    47: ('pomme', 'fruit'),
    48: ('sandwich', 'plat préparé'),
    49: ('orange', 'fruit agrume'),
    50: ('brocoli', 'légume vert'),
    51: ('carotte', 'légume orange'),
    52: ('hot-dog', 'plat rapide'),
    53: ('pizza', 'plat italien'),
    54: ('donut', 'pâtisserie'),
    55: ('gâteau', 'pâtisserie'),
    56: ('chaise', 'siège'),
    57: ('canapé', 'meuble de salon'),
    58: ('plante en pot', 'végétal domestique'),
    59: ('lit', 'meuble de repos'),
    60: ('table à manger', 'meuble de salle à manger'),
    61: ('toilettes', 'sanitaire'),
    62: ('télévision', 'écran d\'affichage'),
    63: ('ordinateur portable', 'appareil informatique'),
    64: ('souris', 'périphérique informatique'),
    65: ('télécommande', 'contrôleur à distance'),
    66: ('clavier', 'périphérique de saisie'),
    67: ('téléphone portable', 'appareil de communication'),
    68: ('micro-ondes', 'appareil de cuisson'),
    69: ('four', 'appareil de cuisson'),
    70: ('grille-pain', 'appareil de cuisson'),
    71: ('évier', 'sanitaire de cuisine'),
    72: ('réfrigérateur', 'appareil de conservation'),
    73: ('livre', 'objet de lecture'),
    74: ('horloge', 'objet de mesure du temps'),
    75: ('vase', 'objet de décoration'),
    76: ('ciseaux', 'outil de coupe'),
    77: ('ours en peluche', 'jouet en peluche'),
    78: ('sèche-cheveux', 'appareil de soin'),
    79: ('brosse à dents', 'outil d\'hygiène'),
}


def get_dominant_color(crop):
    if crop is None or crop.size == 0:
        return "de couleur inconnue"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = s > 30
    if mask.sum() < 10:
        if v.mean() > 180:
            return "de couleur claire (blanc/gris)"
        elif v.mean() < 80:
            return "de couleur sombre (noir)"
        return "de couleur neutre"
    h_mean = h[mask].mean()
    s_mean = s[mask].mean()
    if h_mean < 10 or h_mean > 160:
        return "de couleur rouge"
    elif h_mean < 25:
        return "de couleur orange"
    elif h_mean < 35:
        return "de couleur jaune"
    elif h_mean < 75:
        return "de couleur verte"
    elif h_mean < 100:
        return "de couleur bleue"
    elif h_mean < 130:
        return "de couleur violette"
    elif h_mean < 160:
        return "de couleur rose"
    return "de couleur vive"


def get_size_desc(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    frame_area = frame_shape[0] * frame_shape[1]
    ratio = area / frame_area
    if ratio > 0.15:
        return "très proche"
    elif ratio > 0.05:
        return "à distance moyenne"
    elif ratio > 0.01:
        return "assez éloigné"
    return "très éloigné"


def get_position_desc(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w, h = frame_shape[1], frame_shape[0]
    parts = []
    if cx < w * 0.33:
        parts.append("à gauche")
    elif cx > w * 0.66:
        parts.append("à droite")
    else:
        parts.append("au centre")
    if cy < h * 0.33:
        parts.append("en haut")
    elif cy > h * 0.66:
        parts.append("en bas")
    else:
        parts.append("au milieu")
    return ", ".join(parts)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    img_b64 = data.get('image', '')
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
    results = yolo(frame, conf=0.35, verbose=False, device='cuda:0')
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            crop = frame[y1:y2, x1:x2]
            fr_name, category = COCO_CLASSES_FR.get(cls_id, (r.names[cls_id], 'objet'))
            color_desc = get_dominant_color(crop)
            size_desc = get_size_desc(xyxy, frame.shape)
            pos_desc = get_position_desc(xyxy, frame.shape)
            description = f"{fr_name} {color_desc}, {size_desc}, {pos_desc}"
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': fr_name,
                'class_en': r.names[cls_id],
                'category': category,
                'confidence': round(conf, 3),
                'description': description,
                'color': color_desc,
            })
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return jsonify({'detections': detections, 'count': len(detections)})


if __name__ == '__main__':
    print("=" * 50)
    print("  Object Camera Recognition — GPU Server")
    print("  YOLOv8m + CUDA — RTX 4060 Ti")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
