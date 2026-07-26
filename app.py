import os
import cv2
import numpy as np
import base64
import json
import time
import threading
import logging
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import torch

app = Flask(__name__)

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'app.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('objrec')

session_start = time.time()
session_log = {
    'start_time': datetime.now().isoformat(),
    'frames_processed': 0,
    'total_detections': 0,
    'vlm_calls': 0,
    'vlm_descriptions': [],
    'corrections': [],
    'detection_log': [],
    'errors': [],
    'performance': [],
    'config': {},
}

session_lock = threading.Lock()

print("=" * 60)
print("  Object Camera Recognition — Full GPU Pipeline")
print("  YOLOv8m + ByteTrack + BLIP VLM + TensorRT")
print("=" * 60)

log.info("Chargement du modèle YOLOv8m...")
yolo = YOLO('yolov8m.pt')
if torch.cuda.is_available():
    yolo.to('cuda:0')
    log.info(f"YOLO sur GPU: {torch.cuda.get_device_name(0)}")
else:
    log.warning("YOLO sur CPU - CUDA non disponible")

log.info("Chargement du VLM...")
vlm = None
vlm_processor = None
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    vlm = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("BLIP VLM chargé sur GPU")
except Exception as e:
    log.error(f"VLM non disponible: {e}")

try:
    import tensorrt
    log.info(f"TensorRT {tensorrt.__version__} disponible")
except:
    log.warning("TensorRT non disponible")

session_log['config'] = {
    'model': 'YOLOv8m',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'vlm': 'BLIP' if vlm else 'none',
    'tensorrt': 'tensorrt' in dir(),
}

COCO_CLASSES_FR = {
    0: ('personne', 'humain'), 1: ('vélo', 'véhicule'),
    2: ('voiture', 'véhicule'), 3: ('moto', 'véhicule'),
    4: ('avion', 'aéronef'), 5: ('bus', 'transport'),
    6: ('train', 'transport'), 7: ('camion', 'véhicule'),
    8: ('bateau', 'navire'), 9: ('feu tricolore', 'signal'),
    10: ("bouche d'incendie", 'sécurité'), 11: ('panneau stop', 'signal'),
    12: ('parcmètre', 'stationnement'), 13: ('banc', 'siège'),
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
vlm_frame_counter = 0
VLM_EVERY_N_FRAMES = 3


def get_dominant_color(crop):
    if crop is None or crop.size == 0:
        return "inconnue"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = s > 30
    if mask.sum() < 10:
        return "claire" if v.mean() > 180 else "sombre" if v.mean() < 80 else "neutre"
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
        return None, 0
    try:
        from PIL import Image
        t0 = time.time()
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        inputs = vlm_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(vlm.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = vlm.generate(**inputs, max_new_tokens=50, num_beams=3)
        caption = vlm_processor.decode(generated[0], skip_special_tokens=True)
        elapsed = (time.time() - t0) * 1000
        log.debug(f"VLM: {caption} ({elapsed:.0f}ms)")
        with session_lock:
            session_log['vlm_calls'] += 1
            session_log['vlm_descriptions'].append({
                'caption': caption,
                'latency_ms': round(elapsed, 1),
                'timestamp': time.time(),
            })
        return caption if len(caption) > 5 else None, elapsed
    except Exception as e:
        log.error(f"VLM erreur: {e}")
        return None, 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    frame_start = time.time()
    data = request.json
    img_b64 = data.get('image', '')
    use_tracking = data.get('tracking', True)
    use_vlm = data.get('vlm', True)

    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception:
        log.error("Image invalide reçue")
        return jsonify({'error': 'Image invalide'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        log.error("Impossible de décoder l'image")
        return jsonify({'error': 'Image non décodable'}), 400

    h_frame, w_frame = frame.shape[:2]

    global custom_model
    active_model = custom_model if custom_model else yolo

    try:
        if use_tracking:
            results = active_model.track(
                frame, conf=0.25, verbose=False,
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                tracker="bytetrack.yaml", persist=True,
                iou=0.5, classes=None
            )
        else:
            results = active_model(
                frame, conf=0.25, verbose=False,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            )
    except Exception as e:
        log.error(f"Erreur détection: {e}")
        with session_lock:
            session_log['errors'].append({'type': 'detection', 'message': str(e), 'timestamp': time.time()})
        return jsonify({'error': str(e)}), 500

    det_start = time.time()
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            crop = frame[max(0,y1):min(h_frame,y2), max(0,x1):min(w_frame,x2)]

            fr_name, category = COCO_CLASSES_FR.get(cls_id, (r.names[cls_id], 'objet'))

            with corrections_lock:
                for c in corrections:
                    if c['from'] == fr_name or c.get('from_en') == r.names[cls_id]:
                        fr_name = c['to']
                        break

            color_desc = get_dominant_color(crop)
            vlm_desc = None
            vlm_latency = 0

            global vlm_frame_counter
            vlm_frame_counter += 1
            run_vlm = use_vlm and vlm and crop.size > 100 and (vlm_frame_counter % VLM_EVERY_N_FRAMES == 0)
            if run_vlm:
                vlm_desc, vlm_latency = get_vlm_description(crop)

            track_id = None
            if use_tracking and hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id)

            desc = f"{fr_name} {color_desc}"
            if vlm_desc:
                desc += f" — {vlm_desc}"

            det = {
                'bbox': [x1, y1, x2, y2],
                'class': fr_name,
                'class_en': r.names[cls_id],
                'category': category,
                'confidence': round(conf, 3),
                'description': desc,
                'color': color_desc,
                'vlm_description': vlm_desc,
                'vlm_latency_ms': round(vlm_latency, 1),
                'track_id': track_id,
            }
            detections.append(det)

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    frame_latency = (time.time() - frame_start) * 1000

    with session_lock:
        session_log['frames_processed'] += 1
        session_log['total_detections'] += len(detections)
        session_log['performance'].append({
            'frame': session_log['frames_processed'],
            'latency_ms': round(frame_latency, 1),
            'detections': len(detections),
            'vlm_active': use_vlm,
            'tracking_active': use_tracking,
            'timestamp': time.time(),
        })
        if len(detections) > 0:
            session_log['detection_log'].append({
                'frame': session_log['frames_processed'],
                'objects': [{
                    'class': d['class'],
                    'class_en': d['class_en'],
                    'category': d['category'],
                    'confidence': d['confidence'],
                    'color': d['color'],
                    'track_id': d['track_id'],
                    'vlm_description': d['vlm_description'],
                } for d in detections],
                'timestamp': time.time(),
            })

    log.info(f"Frame #{session_log['frames_processed']}: {len(detections)} objets, {frame_latency:.0f}ms")

    return jsonify({
        'detections': detections,
        'count': len(detections),
        'frame': session_log['frames_processed'],
        'latency_ms': round(frame_latency, 1),
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
        corr = {
            'from': data['from'], 'to': data['to'],
            'from_en': data.get('from_en', ''), 'timestamp': time.time(),
        }
        corrections.append(corr)
        with session_lock:
            session_log['corrections'].append(corr)
    log.info(f"Correction: {data['from']} → {data['to']}")
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

    log.info(f"Fine-tune capture: {label} (#{count+1})")
    return jsonify({'status': 'ok', 'saved': filepath, 'count': count + 1})


@app.route('/export')
def export_session():
    with session_lock:
        export = {
            'session': {
                'start_time': session_log['start_time'],
                'duration_seconds': round(time.time() - session_start, 1),
                'config': session_log['config'],
            },
            'stats': {
                'frames_processed': session_log['frames_processed'],
                'total_detections': session_log['total_detections'],
                'vlm_calls': session_log['vlm_calls'],
                'corrections_count': len(session_log['corrections']),
                'avg_latency_ms': round(
                    np.mean([p['latency_ms'] for p in session_log['performance']])
                    if session_log['performance'] else 0, 1
                ),
                'max_latency_ms': round(
                    max([p['latency_ms'] for p in session_log['performance']])
                    if session_log['performance'] else 0, 1
                ),
                'min_latency_ms': round(
                    min([p['latency_ms'] for p in session_log['performance']])
                    if session_log['performance'] else 0, 1
                ),
            },
            'corrections': session_log['corrections'],
            'vlm_descriptions': session_log['vlm_descriptions'][-50:],
            'detection_log': session_log['detection_log'][-100:],
            'performance': session_log['performance'][-200:],
            'errors': session_log['errors'],
        }

    filepath = os.path.join(LOGS_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    log.info(f"Session exportée: {filepath}")
    return send_file(filepath, as_attachment=True, download_name=os.path.basename(filepath))


@app.route('/export/csv')
def export_csv():
    filepath = os.path.join(LOGS_DIR, f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'timestamp', 'classe', 'classe_en', 'categorie', 'confiance', 'couleur', 'track_id', 'vlm_description'])
        with session_lock:
            for entry in session_log['detection_log']:
                for obj in entry['objects']:
                    writer.writerow([
                        entry['frame'],
                        datetime.fromtimestamp(entry['timestamp']).isoformat(),
                        obj['class'], obj.get('class_en', ''), obj.get('category', ''),
                        obj['confidence'], obj.get('color', ''), obj.get('track_id', ''),
                        obj.get('vlm_description', ''),
                    ])

    log.info(f"CSV exporté: {filepath}")
    return send_file(filepath, as_attachment=True, download_name=os.path.basename(filepath))


@app.route('/export/html')
def export_html():
    with session_lock:
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Session Object Recognition</title>
<style>
body {{ font-family: sans-serif; max-width: 1000px; margin: 2rem auto; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }}
h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ padding: 0.5rem; border: 1px solid #21262d; text-align: left; }}
th {{ background: #161b22; color: #8b949e; }}
.ok {{ color: #3fb950; }}
.err {{ color: #f85149; }}
.stat {{ display: inline-block; background: #161b22; padding: 0.5rem 1rem; margin: 0.3rem; border-radius: 8px; border: 1px solid #21262d; }}
</style></head><body>
<h1>Session Object Recognition — Rapport</h1>
<p>Durée: {round(time.time() - session_start, 0):.0f}s | Frames: {session_log['frames_processed']} | Détections: {session_log['total_detections']}</p>

<h2>Configuration</h2>
<table><tr><th>Paramètre</th><th>Valeur</th></tr>
<tr><td>Modèle</td><td>{session_log['config'].get('model', '?')}</td></tr>
<tr><td>Device</td><td>{session_log['config'].get('device', '?')}</td></tr>
<tr><td>VLM</td><td>{session_log['config'].get('vlm', '?')}</td></tr>
<tr><td>TensorRT</td><td>{session_log['config'].get('tensorrt', '?')}</td></tr></table>

<h2>Performance</h2>
<div>
<span class="stat">Latence moy: {export['stats']['avg_latency_ms'] if 'export' in dir() else '?'}ms</span>
<span class="stat">Latence min: {export['stats']['min_latency_ms'] if 'export' in dir() else '?'}ms</span>
<span class="stat">Latence max: {export['stats']['max_latency_ms'] if 'export' in dir() else '?'}ms</span>
<span class="stat">VLM calls: {session_log['vlm_calls']}</span>
<span class="stat">Corrections: {len(session_log['corrections'])}</span>
</div>

<h2>Détections ({len(session_log['detection_log'])} frames)</h2>
<table><tr><th>Frame</th><th>Objets</th><th>Détails</th></tr>"""
        for entry in session_log['detection_log'][-50:]:
            objs = ', '.join([f"{o['class']} ({o['confidence']:.0%})" for o in entry['objects']])
            html += f"\n<tr><td>#{entry['frame']}</td><td>{len(entry['objects'])}</td><td>{objs}</td></tr>"
        html += "</table>"

        if session_log['corrections']:
            html += "<h2>Corrections</h2><table><tr><th>De</th><th>Vers</th><th>Timestamp</th></tr>"
            for c in session_log['corrections']:
                html += f"\n<tr><td>{c['from']}</td><td>{c['to']}</td><td>{datetime.fromtimestamp(c['timestamp']).isoformat()}</td></tr>"
            html += "</table>"

        if session_log['vlm_descriptions']:
            html += "<h2>Descriptions VLM</h2><table><tr><th>Description</th><th>Latence</th></tr>"
            for v in session_log['vlm_descriptions'][-20:]:
                html += f"\n<tr><td>{v['caption']}</td><td>{v['latency_ms']}ms</td></tr>"
            html += "</table>"

        if session_log['errors']:
            html += "<h2 class='err'>Erreurs</h2><table><tr><th>Type</th><th>Message</th></tr>"
            for e in session_log['errors']:
                html += f"\n<tr><td>{e['type']}</td><td>{e['message']}</td></tr>"
            html += "</table>"

        html += "</body></html>"

    filepath = os.path.join(LOGS_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    log.info(f"HTML exporté: {filepath}")
    return send_file(filepath, as_attachment=True, download_name=os.path.basename(filepath))


@app.route('/stats')
def stats():
    with session_lock:
        perf = session_log['performance']
        return jsonify({
            'gpu': 'RTX 4060 Ti' if torch.cuda.is_available() else 'CPU',
            'model': 'YOLOv8m',
            'vlm': 'BLIP' if vlm else 'none',
            'tensorrt': 'tensorrt' in dir(),
            'corrections': len(corrections),
            'frames': session_log['frames_processed'],
            'total_detections': session_log['total_detections'],
            'avg_latency_ms': round(np.mean([p['latency_ms'] for p in perf]), 1) if perf else 0,
            'vlm_calls': session_log['vlm_calls'],
        })


if __name__ == '__main__':
    log.info("Serveur démarré: http://localhost:5000")
    print("\n✅ Serveur prêt: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
