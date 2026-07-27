import os
import sys
import subprocess
import cv2
import numpy as np
import base64
import json
import time
import threading
import logging
import atexit
import signal
import csv
import re
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import torch

from scene_analyzer import SceneAnalyzer
from vlm_agent import VisionLanguageOrchestrator
from tracker import DetectionTracker, OCRFilter, MultiFrameAggregator
from stream_analyzer import CameraManager
from activity_recorder import ActivityRecorder
from video_exporter import VideoExporter
from audio_analyzer import AudioAnalyzer
from orchestrator import Orchestrator
import license_manager
import coach
from agents.data_collect_agent import DataCollectAgent
from agents.training_agent import TrainingAgent
from agents.export_agent import ExportAgent
from agents.continuous_agent import ContinuousAgent
from agents.video_agent import VideoAgent
from agents.image_gen_agent import ImageGenAgent

app = Flask(__name__)

# Mode léger : désactive les modules lourds (VLM, enregistrement, audio, orchestrateur)
# À utiliser si l'application est lente ou instable au démarrage.
LIGHT_MODE = os.environ.get('VISION_LIGHT', '0') == '1'

# Mode moyen : désactive seulement le VLM (très lourd), garde l'enregistrement et les agents.
# Recommandé comme usage quotidien par défaut.
DISABLE_VLM = os.environ.get('VISION_DISABLE_VLM', '0') == '1' or LIGHT_MODE

# Auto-sauvegarde des sessions et des produits détectés
AUTO_SAVE_PRODUCTS = os.environ.get('VISION_AUTO_SAVE_PRODUCTS', '1') != '0'
AUTO_EXPORT_INTERVAL_SEC = int(os.environ.get('VISION_AUTO_EXPORT_INTERVAL', '60'))

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
PRODUCTS_DB = os.path.join(os.path.dirname(__file__), 'products_db.json')
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
    'ocr_reads': 0,
    'barcodes_found': 0,
    'products_recognized': 0,
    'ocr_results': [],
    'barcode_results': [],
    'corrections': [],
    'detection_log': [],
    'errors': [],
    'performance': [],
    'config': {},
}

session_lock = threading.Lock()

print("=" * 60)
if LIGHT_MODE:
    print("  Systeme Autonome de Reconnaissance Produits (MODE LÉGER)")
    print("  OCR + YOLO + Barcodes (VLM/audio/recorder désactivés)")
else:
    print("  Systeme Autonome de Reconnaissance Produits")
    print("  OCR + YOLO + Barcodes + Base Locale")
print("=" * 60)

log.info("Chargement YOLOv8...")
best_model_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')
if os.path.exists(best_model_path):
    yolo = YOLO(best_model_path)
    log.info("Modele fine-tune charge (5 classes: bottle, cell_phone, cracker, person, e-cig)")
else:
    yolo = YOLO('yolov8m.pt')
    log.info("Modele YOLOv8m COCO standard charge (80 classes)")

if torch.cuda.is_available():
    yolo.to('cuda:0')
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")

log.info("Chargement EasyOCR...")
ocr_reader = None
try:
    import easyocr
    ocr_reader = easyocr.Reader(['fr', 'en'], gpu=torch.cuda.is_available(), verbose=False)
    log.info("EasyOCR pret (fr + en)")
except Exception as e:
    log.error(f"EasyOCR: {e}")

log.info("Verification pyzbar...")
pyzbar_available = False
try:
    from pyzbar import pyzbar
    pyzbar_available = True
    log.info("pyzbar pret")
except Exception as e:
    log.error(f"pyzbar: {e}")

products_db = {}
if os.path.exists(PRODUCTS_DB):
    try:
        with open(PRODUCTS_DB, 'r', encoding='utf-8') as f:
            products_db = json.load(f)
        log.info(f"Base produits chargee: {len(products_db)} produits")
    except:
        pass

def save_products_db():
    try:
        with open(PRODUCTS_DB, 'w', encoding='utf-8') as f:
            json.dump(products_db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Erreur save DB: {e}")

session_log['config'] = {
    'model': 'fine-tune' if os.path.exists(best_model_path) else 'yolov8m',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'ocr': 'EasyOCR' if ocr_reader else 'none',
    'barcode': 'pyzbar' if pyzbar_available else 'none',
    'products_db': len(products_db),
}

# --- Agent VLM + Analyseur de scène multimodal ---
vlm_orchestrator = VisionLanguageOrchestrator(device='cuda' if torch.cuda.is_available() else 'cpu')
scene_analyzer = None

if DISABLE_VLM:
    if LIGHT_MODE:
        log.info("Mode léger : SceneAnalyzer initialisé sans VLM, sans visage/gestes")
        try:
            scene_analyzer = SceneAnalyzer(
                yolo_model=yolo,
                ocr_reader=ocr_reader,
                barcode_enabled=pyzbar_available,
                vlm_orchestrator=None,
                gesture_enabled=False,
                face_enabled=False,
            )
            log.info("SceneAnalyzer prêt (mode léger)")
        except Exception as e:
            log.error(f"Erreur init SceneAnalyzer: {e}")
    else:
        log.info("Mode optimisé : SceneAnalyzer avec visage + gestes, sans VLM")
        try:
            scene_analyzer = SceneAnalyzer(
                yolo_model=yolo,
                ocr_reader=ocr_reader,
                barcode_enabled=pyzbar_available,
                vlm_orchestrator=None,
                gesture_enabled=True,
                face_enabled=True,
            )
            log.info("SceneAnalyzer prêt (mode optimisé)")
        except Exception as e:
            log.error(f"Erreur init SceneAnalyzer: {e}")
else:
    log.info("Initialisation du SceneAnalyzer (gestes + visage + YOLO + VLM)...")
    try:
        scene_analyzer = SceneAnalyzer(
            yolo_model=yolo,
            ocr_reader=ocr_reader,
            barcode_enabled=pyzbar_available,
            vlm_orchestrator=vlm_orchestrator,
            gesture_enabled=True,
            face_enabled=True,
        )
        log.info("SceneAnalyzer prêt")
    except Exception as e:
        log.error(f"Erreur init SceneAnalyzer: {e}")

# --- Gestionnaire multi-caméras ---
camera_manager = CameraManager(scene_analyzer=scene_analyzer)


# --- Orchestrateur d'agents IA ---
if LIGHT_MODE:
    orchestrator = None
    log.info("Mode léger : Orchestrateur IA désactivé")
else:
    orchestrator = Orchestrator(
        agents={
            'data_collect': DataCollectAgent(base_dir=os.path.dirname(os.path.abspath(__file__))),
            'training': TrainingAgent(base_dir=os.path.dirname(os.path.abspath(__file__))),
            'export': ExportAgent(base_dir=os.path.dirname(os.path.abspath(__file__))),
            'continuous': ContinuousAgent(base_dir=os.path.dirname(os.path.abspath(__file__))),
            'video': VideoAgent(base_dir=os.path.dirname(os.path.abspath(__file__))),
            'image_gen': ImageGenAgent(base_dir=os.path.dirname(os.path.abspath(__file__))),
        },
        memory_db=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_memory'),
    )
    orchestrator.start()
    log.info('Orchestrateur IA demarre avec ' + str(len(orchestrator.agents)) + ' agents')

# --- Activite Recorder auto-demarrage ---

def _kill_lingering_python():
    """Tue les processus python zombies qui pourraient verrouiller la webcam."""
    try:
        import subprocess as sp
        result = sp.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        for line in lines[3:]:  # Skip header lines
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pid = int(parts[1])
                    if pid != os.getpid():
                        import os as _os
                        _os.kill(pid, 9)
                        log.info(f"Processus zombie python {pid} terminé")
                except (ValueError, ProcessLookupError, PermissionError, OSError):
                    pass
    except Exception as e:
        log.debug(f"_kill_lingering_python: {e}")

_kill_lingering_python()

if LIGHT_MODE:
    recorder = None
    log.info("Mode léger : ActivityRecorder désactivé")
else:
    recorder = ActivityRecorder(
        output_dir=os.path.join(os.path.dirname(__file__), 'recordings'),
        screen_fps=5,
        webcam_fps=5,
    )
    recorder.start()
    log.info("ActiviteRecorder demarre")

# --- Video Exporter + Audio Analyzer instances ---
video_exporter = VideoExporter(
    recordings_dir=os.path.join(os.path.dirname(__file__), 'recordings'),
    output_dir=os.path.join(os.path.dirname(__file__), 'recordings', 'exports'),
)
if LIGHT_MODE:
    audio_analyzer = None
    log.info("Mode léger : AudioAnalyzer désactivé")
else:
    audio_analyzer = AudioAnalyzer(device='cuda' if torch.cuda.is_available() else 'cpu')
log.info("VideoExporter prêt")

def _export_session(write_timestamp=True):
    """Exporte la session courante dans un fichier JSON.
    Par defaut, ecrit a la fois dans session_latest.json et un fichier horodate."""
    try:
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
                    'ocr_reads': session_log['ocr_reads'],
                    'barcodes_found': session_log['barcodes_found'],
                    'products_recognized': session_log['products_recognized'],
                    'corrections_count': len(session_log['corrections']),
                    'avg_latency_ms': round(np.mean([p['latency_ms'] for p in session_log['performance']]) if session_log['performance'] else 0, 1),
                },
                'products_db': products_db,
                'corrections': session_log['corrections'],
                'ocr_results': session_log['ocr_results'][-200:],
                'barcode_results': session_log['barcode_results'][-50:],
                'performance': session_log['performance'][-500:],
            }
        # Fichier courant toujours ecrase (rapide et fiable)
        latest_path = os.path.join(LOGS_DIR, 'session_latest.json')
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        # Fichier horodate toutes les N secondes
        if write_timestamp:
            filepath = os.path.join(LOGS_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export, f, ensure_ascii=False, indent=2)
            log.info(f"Session exportee: {filepath}")
    except Exception as e:
        log.error(f"Erreur export session: {e}")


def _cleanup_all():
    global recorder
    _export_session(write_timestamp=True)
    if recorder is not None and recorder.is_recording:
        log.info("Nettoyage recorder a l'arret...")
        try:
            recorder.stop()
        except Exception:
            pass
    if video_exporter is not None:
        log.info("VideoExporter cleanup ok")
    log.info("Cleanup complet")


atexit.register(_cleanup_all)
try:
    signal.signal(signal.SIGINT, lambda s, f: _cleanup_all())
    signal.signal(signal.SIGTERM, lambda s, f: _cleanup_all())
except Exception:
    pass

# Auto-export déclenché par les requêtes /detect
_last_auto_export_time = 0


def _maybe_auto_export():
    global _last_auto_export_time
    now = time.time()
    if now - _last_auto_export_time >= max(AUTO_EXPORT_INTERVAL_SEC, 10):
        _last_auto_export_time = now
        log.info("Auto-export declenche")
        try:
            _export_session(write_timestamp=True)
            log.info("Auto-export reussi")
        except Exception as e:
            log.error(f"Erreur auto-export: {e}")


log.info(f"Auto-export session toutes les {AUTO_EXPORT_INTERVAL_SEC}s (sur /detect)")

def _load_vlm_async():
    try:
        log.info("Chargement asynchrone du VLM...")
        ok = vlm_orchestrator.load(prefer='auto')
        if ok:
            log.info("VLM actif : " + str(vlm_orchestrator.active.__class__.__name__))
        else:
            log.warning("Aucun VLM n'a pu être chargé — descriptions désactivées")
    except Exception as e:
        log.error(f"Erreur chargement VLM: {e}")

if DISABLE_VLM:
    log.info("Chargement VLM ignoré (mode optimisé)")
else:
    vlm_thread = threading.Thread(target=_load_vlm_async, daemon=True)
    vlm_thread.start()

# --- Stabilisation multi-frame et filtrage OCR ---
log.info("Initialisation tracker + filtres OCR...")
detection_tracker = DetectionTracker(history_size=5, min_hits=2, iou_threshold=0.3)
ocr_filter = OCRFilter(min_conf=0.30, min_length=3, min_alpha_ratio=0.5)
ocr_aggregator = MultiFrameAggregator(history_size=10, min_occurrences=2)

# --- VLM live captioning ---
last_vlm_caption = None
last_vlm_frame_time = 0
vlm_caption_interval = 3.0
vlm_caption_lock = threading.Lock()

# --- Entraînement en arrière-plan ---
retrain_status = {
    'status': 'idle',
    'epoch': 0,
    'epochs': 0,
    'error': None,
    'start_time': None,
    'end_time': None,
}
retrain_lock = threading.Lock()
retrain_process = None


def _run_training():
    global retrain_status, retrain_process
    with retrain_lock:
        retrain_status = {
            'status': 'running',
            'epoch': 0,
            'epochs': 50,
            'error': None,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
        }
    try:
        import subprocess
        script = os.path.join(os.path.dirname(__file__), 'train.py')
        log_file = os.path.join(LOGS_DIR, 'train.log')
        with open(log_file, 'w', encoding='utf-8') as lf:
            retrain_process = subprocess.Popen(
                [sys.executable, script],
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(__file__),
            )
            retrain_process.wait()
        if retrain_process.returncode == 0:
            with retrain_lock:
                retrain_status = {
                    'status': 'done',
                    'epoch': 50,
                    'epochs': 50,
                    'error': None,
                    'start_time': retrain_status.get('start_time'),
                    'end_time': datetime.now().isoformat(),
                }
            log.info("Réentraînement terminé avec succès")
        else:
            with retrain_lock:
                retrain_status = {
                    'status': 'error',
                    'epoch': 0,
                    'epochs': 50,
                    'error': f'process returned {retrain_process.returncode}',
                    'start_time': retrain_status.get('start_time'),
                    'end_time': datetime.now().isoformat(),
                }
    except Exception as e:
        with retrain_lock:
            retrain_status = {
                'status': 'error',
                'epoch': 0,
                'epochs': 50,
                'error': str(e),
                'start_time': retrain_status.get('start_time'),
                'end_time': datetime.now().isoformat(),
            }
        log.error(f"Réentraînement erreur: {e}")
    finally:
        retrain_process = None


COCO_CLASSES_FR = {
    0: ('personne', 'humain'), 1: ('velo', 'vehicule'),
    2: ('voiture', 'vehicule'), 3: ('moto', 'vehicule'),
    4: ('avion', 'aeronef'), 5: ('bus', 'transport'),
    6: ('train', 'transport'), 7: ('camion', 'vehicule'),
    8: ('bateau', 'navire'), 9: ('feu tricolore', 'signal'),
    13: ('banc', 'siege'),
    14: ('oiseau', 'animal'), 15: ('chat', 'animal'),
    16: ('chien', 'animal'), 17: ('cheval', 'animal'),
    24: ('sac a dos', 'accessoire'), 25: ('parapluie', 'accessoire'),
    26: ('sac a main', 'accessoire'), 27: ('cravate', 'vetement'),
    39: ('bouteille', 'cuisine'), 40: ('verre a vin', 'cuisine'),
    41: ('tasse', 'cuisine'), 42: ('fourchette', 'cuisine'),
    43: ('couteau', 'cuisine'), 44: ('cuillere', 'cuisine'),
    45: ('bol', 'cuisine'), 46: ('banane', 'aliment'),
    47: ('pomme', 'aliment'), 48: ('sandwich', 'aliment'),
    49: ('orange', 'aliment'), 50: ('brocoli', 'aliment'),
    51: ('carotte', 'aliment'), 53: ('pizza', 'aliment'),
    54: ('donut', 'aliment'), 55: ('gateau', 'aliment'),
    56: ('chaise', 'meuble'), 57: ('canape', 'meuble'),
    60: ('table', 'meuble'), 62: ('television', 'electronique'),
    63: ('ordinateur', 'electronique'), 67: ('telephone', 'electronique'),
    68: ('micro-ondes', 'electromenager'), 72: ('refrigerateur', 'electromenager'),
    73: ('livre', 'culture'), 77: ('jouet', 'jouet'),
}

# Traduction générique pour les modèles fine-tunés (noms anglais → français + catégorie)
CLASS_TRANSLATIONS = {
    'person': ('personne', 'humain'),
    'bicycle': ('velo', 'vehicule'),
    'car': ('voiture', 'vehicule'),
    'motorcycle': ('moto', 'vehicule'),
    'airplane': ('avion', 'aeronef'),
    'bus': ('bus', 'transport'),
    'train': ('train', 'transport'),
    'truck': ('camion', 'vehicule'),
    'boat': ('bateau', 'navire'),
    'traffic light': ('feu tricolore', 'signal'),
    'bench': ('banc', 'siege'),
    'bird': ('oiseau', 'animal'),
    'cat': ('chat', 'animal'),
    'dog': ('chien', 'animal'),
    'horse': ('cheval', 'animal'),
    'backpack': ('sac a dos', 'accessoire'),
    'umbrella': ('parapluie', 'accessoire'),
    'handbag': ('sac a main', 'accessoire'),
    'tie': ('cravate', 'vetement'),
    'bottle': ('bouteille', 'cuisine'),
    'wine glass': ('verre a vin', 'cuisine'),
    'cup': ('tasse', 'cuisine'),
    'fork': ('fourchette', 'cuisine'),
    'knife': ('couteau', 'cuisine'),
    'spoon': ('cuillere', 'cuisine'),
    'bowl': ('bol', 'cuisine'),
    'banana': ('banane', 'aliment'),
    'apple': ('pomme', 'aliment'),
    'sandwich': ('sandwich', 'aliment'),
    'orange': ('orange', 'aliment'),
    'broccoli': ('brocoli', 'aliment'),
    'carrot': ('carotte', 'aliment'),
    'pizza': ('pizza', 'aliment'),
    'donut': ('donut', 'aliment'),
    'cake': ('gateau', 'aliment'),
    'chair': ('chaise', 'meuble'),
    'couch': ('canape', 'meuble'),
    'dining table': ('table', 'meuble'),
    'tv': ('television', 'electronique'),
    'laptop': ('ordinateur', 'electronique'),
    'cell phone': ('telephone', 'electronique'),
    'microwave': ('micro-ondes', 'electromenager'),
    'refrigerator': ('refrigerateur', 'electromenager'),
    'book': ('livre', 'culture'),
    'teddy bear': ('jouet', 'jouet'),
    # Classes du modèle fine-tuné
    'cell_phone': ('telephone portable', 'electronique'),
    'cracker': ('biscuit aperitif', 'aliment'),
}

def translate_class(name_en, cls_id=None):
    """Traduit un nom de classe anglais en français.
    Priorité au nom de classe, car un modèle fine-tuné peut réutiliser les IDs COCO."""
    if name_en:
        name_lower = name_en.lower()
        if name_lower in CLASS_TRANSLATIONS:
            return CLASS_TRANSLATIONS[name_lower]
    if cls_id is not None and cls_id in COCO_CLASSES_FR:
        return COCO_CLASSES_FR[cls_id]
    return (name_en or 'objet', 'objet')

corrections = []
corrections_lock = threading.Lock()

PRODUCT_DB_KEYWORDS = {
    'gateau': ['lu', 'petit beurre', 'cookies', 'biscuit', 'cake', 'gateau', 'prince', 'BN', 'speculoos'],
    'chips': ['lays', ' chips', 'pringles', 'crisp', 'frito', 'vico', 'brets'],
    'boisson': ['coca', 'pepsi', 'eau', 'jus', 'soda', 'evian', 'perrier', 'vittel', 'fanta', 'sprite', 'capri-sun'],
    'cereales': ['kellogg', 'cereales', 'corn flakes', 'muesli', 'quaker'],
    'lait': ['lait', 'creme', 'yaourt', 'fromage', 'vache qui rit', 'la vache qui rit', '总统', 'président'],
    'pain': ['pain', 'baguette', 'croissant', 'brioche', 'harrys'],
    'conserve': ['bonduelle', 'cuillier', 'petit pois', 'haricots', 'conserve', 'cougat'],
    'hygiene': ['shampoo', 'shampoing', 'savon', 'dentifrice', 'brosse', 'dove', 'niveaux'],
    'entretien': ['lessive', 'nettoyant', 'javex', 'javel', 'ajax'],
    'viande': ['poulet', 'boeuf', 'porc', 'jambon', 'saucisse', 'herta', 'hachoir'],
    'poisson': ['thon', 'saumon', 'poisson', 'john west', 'petit navire'],
    'boite': ['boite', 'barquette', 'contenant'],
}


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


def get_texture_info(crop):
    if crop is None or crop.size == 0:
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_mean = hsv[:, :, 1].mean()
    val_mean = hsv[:, :, 2].mean()
    return {
        'edge_density': round(float(edge_density), 3),
        'saturation': round(float(sat_mean), 1),
        'brightness': round(float(val_mean), 1),
        'is_textured': bool(edge_density > 0.15),
        'is_colorful': bool(sat_mean > 80),
    }


def extract_prices(texts):
    prices = []
    price_patterns = [
        r'(\d+[.,]\d{2})\s*[\$\u20ac]',
        r'[\$\u20ac]\s*(\d+[.,]\d{2})',
        r'(\d+[.,]\d{2})\s*(?:eur|euros?)',
        r'(\d+[.,]\d{2})\s*(?:xof|fcfa|cfa)',
        r'prix\s*:\s*(\d+[.,]\d{2})',
        r'(\d{1,4}[.,]\d{2})',
    ]
    for text in texts:
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                try:
                    val = float(m.replace(',', '.'))
                    if 0.01 < val < 100000:
                        prices.append({'value': val, 'text': text, 'raw': m})
                except ValueError:
                    pass
    return prices


def identify_product_from_texts(texts, barcodes=None):
    all_text = ' '.join(t['text'].lower() for t in texts)

    for bc in (barcodes or []):
        barcode_data = bc.get('data', '')
        if barcode_data:
            if barcode_data in products_db:
                return {
                    'source': 'barcode_db',
                    'product': products_db[barcode_data],
                    'confidence': 1.0,
                }
            else:
                # Code-barre inconnu -> proposer pour auto-sauvegarde
                return {
                    'source': 'barcode_scan',
                    'product': {
                        'name': f'Article {barcode_data[-8:]}',
                        'category': '',
                        'price': 0,
                        'barcode': barcode_data,
                    },
                    'confidence': 1.0,
                }

    for category, keywords in PRODUCT_DB_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in all_text:
                return {
                    'source': 'text_match',
                    'product': {'category': category, 'matched_keyword': kw, 'raw_text': all_text},
                    'confidence': 0.7,
                }

    if texts:
        best_text = max(texts, key=lambda t: t['confidence'])
        if best_text['confidence'] > 0.5:
            return {
                'source': 'ocr_read',
                'product': {'name': best_text['text'], 'raw_text': all_text},
                'confidence': best_text['confidence'],
            }

    return None


# Cache pour éviter de sauvegarder le même produit 50 fois par seconde
_auto_saved_products = set()


def _auto_save_product(product_info, ocr_texts):
    """Enregistre automatiquement un produit détecté dans products_db."""
    global products_db
    try:
        source = product_info.get('source', '')
        product = product_info.get('product', {})

        if source == 'barcode_db':
            # Déjà connu, pas besoin de resauvegarder
            return

        if source == 'barcode_scan':
            # Si on a un code-barre associé, l'utiliser comme clé
            # product_info devrait contenir le barcode dans le product
            barcode = product.get('barcode', '')
            if barcode and barcode not in products_db and barcode not in _auto_saved_products:
                products_db[barcode] = {
                    'name': product.get('name', f'Article {barcode[-6:]}'),
                    'category': product.get('category', 'non classé'),
                    'price': product.get('price', 0),
                    'barcode': barcode,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'auto_saved': True,
                }
                _auto_saved_products.add(barcode)
                save_products_db()
                log.info(f"Produit auto-sauvegarde (code-barre): {barcode}")

        elif source == 'text_match':
            category = product.get('category', '')
            matched = product.get('matched_keyword', '')
            key = f"auto_{category}_{matched}".lower().replace(' ', '_')
            if key not in products_db and key not in _auto_saved_products:
                products_db[key] = {
                    'name': matched.strip(),
                    'category': category,
                    'price': 0,
                    'barcode': '',
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'auto_saved': True,
                }
                _auto_saved_products.add(key)
                save_products_db()
                log.info(f"Produit auto-sauvegarde (texte): {key}")
    except Exception as e:
        log.error(f"Erreur auto-save produit: {e}")


def run_ocr_fullframe(frame):
    if ocr_reader is None:
        return [], 0
    try:
        t0 = time.time()
        results = ocr_reader.readtext(
            frame,
            detail=1,
            paragraph=False,
            width_ths=0.7,
            text_threshold=0.5,
            low_text=0.3,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,- xX%:;/()',
        )
        elapsed = (time.time() - t0) * 1000

        texts = []
        for (bbox, text, conf) in results:
            if conf > 0.15 and len(text.strip()) > 1:
                x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                texts.append({
                    'text': text.strip(),
                    'confidence': round(conf, 3),
                    'bbox': [x1, y1, x2, y2],
                    'area': (x2-x1) * (y2-y1),
                })

        texts.sort(key=lambda t: t['area'], reverse=True)
        texts = ocr_filter.filter(texts)
        log.debug(f"OCR: {len(texts)} textes ({elapsed:.0f}ms)")
        return texts, elapsed
    except Exception as e:
        log.error(f"OCR erreur: {e}")
        return [], 0


def run_barcode(frame):
    if not pyzbar_available:
        return [], 0
    try:
        from pyzbar import pyzbar
        t0 = time.time()
        barcodes = pyzbar.decode(frame)
        elapsed = (time.time() - t0) * 1000

        results = []
        for bc in barcodes:
            x, y, w, h = bc.rect
            data = bc.data.decode('utf-8', errors='replace')
            results.append({
                'type': bc.type,
                'data': data,
                'bbox': [x, y, x + w, y + h],
                'in_db': data in products_db,
            })

        log.debug(f"Barcodes: {len(results)} ({elapsed:.0f}ms)")
        return results, elapsed
    except Exception as e:
        log.error(f"Barcode erreur: {e}")
        return [], 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cameras')
def cameras_page():
    return render_template('cameras.html')


@app.route('/detect', methods=['POST'])
def detect():
    global last_vlm_caption, last_vlm_frame_time
    frame_start = time.time()

    # Support JSON (ancien) et FormData (nouvel index)
    if request.files:
        file = request.files.get('image')
        use_ocr = request.form.get('ocr', 'true') == 'true'
        use_barcode = request.form.get('barcode', 'true') == 'true'
        use_yolo = request.form.get('yolo', 'true') == 'true'
        if file:
            img_bytes = file.read()
        else:
            return jsonify({'error': 'Image manquante'}), 400
    else:
        data = request.json or {}
        img_b64 = data.get('image', '')
        use_ocr = data.get('ocr', True)
        use_barcode = data.get('barcode', True)
        use_yolo = data.get('yolo', True)
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        try:
            img_bytes = base64.b64decode(img_b64)
        except Exception:
            return jsonify({'error': 'Image invalide'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Image non decodable'}), 400

    h_frame, w_frame = frame.shape[:2]

    ocr_all = []
    ocr_latency = 0
    if use_ocr and ocr_reader:
        ocr_all, ocr_latency = run_ocr_fullframe(frame)

    barcodes_all = []
    barcode_latency = 0
    if use_barcode and pyzbar_available:
        barcodes_all, barcode_latency = run_barcode(frame)

    detections = []
    if use_yolo:
        try:
            results = yolo(frame, conf=0.30, verbose=False,
                           device='cuda:0' if torch.cuda.is_available() else 'cpu')
            raw_detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    crop = frame[max(0,y1):min(h_frame,y2), max(0,x1):min(w_frame,x2)]

                    fr_name, category = translate_class(r.names[cls_id], cls_id)

                    with corrections_lock:
                        for c in corrections:
                            if c['from'] == fr_name or c.get('from_en') == r.names[cls_id]:
                                fr_name = c['to']
                                break

                    color_desc = get_dominant_color(crop)
                    texture = get_texture_info(crop)

                    det_texts = []
                    if use_ocr and ocr_reader and crop.size > 200:
                        crop_texts, _ = run_ocr_fullframe(crop)
                        det_texts = [t['text'] for t in crop_texts]

                    desc = f"{fr_name} {color_desc}"
                    if det_texts:
                        desc += f" -- {' / '.join(det_texts[:3])}"

                    raw_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': fr_name,
                        'class_en': r.names[cls_id],
                        'category': category,
                        'confidence': round(conf, 3),
                        'description': desc,
                        'color': color_desc,
                        'texture': texture,
                        'texts': det_texts,
                        'source': 'yolo',
                    })
            detections = detection_tracker.update(raw_detections)
        except Exception as e:
            log.error(f"YOLO erreur: {e}")

    # Agrégation OCR multi-frame
    ocr_aggregator.add(ocr_all)
    ocr_all = ocr_aggregator.aggregate()

    prices = extract_prices([t['text'] for t in ocr_all])

    product_info = identify_product_from_texts(ocr_all, barcodes_all)

    # Enregistrement automatique des articles détectés
    if AUTO_SAVE_PRODUCTS and product_info:
        _auto_save_product(product_info, ocr_all)

    # Description VLM périodique (live captioning)
    vlm_caption = None
    try:
        now = time.time()
        with vlm_caption_lock:
            if vlm_orchestrator.active is not None and (now - last_vlm_frame_time) >= vlm_caption_interval:
                vlm_caption = vlm_orchestrator.describe(frame)
                if vlm_caption:
                    last_vlm_caption = vlm_caption
                    last_vlm_frame_time = now
        if vlm_caption is None:
            vlm_caption = last_vlm_caption
    except Exception as e:
        log.error(f"VLM live caption erreur: {e}")
        vlm_caption = last_vlm_caption

    if product_info and product_info.get('source') == 'barcode_db':
        product = product_info['product']
        for d in detections:
            if not d.get('product'):
                d['product'] = product
                d['source'] = 'barcode_db'

    frame_latency = (time.time() - frame_start) * 1000

    with session_lock:
        session_log['frames_processed'] += 1
        session_log['total_detections'] += len(detections)
        session_log['ocr_reads'] += len(ocr_all)
        session_log['barcodes_found'] += len(barcodes_all)
        if product_info:
            session_log['products_recognized'] += 1
        if ocr_all:
            session_log['ocr_results'].extend(ocr_all[-20:])
        if barcodes_all:
            session_log['barcode_results'].extend(barcodes_all[-5:])
        session_log['performance'].append({
            'frame': session_log['frames_processed'],
            'latency_ms': round(frame_latency, 1),
            'detections': len(detections),
            'ocr_texts': len(ocr_all),
            'barcodes': len(barcodes_all),
            'product': product_info is not None,
            'timestamp': time.time(),
        })

    # Coach AI : apprend ce que tu fais
    try:
        coach.record_detections(detections)
    except Exception:
        pass

    # Auto-export de la session toutes les X secondes
    _maybe_auto_export()

    return jsonify({
        'detections': detections,
        'count': len(detections),
        'frame': session_log['frames_processed'],
        'latency_ms': round(frame_latency, 1),
        'ocr': ocr_all,
        'ocr_latency_ms': round(ocr_latency, 1),
        'barcodes': barcodes_all,
        'barcode_latency_ms': round(barcode_latency, 1),
        'prices': prices,
        'product': product_info,
        'vlm_caption': vlm_caption,
    })


@app.route('/product/save', methods=['POST'])
def save_product():
    data = request.json
    barcode = data.get('barcode', '')
    name = data.get('name', '')
    category = data.get('category', '')
    price = data.get('price', 0)
    if not barcode and not name:
        return jsonify({'error': 'barcode ou name requis'}), 400
    key = barcode or hashlib.md5(name.encode()).hexdigest()[:12]
    products_db[key] = {
        'name': name,
        'category': category,
        'price': price,
        'barcode': barcode,
        'first_seen': datetime.now().isoformat(),
        'last_seen': datetime.now().isoformat(),
    }
    save_products_db()
    log.info(f"Produit sauvegarde: {name} ({key})")
    return jsonify({'status': 'ok', 'key': key})


@app.route('/product/save-from-ocr', methods=['POST'])
def save_product_from_ocr():
    data = request.json
    texts = data.get('texts', [])
    barcode = data.get('barcode', '')
    all_text = ' '.join(texts)
    name = data.get('name', all_text[:50] if all_text else 'produit inconnu')
    price = data.get('price', 0)
    key = barcode or hashlib.md5(name.encode()).hexdigest()[:12]
    products_db[key] = {
        'name': name,
        'texts': texts,
        'barcode': barcode,
        'price': price,
        'first_seen': datetime.now().isoformat(),
        'last_seen': datetime.now().isoformat(),
    }
    save_products_db()
    return jsonify({'status': 'ok', 'key': key})


@app.route('/products')
def list_products():
    return jsonify(products_db)


@app.route('/products/delete', methods=['POST'])
def delete_product():
    data = request.json
    key = data.get('key', '')
    if key in products_db:
        del products_db[key]
        save_products_db()
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'non trouve'}), 404


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
        corr = {'from': data['from'], 'to': data['to'], 'from_en': data.get('from_en', ''), 'timestamp': time.time()}
        corrections.append(corr)
        with session_lock:
            session_log['corrections'].append(corr)
    log.info(f"Correction: {data['from']} -> {data['to']}")
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

    log.info(f"Fine-tune: {label} (#{count+1})")
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
                'ocr_reads': session_log['ocr_reads'],
                'barcodes_found': session_log['barcodes_found'],
                'products_recognized': session_log['products_recognized'],
                'corrections_count': len(session_log['corrections']),
                'avg_latency_ms': round(np.mean([p['latency_ms'] for p in session_log['performance']]) if session_log['performance'] else 0, 1),
            },
            'products_db': products_db,
            'corrections': session_log['corrections'],
            'ocr_results': session_log['ocr_results'][-200:],
            'barcode_results': session_log['barcode_results'][-50:],
            'performance': session_log['performance'][-500:],
        }
    filepath = os.path.join(LOGS_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    return send_file(filepath, as_attachment=True, download_name=os.path.basename(filepath))


@app.route('/stats')
def stats():
    with session_lock:
        perf = session_log['performance']
        return jsonify({
            'gpu': 'RTX 4060 Ti' if torch.cuda.is_available() else 'CPU',
            'model': session_log['config'].get('model', 'YOLOv8'),
            'ocr': 'EasyOCR' if ocr_reader else 'none',
            'barcode': 'pyzbar' if pyzbar_available else 'none',
            'products_db': len(products_db),
            'corrections': len(corrections),
            'frames': session_log['frames_processed'],
            'total_detections': session_log['total_detections'],
            'ocr_reads': session_log['ocr_reads'],
            'barcodes_found': session_log['barcodes_found'],
            'products_recognized': session_log['products_recognized'],
            'avg_latency_ms': round(np.mean([p['latency_ms'] for p in perf]), 1) if perf else 0,
            'vlm': vlm_orchestrator.status(),
            'scene_analyzer_ready': scene_analyzer is not None,
        })


def _decode_image_from_request():
    data = request.json
    img_b64 = data.get('image', '')
    if ',' in img_b64:
        img_b64 = img_b64.split(',')[1]
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


@app.route('/analyze', methods=['POST'])
def analyze_scene():
    if scene_analyzer is None:
        return jsonify({'error': 'SceneAnalyzer pas encore chargé, réessaie dans quelques secondes.'}), 503
    frame = _decode_image_from_request()
    if frame is None:
        return jsonify({'error': 'Image invalide'}), 400

    data = request.json or {}
    options = {
        'use_yolo': data.get('yolo', True),
        'use_ocr': data.get('ocr', True),
        'use_barcode': data.get('barcode', True),
        'use_gesture': data.get('gesture', True),
        'use_face': data.get('face', True),
        'use_vlm': data.get('vlm', True),
    }
    try:
        result, annotated = scene_analyzer.analyze(frame, options=options)
        return jsonify({
            'analysis': result,
            'annotated_image': _encode_image(annotated),
        })
    except Exception as e:
        log.error(f"/analyze erreur: {e}")
        return jsonify({'error': str(e)}), 500


def _encode_image(frame_bgr):
    _, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')


@app.route('/scan', methods=['POST'])
def scan_object():
    if scene_analyzer is None:
        return jsonify({'error': 'SceneAnalyzer pas encore chargé.'}), 503
    frame = _decode_image_from_request()
    if frame is None:
        return jsonify({'error': 'Image invalide'}), 400
    try:
        summary = scene_analyzer.scan_object(frame)
        return jsonify(summary)
    except Exception as e:
        log.error(f"/scan erreur: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/scan/reset', methods=['POST'])
def reset_scan():
    if scene_analyzer is None:
        return jsonify({'error': 'SceneAnalyzer pas encore chargé.'}), 503
    scene_analyzer.reset_scan()
    return jsonify({'status': 'ok'})


@app.route('/vlm/status', methods=['GET'])
def vlm_status():
    return jsonify(vlm_orchestrator.status())


@app.route('/vlm/load', methods=['POST'])
def vlm_load():
    if vlm_orchestrator.active is not None:
        return jsonify({'status': 'already_loaded', 'active': type(vlm_orchestrator.active).__name__})
    def _load():
        try:
            vlm_orchestrator.load(prefer='auto')
        except Exception as e:
            log.error(f"VLM load on demand error: {e}")
    threading.Thread(target=_load, daemon=True).start()
    return jsonify({'status': 'loading'})


@app.route('/describe', methods=['POST'])
def describe_image():
    if scene_analyzer is None or scene_analyzer.vlm is None or scene_analyzer.vlm.active is None:
        return jsonify({'error': 'VLM non disponible'}), 503
    frame = _decode_image_from_request()
    if frame is None:
        return jsonify({'error': 'Image invalide'}), 400
    data = request.json or {}
    question = data.get('question', 'Décris ce que tu vois en français.')
    try:
        answer = scene_analyzer.vlm.answer(frame, question)
        return jsonify({'question': question, 'answer': answer})
    except Exception as e:
        log.error(f"/describe erreur: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    global retrain_process
    with retrain_lock:
        if retrain_status['status'] == 'running':
            return jsonify({'status': 'running', 'message': 'Entraînement déjà en cours'})
    t = threading.Thread(target=_run_training, daemon=True)
    t.start()
    return jsonify({'status': 'started', 'message': 'Entraînement lancé en arrière-plan'})


@app.route('/retrain/status', methods=['GET'])
def retrain_status_route():
    with retrain_lock:
        return jsonify(retrain_status)


# --- Routes multi-caméras ---
@app.route('/cameras', methods=['GET'])
def list_cameras():
    return jsonify(camera_manager.list_cameras())


@app.route('/cameras', methods=['POST'])
def add_camera():
    data = request.json or {}
    source = data.get('source')
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'name requis'}), 400
    if source is None or source == '':
        return jsonify({'error': 'source requis'}), 400
    # Convertir les sources numériques en int
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    options = {
        'use_yolo': data.get('yolo', True),
        'use_ocr': data.get('ocr', True),
        'use_barcode': data.get('barcode', False),
        'use_gesture': data.get('gesture', False),
        'use_face': data.get('face', False),
        'use_vlm': data.get('vlm', False),
    }
    ok, msg = camera_manager.add_camera(source, name, options)
    if ok:
        return jsonify({'status': 'ok', 'name': name})
    return jsonify({'error': msg}), 400


@app.route('/cameras/<name>/remove', methods=['POST'])
def remove_camera(name):
    if camera_manager.remove_camera(name):
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'camera non trouvee'}), 404


@app.route('/cameras/<name>/latest', methods=['GET'])
def camera_latest(name):
    cam = camera_manager.get_camera(name)
    if cam is None:
        return jsonify({'error': 'camera non trouvee'}), 404
    data = cam.get_latest()
    # Alimenter le coach avec les détections
    if data and data.get('analysis'):
        try:
            coach.record_detections(data['analysis'].get('objects', []))
        except Exception:
            pass
    return jsonify(data)


@app.route('/cameras/<name>/events', methods=['GET'])
def camera_events(name):
    cam = camera_manager.get_camera(name)
    if cam is None:
        return jsonify({'error': 'camera non trouvee'}), 404
    limit = request.args.get('limit', 100, type=int)
    return jsonify(cam.get_events(limit=limit))


@app.route('/cameras/stop_all', methods=['POST'])
def stop_all_cameras():
    camera_manager.stop_all()
    return jsonify({'status': 'ok'})


# --- Endpoints Enregistreur d'activite ---
@app.route('/record/start', methods=['POST'])
def record_start():
    """Demarre l'enregistrement d'activite (screen + webcam)."""
    if LIGHT_MODE:
        return jsonify({'status': 'error', 'message': 'Enregistrement désactivé en mode léger'}), 503
    global recorder
    if recorder is None:
        recorder = ActivityRecorder(
            output_dir=os.path.join(os.path.dirname(__file__), 'recordings'),
            screen_fps=5,
            webcam_fps=5,
            yolo_model_path=os.path.join(os.path.dirname(__file__), 'best_model.pt'),
        )
    ok = recorder.start()
    if ok:
        return jsonify({'status': 'ok', 'action': 'started', 'session': recorder.get_status()})
    return jsonify({'status': 'error', 'message': 'deja en cours'}), 400


@app.route('/record/stop', methods=['POST'])
def record_stop():
    """Arrete l'enregistrement d'activite."""
    if LIGHT_MODE:
        return jsonify({'status': 'error', 'message': 'Enregistrement désactivé en mode léger'}), 503
    global recorder
    if recorder is None:
        return jsonify({'status': 'error', 'message': 'aucun enregistrement en cours'}), 400
    recorder.stop()
    status = recorder.get_status()
    return jsonify({'status': 'ok', 'action': 'stopped', 'session': status})


@app.route('/record/status', methods=['GET'])
def record_status():
    """Retourne le statut de l'enregistreur d'activite."""
    if LIGHT_MODE:
        return jsonify({'recording': False, 'session_id': None, 'mode': 'light'})
    global recorder
    if recorder is None:
        return jsonify({'recording': False, 'session_id': None})
    return jsonify(recorder.get_status())


@app.route('/record/sessions', methods=['GET'])
def record_sessions():
    """Liste toutes les sessions enregistrees."""
    recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
    sessions = []
    if os.path.exists(recordings_dir):
        for name in sorted(os.listdir(recordings_dir), reverse=True):
            full = os.path.join(recordings_dir, name)
            if os.path.isdir(full):
                meta = os.path.join(full, 'detections.json')
                if os.path.exists(meta):
                    try:
                        with open(meta, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        sessions.append(data)
                    except Exception:
                        sessions.append({'session_id': name, 'error': 'metadata unavailable'})
                else:
                    sessions.append({'session_id': name, 'detections': 'no metadata'})
    return jsonify({'sessions': sessions, 'count': len(sessions)})


@app.route('/record/download/<session_id>', methods=['GET'])
def record_download(session_id):
    """Telecharge une session enregistree en ZIP."""
    import zipfile
    from io import BytesIO
    session_dir = os.path.join(os.path.dirname(__file__), 'recordings', session_id)
    if not os.path.exists(session_dir):
        return jsonify({'error': 'session not found'}), 404

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(session_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, session_dir)
                zf.write(fpath, arcname)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'recording_{session_id}.zip',
    )


# --- Endpoints Video + Audio ---
@app.route('/record/export/<session_id>', methods=['POST'])
def record_export_mp4(session_id):
    """Convertit une session enregistree en fichiers MP4."""
    fps = request.json.get('fps') if request.is_json else None
    if fps is None and request.is_json:
        fps = request.json.get('fps')
    meta, err = video_exporter.session_to_mp4(session_id, fps=fps)
    if err:
        return jsonify({'error': err}), 404
    return jsonify({'status': 'ok', 'export': meta})


@app.route('/record/zip/<session_id>', methods=['GET'])
def record_download_zip(session_id):
    """Telecharge une session enregistree en ZIP."""
    zip_path, err = video_exporter.session_to_zip(session_id)
    if err:
        return jsonify({'error': err}), 404
    return send_file(zip_path, mimetype='application/zip', as_attachment=True,
                     download_name=f'recording_{session_id}.zip')


@app.route('/record/analyze_video/<session_id>', methods=['POST'])
def record_analyze_video(session_id):
    """Analyse les frames video d'une session avec le VLM pour en faire une description."""
    session_dir = os.path.join(os.path.dirname(__file__), 'recordings', session_id)
    screen_dir = os.path.join(session_dir, 'screen')
    if not os.path.exists(screen_dir):
        return jsonify({'error': 'screen frames not found'}), 404

    max_frames = request.json.get('max_frames', 30) if request.is_json else 30
    analysis = video_exporter.analyze_video_frames(screen_dir, vlm_orchestrator, max_frames=max_frames)
    return jsonify({'status': 'ok', 'analysis': analysis})


@app.route('/audio/transcribe', methods=['POST'])
def audio_transcribe():
    """Transcrit un fichier audio en texte via Whisper."""
    if LIGHT_MODE or audio_analyzer is None:
        return jsonify({'status': 'error', 'message': 'Analyse audio désactivée en mode léger'}), 503
    data = request.json or {}
    audio_path = data.get('path')
    model_size = data.get('model', 'base')
    if not audio_path:
        return jsonify({'error': 'path required'}), 400
    transcript = audio_analyzer.analyze_file(audio_path, model_size=model_size)
    return jsonify({'status': 'ok', 'transcript': transcript})


@app.route('/audio/analyze', methods=['POST'])
def audio_analyze():
    """Analyse un texte de transcription pour en extraire un resume."""
    if LIGHT_MODE or audio_analyzer is None:
        return jsonify({'status': 'error', 'message': 'Analyse audio désactivée en mode léger'}), 503
    data = request.json or {}
    text = data.get('text', '')
    prompt = data.get('prompt')
    result = audio_analyzer.analyze_text(text, prompt=prompt)
    return jsonify({'status': 'ok', 'analysis': result})


# --- Endpoints Orchestrateur d'agents IA ---
def _agents_disabled_response():
    return jsonify({'status': 'error', 'message': 'Orchestrateur IA désactivé en mode léger'}), 503

@app.route('/agents', methods=['GET'])
def agents_list():
    """Liste tous les agents et leur statut."""
    if LIGHT_MODE or orchestrator is None:
        return _agents_disabled_response()
    return jsonify({
        'agents': list(orchestrator.agents.keys()),
        'health': orchestrator.health_check(),
    })

@app.route('/agents/task', methods=['POST'])
def submit_task():
    """Soumet une tache a un agent specifique."""
    if LIGHT_MODE or orchestrator is None:
        return _agents_disabled_response()
    data = request.json or {}
    task_type = data.get('task_type', 'unknown')
    agent = data.get('agent')
    payload = data.get('payload', {})
    priority = data.get('priority', 0)

    if not agent or agent not in orchestrator.agents:
        return jsonify({'error': f'Agent not found: {agent}'}), 400

    task_id = orchestrator.submit(task_type, agent, payload, priority=priority)
    return jsonify({'status': 'ok', 'task_id': task_id})

@app.route('/agents/task/<task_id>', methods=['GET'])
def get_task(task_id):
    """Retourne le statut d'une tache."""
    if LIGHT_MODE or orchestrator is None:
        return _agents_disabled_response()
    task = orchestrator.get_task(task_id)
    if task is None:
        return jsonify({'error': 'task not found'}), 404
    return jsonify(task.to_dict())

@app.route('/agents/tasks', methods=['GET'])
def list_tasks():
    """Liste toutes les taches."""
    if LIGHT_MODE or orchestrator is None:
        return _agents_disabled_response()
    status = request.args.get('status')
    tasks = orchestrator.get_tasks(status=status)
    return jsonify({'tasks': [t.to_dict() for t in tasks], 'count': len(tasks)})

@app.route('/agents/run', methods=['POST'])
def run_agent():
    """Execute une action specifique sur un agent et attend le resultat."""
    if LIGHT_MODE or orchestrator is None:
        return _agents_disabled_response()
    data = request.json or {}
    agent = data.get('agent')
    payload = data.get('payload', {})
    timeout = data.get('timeout', 60)

    if not agent or agent not in orchestrator.agents:
        return jsonify({'error': f'Agent not found: {agent}'}), 400

    task_type = payload.get('action', 'run')
    task_id = orchestrator.submit(task_type, agent, payload)
    if task_id is None:
        return jsonify({'error': 'Failed to submit task'}), 500

    result = orchestrator.submit_and_wait(task_type, agent, payload, timeout=timeout)
    if result:
        return jsonify({'status': 'ok', 'task_id': task_id, 'result': result.to_dict()})
    return jsonify({'status': 'timeout', 'task_id': task_id})

@app.route('/agents/memory', methods=['GET'])
def get_memory():
    """Liste les cles de memoire de l'orchestrateur."""
    if LIGHT_MODE or orchestrator is None:
        return _agents_disabled_response()
    keys = orchestrator.memory.list_keys()
    return jsonify({'memory_keys': keys})

@app.route('/agents/memory/<key>', methods=['GET'])
def get_memory_key(key):
    """Retourne une valeur de memoire par cle."""
    if LIGHT_MODE or orchestrator is None:
        return _agents_disabled_response()
    value = orchestrator.get_memory(key)
    return jsonify({'key': key, 'value': value})

@app.route('/agents/status', methods=['GET'])
def orchestrator_status():
    """Status complet de l'orchestrateur."""
    if LIGHT_MODE or orchestrator is None:
        return _agents_disabled_response()
    return jsonify(orchestrator.health_check())


# --- PWA & landing page ---
LANDING_DIR = os.path.join(os.path.dirname(__file__), 'landing')

@app.route('/')
def landing_page():
    return send_file(os.path.join(LANDING_DIR, 'index.html'))

@app.route('/app')
def app_page():
    return render_template('index.html')

@app.route('/manifest.json')
def manifest():
    return send_file(os.path.join(LANDING_DIR, 'manifest.json'), mimetype='application/json')

@app.route('/sw.js')
def service_worker():
    return send_file(os.path.join(LANDING_DIR, 'sw.js'), mimetype='application/javascript')

@app.route('/landing/<path:filename>')
def landing_assets(filename):
    return send_file(os.path.join(LANDING_DIR, filename))


@app.route('/api/license', methods=['GET', 'POST'])
def api_license():
    if request.method == 'GET':
        return jsonify(license_manager.get_plan())
    data = request.json or {}
    key = data.get('key', '')
    email = data.get('email', '')
    if not key or not email:
        return jsonify({'success': False, 'error': 'Clé et email requis'}), 400
    return jsonify(license_manager.activate_license(key, email))


@app.route('/coach')
def coach_suggest():
    return jsonify(coach.get_coach_data())


if __name__ == '__main__':
    log.info("Serveur demarre: http://localhost:5000")
    print("\n  Systeme autonome pret")
    print("  OCR en primaire + YOLO + Barcodes + Base locale")
    print("  http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
