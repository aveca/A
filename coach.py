import time
import json
from collections import defaultdict, deque

# Mémoire des détections sur les 30 dernières secondes
DETECTION_HISTORY = deque(maxlen=300)
STATS_WINDOW = 30  # secondes

# Mapping objet → activité probable
ACTIVITY_MAP = {
    'bottle': 'boit / prépare une boisson',
    'cell_phone': 'utilise son téléphone',
    'cracker': 'mange un en-cas',
    'e-cig': 'vapote',
    'person': 'se déplace',
    'book': 'lit',
    'laptop': 'travaille sur ordinateur',
    'keyboard': 'tape au clavier',
    'mouse': 'utilise la souris',
    'cup': 'boit',
    'glass': 'boit',
    'plate': 'mange',
    'food': 'mange',
    'tv': 'regarde la télévision',
    'remote': 'regarde / contrôle la télé',
    'door': 'ouvre/ferme une porte',
    'chair': 'assis',
    'couch': 'assis / se repose',
}

# Conseils par activité
SUGGESTIONS = {
    'boit / prépare une boisson': 'Pense à t\'hydrater régulièrement.',
    'utilise son téléphone': 'Fais une pause écran toutes les 20 minutes.',
    'mange un en-cas': 'Bien. Essaye de privilégier des fruits.',
    'vapote': 'Pense à aérer la pièce.',
    'se déplace': 'Bouger c\'est bon. Profites-en pour t\'étirer.',
    'lit': 'Bonne lecture. Alterne avec une pause yeux fermés.',
    'travaille sur ordinateur': 'Rappel : lève-toi toutes les heures.',
    'tape au clavier': 'Garde les poignets droits.',
    'utilise la souris': 'Change de main si possible.',
    'mange': 'Prends le temps de mâcher lentement.',
    'assis': 'Pense à te lever 5 min toutes les heures.',
}


def get_default_context():
    return {
        'start_time': time.time(),
        'activities': defaultdict(int),
        'last_activity': 'Rien',
        'last_suggestion': 'Analyse en cours...',
        'history': DETECTION_HISTORY,
    }


def record_detections(objects):
    """Enregistre les objets détectés avec timestamp"""
    now = time.time()
    for obj in objects:
        label = obj.get('label', 'inconnu').lower()
        DETECTION_HISTORY.append((now, label))


def infer_activity():
    """Trouve l'activité dominante dans la fenêtre de temps"""
    if not DETECTION_HISTORY:
        return 'Rien'

    now = time.time()
    cutoff = now - STATS_WINDOW
    counts = defaultdict(int)

    for ts, label in DETECTION_HISTORY:
        if ts >= cutoff:
            counts[label] += 1

    if not counts:
        return 'Rien'

    # Activité la plus fréquente
    top_label = max(counts, key=counts.get)
    activity = ACTIVITY_MAP.get(top_label, f'interagit avec {top_label}')
    return activity


def get_suggestion(activity):
    """Retourne le conseil associé à l'activité"""
    base = SUGGESTIONS.get(activity, 'Continue. Tout va bien.')
    # Ajouter du contexte temporel
    if activity == 'Rien':
        return 'Va dans le champ de la caméra pour être détecté.'
    return base


def get_coach_data():
    """Point d'entrée pour /coach"""
    activity = infer_activity()
    suggestion = get_suggestion(activity)
    return {
        'activity': activity,
        'suggestion': suggestion,
    }
