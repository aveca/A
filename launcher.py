"""
Launcher pour l'application Object Recognition en .exe.
Démarre le serveur Flask et ouvre le navigateur.
"""
import os
import sys
import time
import threading
import webbrowser
import logging

# Déterminer le répertoire de l'exécutable ou du script
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    INTERNAL_DIR = os.path.join(BASE_DIR, '_internal')
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INTERNAL_DIR = BASE_DIR

os.chdir(BASE_DIR)

# Désactiver les logs verbeux de transformers/httpx
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Rediriger les logs vers un fichier dans le répertoire de l'app
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'launcher.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout if sys.stdout else open(os.devnull, 'w'))
    ]
)
log = logging.getLogger('launcher')


def wait_for_server(url, timeout=60):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    log.info("=" * 60)
    log.info("  Object Recognition System")
    log.info("  Démarre le serveur local...")
    log.info("=" * 60)

    # Importer app.py après avoir changé de répertoire
    import importlib.util
    app_path = os.path.join(INTERNAL_DIR, "app.py")
    if not os.path.exists(app_path):
        app_path = os.path.join(BASE_DIR, "app.py")
    log.info(f"Chargement app.py depuis: {app_path}")
    spec = importlib.util.spec_from_file_location("app", app_path)
    app_module = importlib.util.module_from_spec(spec)
    sys.modules['app'] = app_module
    spec.loader.exec_module(app_module)

    log.info("Application chargée.")
    log.info("URL: http://localhost:5000/app")
    log.info("Cameras: http://localhost:5000/cameras")

    # Ouvrir le navigateur après quelques secondes
    def open_browser():
        time.sleep(4.0)
        if wait_for_server('http://localhost:5000/app', timeout=60):
            log.info("Ouverture du navigateur...")
            webbrowser.open('http://localhost:5000/app')
        else:
            log.warning("Le serveur n'a pas répondu à temps. Ouvrez http://localhost:5000/ manuellement.")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Le serveur démarre dans app_module
    log.info("Serveur en cours d'exécution. Fermez cette fenêtre pour arrêter.")
    app_module.app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f"Erreur fatale: {e}", exc_info=True)
        try:
            input("Appuyez sur Entrée pour fermer...")
        except Exception:
            pass
        sys.exit(1)
