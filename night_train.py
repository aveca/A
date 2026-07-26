"""
Night Training Orchestrator — Lance la collecte + training automatiquement
Exécute: python night_train.py
"""
import os
import sys
import time
import subprocess
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  NIGHT TRAINING ORCHESTRATOR")
print(f"  Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

PHASES = [
    {"name": "Phase 1: Collecte de données", "script": "auto_collect.py", "args": ["200"], "timeout": 600},
    {"name": "Phase 2: Fine-tuning YOLOv8x", "script": "train.py", "args": [], "timeout": 21600},
]

log_file = os.path.join(PROJECT_DIR, 'training_log.txt')
with open(log_file, 'w') as f:
    f.write(f"Training started: {datetime.now().isoformat()}\n")

for phase in PHASES:
    print(f"\n{'='*60}")
    print(f"  {phase['name']}")
    print(f"  Début: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    with open(log_file, 'a') as f:
        f.write(f"\n{phase['name']} started: {datetime.now().isoformat()}\n")

    cmd = [sys.executable, os.path.join(PROJECT_DIR, phase['script'])] + phase['args']
    try:
        result = subprocess.run(cmd, cwd=PROJECT_DIR, timeout=phase['timeout'])
        with open(log_file, 'a') as f:
            f.write(f"{phase['name']} finished: exit code {result.returncode}\n")
    except subprocess.TimeoutExpired:
        print(f"  Timeout après {phase['timeout']}s")
        with open(log_file, 'a') as f:
            f.write(f"{phase['name']} timeout after {phase['timeout']}s\n")
    except Exception as e:
        print(f"  Erreur: {e}")
        with open(log_file, 'a') as f:
            f.write(f"{phase['name']} error: {e}\n")

print(f"\n{'='*60}")
print(f"  TRAINING TERMINÉ")
print(f"  Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Log: {log_file}")
print(f"{'='*60}")

with open(log_file, 'a') as f:
    f.write(f"\nTraining completed: {datetime.now().isoformat()}\n")
