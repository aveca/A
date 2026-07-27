# Vision Agent — Vision par Ordinateur Locale & Intelligente

Vision Agent est une application Windows de vision par ordinateur qui fonctionne **100 % en local** : détection d'objets, OCR, codes-barres, reconnaissance faciale, enregistrement vidéo et auto-apprentissage.

> **Un seul bouton. Lance. Détecte. Apprend.**

---

## Fonctionnalités clés

- **Détection d'objets temps réel** — YOLOv8, GPU & CPU
- **OCR** — lecture de texte en français et anglais
- **Codes-barres** — scan EAN/UPC/QR
- **Reconnaissance faciale & gestes** — MediaPipe
- **Enregistrement vidéo & écran** — sauvegarde automatique
- **Auto-apprentissage** — collecte d'images et réentraînement automatique
- **Base de produits locale** — JSON, exportable
- **Rapports d'analyse automatiques** — sessions, use cases, recommandations
- **PWA mobile** — accès depuis smartphone/tablette sur le même réseau
- **Mode optimisé** — démarrage rapide, VLM désactivé
- **Mode complet** — VLM (description d'images) activé

---

## Installation rapide

1. Télécharge `VisionAgent_Setup.exe`
2. Double-clique et suis l'assistant
3. Lance le raccourci **Vision Agent** sur ton bureau
4. Ouvre ton navigateur sur `http://localhost:5000`

---

## Use cases commerciaux

| Secteur | Use case |
|---|---|
| Retail | Caisse automatique, inventaire, contrôle des prix |
| Industrie | Contrôle qualité, traçabilité, comptage pièces |
| Sécurité | Détection d'intrusion, comptage de passage |
| Logistique | Scan codes-barres, suivi colis |
| Santé | Vérification de présence, gestion de stock |
| Smart Home | Surveillance intelligente, détection d'anomalies |

---

## Tarification

| Plan | Prix | Inclus |
|---|---|---|
| **Starter** | Gratuit | 1 caméra, OCR, YOLO, export JSON |
| **Pro** | 49€/mois | Multi-caméras, enregistrement cloud, rapports |
| **Enterprise** | Sur devis | Déploiement multi-site, API, support dédié |

---

## Développement

```bash
# Lancer en développement
python app.py

# Build l'exécutable Windows
python build_exe.py

# Créer l'installateur
.\build_installer.bat
```

---

## Stack technique

- Python 3.11+
- Flask
- YOLOv8 (Ultralytics)
- EasyOCR
- MediaPipe
- PyInstaller
- Inno Setup

---

## Licence

© 2026 Vision Agent. Tous droits réservés.
