# Projet Stellantis Mood Detection

Ce projet est une application de détection de l'état du conducteur (fatigue, inconfort, baillement) développée pour Stellantis. Elle utilise l'analyse faciale en temps réel pour ajuster intelligemment la température de l'habitacle.

## Structure du Projet

- **frontend/** : Application React affichant le retour vidéo, l'état détecté et la température.
- **backend/** : Serveur API Flask (Python) utilisant MediaPipe pour l'analyse faciale.
- **back-stellantis/** : (Legacy) Référentiel MediaPipe original et scripts utilisateurs.
- **front-stellantis/** : (Legacy) Anciens dossiers.

## Installation et Lancement

Pour lancer le projet, vous devez démarrer le backend (API d'analyse) et le frontend (Interface Utilisateur) dans deux terminaux séparés.

### 1. Backend (Serveur Python - Port 8000)

1. Ouvrez un terminal à la racine du projet (`ing4-stellantis-mood-detection-master`).
2. Installez les dépendances nécessaires (créez un environnement virtuel si besoin) :
   ```bash
   pip install -r requirements.txt
   ```
3. Allez dans le dossier `backend` et lancez le serveur :
   ```bash
   cd backend
   python app.py
   ```
   *Laissez ce terminal ouvert.*

### 2. Frontend (Interface React - Port 3000)

1. Ouvrez un **nouveau** terminal à la racine du projet.
2. Allez dans le dossier `frontend` :
   ```bash
   cd frontend
   ```
3. Installez les dépendances (nécessaire uniquement la première fois) :
   ```bash
   npm install
   ```
4. Lancez l'application :
   ```bash
   npm start
   ```

L'application devrait s'ouvrir automatiquement sur [http://localhost:3000](http://localhost:3000).

---

## Fonctionnalités et Utilisation

### Calibration
Au démarrage de la détection, une phase de **calibration de 5 secondes** est nécessaire. 
**Gardez une expression neutre et regardez la caméra** pendant cette période. Cela permet au système de s'adapter à votre visage (écart des yeux, formes, etc.).

### États Détectés
Le système analyse votre visage pour déterminer votre état physique ou émotionnel :
- **NEUTRE** : État normal, conduite standard.
- **SOMNOLENCE** : Détectée si vos yeux se ferment involontairement.
- **BAILLEMENT** : Détecté par une grande ouverture de la bouche.
- **INCONFORT** : Détecté par un froncement des sourcils (signe de gêne ou douleur).

### Ajustement de la Température
La température de l'habitacle est ajustée dynamiquement en fonction de votre état :
- **INCONFORT** : Diminution progressive pour rafraîchir.
- **SOMNOLENCE / BAILLEMENT** : Ajustement pour stimuler la vigilance.
- **NEUTRE** : Maintien d'une température de confort stable.

## Technologies Utilisées
- **Backend** : Python, Flask, OpenCV, MediaPipe (Face Mesh).
- **Frontend** : React.js.
- **Méthodologie** : Analyse des points de repère faciaux (Landmarks) pour calculer l'EAR (Eye Aspect Ratio), le MAR (Mouth Aspect Ratio) et les distances sourcils-yeux.

## Auteur
Projet réalisé dans le cadre du stage ING4 Stellantis.
