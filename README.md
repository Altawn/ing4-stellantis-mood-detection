# Projet Stellantis Emotion Analysis

Ce projet a été fusionné pour regrouper le frontend et le backend.

## Structure

- **frontend/** : Application React (anciennement `front-stellantis/frontend`).
- **backend/** : Serveur API Flask (Python) gérant l'analyse faciale avec MediaPipe.
- **back-stellantis/** : (Legacy) Référentiel MediaPipe original et scripts utilisateurs.
- **front-stellantis/** : (Legacy) Anciens dossiers.

## 🚀 Lancement du Projet 5

Pour lancer le projet, il faut démarrer le backend (API) et le frontend (Site Web) dans deux terminaux séparés.

### 1. Backend (Serveur Python)
Ce serveur gère l'analyse des émotions et tourne sur le port `8000`.

1. Ouvrez un terminal (PowerShell ou Cmd).
2. Allez dans le dossier `backend` :
   ```bash
   cd backend
   ```
3. Installez les librairies nécessaires :
   ```bash
   pip install -r requirements.txt
   ```
4. Lancez le serveur :
   ```bash
   python app.py
   ```
   *Laissez ce terminal ouvert.*

### 2. Frontend (Interface React)
C'est le site web que vous verrez. Il tourne sur le port `3000`.

1. Ouvrez un **nouveau** terminal.
2. Allez dans le dossier `frontend` :
   ```bash
   cd frontend
   ```
3. Installez les dépendances (une seule fois suffit) :
   ```bash
   npm install
   ```
4. Lancez le site :
   ```bash
   npm start
   ```

Le site devrait s'ouvrir automatiquement sur [http://localhost:3000](http://localhost:3000).

---

## Informations Techniques (IA)

### États détectés
- 🔴 **ENERVE** : Sourcils froncés
- 🟢 **CONTENT** : Présence d'un sourire
- 🟡 **NEUTRE** : Expression normale

### Métriques calculées
- **EAR (Eye Aspect Ratio)** : Mesure l'ouverture des yeux
- **MAR (Mouth Aspect Ratio)** : Mesure l'ouverture de la bouche
- **Brow Score** : Inclinaison des sourcils (négatif = énervé)
- **Smile Score** : Position des coins de la bouche (positif = sourire)

## Auteur
Projet réalisé dans le cadre du stage ING4 Stellantis.
