# Détection d'Humeur en Temps Réel

Ce projet utilise **MediaPipe Face Mesh** pour détecter l'état émotionnel d'une personne via webcam en temps réel.

## États détectés

- 🔴 **ENERVE** : Sourcils froncés
- 🟢 **CONTENT** : Présence d'un sourire
- 🟡 **NEUTRE** : Expression normale

## Métriques calculées

- **EAR (Eye Aspect Ratio)** : Mesure l'ouverture des yeux
- **MAR (Mouth Aspect Ratio)** : Mesure l'ouverture de la bouche
- **Brow Score** : Inclinaison des sourcils (négatif = énervé)
- **Smile Score** : Position des coins de la bouche (positif = sourire)

## Installation

### Prérequis
- Python 3.9 à 3.12
- Une webcam

### Étapes

1. **Cloner le dépôt**
   ```bash
   git clone git@github.com:Altawn/ing4-stellantis-mood-detection.git
   cd ing4-stellantis-mood-detection
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   ```

3. **Activer l'environnement virtuel**
   - Windows :
     ```bash
     .\venv\Scripts\activate
     ```
   - macOS/Linux :
     ```bash
     source venv/bin/activate
     ```

4. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

Lancer le script de détection :

```bash
python user_scripts/face_tracking.py
```

- Une fenêtre s'ouvrira avec le flux de votre webcam
- Votre état émotionnel s'affichera en haut
- Appuyez sur **ECHAP** pour quitter

## Structure du projet

```
.
├── user_scripts/
│   └── face_tracking.py    # Script principal
├── requirements.txt        # Dépendances Python
├── README.md              # Ce fichier
└── .gitignore            # Fichiers à ignorer
```

## Personnalisation

Vous pouvez ajuster les seuils de détection dans `user_scripts/face_tracking.py` :

- **Ligne ~136** : Seuil pour "ENERVE" (actuellement `-0.3`)
- **Ligne ~139** : Seuil pour "CONTENT" (actuellement `0.03`)

## Dépannage

### La webcam ne s'ouvre pas
Vérifiez que votre webcam est bien connectée et autorisée pour Python.

### Erreur d'installation
Assurez-vous d'utiliser Python 3.9 à 3.12. MediaPipe n'est pas compatible avec Python 3.13+.

## Auteur

Projet réalisé dans le cadre du stage ING4 Stellantis.
