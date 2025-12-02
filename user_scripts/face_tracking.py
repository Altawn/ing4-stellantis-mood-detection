import cv2
import mediapipe as mp
import numpy as np
import math

# Initialisation des modules MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# --- INDICES DES LANDMARKS (Points de repère) ---

# Yeux (6 points par oeil pour le EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380] 
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Bouche (6 points pour le MAR)
MOUTH = [61, 37, 267, 291, 314, 84]
# Points supplémentaires pour le sourire (Coins et Haut/Bas centre)
MOUTH_CORNERS = [61, 291]
MOUTH_CENTER_TOP = 0
MOUTH_CENTER_BOTTOM = 17

# Sourcils (Pour l'inclinaison)
LEFT_EYEBROW_EXTREMES = [55, 46] # [Inner, Outer]
RIGHT_EYEBROW_EXTREMES = [285, 276] # [Inner, Outer]

# Liste de tous les points à dessiner
ALL_POINTS = LEFT_EYE + RIGHT_EYE + MOUTH + LEFT_EYEBROW_EXTREMES + RIGHT_EYEBROW_EXTREMES

def calculate_distance(p1, p2):
    """Calcule la distance euclidienne entre deux points."""
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_ratio_6_points(landmarks, indices):
    """Calcule un ratio basé sur 6 points (EAR/MAR)."""
    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]
    p5 = landmarks[indices[4]]
    p6 = landmarks[indices[5]]

    v1 = calculate_distance(p2, p6)
    v2 = calculate_distance(p3, p5)
    h = calculate_distance(p1, p4)

    if h == 0: return 0
    return (v1 + v2) / (2.0 * h)

def calculate_brow_inclination(landmarks, scale_ref):
    """
    Calcule l'inclinaison des sourcils.
    < -0.3 : Énervé
    """
    l_inner = landmarks[LEFT_EYEBROW_EXTREMES[0]]
    l_outer = landmarks[LEFT_EYEBROW_EXTREMES[1]]
    r_inner = landmarks[RIGHT_EYEBROW_EXTREMES[0]]
    r_outer = landmarks[RIGHT_EYEBROW_EXTREMES[1]]

    l_slope = l_outer.y - l_inner.y
    r_slope = r_outer.y - r_inner.y
    avg_slope = (l_slope + r_slope) / 2.0

    sensitivity = 0.08 
    return max(-1.0, min(1.0, avg_slope / (sensitivity * scale_ref)))

def calculate_smile_score(landmarks, scale_ref):
    """
    Calcule un score de sourire basé sur la position des coins de la bouche par rapport au centre.
    Positif : Coins plus hauts que le centre (Sourire).
    """
    corner_l = landmarks[MOUTH_CORNERS[0]]
    corner_r = landmarks[MOUTH_CORNERS[1]]
    center_top = landmarks[MOUTH_CENTER_TOP]
    
    # Moyenne Y des coins
    avg_corner_y = (corner_l.y + corner_r.y) / 2.0
    
    # Différence : Centre Y - Coins Y
    # Y augmente vers le bas. Si Coins < Centre (plus haut), Diff > 0.
    diff = center_top.y - avg_corner_y
    
    # Normalisation
    raw_score = diff / scale_ref
    
    # Ajout d'un biais pour détecter les "petits sourires" (demande utilisateur)
    # Cela rend le score plus facilement positif même pour des bouches quasi-plates
    sensitivity_bias = 0.02 
    
    return raw_score + sensitivity_bias

# Configuration de la capture vidéo
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

  print("Appuyez sur 'ECHAP' pour quitter.")

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Impossible de lire la vidéo.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    h_img, w_img, _ = image.shape

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        
        # Référence d'échelle (Distance inter-oculaire)
        eye_span = calculate_distance(landmarks[33], landmarks[263])
        if eye_span == 0: eye_span = 1

        # --- CALCULS ---
        left_ear = calculate_ratio_6_points(landmarks, LEFT_EYE)
        right_ear = calculate_ratio_6_points(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = calculate_ratio_6_points(landmarks, MOUTH)
        brow_score = calculate_brow_inclination(landmarks, eye_span)
        smile_score = calculate_smile_score(landmarks, eye_span)

        # --- LOGIQUE D'ÉTAT ---
        state = "NEUTRE"
        color = (255, 255, 0) # Cyan pour neutre

        # Priorité : Énervé > Content > Neutre
        # Seuils ajustables
        if brow_score < -0.3:
            state = "ENERVE"
            color = (0, 0, 255) # Rouge
        elif smile_score > 0.03: # Seuil bas pour détecter les petits sourires
            state = "CONTENT"
            color = (0, 255, 0) # Vert
        
        # --- AFFICHAGE ---
        # Texte
        cv2.putText(image, f'Etat: {state}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Debug values (plus petit)
        cv2.putText(image, f'Brow: {brow_score:.2f} | Smile: {smile_score:.2f}', (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f'EAR: {avg_ear:.2f} | MAR: {mar:.2f}', (30, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # --- DESSIN DES POINTS (Pas de contours, juste des points) ---
        for idx in ALL_POINTS:
            pt = landmarks[idx]
            x = int(pt.x * w_img)
            y = int(pt.y * h_img)
            # Couleur du point dépend de la zone ? Ou uniforme.
            # Faisons simple : Blanc
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)

    cv2.imshow('MediaPipe Face Analysis', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
