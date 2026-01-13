import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

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

def calculate_brow_eye_distance(landmarks, scale_ref):
    """
    Calcule la distance moyenne entre le coin intérieur de l'oeil et le coin intérieur du sourcil.
    Cette mesure est plus robuste à l'inclinaison de la tête (pitch) que la pente brute.
    Plus petit = plus énervé (sourcils froncés vers le bas).
    """
    # Indices:
    # Left Eye Inner: 362, Left Eyebrow Inner: 55
    # Right Eye Inner: 133, Right Eyebrow Inner: 285
    
    l_dist = calculate_distance(landmarks[362], landmarks[55])
    r_dist = calculate_distance(landmarks[133], landmarks[285])
    
    avg_dist = (l_dist + r_dist) / 2.0
    return avg_dist / scale_ref

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

def draw_metric_lines(image, landmarks, indices, w_img, h_img, color=(0, 255, 0)):
    """Dessine les lignes verticales et horizontales pour EAR/MAR."""
    # Points according to calculate_ratio_6_points: p1..p6
    # indices: [p1, p2, p3, p4, p5, p6]
    # Vertical lines: p2-p6, p3-p5
    # Horizontal line: p1-p4
    
    ps = [landmarks[i] for i in indices]
    coords = [(int(p.x * w_img), int(p.y * h_img)) for p in ps]
    
    # Verticals
    cv2.line(image, coords[1], coords[5], color, 1)
    cv2.line(image, coords[2], coords[4], color, 1)
    # Horizontal
    cv2.line(image, coords[0], coords[3], color, 1)

# Configuration de la capture vidéo
# Configuration de la capture vidéo
cap = cv2.VideoCapture(0)

# --- VARIABLES ETAT & CALIBRATION ---
CALIBRATION_DURATION = 5.0 # secondes (Augmenté pour mieux s'adapter)
calibration_start_time = None
is_calibrating = True
calib_brow_vals = []
calib_smile_vals = []
calib_brow_dist_vals = [] # Nouveau pour la distance

# Valeurs de référence (seront mises à jour après calibration)
ref_brow_neutral = 0.0
ref_smile_neutral = 0.0
ref_brow_dist_neutral = 0.0

# Lissage (Smoothing)
alpha = 0.2 # Facteur de lissage (0 < alpha <= 1). Plus petit = plus lisse.
smooth_brow = 0.0
smooth_smile = 0.0
smooth_brow_dist = 0.0

# State Stability
current_state = "NEUTRE"
potential_state = "NEUTRE"
state_start_time = 0.0
STATE_DURATION_THRESHOLD = 0.3 # secondes

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
        brow_score = calculate_brow_inclination(landmarks, eye_span) # Gardé pour info mais on utilise la distance
        brow_dist_score = calculate_brow_eye_distance(landmarks, eye_span)
        smile_score = calculate_smile_score(landmarks, eye_span)

        rel_brow = 0.0
        rel_smile = 0.0
        rel_brow_dist = 0.0

        # --- CALIBRATION LOGIC ---
        if is_calibrating:
            if calibration_start_time is None:
                calibration_start_time = time.time()
            
            elapsed = time.time() - calibration_start_time
            remaining =  max(0, CALIBRATION_DURATION - elapsed)
            
            # Affichage compte à rebours
            cv2.putText(image, f"CALIBRATION: VISAGE NEUTRE ({remaining:.1f}s)", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Collecte des données
            calib_brow_vals.append(brow_score)
            calib_smile_vals.append(smile_score)
            calib_brow_dist_vals.append(brow_dist_score)
            
            if elapsed >= CALIBRATION_DURATION:
                is_calibrating = False
                if calib_brow_vals:
                    ref_brow_neutral = sum(calib_brow_vals) / len(calib_brow_vals)
                    ref_smile_neutral = sum(calib_smile_vals) / len(calib_smile_vals)
                    ref_brow_dist_neutral = sum(calib_brow_dist_vals) / len(calib_brow_dist_vals)
                print(f"Calibration Done! Ref Brow Dist: {ref_brow_dist_neutral:.3f}, Ref Smile: {ref_smile_neutral:.3f}")

            # Pendant la calibration, on ne change pas d'état visuel
            state = "CALIBRATION"
            color = (128, 128, 128)
            
        else:
            # --- CALCULS LISSÉS & RELATIFS ---
            # Lissage Exponentiel
            smooth_brow = alpha * brow_score + (1 - alpha) * smooth_brow
            smooth_smile = alpha * smile_score + (1 - alpha) * smooth_smile
            smooth_brow_dist = alpha * brow_dist_score + (1 - alpha) * smooth_brow_dist
            
            # Valeurs relatives à la calibration
            rel_brow = smooth_brow - ref_brow_neutral
            rel_smile = smooth_smile - ref_smile_neutral
            rel_brow_dist = smooth_brow_dist - ref_brow_dist_neutral
    
            # --- LOGIQUE D'ÉTAT PROVISOIRE ---
            detected_state = "NEUTRE"
            detected_color = (255, 255, 0) # Cyan pour neutre

            # Seuils ajustés sur le relatif
            # Angry detection based on BROW DISTANCE now
            # Si distance diminue -> Froncement -> Angry
            # Threshold experimental: -0.02 (depends on metric scale)
            
            if rel_brow_dist < -0.015: # Seuil distance sourcil-oeil (plus robuste au pitch)
                detected_state = "ENERVE"
                detected_color = (0, 0, 255) # Rouge
            elif rel_smile > 0.02: 
                detected_state = "CONTENT"
                detected_color = (0, 255, 0) # Vert
            
            # --- STABILITE TEMPORELLE ---
            if detected_state == potential_state:
                # Si l'état détecté est le même que le potentiel, on incrémente le temps
                if time.time() - state_start_time > STATE_DURATION_THRESHOLD:
                     current_state = detected_state
                     if current_state == "ENERVE": color = (0, 0, 255)
                     elif current_state == "CONTENT": color = (0, 255, 0)
                     else: color = (255, 255, 0)
            else:
                # Changement de candidat -> Reset timer
                potential_state = detected_state
                state_start_time = time.time()
                
            # Fallback color for display if state hasn't changed yet
            if current_state == "ENERVE": color = (0, 0, 255)
            elif current_state == "CONTENT": color = (0, 255, 0)
            else: color = (255, 255, 0)
                
        # --- AFFICHAGE ---
        if not is_calibrating:
             cv2.putText(image, f'Etat: {current_state}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Debug values (plus petit)
        cv2.putText(image, f'B-Dist: {smooth_brow_dist:.3f} (Rel: {rel_brow_dist:.3f})', (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f'Smile: {smile_score:.2f} (Rel: {rel_smile:.2f})', (30, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f'EAR: {avg_ear:.2f} | MAR: {mar:.2f}', (30, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # --- DESSIN DES POINTS (Pas de contours, juste des points) ---
        # --- DESSIN DES LIGNES (EAR / MAR) ---
        draw_metric_lines(image, landmarks, LEFT_EYE, w_img, h_img, (0, 255, 255))
        draw_metric_lines(image, landmarks, RIGHT_EYE, w_img, h_img, (0, 255, 255))
        draw_metric_lines(image, landmarks, MOUTH, w_img, h_img, (0, 100, 255))
        
        # --- DESSIN DES POINTS (Optionnel: Tous les points) ---
        for idx in ALL_POINTS:
            pt = landmarks[idx]
            x = int(pt.x * w_img)
            y = int(pt.y * h_img)
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow('MediaPipe Face Analysis', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
