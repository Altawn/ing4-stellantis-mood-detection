import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- INDICES DES p_obj (Points de repère MediaPipe Face Mesh) ---
# MediaPipe fournit 468 (ou 478 avec iris) points en 3D sur le visage.
# On utilise des indices spécifiques pour calculer des ratios de distance.

# Yeux (6 points par oeil pour le calcul de l'EAR - Eye Aspect Ratio)
# Ces points permettent de mesurer l'ouverture de l'oeil.
LEFT_EYE = [362, 385, 387, 263, 373, 380] 
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Bouche (6 points pour le MAR - Mouth Aspect Ratio)
# Utilisé pour détecter l'ouverture de la bouche.
MOUTH = [61, 37, 267, 291, 314, 84]

# Points supplémentaires pour détecter le sourire
MOUTH_CORNERS = [61, 291] # Coins des lèvres
MOUTH_CENTER_TOP = 0       # Milieu lèvre supérieure
MOUTH_CENTER_BOTTOM = 17   # Milieu lèvre inférieure

# Sourcils (utilisés pour détecter la colère/froncement)
LEFT_EYEBROW_EXTREMES = [55, 46]   # [Intérieur, Extérieur]
RIGHT_EYEBROW_EXTREMES = [285, 276] # [Intérieur, Extérieur]

# Liste regroupant tous les points d'intérêt pour affichage ou calcul groupé
ALL_POINTS = LEFT_EYE + RIGHT_EYE + MOUTH + LEFT_EYEBROW_EXTREMES + RIGHT_EYEBROW_EXTREMES

"""Calcule la distance euclidienne entre deux points (x, y) normalisés."""
def calcul_distance(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

"""
Calcule le ratio d'aspect (EAR pour yeux, MAR pour bouche).
Le ratio compare la hauteur moyenne des points verticaux par rapport à la largeur.
Si le ratio diminue, l'oeil se ferme. Si le MAR augmente, la bouche s'ouvre.
"""
def calcul_EAR_MAR(p_obj, i):

    # Distance verticale - Haut 1 et Bas 2
    v1 = calcul_distance(p_obj[i[1]], p_obj[i[5]])

    # Distance verticale - Haut 2 et Bas 1
    v2 = calcul_distance(p_obj[i[2]], p_obj[i[4]])

    # Distance horizontale - Coin gauche et Coin droit
    h = calcul_distance(p_obj[i[0]], p_obj[i[3]])

    # Formule standard: moyenne des hauteurs divisée par la largeur
    return h == 0 ? 0 : (v1 + v2) / (2.0 * h)

"""
Calcule l'inclinaison des sourcils.
Plus le score est négatif, plus les sourcils pointent vers le bas (froncement).
"""
def calculate_brow_inclination(p_obj, scale_ref):
    
    # Distance intérieur extérieur
    l_slope = p_obj[LEFT_EYEBROW_EXTREMES[1]].y - p_obj[LEFT_EYEBROW_EXTREMES[0]].y
    r_slope = p_obj[RIGHT_EYEBROW_EXTREMES[1]].y - p_obj[RIGHT_EYEBROW_EXTREMES[0]].y
    avg_slope = (l_slope + r_slope) / 2.0

    # Normalisation par rapport à la taille du visage (scale_ref)
    sensitivity = 0.08 
    return max(-1.0, min(1.0, avg_slope / (sensitivity * scale_ref)))

def calculate_brow_eye_distance(p_obj, scale_ref):
    """
    Calcule la distance entre les sourcils et les yeux.
    Une réduction de cette distance est souvent signe de colère ou de concentration intense.
    """
    # Distance entre le coin interne de l'oeil et le coin interne du sourcil
    l_dist = calcul_distance(p_obj[362], p_obj[55])
    r_dist = calcul_distance(p_obj[133], p_obj[285])
    avg_dist = (l_dist + r_dist) / 2.0
    return avg_dist / scale_ref

def calculate_smile_score(p_obj, scale_ref):
    """
    Détecte le sourire en mesurant l'élévation des coins de la bouche 
    par rapport au centre de la lèvre supérieure.
    """
    corner_l = p_obj[MOUTH_CORNERS[0]]
    corner_r = p_obj[MOUTH_CORNERS[1]]
    center_top = p_obj[MOUTH_CENTER_TOP]
    
    # Moyenne de la hauteur des coins
    avg_corner_y = (corner_l.y + corner_r.y) / 2.0
    # Différence avec le centre (si les coins montent, diff augmente positivement car y descend)
    # Attention: en image y=0 est en haut, donc si un point monte, son y diminue.
    diff = center_top.y - avg_corner_y
    raw_score = diff / scale_ref
    
    sensitivity_bias = 0.02 
    return raw_score + sensitivity_bias

def draw_metric_lines(image, p_obj, indices, w_img, h_img, color=(0, 255, 0)):
    """Dessine les lignes de mesure sur l'image (pour le debug/feedback visuel)."""
    ps = [p_obj[i] for i in indices]
    coords = [(int(p.x * w_img), int(p.y * h_img)) for p in ps]
    
    # Dessine les segments verticaux et le segment horizontal
    cv2.line(image, coords[1], coords[5], color, 1)
    cv2.line(image, coords[2], coords[4], color, 1)
    cv2.line(image, coords[0], coords[3], color, 1)

class FaceAnalyzer:
    """Classe principale pour gérer le mesh facial et l'analyse d'émotions."""
    
    def __init__(self):
        # Initialisation de MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,                # On n'analyse qu'un seul visage
            refine_p_obj=True,          # Active le suivi précis (iris, lèvres)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Paramètres de Calibration
        # Le système a besoin de 5 secondes de visage "neutre" pour établir une référence.
        self.CALIBRATION_DURATION = 5.0
        self.calibration_start_time = None
        self.is_calibrating = True
        
        # Stockage des valeurs pendant la calibration pour faire une moyenne
        self.calib_brow_vals = []
        self.calib_smile_vals = []
        self.calib_brow_dist_vals = []
        
        # Valeurs de référence (le "neutre" de l'utilisateur)
        self.ref_brow_neutral = 0.0
        self.ref_smile_neutral = 0.0
        self.ref_brow_dist_neutral = 0.0
        
        # Paramètres de Lissage (Exponential Moving Average)
        # alpha = 0.2 signifie que la nouvelle valeur compte pour 20%, l'ancienne pour 80%
        # Cela évite que l'affichage "saute" si les points bougent un peu.
        self.alpha = 0.2
        self.smooth_brow = 0.0
        self.smooth_smile = 0.0
        self.smooth_brow_dist = 0.0
        
        # Gestion de la stabilité des états (Hystérésis)
        # On ne change d'état que si la détection est stable pendant X secondes.
        self.current_state = "NEUTRE"
        self.potential_state = "NEUTRE"
        self.state_start_time = 0.0
        self.STATE_DURATION_THRESHOLD = 0.3 # 300ms de stabilité requis

    def process(self, image):
        """Fonction principale traitant une frame image."""
        
        # Conversion BGR (OpenCV) vers RGB (MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Optimisation performance
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        
        h_img, w_img, _ = image.shape
        metrics = {}

        if results.multi_face_p_obj:
            for face_p_obj in results.multi_face_p_obj:
                p_obj = face_p_obj.landmark
                
                # Référence de taille : distance entre les deux coins externes des yeux
                # Permet de rendre les calculs indépendants de la distance caméra/visage.
                eye_span = calcul_distance(p_obj[33], p_obj[263])
                if eye_span == 0: eye_span = 1
                
                # Calcul des scores bruts
                brow_score = calculate_brow_inclination(p_obj, eye_span)
                brow_dist_score = calculate_brow_eye_distance(p_obj, eye_span)
                smile_score = calculate_smile_score(p_obj, eye_span)
                
                # Calcul des ratios pour les clignements et la bouche (optionnel pour l'instant)
                left_ear = calcul_EAR_MAR(p_obj, LEFT_EYE)
                right_ear = calcul_EAR_MAR(p_obj, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calcul_EAR_MAR(p_obj, MOUTH)

                rel_brow = 0.0
                rel_smile = 0.0
                rel_brow_dist = 0.0
                
                # --- PHASE DE CALIBRATION ---
                if self.is_calibrating:
                    if self.calibration_start_time is None:
                        self.calibration_start_time = time.time()
                    
                    elapsed = time.time() - self.calibration_start_time
                    remaining = max(0, self.CALIBRATION_DURATION - elapsed)
                    
                    # Message d'instruction à l'écran
                    cv2.putText(image, f"CALIBRATION: VISAGE NEUTRE ({remaining:.1f}s)", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
                    # Accumulation des données
                    self.calib_brow_vals.append(brow_score)
                    self.calib_smile_vals.append(smile_score)
                    self.calib_brow_dist_vals.append(brow_dist_score)
                    
                    # Fin de calibration
                    if elapsed >= self.CALIBRATION_DURATION:
                        self.is_calibrating = False
                        if self.calib_brow_vals:
                            self.ref_brow_neutral = sum(self.calib_brow_vals) / len(self.calib_brow_vals)
                            self.ref_smile_neutral = sum(self.calib_smile_vals) / len(self.calib_smile_vals)
                            self.ref_brow_dist_neutral = sum(self.calib_brow_dist_vals) / len(self.calib_brow_dist_vals)
                        print(f"Calibration Terminee! Refs: {self.ref_brow_neutral}, {self.ref_smile_neutral}")
                    
                    self.current_state = "CALIBRATION"
                    color = (128, 128, 128)
                
                # --- PHASE D'ANALYSE ---
                else:
                    # 1. Lissage des valeurs pour éviter les tremblements
                    self.smooth_brow = self.alpha * brow_score + (1 - self.alpha) * self.smooth_brow
                    self.smooth_smile = self.alpha * smile_score + (1 - self.alpha) * self.smooth_smile
                    self.smooth_brow_dist = self.alpha * brow_dist_score + (1 - self.alpha) * self.smooth_brow_dist
                    
                    # 2. Calcul des écarts par rapport au neutre (Relative scores)
                    rel_brow = self.smooth_brow - self.ref_brow_neutral
                    rel_smile = self.smooth_smile - self.ref_smile_neutral
                    rel_brow_dist = self.smooth_brow_dist - self.ref_brow_dist_neutral
                    
                    # 3. Logique de décision (Heuristiques)
                    detected_state = "NEUTRE"
                    
                    # Si le sourire augmente de plus de 0.03 par rapport au neutre -> CONTENT
                    if rel_smile > 0.03:
                        detected_state = "CONTENT"
                    # Si la distance sourcil-oeil diminue significativement -> ENERVE
                    elif rel_brow_dist < -0.010:
                        detected_state = "ENERVE"
                        
                    # 4. Mécanisme de confirmation (évite les switchs trop rapides)
                    if detected_state == self.potential_state:
                        # Si l'état détecté est le même que le potentiel depuis X secondes, on valide
                        if time.time() - self.state_start_time > self.STATE_DURATION_THRESHOLD:
                            self.current_state = detected_state
                    else:
                        # Sinon, on définit un nouvel état potentiel et on reset le timer
                        self.potential_state = detected_state
                        self.state_start_time = time.time()
                        
                    # Choix de la couleur d'affichage selon l'état final
                    if self.current_state == "ENERVE": color = (0, 0, 255) # Rouge
                    elif self.current_state == "CONTENT": color = (0, 255, 0) # Vert
                    else: color = (255, 255, 0) # Cyan/Jaune
                    
                    # Affichage du texte principal
                    cv2.putText(image, f'Etat: {self.current_state}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    
                    # --- DEBUG INFO ---
                    # Affiche les valeurs numériques pour ajuster les seuils si besoin
                    cv2.putText(image, f'B-Dist: {self.smooth_brow_dist:.3f} (Rel: {rel_brow_dist:.3f})', (30, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.putText(image, f'Smile: {self.smooth_smile:.2f} (Rel: {rel_smile:.2f})', (30, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Dessin des points de repère sur l'image
                draw_metric_lines(image, p_obj, LEFT_EYE, w_img, h_img, (0, 255, 255))
                draw_metric_lines(image, p_obj, RIGHT_EYE, w_img, h_img, (0, 255, 255))
                draw_metric_lines(image, p_obj, MOUTH, w_img, h_img, (0, 100, 255))
                
                # On remplit le dictionnaire de retour pour l'API backend
                metrics['emotion'] = self.current_state
                
        return image, metrics
