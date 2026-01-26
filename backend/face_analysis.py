import cv2
import sys
from unittest.mock import MagicMock
# Mock sounddevice to avoid PortAudio initialization error on Windows
sys.modules['sounddevice'] = MagicMock()
import mediapipe as mp
import numpy 
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
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

"""Calcule le ratio d'aspect (EAR pour yeux, MAR pour bouche)."""
def _calculate_aspect_ratio(p_obj, i):

    # Distance verticale - Haut 1 et Bas 2
    v1 = calcul_distance(p_obj[i[1]], p_obj[i[5]])

    # Distance verticale - Haut 2 et Bas 1
    v2 = calcul_distance(p_obj[i[2]], p_obj[i[4]])

    # Distance horizontale - Coin gauche et Coin droit
    h = calcul_distance(p_obj[i[0]], p_obj[i[3]])

    # Formule standard: moyenne des hauteurs divisée par la largeur
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

""" --- SECTION YEUX --- """
"""Calcule les métriques des yeux (actuellement EAR)."""
def calculate_eye_metrics(p_obj):
    ear_l = _calculate_aspect_ratio(p_obj, LEFT_EYE)
    ear_r = _calculate_aspect_ratio(p_obj, RIGHT_EYE)
    return (ear_l + ear_r) / 2.0

"""Calcule à la fois l'inclinaison des sourcils et la distance sourcils-yeux. Retourne (inclination_score, proximity_score)."""
def calculate_brow_metrics(p_obj, scale_ref):
    # 1. Inclinaison (Brow Inclination)
    # Distance intérieur extérieur
    l_slope = p_obj[LEFT_EYEBROW_EXTREMES[1]].y - p_obj[LEFT_EYEBROW_EXTREMES[0]].y
    r_slope = p_obj[RIGHT_EYEBROW_EXTREMES[1]].y - p_obj[RIGHT_EYEBROW_EXTREMES[0]].y
    avg_slope = (l_slope + r_slope) / 2.0

    # Normalisation par rapport à la taille du visage (scale_ref)
    sensitivity = 0.1 #Valeur permettant d'avoir la aprfaite inclinaison des sourcils, par rapport à moi en tout cas
    inclination_score = max(-1.0, min(1.0, avg_slope / (sensitivity * scale_ref)))

    # 2. Distance Sourcils-Yeux (Brow-Eye Distance)
    # Distance entre le coin interne de l'oeil et le coin interne du sourcil
    l_dist = calcul_distance(p_obj[362], p_obj[55])
    r_dist = calcul_distance(p_obj[133], p_obj[285])
    avg_dist = (l_dist + r_dist) / 2.0
    proximity_score = avg_dist / scale_ref
    
    return inclination_score, proximity_score

""" --- SECTION BOUCHE --- """
"""Calcule les métriques de la bouche (MAR et Sourire). Retourne (smile_score, mar_score)."""
def calculate_mouth_metrics(p_obj, scale_ref):
    # 1. MAR (Mouth Aspect Ratio)
    mar_score = _calculate_aspect_ratio(p_obj, MOUTH)
    
    # 2. Score Sourire (Smile)
    corner_l = p_obj[MOUTH_CORNERS[0]]
    corner_r = p_obj[MOUTH_CORNERS[1]]
    center_top = p_obj[MOUTH_CENTER_TOP]
    
    # Moyenne de la hauteur des coins
    avg_corner_y = (corner_l.y + corner_r.y) / 2.0
    
    # Différence avec le centre (si les coins montent, diff augmente positivement car y descend)
    diff = center_top.y - avg_corner_y
    raw_score = diff / scale_ref
    
    sensitivity_bias = 0.02 
    return raw_score + sensitivity_bias, mar_score

"""Dessine les lignes de mesure sur l'image (pour le debug/feedback visuel)."""
def draw_metric_lines(image, p_obj, indices, w_img, h_img, color=(0, 255, 0)):
    ps = [p_obj[i] for i in indices]
    coords = [(int(p.x * w_img), int(p.y * h_img)) for p in ps]
    
    # Dessine les segments verticaux et le segment horizontal
    cv2.line(image, coords[1], coords[5], color, 1)
    cv2.line(image, coords[2], coords[4], color, 1)
    cv2.line(image, coords[0], coords[3], color, 1)

"""Classe principale pour gérer le mesh facial et l'analyse d'émotions."""
class FaceAnalyzer:
    
    def __init__(self):
        # Initialisation de MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,                # On n'analyse qu'un seul visage
            refine_landmarks=True,          # Active le suivi précis (iris, lèvres)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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
        self.calib_ear_vals = [] # Nouvelle liste pour les yeux
        
        # Valeurs de référence (le "neutre" de l'utilisateur)
        self.ref_brow_neutral = 0.0
        self.ref_smile_neutral = 0.0
        self.ref_brow_dist_neutral = 0.0
        self.ref_ear_neutral = 0.4  # Valeur par défaut avant calibration
        
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

        self.eye_closed_start = None
        self.mouth_open_start = None
        self.fatigue_score = 0

    def process(self, image):
        """Fonction principale traitant une frame image."""
        
        # Conversion BGR (OpenCV) vers RGB (MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Optimisation performance
        try :  #version qui ignore le flux temporel de l'image si besoin pour éviter bug lié au temps de l'image egal ou inférieur à la précédente 
            results = self.face_mesh.process(image_rgb)
        except Exception as e:
            print(f"Erreur de Mediapipe : {e}")
            return image, {}

        image_rgb.flags.writeable = True
        
        h_img, w_img, _ = image.shape
        metrics = {}

        if results.multi_face_landmarks:
            p_obj = results.multi_face_landmarks[0].landmark
            
            # Référence de taille : distance entre les deux coins externes des yeux
            # Permet de rendre les calculs indépendants de la distance caméra/visage.
            eye_span = calcul_distance(p_obj[33], p_obj[263])
            if eye_span == 0: eye_span = 1
            
            # Calcul des scores bruts
            brow_score, brow_dist_score = calculate_brow_metrics(p_obj, eye_span)
            smile_score, mar_score = calculate_mouth_metrics(p_obj, eye_span)
            # eye_score = calculate_eye_metrics(p_obj) # Disponible si besoin

            avg_ear = calculate_eye_metrics(p_obj)

            rel_brow = 0.0
            rel_smile = 0.0
            rel_brow_dist = 0.0
            
            # --- PHASE DE CALIBRATION ---
            if self.is_calibrating:
                if self.calibration_start_time is None:
                    self.calibration_start_time = time.time()
                
                elapsed = time.time() - self.calibration_start_time
                remaining = max(0, self.CALIBRATION_DURATION - elapsed)
                
                # Accumulation des données
                self.calib_brow_vals.append(brow_score)
                self.calib_smile_vals.append(smile_score)
                self.calib_brow_dist_vals.append(brow_dist_score)
                self.calib_ear_vals.append(avg_ear) # Sauvegarde de l'ouverture des yeux
                
                # Fin de calibration
                if elapsed >= self.CALIBRATION_DURATION:
                    self.is_calibrating = False
                    if self.calib_brow_vals:
                        self.ref_brow_neutral = sum(self.calib_brow_vals) / len(self.calib_brow_vals)
                        self.ref_smile_neutral = sum(self.calib_smile_vals) / len(self.calib_smile_vals)
                        self.ref_brow_dist_neutral = sum(self.calib_brow_dist_vals) / len(self.calib_brow_dist_vals)
                        self.ref_ear_neutral = sum(self.calib_ear_vals) / len(self.calib_ear_vals)
                    print(f"Calibration Terminee! Refs EAR: {self.ref_ear_neutral:.3f}")
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

                #Nouvelle logique de décision pour états physiologiques (fatigue et inconfort)

                detected_state = "NEUTRE"

                # 3. Détection Somnolence
                # On détecte si l'ouverture des yeux descend en dessous de 80% de l'ouverture normale
                # OU si elle passe sous un seuil absolu de 0.22 (plus sensible que 0.2)
                ear_threshold = min(0.22, self.ref_ear_neutral * 0.8)
                
                if avg_ear < ear_threshold: 
                    if self.eye_closed_start is None:
                        self.eye_closed_start = time.time()
                    elif time.time() - self.eye_closed_start > 0.5:  # pendant plus de 0.5 secondes
                        detected_state = "SOMNOLENCE"   
                else :
                    self.eye_closed_start = None

                #detection Baillement
                if mar_score > 0.8 : #bouche grande ouverte
                    if self.mouth_open_start is None:
                        self.mouth_open_start = time.time()
                    elif time.time() - self.mouth_open_start > 1.5: #pendant plus de 1.5 secondes
                        detected_state = "BAILLEMENT"
                else:
                    self.mouth_open_start = None

                # 5. Détection Inconfort (uniquement si les yeux sont ouverts)
                if detected_state == "NEUTRE":
                    if rel_brow_dist < -0.012 and avg_ear >= ear_threshold: 
                        detected_state = "INCONFORT"


                #Mise à jour de l'état avec lissage temporel 

                if detected_state == self.potential_state: #si etat potention et etat detecté sont les memes
                    if time.time() - self.state_start_time > self.STATE_DURATION_THRESHOLD: #pendant assez de temps
                        self.current_state = detected_state #alors etat = etat détecté
                else : 
                    self.potential_state = detected_state #sinon etat potentiel devient etat detecté 
                    self.state_start_time = time.time() #et on reset le timer

                # --- DEBUG INFO ---
                # Affiche les valeurs numériques pour ajuster les seuils si besoin
                # cv2.putText(image, f'B-Dist: {self.smooth_brow_dist:.3f} (Rel: {rel_brow_dist:.3f})', (30, 90), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                # cv2.putText(image, f'Smile: {self.smooth_smile:.2f} (Rel: {rel_smile:.2f})', (30, 110), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Dessin des points de repère sur l'image
            draw_metric_lines(image, p_obj, LEFT_EYE, w_img, h_img, (0, 255, 255))
            draw_metric_lines(image, p_obj, RIGHT_EYE, w_img, h_img, (0, 255, 255))
            draw_metric_lines(image, p_obj, MOUTH, w_img, h_img, (0, 100, 255))
            
            # On remplit le dictionnaire de retour pour l'API backend
            metrics['emotion'] = self.current_state
        return image, metrics
            
