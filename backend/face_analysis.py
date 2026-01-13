import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- INDICES DES LANDMARKS (Points de repère) ---
# Yeux (6 points par oeil pour le EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380] 
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Bouche (6 points pour le MAR)
MOUTH = [61, 37, 267, 291, 314, 84]
# Points supplémentaires pour le sourire
MOUTH_CORNERS = [61, 291]
MOUTH_CENTER_TOP = 0
MOUTH_CENTER_BOTTOM = 17

# Sourcils
LEFT_EYEBROW_EXTREMES = [55, 46] # [Inner, Outer]
RIGHT_EYEBROW_EXTREMES = [285, 276] # [Inner, Outer]

ALL_POINTS = LEFT_EYE + RIGHT_EYE + MOUTH + LEFT_EYEBROW_EXTREMES + RIGHT_EYEBROW_EXTREMES

def calculate_distance(p1, p2):
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_ratio_6_points(landmarks, indices):
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
    l_dist = calculate_distance(landmarks[362], landmarks[55])
    r_dist = calculate_distance(landmarks[133], landmarks[285])
    avg_dist = (l_dist + r_dist) / 2.0
    return avg_dist / scale_ref

def calculate_smile_score(landmarks, scale_ref):
    corner_l = landmarks[MOUTH_CORNERS[0]]
    corner_r = landmarks[MOUTH_CORNERS[1]]
    center_top = landmarks[MOUTH_CENTER_TOP]
    
    avg_corner_y = (corner_l.y + corner_r.y) / 2.0
    diff = center_top.y - avg_corner_y
    raw_score = diff / scale_ref
    sensitivity_bias = 0.02 
    return raw_score + sensitivity_bias

def draw_metric_lines(image, landmarks, indices, w_img, h_img, color=(0, 255, 0)):
    ps = [landmarks[i] for i in indices]
    coords = [(int(p.x * w_img), int(p.y * h_img)) for p in ps]
    
    cv2.line(image, coords[1], coords[5], color, 1)
    cv2.line(image, coords[2], coords[4], color, 1)
    cv2.line(image, coords[0], coords[3], color, 1)

class FaceAnalyzer:
    def __init__(self):
        # Try to use the classic `mp.solutions.face_mesh` API.
        # If not available, try to use the newer MediaPipe Tasks API (FaceLandmarker).
        # If neither works, fall back to a dummy analyzer.
        self.use_dummy = False
        self.use_tasks_api = False
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_tasks_api = False
        except Exception:
            # Try Tasks API
            try:
                import os
                import urllib.request
                # Use the higher-level vision API which provides factory helpers
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks.python.vision.core import image as mp_image

                # Ensure model file exists locally (download from MediaPipe assets if needed)
                model_dir = os.path.join(os.path.dirname(__file__), 'models')
                os.makedirs(model_dir, exist_ok=True)
                model_asset_name = 'face_landmarker_v2.task'
                model_path = os.path.join(model_dir, model_asset_name)
                if not os.path.exists(model_path):
                    try:
                        url = 'https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task'
                        print(f"Downloading face landmarker model to {model_path}...")
                        urllib.request.urlretrieve(url, model_path)
                        print('Model downloaded')
                    except Exception as dl_e:
                        print('Failed to download model asset:', dl_e)
                        raise

                base_options = mp_vision.BaseOptions(model_asset_path=model_path)
                # Use IMAGE running mode and create options
                try:
                    options = mp_vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        num_faces=1,
                        min_detection_confidence=0.5,
                        running_mode=mp_vision.RunningMode.IMAGE
                    )
                except Exception:
                    # Fallback without running_mode
                    options = mp_vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        num_faces=1,
                        min_detection_confidence=0.5
                    )

                try:
                    self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                except Exception as e_create:
                    print('FaceLandmarker.create_from_options failed:', e_create)
                    raise

                self.mp_tasks_image = mp_image.Image
                self.use_tasks_api = True
            except Exception:
                print("mediapipe 'solutions' API not available — using dummy analyzer")
                self.use_dummy = True
        
        # State & Calibration
        self.CALIBRATION_DURATION = 5.0
        self.calibration_start_time = None
        self.is_calibrating = True
        self.calib_brow_vals = []
        self.calib_smile_vals = []
        self.calib_brow_dist_vals = []
        
        self.ref_brow_neutral = 0.0
        self.ref_smile_neutral = 0.0
        self.ref_brow_dist_neutral = 0.0
        
        # Smoothing
        self.alpha = 0.2
        self.smooth_brow = 0.0
        self.smooth_smile = 0.0
        self.smooth_brow_dist = 0.0
        
        # State Stability
        self.current_state = "NEUTRE"
        self.potential_state = "NEUTRE"
        self.state_start_time = 0.0
        self.STATE_DURATION_THRESHOLD = 0.3

    def process(self, image):
        if getattr(self, 'use_dummy', False):
            return image, {'emotion': 'NEUTRE'}

        # If using Tasks API
        if getattr(self, 'use_tasks_api', False):
            try:
                # Convert BGR->RGB and to Tasks Image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_img = self.mp_tasks_image.create_from_array(image_rgb)

                # Detect
                result = self.face_landmarker.detect(mp_img)

                # Create a minimal results-like structure for compatibility
                multi_face_landmarks = None
                if hasattr(result, 'face_landmarks') and result.face_landmarks:
                    multi_face_landmarks = []
                    for fl in result.face_landmarks:
                        # fl is a NormalizedLandmarkList-like object with .landmark
                        lm_list = []
                        if hasattr(fl, 'landmark'):
                            for lm in fl.landmark:
                                # Each lm has x,y (normalized)
                                class LM:
                                    pass
                                p = LM()
                                p.x = lm.x
                                p.y = lm.y
                                lm_list.append(p)
                        multi_face_landmarks.append(type('Face', (), {'landmark': lm_list}))

                # Now set up a pseudo-results object to reuse existing logic
                class Results:
                    pass
                results = Results()
                results.multi_face_landmarks = multi_face_landmarks

            except Exception as e:
                print('Tasks API face_landmarker failed:', e)
                return image, {'emotion': 'NEUTRE'}

            image.flags.writeable = True
        else:
            # Classic FaceMesh
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.face_mesh.process(image_rgb)
            image_rgb.flags.writeable = True

        # image stays BGR for drawing if we want, but we should return RGB or BGR? 
        # Usually frontend expects image to display.
        
        h_img, w_img, _ = image.shape
        
        metrics = {}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                eye_span = calculate_distance(landmarks[33], landmarks[263])
                if eye_span == 0: eye_span = 1
                
                brow_score = calculate_brow_inclination(landmarks, eye_span)
                brow_dist_score = calculate_brow_eye_distance(landmarks, eye_span)
                smile_score = calculate_smile_score(landmarks, eye_span)
                
                left_ear = calculate_ratio_6_points(landmarks, LEFT_EYE)
                right_ear = calculate_ratio_6_points(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_ratio_6_points(landmarks, MOUTH)

                rel_brow = 0.0
                rel_smile = 0.0
                rel_brow_dist = 0.0
                
                if self.is_calibrating:
                    if self.calibration_start_time is None:
                        self.calibration_start_time = time.time()
                    
                    elapsed = time.time() - self.calibration_start_time
                    remaining = max(0, self.CALIBRATION_DURATION - elapsed)
                    
                    cv2.putText(image, f"CALIBRATION: VISAGE NEUTRE ({remaining:.1f}s)", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
                    self.calib_brow_vals.append(brow_score)
                    self.calib_smile_vals.append(smile_score)
                    self.calib_brow_dist_vals.append(brow_dist_score)
                    
                    if elapsed >= self.CALIBRATION_DURATION:
                        self.is_calibrating = False
                        if self.calib_brow_vals:
                            self.ref_brow_neutral = sum(self.calib_brow_vals) / len(self.calib_brow_vals)
                            self.ref_smile_neutral = sum(self.calib_smile_vals) / len(self.calib_smile_vals)
                            self.ref_brow_dist_neutral = sum(self.calib_brow_dist_vals) / len(self.calib_brow_dist_vals)
                        print(f"Calibration Done! Refs: {self.ref_brow_neutral}, {self.ref_smile_neutral}")
                    
                    self.current_state = "CALIBRATION"
                    color = (128, 128, 128)
                else:
                    self.smooth_brow = self.alpha * brow_score + (1 - self.alpha) * self.smooth_brow
                    self.smooth_smile = self.alpha * smile_score + (1 - self.alpha) * self.smooth_smile
                    self.smooth_brow_dist = self.alpha * brow_dist_score + (1 - self.alpha) * self.smooth_brow_dist
                    
                    rel_brow = self.smooth_brow - self.ref_brow_neutral
                    rel_smile = self.smooth_smile - self.ref_smile_neutral
                    rel_brow_dist = self.smooth_brow_dist - self.ref_brow_dist_neutral
                    
                    detected_state = "NEUTRE"
                    detected_color = (255, 255, 0)
                    
                    if rel_smile > 0.03:
                        detected_state = "CONTENT"
                        detected_color = (0, 255, 0)
                    elif rel_brow_dist < -0.010:
                        detected_state = "ENERVE"
                        detected_color = (0, 0, 255)
                        
                    if detected_state == self.potential_state:
                        if time.time() - self.state_start_time > self.STATE_DURATION_THRESHOLD:
                            self.current_state = detected_state
                    else:
                        self.potential_state = detected_state
                        self.state_start_time = time.time()
                        
                    if self.current_state == "ENERVE": color = (0, 0, 255)
                    elif self.current_state == "CONTENT": color = (0, 255, 0)
                    else: color = (255, 255, 0)
                    
                    cv2.putText(image, f'Etat: {self.current_state}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    
                    # Debug
                    cv2.putText(image, f'B-Dist: {self.smooth_brow_dist:.3f} (Rel: {rel_brow_dist:.3f})', (30, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.putText(image, f'Smile: {self.smooth_smile:.2f} (Rel: {rel_smile:.2f})', (30, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Draw
                draw_metric_lines(image, landmarks, LEFT_EYE, w_img, h_img, (0, 255, 255))
                draw_metric_lines(image, landmarks, RIGHT_EYE, w_img, h_img, (0, 255, 255))
                draw_metric_lines(image, landmarks, MOUTH, w_img, h_img, (0, 100, 255))
                
                # Primary emotion for the response (backward compatibility)
                metrics['emotion'] = self.current_state
                
        return image, metrics
