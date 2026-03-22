from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
from face_analysis import FaceAnalyzer

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

analyzer = FaceAnalyzer()
last_secheresse_time = 0
global_temp = 20.0

@app.route('/api/temperature', methods=['POST'])
def adjust_temperature():
    global global_temp
    try:
        data = request.json or {}
        delta = data.get('delta', 0)
        global_temp += delta
        return jsonify({"temperature": global_temp})
    except Exception as e:
        print(f"Error adjusting temp: {e}")
        return jsonify({"error": str(e)}), 500

def decode_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(cv2_img):
    _, buffer = cv2.imencode('.jpg', cv2_img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

@app.route('/api/frame', methods=['POST'])
def process_frame():
    global last_secheresse_time
    global global_temp
    
    try:
        data = request.json or {}
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Decode
        frame = decode_image(image_data)
        
        # Process (image non modifiée, on récupère juste les métriques)
        _, metrics = analyzer.process(frame)
        
        emotion = metrics.get('emotion', 'PAS DE VISAGE')
        annotations = metrics.get('annotations', {})
        
        # Temperature logic based on emotion (simple example)
        # If ENERVE, lower temp? If CONTENT, ?
        # App.jsx comments: "backend gérera l'ajustement progressif" for VLM, 
        # but here we can just do simple reactive temp.
        
        if emotion == "INCONFORT":
            global_temp = max(16.0, global_temp - 0.05)
        elif emotion == "NEUTRE":
            pass # Stable temperature for happiness
        elif emotion == "SOMNOLENCE":
            if global_temp < 20.0: 
                global_temp += 0.05
        elif emotion == "BAILLEMENT":
            global_temp = min(20.0, global_temp + 0.05)
        elif emotion == "SÉCHERESSE OCULAIRE":
            if time.time() - last_secheresse_time > 5.0:
                print(f"[APP] Sécheresse Oculaire active. Ancienne temp: {global_temp}")
                global_temp -= 2.0
                print(f"[APP] Nouvelle temp: {global_temp}")
                last_secheresse_time = time.time()
            
        return jsonify({
            "emotion": emotion,
            "primary_emotion": emotion,
            "annotations": annotations,
            "temperature": round(global_temp, 2)
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/vlm-check', methods=['GET'])
def vlm_check():
    # Placeholder for VLM check
    return jsonify({"question": None})

@app.route('/api/vlm-response', methods=['POST'])
def vlm_response():
    data = request.json
    print(f"VLM Response received: {data}")
    return jsonify({"status": "received"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
