from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from face_analysis import FaceAnalyzer

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

analyzer = FaceAnalyzer()

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
    data = request.json
    image_data = data.get('image')
    current_temp = data.get('temperature', 20)
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode
        frame = decode_image(image_data)
        
        # Process
        annotated_frame, metrics = analyzer.process(frame)
        
        # Encode back
        annotated_image_b64 = encode_image(annotated_frame)
        
        emotion = metrics.get('emotion', 'PAS DE VISAGE')
        
        # Temperature logic based on emotion (simple example)
        # If ENERVE, lower temp? If CONTENT, ?
        # App.jsx comments: "backend gérera l'ajustement progressif" for VLM, 
        # but here we can just do simple reactive temp.
        
        new_temp = current_temp
        if emotion == "INCONFORT" or emotion == "BAILLEMENT":
            new_temp = max(16, current_temp - 0.05)
        elif emotion == "NEUTRE":
            pass # Stable temperature for happiness
        else:
            # Return towards 20?
            if current_temp > 20: new_temp -= 0.1
            elif current_temp < 20: new_temp += 0.1
            
        return jsonify({
            "emotion": emotion,
            "primary_emotion": emotion,
            "annotated_image": annotated_image_b64,
            "temperature": round(new_temp, 2)
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
