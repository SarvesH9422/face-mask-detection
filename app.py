from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
from PIL import Image
import io
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import os
from datetime import datetime

# Create Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = "models/mask_detector.keras"
FACE_PROTOTXT = "face_detector/deploy.prototxt"
FACE_MODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
CAPTURE_FOLDER = "captures"

# Global variables
face_net = None
mask_net = None
detection_enabled = True

# Create captures folder
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# Helper functions
def load_models():
    """Load face detector and mask detector models"""
    global face_net, mask_net
    
    if face_net is None:
        print("[INFO] Loading face detector...")
        face_net = cv2.dnn.readNet(FACE_PROTOTXT, FACE_MODEL)
    
    if mask_net is None:
        print("[INFO] Loading mask detector...")
        mask_net = load_model(MODEL_PATH)
    
    return face_net, mask_net

def detect_and_predict_mask(frame, face_net, mask_net):
    """Detect faces and predict mask/no-mask"""
    global detection_enabled
    
    if not detection_enabled:
        return frame, []
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()
    
    results = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            
            face_resized = cv2.resize(face, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = img_to_array(face_rgb)
            face_preprocessed = preprocess_input(face_array)
            face_batch = np.expand_dims(face_preprocessed, axis=0)
            
            (mask, withoutMask) = mask_net.predict(face_batch, verbose=0)[0]
            
            label = "Mask" if mask > withoutMask else "No Mask"
            confidence_val = max(mask, withoutMask) * 100
            
            results.append({
                'label': label,
                'confidence': float(confidence_val),
                'box': [int(startX), int(startY), int(endX), int(endY)]
            })
    
    return frame, results

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """Detect masks in frame from webcam stream"""
    try:
        data = request.json
        
        if not data or 'frame' not in data:
            return jsonify({'status': 'error', 'message': 'No frame data'}), 400
        
        frame_data = data['frame']
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        frame_pil = Image.open(io.BytesIO(frame_bytes))
        frame = np.array(frame_pil)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if face_net is None or mask_net is None:
            load_models()
        
        frame_processed, detections = detect_and_predict_mask(frame, face_net, mask_net)
        
        return jsonify({'status': 'success', 'detections': detections})
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Detect masks in uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_net is None or mask_net is None:
            load_models()
        
        result_frame, detections = detect_and_predict_mask(frame, face_net, mask_net)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        filepath = os.path.join(CAPTURE_FOLDER, filename)
        cv2.imwrite(filepath, result_frame)
        
        _, buffer = cv2.imencode('.jpg', result_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'image': f"data:image/jpeg;base64,{img_base64}",
            'detections': detections,
            'filename': filename
        })
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection on/off"""
    global detection_enabled
    detection_enabled = not detection_enabled
    return jsonify({
        'status': 'success',
        'detection_enabled': detection_enabled
    })

@app.route('/get_captures')
def get_captures():
    """Get list of captured images"""
    files = os.listdir(CAPTURE_FOLDER)
    images = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort(reverse=True)
    return jsonify({'captures': images})

@app.route('/captures/<filename>')
def serve_capture(filename):
    """Serve captured image"""
    return send_from_directory(CAPTURE_FOLDER, filename)

# ==================== MAIN ====================

if __name__ == '__main__':
    print("="*70)
    print("🎭 Face Mask Detection - Web UI")
    print("="*70)
    print("Starting Flask server...")
    print("Open browser: http://127.0.0.1:5000")
    print("="*70)
    
    # Load models at startup
    load_models()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
