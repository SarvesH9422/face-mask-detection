"""
Flask Web UI for Face Mask Detection
Real-time webcam streaming with controls
"""

from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import os
from datetime import datetime
import base64
import urllib.request
# Google Drive direct download links (replace with your links)
MODEL_URLS = {
    'models/mask_detector.keras': 'https://drive.google.com/uc?export=download&id=1AbZChTE0LRV8YK_P_LB6L07-Dui2tFbF',
    'face_detector/res10_300x300_ssd_iter_140000.caffemodel': 'https://drive.google.com/uc?export=download&id=1aU6Ll0-NKAtDhaPWebI47BQpBIAuD6ud',
    'face_detector/deploy.prototxt': 'https://drive.google.com/uc?export=download&id=1X-Ky_V1UdXpAq9-zTqm8qT_o4g-2IyZW'
}

def download_models():
    """Download models on first run"""
    for filepath, url in MODEL_URLS.items():
        if not os.path.exists(filepath):
            print(f"[INFO] Downloading {filepath}...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"[SUCCESS] Downloaded {filepath}")
            except Exception as e:
                print(f"[ERROR] Failed to download {filepath}: {e}")

# Download models before starting app
print("Checking models...")
download_models()
print("Models ready!")

app = Flask(__name__)

# ============ CONFIGURATION ============
MODEL_PATH = "models/mask_detector.keras"
FACE_PROTOTXT = "face_detector/deploy.prototxt"
FACE_MODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = (224, 224)
CAPTURE_FOLDER = "captures"

COLOR_MASK = (0, 255, 0)
COLOR_NO_MASK = (0, 0, 255)

# Global variables
camera = None
is_streaming = False
face_net = None
mask_net = None
detection_enabled = True

# Create captures folder
os.makedirs(CAPTURE_FOLDER, exist_ok=True)


def load_models():
    global face_net, mask_net

    if face_net is None:
        print("[INFO] Loading face detector...")
        face_net = cv2.dnn.readNet(FACE_PROTOTXT, FACE_MODEL)

    if mask_net is None:
        print("[INFO] Loading mask detector...")
        model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH.replace('.keras', '.h5')
        mask_net = load_model(model_path)

    return face_net, mask_net


def detect_faces(image, face_net, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            faces.append((startX, startY, endX, endY))

    return faces


def detect_and_predict_mask(frame, face_net, mask_net):
    global detection_enabled

    if not detection_enabled:
        return frame, []

    faces = detect_faces(frame, face_net, CONFIDENCE_THRESHOLD)
    results = []

    for (startX, startY, endX, endY) in faces:
        face = frame[startY:endY, startX:endX]

        if face.shape[0] < 20 or face.shape[1] < 20:
            continue

        face_resized = cv2.resize(face, IMAGE_SIZE)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_array = img_to_array(face_rgb)
        face_preprocessed = preprocess_input(face_array)
        face_batch = np.expand_dims(face_preprocessed, axis=0)

        (mask, withoutMask) = mask_net.predict(face_batch, verbose=0)[0]

        label = "Mask" if mask > withoutMask else "No Mask"
        color = COLOR_MASK if label == "Mask" else COLOR_NO_MASK
        confidence = max(mask, withoutMask) * 100

        results.append({
            'label': label,
            'confidence': float(confidence),
            'box': [int(startX), int(startY), int(endX), int(endY)]
        })

        label_text = f"{label}: {confidence:.2f}%"
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label_text, (startX, startY - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, results


def generate_frames():
    global camera, face_net, mask_net

    while is_streaming:
        success, frame = camera.read()
        if not success:
            break

        # Detect and draw
        frame, _ = detect_and_predict_mask(frame, face_net, mask_net)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, is_streaming, face_net, mask_net

    try:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({'status': 'error', 'message': 'Camera not found'}), 400

        # Load models
        face_net, mask_net = load_models()
        is_streaming = True

        return jsonify({'status': 'success', 'message': 'Camera started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, is_streaming

    is_streaming = False
    if camera is not None:
        camera.release()
        camera = None

    return jsonify({'status': 'success', 'message': 'Camera stopped'})


@app.route('/capture_image', methods=['POST'])
def capture_image():
    global camera

    if camera is None or not camera.isOpened():
        return jsonify({'status': 'error', 'message': 'Camera not active'}), 400

    success, frame = camera.read()
    if not success:
        return jsonify({'status': 'error', 'message': 'Failed to capture'}), 500

    # Detect on captured frame
    frame, results = detect_and_predict_mask(frame, face_net, mask_net)

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(CAPTURE_FOLDER, filename)
    cv2.imwrite(filepath, frame)

    return jsonify({
        'status': 'success',
        'message': 'Image captured',
        'filename': filename,
        'results': results
    })


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    return jsonify({
        'status': 'success',
        'detection_enabled': detection_enabled
    })


@app.route('/get_captures')
def get_captures():
    files = os.listdir(CAPTURE_FOLDER)
    images = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort(reverse=True)
    return jsonify({'captures': images})


@app.route('/captures/<filename>')
def serve_capture(filename):
    return send_from_directory(CAPTURE_FOLDER, filename)


@app.route('/delete_capture/<filename>', methods=['DELETE'])
def delete_capture(filename):
    filepath = os.path.join(CAPTURE_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'status': 'success', 'message': 'Image deleted'})
    return jsonify({'status': 'error', 'message': 'File not found'}), 404


if __name__ == '__main__':
    print("="*70)
    print("🎭 Face Mask Detection - Web UI")
    print("="*70)
    print("Starting Flask server...")
    print("Open browser: http://127.0.0.1:5000")
    print("="*70)
    app.run(debug=True, threaded=True)
