"""
Helper functions for Face Mask Detection System
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_face_detector(prototxt_path, model_path):
    """
    Load OpenCV DNN face detector model

    Args:
        prototxt_path: Path to deploy.prototxt file
        model_path: Path to .caffemodel file

    Returns:
        Face detector model
    """
    print("[INFO] Loading face detector model...")
    face_net = cv2.dnn.readNet(prototxt_path, model_path)
    return face_net


def detect_faces(image, face_net, confidence_threshold=0.5):
    """
    Detect faces in an image using OpenCV DNN

    Args:
        image: Input image
        face_net: Pre-loaded face detector model
        confidence_threshold: Minimum confidence for detection

    Returns:
        List of face locations and confidences
    """
    (h, w) = image.shape[:2]

    # Construct blob from image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # Pass blob through network
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []

    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > confidence_threshold:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding boxes are within image dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            faces.append((startX, startY, endX, endY))
            locs.append(confidence)

    return faces, locs


def preprocess_face(face, target_size=(224, 224)):
    """
    Preprocess face image for mask detection model

    Args:
        face: Face ROI image
        target_size: Target size for model input

    Returns:
        Preprocessed face array
    """
    # Resize face
    face = cv2.resize(face, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Convert to array and preprocess
    face = img_to_array(face)
    face = preprocess_input(face)

    return face


def draw_prediction(image, box, label, confidence, color):
    """
    Draw bounding box and label on image

    Args:
        image: Input image
        box: Bounding box coordinates (startX, startY, endX, endY)
        label: Class label
        confidence: Prediction confidence
        color: Box color in BGR format

    Returns:
        Image with drawn predictions
    """
    (startX, startY, endX, endY) = box

    # Draw bounding box
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # Draw label with confidence
    label_text = f"{label}: {confidence:.2f}%"
    cv2.putText(image, label_text, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image
