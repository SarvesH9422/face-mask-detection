"""
Configuration file for Face Mask Detection System
"""

# Model Configuration
MODEL_PATH = "models/mask_detector.keras"
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# Dataset Configuration
DATASET_PATH = "dataset"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Face Detector Configuration
FACE_PROTOTXT = "face_detector/deploy.prototxt"
FACE_MODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5

# Class Labels
CLASSES = ["with_mask", "without_mask"]

# Colors for visualization (BGR format)
COLORS = {
    "with_mask": (0, 255, 0),      # Green
    "without_mask": (0, 0, 255)     # Red
}

# Image Processing
IMAGE_SIZE = (224, 224)
MEAN_VALUES = (104.0, 177.0, 123.0)
