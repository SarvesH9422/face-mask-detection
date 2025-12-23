#!/bin/bash

echo "========================================="
echo "Face Mask Detection - Quick Start Setup"
echo "========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run dataset preparation
echo "Setting up dataset and models..."
python scripts/prepare_dataset.py

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Add training images to dataset/with_mask and dataset/without_mask"
echo "2. Run training: python scripts/train.py"
echo "3. Test detection: python scripts/detect_mask_video.py"
echo ""
echo "For more info, see README.md"
