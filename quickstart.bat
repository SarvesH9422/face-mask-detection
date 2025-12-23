@echo off

echo =========================================
echo Face Mask Detection - Quick Start Setup
echo =========================================

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Run dataset preparation
echo Setting up dataset and models...
python scripts/prepare_dataset.py

echo.
echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Add training images to dataset/with_mask and dataset/without_mask
echo 2. Run training: python scripts/train.py
echo 3. Test detection: python scripts/detect_mask_video.py
echo.
echo For more info, see README.md

pause
