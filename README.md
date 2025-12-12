# CECS-456-Project
Chest X-Ray Images (Pneumonia) Classification Using CNN

## Overview
Deep learning model using Convolutional Neural Networks (CNN) to classify chest X-ray images as Normal or Pneumonia.

**Model 1 Test Accuracy: 85.58%**
**Model 2 Test Accuracy: 90.38%**

## Setup Instructions

### 1. Create Virtual Environment to Install Dependencies
```bash
python -m venv venv
.\venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 2. Download Dataset
1. Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
   - Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Extract the downloaded zip file
3. Place the `chest_xray` folder in the project root directory

### 3. Expected Directory Structure
```
CECS-456-Project/
├── venv/                    # Virtual environment (ignored by git)
├── requirements.txt
├── Model1_CECS_456_Pneumonia_CNN.ipynb
├── Model2_CECS_456_Pneumonia_CNN.ipynb
├── README.md
└── chest_xray/              # Dataset folder (not in git)
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

### 4. Run the Notebook
Open `Model1_CECS_456_Pneumonia_CNN.ipynb` and run cells in order from top to bottom.

Open `Model2_CECS_456_Pneumonia_CNN.ipynb` and run cells in order from top to bottom.

**Note:** Training takes approximately 10-20 minutes depending on your hardware.

## Model 1 Architecture
- 3 Convolutional Blocks with BatchNormalization
- MaxPooling and Dropout layers for regularization
- Dense layers with 256 and 128 units
- Binary classification output (Sigmoid activation)

## Model 2 Architecture
- 3 Convolutional Blocks with BatchNormalization
- MaxPooling and Dropout layers for regularization
- Dense layers with 256 and 128 units
- Binary classification output (Sigmoid activation)

## Dataset
- **Training**: 5,216 images
- **Validation**: 16 images
- **Test**: 624 images
- **Classes**: Normal (0), Pneumonia (1)

## Results: Model 1
- **Test Accuracy**: 85.58%
- **Image Size**: 150x150 pixels
- **Data Augmentation**: Rotation, shifts, zoom, horizontal flip

## Results: Model 2
- **Test Accuracy**: 90.38%
- **Image Size**: 150x150 pixels
- **Data Augmentation**: Rotation, shifts, zoom, horizontal flip

## Files Generated
- `Model1_best_pneumonia_model.h5` - Best model during training
- `Model1_pneumonia_cnn_final_model.h5` - Final trained model
- `Model1_model_architecture.json` - Model structure
- `Model2_best_pneumonia_model.h5` - Best model during training
- `Model2_pneumonia_cnn_final_model.h5` - Final trained model
- `Model2_model_architecture.json` - Model structure