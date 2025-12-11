# CECS-456-Project
Chest X-Ray Images (Pneumonia) Classification Using CNN

## Overview
Deep learning model using Convolutional Neural Networks (CNN) to classify chest X-ray images as Normal or Pneumonia.

**Test Accuracy: 85.58%**

## Setup Instructions

### 1. Install Dependencies
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### 2. Download Dataset
1. Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
   - Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Extract the downloaded zip file
3. Place the `chest_xray` folder in the project root directory

### 3. Expected Directory Structure
```
CECS-456-Project/
├── CECS_456_Pneumonia_CNN.ipynb
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
Open `CECS_456_Pneumonia_CNN.ipynb` and run cells in order from top to bottom.

**Note:** Training takes approximately 10-20 minutes depending on your hardware.

## Model Architecture
- 3 Convolutional Blocks with BatchNormalization
- MaxPooling and Dropout layers for regularization
- Dense layers with 256 and 128 units
- Binary classification output (Sigmoid activation)

## Dataset
- **Training**: 5,216 images
- **Validation**: 16 images
- **Test**: 624 images
- **Classes**: Normal (0), Pneumonia (1)

## Results
- **Test Accuracy**: 85.58%
- **Image Size**: 150x150 pixels
- **Data Augmentation**: Rotation, shifts, zoom, horizontal flip

## Files Generated
- `best_pneumonia_model.h5` - Best model during training
- `pneumonia_cnn_final_model.h5` - Final trained model
- `model_architecture.json` - Model structure
