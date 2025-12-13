# CECS-456-Project
Chest X-Ray Images (Pneumonia) Classification Using CNN

## Overview
Deep learning model using Convolutional Neural Networks (CNN) to classify chest X-ray images as Normal or Pneumonia.

**Model 1 Test Accuracy: 85.58%**
**Model 2 Test Accuracy: 90.38%**

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or VS Code with Jupyter extension

## Setup Instructions

**Important:** All commands should be run in a terminal/command prompt from the `CECS-456-Project` directory.

### 1. Clone the Repository
```bash
git clone https://github.com/brandon-smsn/CECS-456-Project.git
cd CECS-456-Project
```

### 2. Create Virtual Environment and Install Dependencies

#### Windows (Command Prompt or PowerShell)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Command Prompt:
venv\Scripts\activate.bat
# For PowerShell:
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### Mac/Linux (Terminal)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Verify Installation:**
```bash
python --version  # Should show Python 3.8+
pip list          # Should show installed packages (tensorflow, keras, numpy, etc.)
```

### 3. Download Dataset
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

### 4. Run the Notebooks

#### Option A: Using VS Code (Recommended)
1. Open VS Code in the project directory
2. Open `Model1_CECS_456_Pneumonia_CNN.ipynb` or `Model2_CECS_456_Pneumonia_CNN.ipynb`
3. Select the virtual environment kernel (venv)
4. Run cells sequentially from top to bottom

#### Option B: Using Jupyter Notebook
```bash
# Make sure virtual environment is activated first!

# Launch Jupyter Notebook
jupyter notebook

# This will open in your browser
# Navigate to Model1_CECS_456_Pneumonia_CNN.ipynb or Model2_CECS_456_Pneumonia_CNN.ipynb
# Run cells sequentially from top to bottom
```

#### Option C: Run as Python Script (Advanced)
If you want to convert and run the notebook as a script:
```bash
# Install nbconvert if not already installed
pip install nbconvert

# Convert notebook to Python script
jupyter nbconvert --to script Model1_CECS_456_Pneumonia_CNN.ipynb

# Run the script
python Model1_CECS_456_Pneumonia_CNN.py
```

**Note:** Training takes approximately 10-20 minutes depending on your hardware.

## Model 1 Architecture
- 3 Convolutional Blocks with BatchNormalization
- MaxPooling and Dropout layers for regularization
- Dense layers with 256 and 128 units
- Binary classification output (Sigmoid activation)

## Model 2 Architecture
- 4 Convolutional Blocks with BatchNormalization and increasing filter sizes (32, 64, 128)
- MaxPooling, AveragePooling and Dropout layers for regularization
- Progressive Dropout regularization (0.1 -> 0.15 -> 0.25 -> 0.25)
- Single Dense layer with 256 units (ReLU activation)
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

## Troubleshooting

### Virtual Environment Not Activating (Windows PowerShell)
If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "python command not found" (Mac/Linux)
Try using `python3` instead of `python`:
```bash
python3 -m venv venv
```

### Module Not Found Errors
Make sure your virtual environment is activated and dependencies are installed:
```bash
# Check if venv is activated (you should see (venv) in your terminal prompt)
# If not, activate it again (see Step 2 above)

# Reinstall dependencies
pip install -r requirements.txt
```

### Jupyter Kernel Issues
If the notebook can't find packages:
```bash
# With venv activated:
pip install ipykernel
python -m ipykernel install --user --name=venv
# Then select the "venv" kernel in Jupyter/VS Code
```