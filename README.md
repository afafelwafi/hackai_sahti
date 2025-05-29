# Heartbeat Audio Classification with Pre-trained Models

This repository contains code for heartbeat audio classification using pre-trained audio encoders (Wav2Vec2, WavLM) enhanced with multi-label classification and specific disease detection capabilities. The project was developed during the Hack AI Sahti hackathon.

## 🎯 Project Overview

The system classifies heartbeat audio recordings to detect various heart conditions:
- **Binary Classification**: N : (0 : disease, 1: Normal)
- **Multi-label Classification**: Multiple heart conditions simultaneously
  - N: Normal
  - AS: Aortic Stenosis  
  - AR: Aortic Regurgitation
  - MR: Mitral Regurgitation
  - MS: Mitral Stenosis

## 📁 Project Structure

```
├── main.py                 # Main training and evaluation script
├── config.json            # Configuration file with all parameters
├── data_utils.py          # Dataset loading and preprocessing utilities
├── dataset.py             # PyTorch dataset and dataloader implementations
├── models.py              # Neural network model definitions
├── train.py               # Training and evaluation functions
├── extract_data.py        # Data downloading utilities
├── config_loader.py       # Configuration management
├── requirements.txt       # Python dependencies
└── models/                # Directory for saved models and results
```

## 🔧 Key Components

### Data Processing (`data_utils.py`)
- Loads audio files (.wav) and metadata from CSV files
- Performs subject-level train/validation splits to prevent data leakage
- Normalizes demographic features (age, gender, etc.)
- Supports both binary and multi-label classification

### Dataset (`dataset.py`)
- Custom PyTorch dataset for heartbeat audio
- Handles audio preprocessing (resampling, normalization, padding/truncation)
- Integrates demographic information with audio features

### Models (`models.py`)
- **PretrainedAudioEncoder**: Uses pre-trained models for feature extraction
- **HeartbeatLinearProbe**: Classification head with different architectures for binary/multi-label tasks
- Supports multiple pre-trained models (Wav2Vec2, WavLM)

### Training (`train.py`)
- Feature extraction using pre-trained encoders
- Training loop with early stopping and learning rate scheduling
- Comprehensive evaluation with confusion matrices and classification reports

## 🚀 Setup and Installation

### Prerequisites
- Hugging Face Token
- Python 3.7+
- CUDA-compatible GPU (recommended)

### Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
torch>=1.9.0
torchaudio>=0.9.0
transformers>=4.15.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
tqdm>=4.62.0
```

## 📊 Configuration
First set your HF token
```bash
import os
os.environ['HF_TOKEN'] = 'YOUR_API_TOKEN_HERE'
```
The `config.json` file contains all training parameters:

```json
{
  "dataset": {
    "disease_classes": ["N", "AS", "AR", "MR", "MS"],
    "train_csv": "train.csv",
    "metadata_csv": "additional_metadata.csv"
  },
  "training": {
    "batch_size": 16,
    "epochs": 200,
    "learning_rate": 0.001,
    "multi_label": false
  },
  "models": {
    "default_model": "facebook/wav2vec2-base"
  }
}
```

## 🏃‍♂️ Usage

### Basic Training
```bash
# Binary classification (Normal vs Abnormal)
python main.py

# Multi-label classification
python main.py --multi_label

# Custom configuration
python main.py --config custom_config.json
```

### Advanced Options
```bash
python main.py \
    --data_dir /path/to/your/data \
    --model_name facebook/wav2vec2-large \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.0005 \
    --multi_label \
    --freeze_encoder
```

### Available Pre-trained Models
- `facebook/wav2vec2-base`
- `facebook/wav2vec2-large`  
- `facebook/wav2vec2-large-960h`
- `microsoft/wavlm-base-plus`

## 📈 Model Architecture

### Feature Extraction
1. **Pre-trained Audio Encoder**: Extracts rich audio representations
2. **Global Average Pooling**: Reduces temporal dimension
3. **Demographic Integration**: Concatenates demographic features

### Classification Head
- **Binary**: Simple 3-layer MLP
- **Multi-label**: Enhanced 4-layer architecture with BatchNorm and dropout

### Training Strategy
- **Linear Probing**: Frozen pre-trained encoder (default)
- **Fine-tuning**: Trainable encoder (optional)
- **Subject-level Split**: Prevents patient data leakage
- **Early Stopping**: Prevents overfitting

## 📊 Data Format

### Expected Directory Structure
```
dataset/
├── train.csv                    # Main metadata file
├── additional_metadata.csv      # Demographic information
└── audio_files/
    ├── N_001.wav               # Normal heartbeat samples
    ├── AS_002.wav              # Aortic Stenosis samples
    └── ...
```

### CSV Format
- **train.csv**: Contains patient_id, recording columns, and disease labels
- **additional_metadata.csv**: Contains demographic information (age, gender, etc.)

## 🎯 Performance Metrics

### Binary Classification
- Accuracy
- Sensitivity (Abnormal Detection Rate)
- Specificity (Normal Detection Rate)
- F1-Score

### Multi-label Classification
- Per-class metrics
- Micro/Macro F1-Score
- Confusion matrices per class

## 📁 Output Files

After training, the following files are generated:
- `models/best_heartbeat_classifier.pth` - Best binary model
- `models/best_multi_label_heartbeat_classifier.pth` - Best multi-label model
- `models/results_*.json` - Training results and metrics

## 🔍 Key Features

### Data Handling
- **Subject-level Splitting**: Ensures no patient data leakage between train/val sets
- **Audio Preprocessing**: Automatic resampling, normalization, and length standardization
- **Demographic Integration**: Incorporates patient metadata (age, gender, etc.)

### Training Features
- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Gradient Clipping**: Stabilizes training
- **Class Balancing**: Weighted loss for imbalanced datasets

### Model Flexibility
- **Multiple Pre-trained Models**: Support for various Wav2Vec2 and WavLM variants
- **Configurable Architecture**: Easy switching between binary and multi-label
- **Frozen vs Fine-tuning**: Option to freeze or train encoder weights


## Notes

This project was shared for educational purposes with hackathon contestants. Feel free to:
- Experiment with different pre-trained models
- Modify the classification architecture
- Add new evaluation metrics
- Improve data preprocessing


## 🔗 Dataset

The project uses the BMD-HS Dataset. If no local data directory is provided, the system will automatically download the dataset from the configured repository.

---

**Disclaimer**: This is an educational project developed during a hackathon. It should not be used for actual medical diagnosis without proper validation and regulatory approval.
