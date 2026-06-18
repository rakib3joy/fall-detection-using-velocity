# Human Fall Detection Through Velocity-Driven Temporal Deep Learning

An IEEE ICCIT 2025 published research project that combines YOLOv8 pose estimation and LSTM-based temporal modeling to detect falls, unstable movements, and normal activities from video data.

## 🎯 Overview

This project implements an advanced fall detection system using computer vision and machine learning techniques. The system processes video data to extract human pose keypoints, calculates velocity metrics, and uses deep learning models to classify movement patterns as normal, unstable, or fall events.

## 📖 Publication Status

🎉 **This research has been published in ICCIT 2025.** The code is made available for academic and research purposes. Please cite our work if you use this code in your research.

**Paper:**
Human Fall Detection Through Velocity-Driven Temporal Deep Learning

**Venue:**
28th International Conference on Computer and Information Technology (ICCIT 2025)

**Status:** Published

**DOI:**
[10.1109/ICCIT68739.2025.11491387](https://doi.org/10.1109/ICCIT68739.2025.11491387)

**IEEE Xplore:**
https://ieeexplore.ieee.org/document/11491387

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Manual Installation](#-manual-installation-alternative)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Data Processing](#-data-processing-pipeline)
- [Models](#-models)
- [Results](#-results)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#%EF%B8%8F-license)
- [Performance Monitoring](#-performance-monitoring)
- [Technical Details](#-technical-details)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

## 🚀 Features

- **Real-time Pose Detection**: Uses YOLOv8-pose for accurate human keypoint detection
- **Velocity Analysis**: Calculates vertical velocity using shoulder keypoint movements
- **Multi-class Classification**: Distinguishes between fall, normal, and unstable movements
- **LSTM-based Prediction**: Employs sequence modeling for temporal pattern recognition
- **Comprehensive Preprocessing**: Includes data normalization, interpolation, and smoothing
- **Robust Evaluation**: Multiple metrics and visualization tools for model assessment

### Robust Keypoint Selection
- Intelligent selection between left/right shoulder keypoints
- Confidence threshold filtering
- Fallback mechanisms for missing data

### Advanced Data Processing
- Height-based coordinate normalization
- Temporal smoothing for noise reduction
- Derivative feature calculation (velocity, acceleration)

### Model Reliability
- Video-level data splitting to prevent leakage
- Early stopping and regularization
- Comprehensive evaluation metrics

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/rakib3joy/fall-detection-using-velocity.git
cd fall-detection-using-velocity
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the preprocessing pipeline:**
```bash
python fall_detection_preprocessor.py
```

4. **Train models using Jupyter notebooks:**
- Open `velocity_prediction.ipynb` for velocity prediction model
- Open `lstm-with-custom-min-max-norm.ipynb` for classification model

## 🛠 Manual Installation (Alternative)

If you prefer to install packages individually:

```bash
# Core Machine Learning and Deep Learning
pip install tensorflow>=2.10.0
pip install torch>=1.12.0 torchvision>=0.13.0

# Computer Vision and Pose Estimation
pip install ultralytics>=8.0.0 opencv-python>=4.6.0

# Data Processing and Analysis
pip install numpy>=1.21.0 pandas>=1.4.0 scikit-learn>=1.1.0 scipy>=1.8.0

# Signal Processing and Filtering
pip install filterpy>=1.4.5

# Visualization
pip install matplotlib>=3.5.0 seaborn>=0.11.0 plotly>=5.0.0

# Model Persistence and Utilities
pip install joblib>=1.1.0 h5py>=3.7.0

# Jupyter Notebook Support
pip install jupyter>=1.0.0 ipykernel>=6.0.0 notebook>=6.4.0

# Progress bars and utilities
pip install tqdm>=4.64.0
```

## 🔧 Usage

### 1. Data Preprocessing

Process video files to extract pose keypoints and velocity data:

```bash
python fall_detection_preprocessor.py
```

This script will:
- Process all `.mp4` files in the `./manual testing` directory
- Extract 17 keypoints using YOLOv8-pose
- Calculate vertical velocity using shoulder keypoints
- Output CSV files with pose data and velocity measurements

### 2. Model Training

#### LSTM Classification Model

Train the LSTM model for fall classification:

```bash
# Run the lstm-with-custom-min-max-norm.ipynb notebook
# This will train a model to classify movements as fall/normal/unstable
```

#### Velocity Prediction Model

Train the velocity prediction LSTM model:

```bash
# Run the velocity_prediction.ipynb notebook
# This will train a model to predict vertical velocity from pose sequences
```

## 📁 Repository Structure

```
fall-detection-using-velocity/
├── fall_detection_preprocessor.py    # Main preprocessing pipeline
├── lstm-with-custom-min-max-norm.ipynb    # Classification model
├── velocity_prediction.ipynb         # Velocity prediction model  
├── velocity_yolo.py                  # YOLO-based velocity detection
├── graph.py                          # Visualization utilities
├── requirements.txt                  # Package dependencies
├── README.md                         # Documentation
└── .gitignore                        # Git exclusions
```

## 🔄 Data Processing Pipeline

### 1. Pose Extraction
- **Input**: Video files (.mp4)
- **Process**: YOLOv8-pose keypoint detection
- **Output**: 17 keypoints with confidence scores per frame

### 2. Velocity Calculation
- **Method**: Frame-to-frame shoulder keypoint displacement
- **Formula**: `velocity = (current_y - previous_y) / time_interval`
- **Normalization**: Absolute velocity values

### 3. Data Preprocessing
- **Interpolation**: Fill missing keypoints using neighboring valid points
- **Height Normalization**: Normalize coordinates relative to body height
- **Smoothing**: Apply moving average to reduce noise
- **Sequence Generation**: Create temporal windows for LSTM input

## 🧠 Models

### 1. Classification Model (LSTM)

**Architecture:**
- Input: Normalized pose keypoints (shoulder, hip, knee, ankle Y-coordinates)
- LSTM layers with masking for variable-length sequences
- Dropout for regularization
- Softmax output for 3-class classification

**Features:**
- Min-max normalization per sequence
- Class weight balancing
- Early stopping to prevent overfitting

### 2. Velocity Prediction Model (LSTM)

**Architecture:**
- Input: X and Y coordinates of all 17 keypoints + velocity/acceleration features
- Multi-layer LSTM with dropout
- Dense output layer for velocity prediction

**Features:**
- Standard scaling normalization
- Video-based train/test split to prevent data leakage
- Comprehensive evaluation metrics (MAE, R², MSE, MAPE)

## 📊 Results

The proposed two-stage framework combines YOLOv8 pose estimation with LSTM-based velocity prediction and activity classification to achieve robust fall detection in real-world scenarios.

### Activity Classification Performance
- Overall Classification Accuracy: **90%**
- Macro F1-Score: **0.87**
- Fall Detection F1-Score: **0.91**
- Normal Activity F1-Score: **0.95**
- Unstable Activity F1-Score: **0.76**

### Velocity Prediction Performance
- Mean Absolute Error (MAE): **97.65 pixels/second**
- Equivalent to approximately 3.91 pixels per frame at 25 FPS
- Error corresponds to approximately 0.26% of frame height

### Dataset Statistics

| Category | Count |
|----------|-------|
| Total Videos | 3,449 |
| Fall Samples | 655 |
| Unstable Samples | 586 |
| Normal Samples | 2,211 |

- Resolution: 2688 × 1520
- Frame Rate: 25 FPS

### Key Contributions
- Two-stage deep learning framework combining velocity prediction and activity classification
- YOLOv8-based pose estimation with temporal LSTM modeling
- Non-wearable vision-based fall detection system
- Published at ICCIT 2025 (IEEE)

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@INPROCEEDINGS{11491387,
  title={Human Fall Detection Through Velocity-Driven Temporal Deep Learning},
  author={Nohan Ahmed, MD Rakib Hasan, Farhan Haque Shakib, Hosni Rabbani, Rawhatur Rabbi, Sumaiya Tanjil Khan and Aniqua Nusrat Zereen},
  booktitle={2025 28th International Conference on Computer and Information Technology (ICCIT)},
  year={2025},
  doi={10.1109/ICCIT68739.2025.11491387}
}
```

## 📈 Performance Monitoring

The system includes extensive logging and visualization:
- Frame processing statistics
- Model training curves
- Prediction accuracy metrics
- Error distribution analysis

## 🔍 Technical Details

### Keypoint Indices
```python
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
```

### Configuration Parameters
- **Confidence Threshold**: 0.5 for keypoint validity
- **Sequence Length**: 10 frames for LSTM input (velocity prediction)
- **Max Sequence Length**: 128 frames (classification)
- **Smoothing Window**: 3 frames for noise reduction

## 🤝 Contributing

This is a research publication repository. For questions or collaboration:
- Open an issue for bug reports
- Contact authors for research collaboration

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dr. Aniqua Nusrat Zereen for research supervision and guidance
- Ultralytics YOLOv8 for pose estimation
- TensorFlow/Keras for deep learning model development

## 📞 Contact

- **Author:** MD Rakib Hasan - rakibhasanjoy286@gmail.com
- **Research Supervisor:** Dr. Aniqua Nusrat Zereen - aniqua@bracu.ac.bd

---

**Note**: This code is provided for research and educational purposes. Ensure proper validation for production use.