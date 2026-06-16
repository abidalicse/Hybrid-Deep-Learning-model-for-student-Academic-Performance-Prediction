# Student Academic Performance Prediction using Autoencoder + Mutual Information + CNN-GRU

## Overview
This project predicts **student academic performance (Final Grade)** using a hybrid deep learning pipeline that combines:

1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Statistical Analysis (T-Test / ANOVA)
4. Autoencoder-based Feature Extraction
5. Mutual Information Feature Selection
6. SMOTE for Class Balancing
7. CNN-GRU Deep Learning Model
8. Performance Evaluation

## Dataset

The dataset contains 5,000 student records with features such as:
- Age
- Gender
- Parental Education
- Income
- Previous Grade
- Attendance
- Study Hours
- Motivation
- Stress
- Social Media Usage
- Sleep Duration
- Final Grade (Target)

## Methodology

### Data Preprocessing
- Missing value imputation
- Duplicate removal
- Label encoding
- Min-Max normalization

### Autoencoder Feature Extraction
- Latent feature size: 20
- Optimizer: Adam
- Loss: MSE
- Epochs: 50

### Mutual Information Feature Selection
- Selects top-ranked latent features
- Top features used for classification

### SMOTE Balancing
- Handles class imbalance
- Improves minority-class prediction performance

### CNN-GRU Model
Architecture:

Input → Conv1D → MaxPooling1D → GRU → Dense → Dropout → Softmax

## Training Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 80 |
| Batch Size | 128 |
| Loss Function | Categorical Crossentropy |

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

## Installation

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn tensorflow
```

## Project Workflow

Dataset → Preprocessing → EDA → Statistical Analysis → Autoencoder → Mutual Information → SMOTE → CNN-GRU → Evaluation

## Files

```text
├── Abid paper code final.ipynb
├── Dataset(1).csv
├── selected_features.pkl
├── scaler.pkl
├── encoder_model.h5
├── original_features.pkl
├── processed_feature_names.pkl
└── README.md
```

## Author

**Abid Ali**

Student Performance Prediction using Deep Learning and Feature Engineering Techniques.
