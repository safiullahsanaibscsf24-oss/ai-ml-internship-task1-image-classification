# Image Classification Model — AI/ML Internship, Week 01

## Overview
This project implements a Convolutional Neural Network (CNN) to classify handwritten digit images (0–9) from the **MNIST** dataset. It covers the full pipeline: dataset preprocessing, CNN architecture design, training, and performance evaluation.

## Objective
Develop a deep learning model capable of accurately classifying images into predefined categories, gaining hands-on experience in computer vision, image preprocessing, neural networks, and model evaluation.

## Dataset
- **Source:** MNIST handwritten digits dataset
- **Size:** 70,000 grayscale images (60,000 train / 10,000 test), 28×28 pixels
- **Classes:** 10 (digits 0–9)

## Project Structure
```
.
├── data/                       # Raw MNIST dataset files (idx format, gzip-compressed)
├── models/
│   └── mnist_cnn_model.keras   # Trained model file
├── notebooks/
│   └── Image_Classification_Model.ipynb   # Full walkthrough notebook
├── reports/
│   ├── metrics.json            # Accuracy / precision / recall / F1 / per-class report
│   ├── confusion_matrix.png    # Confusion matrix visualization
│   └── training_curves.png     # Accuracy & loss curves
├── src/
│   └── train_model.py          # End-to-end training + evaluation script
├── Performance_Evaluation_Report.md
└── README.md
```

## Model Architecture
A CNN built with TensorFlow/Keras:
- 2 Convolutional blocks (Conv2D → BatchNormalization → MaxPooling → Dropout)
- Fully connected layers with Dropout for regularization
- Softmax output layer for 10-class classification

## How to Run
```bash
# Install dependencies
pip install tensorflow-cpu scikit-learn matplotlib seaborn

# Train and evaluate the model
python src/train_model.py
```
This will preprocess the data, train the CNN, save the trained model to `models/`, and write evaluation metrics/plots to `reports/`.

Alternatively, open `notebooks/Image_Classification_Model.ipynb` for an annotated, step-by-step walkthrough.

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 99.02% |
| Precision (macro) | 99.02% |
| Recall (macro) | 99.00% |
| F1-Score (macro) | 99.01% |

See `Performance_Evaluation_Report.md` and `reports/` for full details, the confusion matrix, and training curves.

## Deliverables
1. Source Code (`src/train_model.py`)
2. Trained Model File (`models/mnist_cnn_model.keras`)
3. Jupyter Notebook (`notebooks/Image_Classification_Model.ipynb`)
4. Performance Evaluation Report (`Performance_Evaluation_Report.md`)
5. GitHub Repository Link (this repo)

## Author
** Safiullah Sanai **
AI/ML Internship — Week 01 submission.
