# Performance Evaluation Report
### AI/ML Internship Task – Week 01 | Task 1: Image Classification Model

## 1. Project Summary
A Convolutional Neural Network (CNN) was developed using TensorFlow/Keras to classify handwritten digit images (0–9) from the MNIST dataset. The pipeline covered dataset preprocessing, CNN architecture design, model training with early stopping and learning rate scheduling, and evaluation on a held-out test set.

## 2. Dataset
- **Source:** MNIST handwritten digits
- **Size:** 70,000 grayscale 28×28 images, 10 classes (digits 0–9)
- **Split:** 54,000 train / 6,000 validation / 10,000 test
- **Preprocessing:** Pixel values normalized to [0, 1]

## 3. Model Architecture
Two convolutional blocks (Conv2D → BatchNormalization → MaxPooling → Dropout), followed by a dense layer with dropout and a softmax output layer. ~422,000 trainable parameters.

## 4. Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 256 |
| Epochs | 8 (EarlyStopping + ReduceLROnPlateau) |

## 5. Evaluation Metrics (Test Set)
| Metric | Score |
|---|---|
| Test Accuracy | 99.02% |
| Precision (macro avg) | 99.02% |
| Recall (macro avg) | 99.00% |
| F1-Score (macro avg) | 99.01% |
| Test Loss | 0.0339 |

## 6. Visualizations
- `reports/training_curves.png` — Accuracy & loss curves
- `reports/confusion_matrix.png` — Confusion matrix on test set

## 7. Results Analysis
The model achieved 99.02% test accuracy with macro-averaged precision, recall, and F1-score all near 99%. Training and validation curves converge closely, indicating good generalization with no significant overfitting. The confusion matrix shows very few misclassifications, mostly between visually similar digits (e.g., 4/9, 3/5).

## 8. Suggested Improvements
- Add data augmentation (rotation, shift, zoom) for robustness.
- Try deeper/residual architectures for further gains.
- Apply advanced LR scheduling (cosine decay, warm-up).
- Ensemble multiple models.
- Test on out-of-distribution handwriting samples.

*A formatted PDF version of this report is available as `Performance_Evaluation_Report.pdf`.*
