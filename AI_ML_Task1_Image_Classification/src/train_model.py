"""
AI/ML Internship Task - Week 01
Task 1: Image Classification Model
------------------------------------
Trains a Convolutional Neural Network (CNN) on the MNIST handwritten
digits dataset to classify images into 10 categories (digits 0-9),
then evaluates performance using Accuracy, Precision, Recall,
F1-Score, and a Confusion Matrix.

Dataset: MNIST (70,000 28x28 grayscale images of handwritten digits, 0-9)
Framework: TensorFlow / Keras
"""

import gzip
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [str(i) for i in range(10)]

DATA_DIR = "data"
MODEL_DIR = "models"
REPORT_DIR = "reports"
EPOCHS = 8
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Step 1 & 2: Dataset Selection + Preprocessing
# ---------------------------------------------------------------------------
def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    num_images = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(num_images, rows, cols)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def load_and_preprocess_data():
    print("Loading MNIST dataset...")
    x_train_full = _read_idx_images(f"{DATA_DIR}/train-images-idx3-ubyte.gz")
    y_train_full = _read_idx_labels(f"{DATA_DIR}/train-labels-idx1-ubyte.gz")
    x_test = _read_idx_images(f"{DATA_DIR}/t10k-images-idx3-ubyte.gz")
    y_test = _read_idx_labels(f"{DATA_DIR}/t10k-labels-idx1-ubyte.gz")

    # Add channel dimension (grayscale) and normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

    y_train_full = y_train_full.astype("int64")
    y_test = y_test.astype("int64")

    # Split training data into train/validation sets (90/10)
    val_split = int(0.9 * len(x_train_full))
    x_train, x_val = x_train_full[:val_split], x_train_full[val_split:]
    y_train, y_val = y_train_full[:val_split], y_train_full[val_split:]

    print(f"Training samples:   {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples:       {x_test.shape[0]}")

    return x_train, y_train, x_val, y_val, x_test, y_test


# ---------------------------------------------------------------------------
# Step 3: Model Development (CNN Architecture)
# ---------------------------------------------------------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="MNIST_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ---------------------------------------------------------------------------
# Step 4: Model Training
# ---------------------------------------------------------------------------
def train_model(model, x_train, y_train, x_val, y_val):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )
    return history


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/training_curves.png", dpi=150)
    plt.close()
    print(f"Saved training curves to {REPORT_DIR}/training_curves.png")


# ---------------------------------------------------------------------------
# Step 5: Model Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test):
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix - MNIST Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {REPORT_DIR}/confusion_matrix.png")

    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "per_class_report": report_dict,
    }

    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {REPORT_DIR}/metrics.json")

    return metrics, cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data()

    model = build_model()
    model.summary()

    history = train_model(model, x_train, y_train, x_val, y_val)
    plot_training_curves(history)

    metrics, cm = evaluate_model(model, x_test, y_test)

    # Deliverable: Trained Model File
    model.save(f"{MODEL_DIR}/mnist_cnn_model.keras")
    print(f"\nModel saved to {MODEL_DIR}/mnist_cnn_model.keras")

    print("\n=== DONE ===")
    return metrics


if __name__ == "__main__":
    main()
