"""Generates the Performance Evaluation Report PDF (Deliverable #4)."""

import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

with open("reports/metrics.json") as f:
    metrics = json.load(f)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=20, spaceAfter=6)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11,
                                 textColor=colors.grey, alignment=TA_CENTER, spaceAfter=20)
h2 = styles["Heading2"]
body = styles["Normal"]

doc = SimpleDocTemplate("Performance_Evaluation_Report.pdf", pagesize=letter,
                         topMargin=0.75 * inch, bottomMargin=0.75 * inch)
story = []

story.append(Paragraph("Performance Evaluation Report", title_style))
story.append(Paragraph("AI/ML Internship Task – Week 01 | Task 1: Image Classification Model", subtitle_style))

story.append(Paragraph("1. Project Summary", h2))
story.append(Paragraph(
    "A Convolutional Neural Network (CNN) was developed using TensorFlow/Keras to classify "
    "handwritten digit images (0-9) from the MNIST dataset. The pipeline covered dataset "
    "preprocessing, CNN architecture design, model training with early stopping and learning "
    "rate scheduling, and thorough performance evaluation on a held-out test set.", body))
story.append(Spacer(1, 12))

story.append(Paragraph("2. Dataset", h2))
story.append(Paragraph(
    "MNIST dataset: 70,000 grayscale images of size 28x28 pixels across 10 classes (digits 0-9). "
    "60,000 images used for training (split 90/10 into training/validation), and 10,000 images "
    "held out as the independent test set. Pixel values were normalized to the [0, 1] range.", body))
story.append(Spacer(1, 12))

story.append(Paragraph("3. Model Architecture", h2))
story.append(Paragraph(
    "A CNN with two convolutional blocks (Conv2D -> BatchNormalization -> MaxPooling -> Dropout), "
    "followed by a fully connected dense layer with dropout regularization and a softmax output "
    "layer for 10-class classification. Total trainable parameters: ~422,000.", body))
story.append(Spacer(1, 12))

story.append(Paragraph("4. Training Configuration", h2))
config_data = [
    ["Parameter", "Value"],
    ["Optimizer", "Adam (lr=0.001)"],
    ["Loss Function", "Sparse Categorical Crossentropy"],
    ["Batch Size", "256"],
    ["Epochs", "8 (with EarlyStopping & ReduceLROnPlateau)"],
]
t = Table(config_data, colWidths=[2.5 * inch, 3.5 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(t)
story.append(Spacer(1, 16))

story.append(Paragraph("5. Evaluation Metrics (Test Set)", h2))
metrics_data = [
    ["Metric", "Score"],
    ["Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%"],
    ["Precision (macro avg)", f"{metrics['precision_macro']*100:.2f}%"],
    ["Recall (macro avg)", f"{metrics['recall_macro']*100:.2f}%"],
    ["F1-Score (macro avg)", f"{metrics['f1_macro']*100:.2f}%"],
    ["Test Loss", f"{metrics['test_loss']:.4f}"],
]
t2 = Table(metrics_data, colWidths=[3 * inch, 3 * inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(t2)
story.append(Spacer(1, 16))

story.append(Paragraph("6. Per-Class Performance", h2))
per_class = metrics["per_class_report"]
class_rows = [["Digit", "Precision", "Recall", "F1-Score", "Support"]]
for cls in [str(i) for i in range(10)]:
    r = per_class[cls]
    class_rows.append([
        cls, f"{r['precision']:.3f}", f"{r['recall']:.3f}",
        f"{r['f1-score']:.3f}", str(int(r["support"]))
    ])
t3 = Table(class_rows, colWidths=[0.8 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.1 * inch])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(t3)
story.append(PageBreak())

story.append(Paragraph("7. Training & Validation Curves", h2))
story.append(Image("reports/training_curves.png", width=6.5 * inch, height=2.44 * inch))
story.append(Spacer(1, 16))

story.append(Paragraph("8. Confusion Matrix", h2))
story.append(Image("reports/confusion_matrix.png", width=5 * inch, height=3.9 * inch))
story.append(Spacer(1, 16))

story.append(Paragraph("9. Results Analysis", h2))
story.append(Paragraph(
    "The model achieved 99.02% accuracy on the held-out test set, with macro-averaged precision, "
    "recall, and F1-score all at approximately 99%. Training and validation accuracy curves "
    "converge closely with no significant divergence, indicating the model generalizes well and "
    "is not overfitting - aided by BatchNormalization and Dropout layers. The confusion matrix "
    "shows very few misclassifications, with the small remaining confusions occurring between "
    "visually similar digit pairs (e.g., 4 and 9, 3 and 5).", body))
story.append(Spacer(1, 12))

story.append(Paragraph("10. Suggested Improvements", h2))
improvements = [
    "Introduce data augmentation (rotation, shift, zoom) to improve robustness to handwriting variation.",
    "Experiment with deeper or residual architectures for further accuracy gains.",
    "Apply advanced learning rate scheduling (e.g., cosine decay, warm-up).",
    "Ensemble multiple CNN models to boost overall accuracy.",
    "Test on out-of-distribution handwritten samples to evaluate real-world generalization.",
]
for imp in improvements:
    story.append(Paragraph(f"- {imp}", body))

doc.build(story)
print("Report generated: Performance_Evaluation_Report.pdf")
