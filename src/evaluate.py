import os
import json
import numpy as np
from tensorflow import keras
from dataset import load_datasets

# Load datasets (we only need test set and class names)
_, _, test_ds, class_names = load_datasets()

# Load trained model
MODEL_PATH = "brain_tumor_model.keras" if os.path.exists("brain_tumor_model.keras") else "brain_tumor_model.h5"
model = keras.models.load_model(MODEL_PATH)

# Collect predictions and ground-truth labels
y_true = []
y_pred = []
for batch_images, batch_labels in test_ds:
    preds = model.predict(batch_images, verbose=0)
    y_true.extend(batch_labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Metrics and confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

print("\nTest accuracy:", f"{acc:.4f}")
print("\nClassification report:\n")
print(report)

# Plot and save confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Acc: {acc:.3f})')
plt.tight_layout()

out_path = 'confusion_matrix.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved confusion matrix to {out_path}")
# Optional: show if running interactively
# plt.show()
plt.close()
