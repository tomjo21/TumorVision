import os
# Use 128 size for original CNN
os.environ["IMG_SIZE"] = "128"

from dataset import load_datasets
from model import build_model
import tensorflow as tf
import json

# Load datasets
train_ds, val_ds, test_ds, class_names = load_datasets()
num_classes = len(class_names)

# Persist class names for inference
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Build model (128x128 input)
model = build_model(input_shape=(128, 128, 3), num_classes=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    tf.keras.callbacks.ModelCheckpoint("brain_tumor_model.keras", monitor="val_accuracy", save_best_only=True),
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

print("✅ Best model saved as brain_tumor_model.keras")
