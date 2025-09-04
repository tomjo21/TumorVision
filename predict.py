import json
import os
import numpy as np
from tensorflow import keras
import sys
sys.path.append('src')
from dataset import IMG_SIZE

# Load model (prefer modern .keras format)
MODEL_PATH = "brain_tumor_model.keras" if os.path.exists("brain_tumor_model.keras") else "brain_tumor_model.h5"
model = keras.models.load_model(MODEL_PATH)

# Load class names saved during training if available, else fallback to dataset scan
CLASS_NAMES_PATH = "class_names.json"
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
else:
    # Fallback: scan dataset folders (slower, but keeps compatibility)
    from dataset import load_datasets
    _, _, _, class_names = load_datasets()


def predict_image(image_path: str):
    # Load and preprocess image
    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch of 1; model has Rescaling layer

    # Predict
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    return class_names[predicted_class], confidence


if __name__ == "__main__":
    # Use CLI arg if provided, otherwise default to a sample test image
    default_img = "data/dataset/Testing/glioma/Te-gl_0010.jpg"
    image_path = sys.argv[1] if len(sys.argv) > 1 else default_img
    label, confidence = predict_image(image_path)
    print(f"✅ Prediction: {label} ({confidence*100:.2f}% confidence)")
