import tensorflow as tf
from tensorflow import keras

def build_model(input_shape=(128,128,3), num_classes=4):
    model = keras.models.Sequential([
        # Normalize pixel values 0-1
        keras.layers.Rescaling(1./255, input_shape=input_shape),

        # Convolution + Pooling layers
        keras.layers.Conv2D(32, (3,3), activation="relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(64, (3,3), activation="relu"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(128, (3,3), activation="relu"),
        keras.layers.MaxPooling2D(),

        # Flatten + Fully Connected Layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax")   # 4 classes
    ])
    return model
