import tensorflow as tf
from tensorflow import keras

def build_model(input_shape=(128,128,3), num_classes=4):
    model = keras.models.Sequential([
        # Preprocessing: Center crop a bit and per-image standardization
        keras.layers.Input(shape=input_shape),
        keras.layers.CenterCrop(int(input_shape[0]*0.9), int(input_shape[1]*0.9)),
        keras.layers.Resizing(input_shape[0], input_shape[1]),
        keras.layers.Rescaling(1.0/255.0),
        keras.layers.Normalization(axis=-1),  # will be adapted at build time to per-batch; still useful

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
