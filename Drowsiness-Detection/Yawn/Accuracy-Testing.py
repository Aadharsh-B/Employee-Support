import os
import cv2
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Force TensorFlow to use CPU only (Faster startup, avoids GPU errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Paths to preprocessed image folders
data_path = r"E:\\action_lm"
yawn_path = os.path.join(data_path, "yawn_lm")
non_yawn_path = os.path.join(data_path, "non_yawn_lm")

# Image size
target_size = (64, 64)

# Load dataset
def load_data():
    images, labels = [], []

    # Load yawn images
    for file in os.listdir(yawn_path):
        img_path = os.path.join(yawn_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(1)  # Yawning

    # Load non-yawn images
    for file in os.listdir(non_yawn_path):
        img_path = os.path.join(non_yawn_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(0)  # Not Yawning

    return np.array(images), np.array(labels)

# Load and preprocess data
X, y = load_data()
X = X / 255.0  # Normalize pixel values
X = X.reshape(-1, 64, 64, 1)  # Reshape for CNN input
y = to_categorical(y, 2)  # Convert labels to categorical (0: Not Yawning, 1: Yawning)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN Model
def build_model():
    tf.keras.backend.clear_session()  # Reset TensorFlow state for consistency

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),  # Reduced Dropout for better stability
        Dense(2, activation='softmax', kernel_initializer='he_normal')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
model = build_model()
print("Training Model...")
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("yawn_trained_model.h5")
print("Model saved as yawn_trained_model.h5")

