import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths to dataset
yawn_path = r"E:\action_lm\yawn_lm"
non_yawn_path = r"E:\action_lm\non_yawn_lm"

# Image size and dataset preparation
IMG_SIZE = (64, 64)  # Resize all images to this size
data = []
labels = []

# Load and preprocess images
def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if image is not None:
            image = cv2.resize(image, IMG_SIZE)
            data.append(image)
            labels.append(label)

# Load dataset
load_images_from_folder(yawn_path, 1)       # 1 for yawn
load_images_from_folder(non_yawn_path, 0)   # 0 for non-yawn

# Convert to NumPy arrays
data = np.array(data).reshape(-1, 64, 64, 1)  # Reshape for CNN (add channel dimension)
labels = np.array(labels)

# Normalize pixel values (0 to 1)
data = data / 255.0

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Helps prevent overfitting
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)

# Save the model
model.save(r"E:\yawn_model.h5")
print("Model saved successfully!")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
