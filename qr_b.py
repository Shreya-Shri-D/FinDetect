import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import sys

# Set console encoding to UTF-8 to handle all character outputs
sys.stdout.reconfigure(encoding='utf-8')

# Load images and labels function
def load_data(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        # Load each image and resize it to 224x224
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        # Convert image to an array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        # Assign labels based on filename: 1 for 'real' QR codes, 0 for 'fake'
        labels.append(1 if filename.startswith('benign') else 0)
    return np.array(images), np.array(labels)

# Path to the dataset directory (images not stored in git)
_ROOT = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(_ROOT, "qr_dataset")

# Load data
images, labels = load_data(dataset_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize pixel values between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define the CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with binary cross-entropy loss and accuracy as metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model and suppress console verbose output to avoid encoding errors
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model on the test set and print results
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Save the trained model
model.save(os.path.join(_ROOT, "qr_classifier_model.h5"))
print("Model saved successfully as 'qr_classifier_model.h5'")
