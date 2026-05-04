import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys

# Set console encoding to UTF-8 to handle all character outputs
sys.stdout.reconfigure(encoding='utf-8')

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load images and labels
def load_data(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory, filename), target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(1 if filename.startswith('benign') else 0)  # 1 for real QR codes, 0 for fake
    return np.array(images), np.array(labels)

# Load and preprocess images
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model input shape
    return img_array

# Load the trained model
def load_model():
    model = tf.keras.models.load_model(os.path.join(_ROOT, "qr_classifier_model.h5"))
    return model

# Predict QR code type
def predict_qr(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    if prediction < 0.5:
        return "Fake QR"
    else:
        return "Correct QR"

# QR code classification page
def qr_page():
    st.title("QR Code Classifier")
    uploaded_file = st.file_uploader("Upload QR Code Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded QR Code Image', use_column_width=True)
        model = load_model()
        prediction = predict_qr(uploaded_file, model)
        st.write("Prediction:", prediction)

# Main function to control the flow of the app
def main():
    # Call the appropriate page function
    qr_page()

if __name__ == "__main__":
    main()
