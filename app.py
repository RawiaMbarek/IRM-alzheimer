import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.layers import Flatten



# Load the trained model
model = load_model('irm.h5')  # Replace with your model filename

# Class names for your model
CLASS_NAMES = ['Non Demented','Mild Demented', 'Moderate Demented','Very Mild Demented']

# Set the title of the app
st.title("Alzheimer's Disease MRI Classification")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # Display the image
    st.image(opencv_image, channels="BGR", caption="Uploaded MRI Image")
    
    # Preprocess the image
    opencv_image = cv2.resize(opencv_image, (128, 128))  # Adjust to (128, 128) as discussed
    opencv_image = np.expand_dims(opencv_image, axis=0)
    
    # Make prediction
    prediction = model.predict(opencv_image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    
    # Display the prediction result
    st.write(f"Prediction: {predicted_class}")
