import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# -------------------------------
# Load Model and Files
# -------------------------------

model = load_model("cnn_gru_model.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# -------------------------------
# App Title
# -------------------------------

st.title("Student Performance Prediction (CNN+GRU)")

st.write("Enter Feature Values Below:")

# -------------------------------
# Input Fields Dynamically
# -------------------------------

input_data = []

for feature in selected_features:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(value)

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("Predict"):
    
    input_array = np.array(input_data).reshape(1, -1)
    
    # reshape for CNN+GRU
    input_array = input_array.reshape(input_array.shape[0],
                                      input_array.shape[1],
                                      1)
    
    prediction = model.predict(input_array)
    pred_class = np.argmax(prediction, axis=1)
    
    final_prediction = le.inverse_transform(pred_class)
    
    st.success(f"Predicted Class: {final_prediction[0]}")