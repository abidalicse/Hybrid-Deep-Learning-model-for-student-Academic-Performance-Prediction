import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --------------------------------
# Load All Saved Components
# --------------------------------

model = load_model("cnn_gru_model.h5")
encoder_model = load_model("encoder_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

with open("original_features.pkl", "rb") as f:
    original_features = pickle.load(f)

with open("processed_feature_names.pkl", "rb") as f:
    processed_feature_names = pickle.load(f)

with open("feature_ranges.pkl", "rb") as f:
    feature_ranges = pickle.load(f)

# --------------------------------
# Dashboard UI
# --------------------------------

st.set_page_config(page_title="Student Performance Dashboard",
                   layout="wide")

st.title("🎓 Student Performance Prediction Dashboard")

st.markdown("### Enter Student Academic Details")

# Use columns layout
cols = st.columns(2)

input_data = []

for i, feature in enumerate(original_features):
    min_val, max_val = feature_ranges[feature]
    
    with cols[i % 2]:
        value = st.slider(
            feature,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2)
        )
    input_data.append(value)

# --------------------------------
# Prediction Button
# --------------------------------

if st.button("Predict Grade"):

    # Create dataframe from raw input
    input_df = pd.DataFrame([input_data], columns=original_features)

    # Scaling directly (same raw feature order)
    scaled = scaler.transform(input_df)

    # Autoencoder transform
    ae_features = encoder_model.predict(scaled)
    ae_df = pd.DataFrame(ae_features)

    # Select MI features (by index, NOT by name)
    final_input = ae_df.iloc[:, :len(selected_features)]

    final_input = final_input.values.reshape(
        final_input.shape[0],
        final_input.shape[1],
        1
    )

    # Prediction
    prediction = model.predict(final_input)
    pred_class = np.argmax(prediction, axis=1)

    grade = le.inverse_transform(pred_class)
    confidence = np.max(prediction) * 100

    st.success(f"🎯 Predicted Grade: {grade[0]}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Probability Graph
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(le.classes_, prediction[0])
    ax.set_ylabel("Probability")
    ax.set_xlabel("Grade")
    st.pyplot(fig)
