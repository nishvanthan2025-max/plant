import os, json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

MODEL = "models/plant_disease_model.keras"
CLASSES = "models/class_names.json"
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌱")
st.title("🌱 Plant Disease Detection")
st.write("Upload a tomato leaf image to predict its class.")

if not os.path.exists(MODEL):
    st.warning("Train the model first using: python train_model.py")
else:
    model = tf.keras.models.load_model(MODEL)
    with open(CLASSES) as f:
        names = json.load(f)
    file = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Predict Disease"):
            arr = np.expand_dims(np.array(image.resize((128,128))), axis=0)
            probs = model.predict(arr, verbose=0)[0]
            i = int(np.argmax(probs))
            st.success(f"Prediction: {names[i]}")
            st.info(f"Confidence: {probs[i]*100:.2f}%")
            st.caption("Educational prototype; consult an agricultural expert for real diagnosis.")
