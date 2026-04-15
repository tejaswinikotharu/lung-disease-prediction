import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ============================
# LOAD MODEL
# ============================

model = tf.keras.models.load_model("model.h5")

class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

# ============================
# UI
# ============================

st.set_page_config(page_title="X-ray Detection", layout="centered")

st.title("🩺 X-ray Disease Detection System")
st.write("Upload a chest X-ray image to detect disease")

file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

# ============================
# PROCESS IMAGE
# ============================

if file is not None:
    img = Image.open(file).resize((224, 224)).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

    # ============================
    # PREPROCESS
    # ============================

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ============================
    # PREDICTION (REAL + FIXED)
    # ============================

    prediction = model.predict(img_array)

    # 🔥 Temperature scaling (keeps it realistic, not fake)
    temperature = 2.0
    prediction = np.exp(np.log(prediction + 1e-8) / temperature)
    prediction = prediction / np.sum(prediction)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result = class_names[predicted_index]

    # ============================
    # RESULT (SIMPLE)
    # ============================

    st.markdown("---")

    if result == "NORMAL":
        st.success(f"🧾 Prediction: {result}")
    else:
        st.error(f"🧾 Prediction: {result}")

    st.write(f"📊 Confidence: {confidence:.2f}%")

    # ============================
    # DISCLAIMER
    # ============================

    st.warning("⚠️ This is an AI-based prediction. Please consult a doctor for confirmation.")