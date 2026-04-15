import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

@st.cache_resource
def load_model():
    return tf.saved_model.load("model_saved")

model = load_model()

class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

st.set_page_config(page_title="X-ray Detection", layout="centered")

st.title("🩺 X-ray Disease Detection System")
st.write("Upload a chest X-ray image to detect disease")

file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        st.image(img, caption="Uploaded Image", width=400)

        # ============================
        # PREPROCESS
        # ============================

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ============================
        # PREDICTION
        # ============================

        infer = model.signatures["serving_default"]
        prediction = infer(tf.constant(img_array))
        prediction = list(prediction.values())[0].numpy()

        # Normalize
        prediction = prediction / np.sum(prediction)

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        # ============================
        # 🔥 ENTROPY CHECK (KEY FIX)
        # ============================

        entropy = -np.sum(prediction * np.log(prediction + 1e-8))

        if confidence < 75 or entropy > 1.0:
            st.error("❌ Not a valid chest X-ray")
            st.stop()

        result = class_names[predicted_index]

        # ============================
        # RESULT
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

    except Exception as e:
        st.error("❌ Error processing image. Please upload a valid X-ray image.")
        st.write(str(e))