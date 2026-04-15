import streamlit as st
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ============================
# LOAD MODEL (SAFE)
# ============================

@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    return load_model("model.h5", compile=False)

model = load_model()

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

        prediction = model.predict(img_array)

        # 🔥 Temperature scaling (stable)
        temperature = 2.0
        prediction = np.log(prediction + 1e-8) / temperature
        prediction = np.exp(prediction)
        prediction = prediction / np.sum(prediction)

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

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
