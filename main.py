# ==========================================
# MICROPLASTIC DETECTION SYSTEM USING MICROSCOPIC CAMERA
#MobileNetV2 + Fine Tuning
# TensorFlow 2.16+ 
# Enhanced streamlit UI
# ==========================================

import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# ================================
# CONFIG
# ================================
IMG_SIZE = (224, 224)
MODEL_SAVE_PATH = "microplastic_best_model.keras"
CLASS_MAP_PATH = "class_mapping.json"

LABEL_MEANINGS = {
    "a": "No Microplastic ✅",
    "b": "Low Microplastic ⚠️",
    "c": "Medium Microplastic ⚠️",
    "d": "High Microplastic 🚨",
    "f": "Severe Contamination ☠️"
}

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_saved_model():
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(CLASS_MAP_PATH):
        model = load_model(MODEL_SAVE_PATH)

        with open(CLASS_MAP_PATH, "r") as f:
            class_indices = json.load(f)

        idx_to_class = {int(v): k for k, v in class_indices.items()}
        return model, idx_to_class
    return None, None


# ================================
# PREDICT
# ================================
def predict_image(model, idx_to_class, image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]

    best_idx = np.argmax(preds)
    best_label = idx_to_class[best_idx]
    confidence = preds[best_idx]

    result = LABEL_MEANINGS.get(best_label, best_label)
    return result, confidence, preds


# ================================
# UI
# ================================
st.set_page_config(page_title="Microplastic Detector", layout="centered")

st.title("💧 Microplastic Detection AI")
st.markdown("### Detect contamination in water samples using AI")

# 🖼️ SAMPLE IMAGE (default display)

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/3/3d/Microplastics_in_sediment.jpg",
    width="stretch"
)

# Load model
model, idx_to_class = load_saved_model()

if model is None:
    st.error("❌ Model not found. Train and save model first.")
    st.stop()
else:
    st.success("✅ Model Loaded Successfully")
# ================================
# 📌 INSTRUCTIONS SECTION
# ================================
st.markdown("## 📋 Instructions for Best Results")

st.info("""
📸 Follow these steps while capturing the image:

• Keep the background dark or plain 🖤  
• Ensure proper lighting on the sample 💡  
• Avoid shadows and reflections 🚫  
• Keep the camera steady 📷  
• Capture clear and focused image 🔍  

👉 This improves accuracy of microplastic detection.
""")
# Upload
uploaded_file = st.file_uploader("📤 Upload Your Water Sample Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, width='stretch')

    if st.button("🔍 Analyze Sample"):
        with st.spinner("Analyzing microplastics..."):
            result, confidence, preds = predict_image(model, idx_to_class, image)

        # 🎨 COLOR BASED RESULT
        if "No" in result:
            st.success(f"### {result}")
        elif "Low" in result:
            st.info(f"### {result}")
        elif "Medium" in result:
            st.warning(f"### {result}")
        else:
            st.error(f"### {result}")

        st.markdown(f"**Confidence:** {confidence:.2%}")

        # 📊 TOP 3
        st.markdown("### 🔍 Other Possibilities")
        top3 = np.argsort(preds)[-3:][::-1]

        for i, idx in enumerate(top3):
            label = idx_to_class[idx]
            meaning = LABEL_MEANINGS.get(label, label)
            st.write(f"{i+1}. {meaning} → {preds[idx]:.2%}")

# Footer
st.markdown("---")
st.markdown("💡 *AI-based Microplastic Detection System using MobileNetV2*")