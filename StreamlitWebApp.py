# SlideLab AI - Streamlit app

# Import Libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
from tensorflow.keras.applications.efficientnet import preprocess_input

# Branding
BRAND_NAME = "SlideLab AI"
BRAND_COLOR = "#0077B6"
ACCENT_COLOR = "#90EE90"
IMAGE_DISPLAY_WIDTH = 350
IMG_SIZE = 180

# Page config
st.set_page_config(
    page_title=f"{BRAND_NAME} — NTD Vision",
    page_icon="🔬",
    layout="wide",
)

# Enhanced CSS for professional look
st.markdown("""
    <style>
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    [data-testid="stFileUploader"] section {
        padding: 60px 40px;
        min-height: 200px;
        border: 2px dashed #0077B6;
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] section > button {
        font-size: 1.1rem;
    }

    /* Clean up spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Progress bar color */
    .stProgress > div > div > div > div {
        background-color: #0077B6;
    }
    </style>
""", unsafe_allow_html=True)

# Constants & Model path
MODEL_PATH = "Malaria_Cell_Classification_Model.h5"
CLASS_NAMES = ["parasitized", "uninfected"]

# Sidebar - Streamlined
with st.sidebar:
    st.markdown(f"### 🔬 {BRAND_NAME}")
    st.caption(
        "AI-powered blood-smear diagnostics for malaria detection, with future support for filariasis, loiasis, and other NTDs.")

    st.divider()

    st.markdown("#### 📋 Quick Guide")
    st.markdown("""
    1. **Upload** a blood-smear image
    2. **Wait** for AI analysis
    3. **Review** results & confidence
    """)

    st.divider()

    with st.expander("⚙️ Advanced Options"):
        debug_mode = st.checkbox("Show preprocessing debug", value=False)
        st.caption("💡 For best results, use clear 180×180+ pixel images of single cells.")

    st.divider()

    st.markdown("#### 🖼️ Need Sample Images?")
    st.markdown(
        "[Download test images](https://drive.google.com/drive/folders/1lg7IwN6EhIgmmxBbdkYLLilEkLZ_xXDg) from our collection of parasitized and uninfected samples.")
    st.caption("Perfect for testing the model")


# Load model
@st.cache_resource
def load_model(path: str):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


model = load_model(MODEL_PATH)


# Preprocess image
def preprocess_image(uploaded_file, show_debug: bool = False):
    display_img = Image.open(uploaded_file).convert("RGB")
    img = tf.keras.utils.img_to_array(display_img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img_manual = (img / 127.5) - 1.0
    img_pre = preprocess_input(img)
    img_pre = tf.expand_dims(img_pre, 0)

    if show_debug:
        st.markdown("##### 🔍 Preprocessing Debug")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original", f"{float(tf.reduce_min(img)):.1f} - {float(tf.reduce_max(img)):.1f}")
        with col2:
            st.metric("EfficientNet", f"{float(tf.reduce_min(img_pre)):.1f} - {float(tf.reduce_max(img_pre)):.1f}")
        with col3:
            st.metric("Manual Norm", f"{float(tf.reduce_min(img_manual)):.1f} - {float(tf.reduce_max(img_manual)):.1f}")

    return img_pre, display_img


# Header
st.markdown(f"# 🔬 {BRAND_NAME}")
st.caption("AI-assisted slide microscopy for malaria & beyond")

# Status badge in top right
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if model:
        st.markdown("""
            <div style="text-align: right;">
                <span style="background-color: #10b981; color: white; padding: 4px 12px; 
                border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                ✓ READY
                </span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align: right;">
                <span style="background-color: #ef4444; color: white; padding: 4px 12px; 
                border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                ✗ ERROR
                </span>
            </div>
        """, unsafe_allow_html=True)

st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "Drop your blood-smear image here or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    # Welcome card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 👋 Welcome to SlideLab AI")
        st.markdown("""
        Upload a blood-smear image to get started. Our AI model will analyze the slide 
        and detect whether cells are parasitized or uninfected.
        """)

        st.markdown("##### ✨ Best Practices")
        st.markdown("""
        - Use images at least **180×180 pixels**
        - Ensure clear, focused **Giemsa-stained** samples
        - Upload **single-cell or small patch** crops
        """)

        st.info("**Model:** EfficientNetB0 • **Classes:** Parasitized, Uninfected", icon="ℹ️")

        st.markdown("##### 🖼️ Don't have an image?")
        st.markdown("""
        [Access sample images →](https://drive.google.com/drive/folders/1lg7IwN6EhIgmmxBbdkYLLilEkLZ_xXDg)

        Try our collection of parasitized and uninfected blood-smear samples.
        """)

else:
    if model is None:
        st.error("⚠️ Model not loaded. Please check the model path or server logs.", icon="🚨")
        st.stop()

    img_tensor, display_img = preprocess_image(uploaded_file, debug_mode)

    with st.spinner("🔬 Analyzing slide with SlideLab AI..."):
        raw_preds = model.predict(img_tensor)

    # Interpret prediction
    preds_arr = np.asarray(raw_preds)
    if preds_arr.ndim == 2 and preds_arr.shape[1] == 2:
        probs = preds_arr[0].astype(float).tolist()
    elif preds_arr.ndim == 1 and preds_arr.size == 2:
        probs = preds_arr.astype(float).tolist()
    elif preds_arr.size == 1:
        p = float(np.squeeze(preds_arr))
        probs = [1.0 - p, p]
    else:
        flat = preds_arr.flatten()
        probs = flat[:2].astype(float).tolist()

    preds_list = [float(probs[0]), float(probs[1])]
    top_index = int(np.argmax(preds_list))
    predicted_label = CLASS_NAMES[top_index]
    confidence = float(preds_list[top_index] * 100.0)

    # Results layout
    st.markdown("### 📊 Analysis Results")

    colA, colB = st.columns([1, 1])

    with colA:
        st.image(display_img, caption="Uploaded Cell Image", use_container_width=True)

        # Prediction card
        prediction_color = "#ef4444" if predicted_label == "parasitized" else "#10b981"
        st.markdown(f"""
            <div style="background-color: {prediction_color}15; padding: 20px; 
            border-radius: 10px; border-left: 4px solid {prediction_color}; margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.9rem; color: #666;">Prediction</p>
                <h2 style="margin: 5px 0; color: {prediction_color};">{predicted_label.upper()}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">
                    Confidence: <strong>{confidence:.2f}%</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.progress(confidence / 100.0)

    with colB:
        # Probabilities
        st.markdown("##### 📈 Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            prob_pct = preds_list[i] * 100
            st.metric(
                label=class_name.capitalize(),
                value=f"{prob_pct:.2f}%",
                delta=None
            )

        st.divider()

        # Model info
        st.markdown("##### 🤖 Model Information")
        st.markdown(f"""
        - **Architecture:** EfficientNetB0
        - **Input size:** {IMG_SIZE} × {IMG_SIZE}
        - **Classes:** {len(CLASS_NAMES)}
        - **Analyzed:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)

        st.divider()

        # Download button
        result_txt = (
            f"SlideLab AI Analysis Report\n"
            f"{'=' * 40}\n"
            f"Prediction: {predicted_label.upper()}\n"
            f"Confidence: {confidence:.2f}%\n"
            f"\nProbabilities:\n"
            f"  - Parasitized: {preds_list[0]:.4f}\n"
            f"  - Uninfected: {preds_list[1]:.4f}\n"
            f"\nTimestamp: {datetime.datetime.now().isoformat()}\n"
            f"Model: EfficientNetB0\n"
        )
        st.download_button(
            "📥 Download Report",
            result_txt,
            file_name=f"slidelab_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            use_container_width=True
        )

# Footer
st.divider()
st.markdown("""
    <p style='text-align: center; color: #94a3b8; font-size: 0.9rem; margin-top: 2rem;'>
    Prepared for 3MTT Hackathon • <strong>SlideLab AI</strong> • {year}
    </p>
""".format(year=datetime.datetime.now().year), unsafe_allow_html=True)