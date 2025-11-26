# Import Libraries

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input

# Branding 
BRAND_NAME = "SlideLab AI"
BRAND_COLOR = "#0077B6"       
ACCENT_COLOR = "#90EE90"
BG_GRAY = "#F6F8FA"
TEXT_MUTED = "#6B7280"
IMAGE_DISPLAY_WIDTH = 500
IMG_SIZE = 180

# Page config
st.set_page_config(
    page_title=f"{BRAND_NAME} — NTD Vision",
    page_icon="🔬",
    layout="wide",
)

# CSS for styling 
st.markdown(
    f"""
    <style>
    /* Page background */
    .stApp {{
        background-color: {BG_GRAY};
    }}

    /* Header - enforce brand color */
    .brand-title {{
        color: {BRAND_COLOR} !important;
        font-size:38px !important;
        font-weight:700 !important;
        margin: 0;
        padding: 0;
        line-height: 1.0;
        text-align: left;
    }}
    .brand-sub {{
        color: {TEXT_MUTED} !important;
        font-size:14px;
        margin-top:4px;
        margin-bottom:12px;
    }}

    /* Card style */
    .card {{
        background: white;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 1px 4px rgba(16,24,40,0.06);
        border: 1px solid rgba(16,24,40,0.04);
    }}

    /* Debug small cards */
    .debug-card {{
        background: linear-gradient(180deg,#ffffff,#fbfcff);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.04);
    }}

    /* Footer */
    .footer {{
        color: #9CA3AF;
        font-size:12px;
        padding-top:10px;
        padding-bottom:30px;
    }}

    /* Make sure markdown h1 elements with brand-title class keep color */
    h1.brand-title {{
        color: {BRAND_COLOR} !important;
    }}

    /* Hide Streamlit default menu (optional) */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants & Model path
MODEL_PATH = "Malaria_Cell_Classification_Model.h5"
CLASS_NAMES = ["parasitized", "uninfected"]

# Sidebar - Info + options
with st.sidebar:
    st.markdown(f"# {BRAND_NAME}")
    st.write(
        "AI-assisted microscopy for blood-smear diagnostics — built for malaria now, "
        "designed to scale to other NTDs (filariasis, loiasis, etc.)."
    )
    st.divider()
    st.markdown("### How to use")
    st.write(
        "1. Upload a stained blood-smear image (jpg/png).\n"
        "2. Wait for the model to analyze the slide.\n"
        "3. View prediction, confidence, and model info."
    )
    st.divider()
    debug_mode = st.checkbox("Show preprocessing debug", value=False)
    st.write(" ")
    st.markdown("### Project")
    st.write("SlideLab AI — NTD Vision (Hackathon submission)")
    st.write("Lead: Micheal Adeniyi")
    st.write("Contact: oluwafemiadeniyi772@gmail.com")
    st.write(" ")
    st.caption("Tip: For best results upload clear 180×180+ crop of a single cell region.")

# Utility: load model (cached)
@st.cache_resource
def load_model(path: str):
    try:
        # try loading model using keras load_model
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        # show a friendly message, return None
        st.error(f" Model load failed: {e}")
        return None

model = load_model(MODEL_PATH)

# Preprocess function 
def preprocess_image(uploaded_file, show_debug: bool = False):
    display_img = Image.open(uploaded_file).convert("RGB")
    img = tf.keras.utils.img_to_array(display_img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)

    # manual normalization & standard preprocess
    img_manual = (img / 127.5) - 1.0
    img_pre = preprocess_input(img)  # EfficientNet preprocessing usually [-1, +1]
    img_pre = tf.expand_dims(img_pre, 0)

    if show_debug:
        # show horizontal debug cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="debug-card">', unsafe_allow_html=True)
            st.markdown("**BEFORE**")
            st.write("min:", float(tf.reduce_min(img)))
            st.write("max:", float(tf.reduce_max(img)))
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="debug-card">', unsafe_allow_html=True)
            st.markdown("**AFTER preprocess_input**")
            try:
                st.write("min:", float(tf.reduce_min(img_pre)))
                st.write("max:", float(tf.reduce_max(img_pre)))
            except Exception:
                st.write("min/max: (n/a)")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="debug-card">', unsafe_allow_html=True)
            st.markdown("**MANUAL**")
            st.write("min:", float(tf.reduce_min(img_manual)))
            st.write("max:", float(tf.reduce_max(img_manual)))
            st.markdown("</div>", unsafe_allow_html=True)

    return img_pre, display_img

# Header / Hero
st.container()
st.markdown(
    f"""
    <div class="card" style="margin-bottom:14px;">
      <div style="display:flex; align-items:center; gap:18px;">
        <div style="flex:1;">
          <h1 class="brand-title">🔬 {BRAND_NAME}</h1>
          <div class="brand-sub">NTD Vision — AI-assisted slide microscopy for malaria & beyond</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:13px;color:{TEXT_MUTED}">Model status</div>
          <div style="font-weight:700;color:{BRAND_COLOR}; margin-top:6px;">
            {"Loaded" if model is not None else "Not loaded"}
          </div>
          <div style="font-size:12px;color:#9CA3AF;margin-top:8px;">Updated: {datetime.datetime.now().strftime("%Y-%m-%d")}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# File uploader & main area
uploaded_file = st.file_uploader("Upload a blood-smear image (jpg, png)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

# If no file, show example card and instructions
if uploaded_file is None:
    st.markdown(
        """
        <div class="card">
            <h3 style="margin-top:0;">Get started</h3>
            <p style="color:{TEXT_MUTED};">
                Upload a crop of a blood-smear slide (single-cell or small patch). The app processes the image and
                returns the predicted label and confidence. Use the debug toggle in the sidebar to inspect preprocessing ranges.
            </p>
            <ul style="color:#374151;">
                <li>Best crop size: at least 180×180 pixels</li>
                <li>Prefer clear, focused Giemsa-stained images</li>
                <li>Model: EfficientNetB0 (trained for parasitized vs uninfected)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")  # spacing
else:
    # ensure model is loaded
    if model is None:
        st.error("Model not loaded. Please check the model path or server logs.")
        st.stop()

    # preprocess (show debug only if checkbox enabled)
    try:
        img_tensor, display_img = preprocess_image(uploaded_file, show_debug=debug_mode)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # Prediction with spinner
    with st.spinner("Analyzing slide with SlideLab AI..."):
        try:
            raw_preds = model.predict(img_tensor)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Interpret various output shapes (softmax vs sigmoid)
    preds_arr = np.asarray(raw_preds)
    # show raw debug if debug mode
    if debug_mode:
        st.markdown("<div class='card'><b>Raw model output</b></div>", unsafe_allow_html=True)
        st.write(preds_arr)
        st.write("Shape:", preds_arr.shape)

    try:
        if preds_arr.ndim == 2 and preds_arr.shape[1] == 2:
            probs = preds_arr[0].astype(float).tolist()
        elif preds_arr.ndim == 1 and preds_arr.size == 2:
            probs = preds_arr.astype(float).tolist()
        elif preds_arr.size == 1:
            p = float(np.squeeze(preds_arr))
            probs = [1.0 - p, p]
        else:
            flat = preds_arr.flatten()
            if flat.size >= 2:
                probs = flat[:2].astype(float).tolist()
            else:
                raise ValueError("Unhandled prediction shape.")
    except Exception as e:
        st.error(f"Could not interpret model output: {e}")
        st.stop()

    preds_list = [float(probs[0]), float(probs[1])]
    top_index = int(np.argmax(preds_list))
    predicted_label = CLASS_NAMES[top_index]
    confidence = float(preds_list[top_index] * 100.0)

    # Results layout
    st.markdown('<div class="card">', unsafe_allow_html=True)
    colA, colB = st.columns([1, 1])

    with colA:
        # image preview on left
        st.image(display_img, caption="Uploaded Cell Image", width=IMAGE_DISPLAY_WIDTH, use_column_width=False)

        # quick metadata
        st.write("")
        st.write("**Prediction**")
        st.markdown(f"<div style='font-size:18px; font-weight:700; color:{BRAND_COLOR};'>{predicted_label.upper()}</div>", unsafe_allow_html=True)
        st.write(f"Confidence: **{confidence:.2f}%**")
        st.progress(min(max(float(confidence) / 100.0, 0.0), 1.0))

    with colB:
        # probabilities and action area on right
        st.subheader("Prediction details")
        st.write("Raw probabilities (class → value):")
        st.json({CLASS_NAMES[i]: preds_list[i] for i in range(len(preds_list))})

        st.write("")  # spacing
        st.markdown("### Model information")
        st.write(f"**Model path:** `{MODEL_PATH}`")
        st.write("**Architecture:** EfficientNetB0 (expected)")
        st.write(f"**Input size:** {IMG_SIZE} × {IMG_SIZE}")
        st.write(f"**Inferred at:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.write("")  # spacing
        st.markdown("### Actions")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("Download result summary"):
                # create a small result text and allow download
                result_txt = (
                    f"SlideLab AI result\n"
                    f"Predicted: {predicted_label}\n"
                    f"Confidence: {confidence:.2f}%\n"
                    f"Timestamp: {datetime.datetime.now().isoformat()}\n"
                )
                st.download_button("Download .txt", result_txt, file_name="slidelab_result.txt")
        with col_btn2:
            st.write("")  # placeholder for spacing
            st.write("") 
            st.markdown(f"<small style='color:{TEXT_MUTED}'>Model ready for inference.</small>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer 
st.markdown(
    f"""
    <div class="card" style="margin-top:16px;">
      <strong>About {BRAND_NAME}</strong>
      <p style="color:#374151;">
        SlideLab AI provides rapid, AI-powered microscopy review for blood-smear slides. Built initially for malaria detection,
        the system is designed to scale to other blood-slide NTDs such as filariasis and loiasis. This submission demonstrates
        clinical-grade model architecture, careful preprocessing, and a user-friendly interface suitable for field and lab use.
      </p>
      <div class="footer">Prepared for Hackathon — SlideLab AI | Contact: oluwafemiadeniyi772@gmail.com</div>
    </div>
    """,
    unsafe_allow_html=True,
)
