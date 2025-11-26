# Import Libraries

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
from tensorflow.keras.applications.efficientnet import preprocess_input


# Page configuration

st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon="🩸🔬",
    layout="wide",
)


# Constants

MODEL_PATH = "https://github.com/OfemiAdeniyi/malaria-app/blob/master/Malaria_Cell_Classification_Model.h5"
CLASS_NAMES = ["parasitized", "uninfected"]
IMG_SIZE = 180
IMAGE_DISPLAY_WIDTH = 250 


# Preprocess 

def preprocess_image(uploaded_file):
    """
    Load, resize and preprocess a malaria image. Print debug info in three horizontal columns:
    BEFORE preprocess_input, AFTER preprocess_input, and AFTER manual normalization.
    Returns: (preprocessed_batch_tensor, display_image_pil)
    """
    display_img = Image.open(uploaded_file).convert("RGB")

    # Convert to array and resize
    img = tf.keras.utils.img_to_array(display_img)  # shape (H, W, C), values 0..255
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)

    # Horizontal columns for Preprocessing outputs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### BEFORE `preprocess_input`")
        st.write("min:", float(tf.reduce_min(img)))
        st.write("max:", float(tf.reduce_max(img)))

    # preprocess_input (EfficientNet expected) and results
    img_pre = preprocess_input(img)
    with col2:
        st.markdown("### AFTER `preprocess_input`")
        # Wrap in try/except in case tf ops cause issues
        try:
            st.write("min:", float(tf.reduce_min(img_pre)))
            st.write("max:", float(tf.reduce_max(img_pre)))
        except Exception:
            st.write("Could not compute min/max after preprocess_input")

    # Normalization for comparison
    img_manual = (img / 127.5) - 1.0
    with col3:
        st.markdown("### MANUAL Normalization (img/127.5 - 1)")
        st.write("min:", float(tf.reduce_min(img_manual)))
        st.write("max:", float(tf.reduce_max(img_manual)))

    # return preprocess_input version (batch dimension added)
    img_pre = tf.expand_dims(img_pre, 0)
    return img_pre, display_img


# Load model (cached)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        # show error and return None
        st.error(f" Model loading failed: {e}")
        return None

model = load_model()


# Header

st.markdown(
    """
    <h1 style="text-align:center;color:#DCDCDC;">🩸🔬 Malaria Cell Classification App</h1>
    <p style="text-align:center;color:#AAAAAA;">
        Upload a blood smear image to classify it as <b>Parasitized</b> or <b>Uninfected</b>.
    </p>
    """,
    unsafe_allow_html=True
)


# File uploader

uploaded_file = st.file_uploader("Upload a blood smear image", type=["jpg", "jpeg", "png"])


# Main

if uploaded_file is not None:

    if model is None:
        st.error("Model not loaded. Check model path or logs above.")
        st.stop()

    # Preprocess and display debug cards horizontally inside the function
    try:
        img_array, display_img = preprocess_image(uploaded_file)
    except Exception as e:
        st.error(f"Failed during preprocessing: {e}")
        st.stop()

    # Prediction
    try:
        raw_preds = model.predict(img_array)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    #  show raw preds and shape
    st.write("DEBUG: raw model output (numpy):", np.asarray(raw_preds))
    st.write("DEBUG: model output shape:", np.asarray(raw_preds).shape)

    # Convert to numpy and normalize shapes:
    preds_arr = np.asarray(raw_preds)
    # Handle typical possibilities:
    # - softmax (1,2) or (2,)
    # - sigmoid (1,) or scalar
    try:
        if preds_arr.ndim == 2 and preds_arr.shape[1] == 2:
            probs = preds_arr[0].astype(float).tolist()
        elif preds_arr.ndim == 1 and preds_arr.size == 2:
            probs = preds_arr.astype(float).tolist()
        elif preds_arr.size == 1:
            # single probability (sigmoid) -> assume positive == parasitized
            p = float(np.squeeze(preds_arr))
            probs = [1.0 - p, p]
        else:
            # fallback: flatten and attempt to use first two entries
            flat = preds_arr.flatten()
            if flat.size >= 2:
                probs = flat[:2].astype(float).tolist()
            else:
                raise ValueError(f"Unhandled prediction shape: {preds_arr.shape}")
    except Exception as e:
        st.error(f"Could not interpret model output: {e}")
        st.stop()

    # Finalize results
    preds_list = [float(probs[0]), float(probs[1])]
    top_index = int(np.argmax(preds_list))
    predicted_label = CLASS_NAMES[top_index]
    confidence = float(preds_list[top_index] * 100.0)

    # Tabs for prediction + model info
    tab1, tab2 = st.tabs(["📊 Prediction", "⚙ Model Info"])

    with tab1:
        colA, colB = st.columns([1, 1])

        with colA:
            # smaller preview (fixed width)
            st.image(display_img, caption="Uploaded Cell Image", width=IMAGE_DISPLAY_WIDTH, use_column_width=False)

        with colB:
            st.subheader("Prediction Result")
            st.markdown(
                f"""
                <h2 style="color:#90EE90;">{predicted_label.upper()}</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
                """,
                unsafe_allow_html=True
            )

            # cast to plain python float for progress
            st.progress(float(confidence) / 100.0)

            st.write("Raw probabilities:")
            st.json({CLASS_NAMES[i]: preds_list[i] for i in range(len(preds_list))})

    with tab2:
        st.subheader("Model Information")
        st.write(f"📌 Model path: `{MODEL_PATH}`")
        st.write(f"📌 Loaded at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"📌 Input Size: {IMG_SIZE} × {IMG_SIZE}")
        st.write("📌 Architecture: EfficientNetB0 (expected)")
        st.success("Model ready for inference.")

else:
    st.info("📤 Upload a malaria cell image to get started.")
