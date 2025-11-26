# Import Libraries

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input
import io
import pandas as pd

# Branding

BRAND_NAME = "SlideLab AI"
BRAND_COLOR = "#0077B6"       # Blue accent
ACCENT_COLOR = "#90EE90"      # Green highlight
BG_COLOR = "#F7F9FA"          # Light gray background
TEXT_COLOR = "#111111"         # Dark text for readability
TEXT_MUTED = "#6B7280"         # Muted gray
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
f""" <style>
.stApp {{
background-color: {BG_COLOR};
color: {TEXT_COLOR};
}}
.brand-title {{
color: {BRAND_COLOR} !important;
font-size:38px !important;
font-weight:700 !important;
margin: 0;
padding: 0;
line-height: 1.0;
}}
.brand-sub {{
color: {TEXT_COLOR} !important;
font-size:14px;
margin-top:4px;
margin-bottom:12px;
}}
.card {{
background: white;
border-radius: 10px;
padding: 14px;
box-shadow: 0 1px 4px rgba(16,24,40,0.06);
border: 1px solid rgba(16,24,40,0.04);
color: {TEXT_COLOR};
}}
.debug-card {{
background: linear-gradient(180deg,#ffffff,#fbfcff);
border-radius: 8px;
padding: 10px;
text-align: center;
border: 1px solid rgba(0,0,0,0.04);
color: {TEXT_COLOR};
}}
.footer {{
color: {TEXT_MUTED};
font-size:12px;
padding-top:10px;
padding-bottom:30px;
}}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }} </style>
""",
unsafe_allow_html=True,
)

# Constants & Model path

MODEL_PATH = "Malaria_Cell_Classification_Model.h5"
CLASS_NAMES = ["parasitized", "uninfected"]

# Sidebar

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
st.write("Contact: [oluwafemiadeniyi772@gmail.com](mailto:oluwafemiadeniyi772@gmail.com)")
st.write(" ")
st.caption("Tip: For best results upload clear 180×180+ crop of a single cell region.")

# Load model

@st.cache_resource
def load_model(path: str):
try:
model = tf.keras.models.load_model(path)
return model
except Exception as e:
st.error(f" Model load failed: {e}")
return None

model = load_model(MODEL_PATH)

# Preprocess function

def preprocess_image(uploaded_file, show_debug: bool = False):
display_img = Image.open(uploaded_file).convert("RGB")
img = tf.keras.utils.img_to_array(display_img)
img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
img = tf.cast(img, tf.float32)

```
img_manual = (img / 127.5) - 1.0
img_pre = preprocess_input(img)
img_pre = tf.expand_dims(img_pre, 0)

if show_debug:
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
```

# Header / Hero

st.container()
st.markdown(
f""" <div class="card" style="margin-bottom:14px;"> <div style="display:flex; align-items:center; gap:18px;"> <div style="flex:1;"> <h1 class="brand-title">🔬 {BRAND_NAME}</h1> <div class="brand-sub">NTD Vision — AI-assisted slide microscopy for malaria & beyond</div> </div> <div style="text-align:right;"> <div style="font-size:13px;color:{TEXT_MUTED}">Model status</div> <div style="font-weight:700;color:{BRAND_COLOR}; margin-top:6px;">
{"Loaded" if model is not None else "Not loaded"} </div> <div style="font-size:12px;color:{TEXT_MUTED};margin-top:8px;">Updated: {datetime.datetime.now().strftime("%Y-%m-%d")}</div> </div> </div> </div>
""",
unsafe_allow_html=True,
)

# File uploader

uploaded_file = st.file_uploader("Upload a blood-smear image (jpg, png)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is None:
st.markdown(
f""" <div class="card"> <h3 style="margin-top:0;">Get started</h3> <p>
Upload a crop of a blood-smear slide (single-cell or small patch). The app processes the image and
returns the predicted label and confidence. Use the debug toggle in the sidebar to inspect preprocessing ranges. </p> <ul> <li>Best crop size: at least 180×180 pixels</li> <li>Prefer clear, focused Giemsa-stained images</li> <li>Model: EfficientNetB0 (trained for parasitized vs uninfected)</li> </ul> </div>
""",
unsafe_allow_html=True,
)
st.write("")  # spacing
else:
if model is None:
st.error("Model not loaded. Please check the model path or server logs.")
st.stop()

```
try:
    img_tensor, display_img = preprocess_image(uploaded_file, show_debug=debug_mode)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

with st.spinner("Analyzing slide with SlideLab AI..."):
    try:
        raw_preds = model.predict(img_tensor)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

preds_arr = np.asarray(raw_preds)
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
    st.image(display_img, caption="Uploaded Cell Image", width=IMAGE_DISPLAY_WIDTH, use_column_width=False)
    st.write("")
    st.write("**Prediction**")
    st.markdown(f"<div style='font-size:18px; font-weight:700; color:{BRAND_COLOR};'>{predicted_label.upper()}</div>", unsafe_allow_html=True)
    st.write(f"Confidence: **{confidence:.2f}%**")
    st.progress(min(max(float(confidence) / 100.0, 0.0), 1.0))

# Downloadable summary
with colB:
    summary_df = pd.DataFrame({
        "Predicted Label": [predicted_label],
        "Confidence (%)": [confidence]
    })
    csv_buffer = io.StringIO()
    summary_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Prediction Summary",
        data=csv_buffer.getvalue(),
        file_name="prediction_summary.csv",
        mime="text/csv"
    )
```

# Footer

st.markdown(
f""" <div class="footer">
SlideLab AI — NTD Vision | Lead: Micheal Adeniyi | Contact: [oluwafemiadeniyi772@gmail.com](mailto:oluwafemiadeniyi772@gmail.com) </div>
""",
unsafe_allow_html=True,
)
