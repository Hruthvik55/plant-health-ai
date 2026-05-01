import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Plant Health AI",
    layout="wide",
    page_icon="🌱"
)

# ======================================
# LOAD MODEL
# ======================================
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# ======================================
# CLASS NAMES
# ======================================
class_names = [
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Healthy"
]

# ======================================
# DISEASE INFORMATION
# ======================================
disease_info = {

    "Tomato Early Blight": {
        "cause": "Fungal infection caused by Alternaria solani.",
        "solution": "Use fungicides and remove infected leaves."
    },

    "Tomato Late Blight": {
        "cause": "Caused by Phytophthora infestans.",
        "solution": "Avoid overwatering and apply fungicide."
    },

    "Tomato Leaf Mold": {
        "cause": "High humidity fungal disease.",
        "solution": "Improve ventilation and reduce moisture."
    },

    "Tomato Healthy": {
        "cause": "Healthy plant detected.",
        "solution": "Maintain regular crop care and monitoring."
    },

    "Potato Early Blight": {
        "cause": "Alternaria fungal infection.",
        "solution": "Use certified seeds and fungicides."
    },

    "Potato Late Blight": {
        "cause": "Water mold infection.",
        "solution": "Remove infected plants immediately."
    },

    "Potato Healthy": {
        "cause": "Healthy potato plant detected.",
        "solution": "Continue proper watering and nutrient management."
    },

    "Pepper Bacterial Spot": {
        "cause": "Bacterial infection in pepper leaves.",
        "solution": "Use disease-free seeds and copper sprays."
    },

    "Pepper Healthy": {
        "cause": "Healthy pepper plant detected.",
        "solution": "Maintain proper sunlight and watering."
    }
}

# ======================================
# CUSTOM CSS
# ======================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f7faf7;
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* HERO SECTION */

.hero {
    background: linear-gradient(90deg, #163d2a, #2f855a);
    padding: 70px 60px;
    border-radius: 22px;
    color: white;
    margin-bottom: 35px;
}

.hero h1 {
    font-size: 64px;
    font-weight: 700;
    margin-bottom: 12px;
}

.hero p {
    font-size: 22px;
    color: #e2f5e9;
    line-height: 1.6;
    max-width: 850px;
}

/* SECTION TITLES */

.section-title {
    font-size: 38px;
    font-weight: 700;
    color: #163d2a;
    margin-top: 30px;
    margin-bottom: 15px;
}

.section-text {
    font-size: 18px;
    color: #4b5563;
    line-height: 1.8;
}

/* CARDS */

.card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.06);
    margin-top: 20px;
}

.feature-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    text-align: left;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.06);
    height: 200px;
}

.feature-card h3 {
    color: #14532d;
    font-size: 24px;
}

.feature-card p {
    color: #4b5563;
    line-height: 1.7;
    font-size: 16px;
}

/* RESULT */

.result-card {
    background: #ecfdf3;
    border-left: 8px solid #16a34a;
    padding: 30px;
    border-radius: 20px;
    margin-top: 15px;
}

.result-card h1 {
    color: #14532d;
    font-size: 42px;
}

.result-card h3 {
    color: #374151;
}

/* IMAGE */

.stImage img {
    border-radius: 16px;
    max-width: 85%;
    margin: auto;
    display: block;
}

/* FOOTER */

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    padding-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# HERO SECTION
# ======================================
st.markdown("""
<div class="hero">
    <h1>Plant Health AI</h1>
    <p>
        An AI-powered plant disease detection platform designed to help identify crop diseases early
        using deep learning and computer vision technology.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================
# ABOUT SECTION
# ======================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">
About The Platform
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-text">
This system analyzes uploaded plant leaf images and predicts potential diseases using a trained
deep learning model. The application currently supports disease detection for Tomato, Potato,
and Pepper plants.
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# FEATURE SECTION
# ======================================
st.markdown("""
<div class="section-title">
Core Features
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>Fast Detection</h3>
        <p>
        Upload a plant leaf image and receive AI-based predictions within seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>Multi Crop Support</h3>
        <p>
        Supports multiple plant categories including Tomato, Potato, and Pepper crops.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>Deep Learning Powered</h3>
        <p>
        Built using TensorFlow and MobileNetV2 transfer learning architecture.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================================
# IMPORTANCE SECTION
# ======================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">
Why Plant Health Matters
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-text">
Early disease detection helps reduce crop loss, improve agricultural productivity,
and support healthier farming practices. AI-assisted monitoring enables faster
identification of infections before they spread across fields.
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# UPLOAD SECTION
# ======================================
st.markdown("""
<div class="section-title">
Disease Detection
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# PREDICTION SECTION
# ======================================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Leaf Image")

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with col2:

        if st.button("Analyze Plant"):

            with st.spinner("Analyzing image..."):

                prediction = model.predict(img_array)

                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                disease = class_names[predicted_class]

            st.markdown(f"""
            <div class="result-card">
                <h2>Prediction Result</h2>
                <h1>{disease}</h1>
                <h3>Confidence Score: {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

            info = disease_info[disease]

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("Cause")
            st.write(info["cause"])

            st.subheader("Recommended Action")
            st.write(info["solution"])

            st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# FOOTER
# ======================================
st.markdown("""
<div class="footer">
Built with TensorFlow, Streamlit, and Deep Learning
</div>
""", unsafe_allow_html=True)