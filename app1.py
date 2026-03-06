import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AgroDetect AI",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------
# MODERN UI STYLE
# -----------------------------

st.markdown("""
<style>

.main {
background-color:#f4fff6;
}

h1 {
color:#2e7d32;
text-align:center;
}

.stButton>button {
background-color:#2e7d32;
color:white;
font-size:18px;
border-radius:10px;
padding:10px 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():

    classifier = pipeline(
        "image-classification",
        model="microsoft/resnet-50"
    )

    return classifier

classifier = load_model()

# -----------------------------
# TRANSLATOR
# -----------------------------

def translate(text, lang):

    languages = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te"
    }

    if lang == "English":
        return text

    return GoogleTranslator(
        source="auto",
        target=languages[lang]
    ).translate(text)

# -----------------------------
# TREATMENT DATABASE
# -----------------------------

def suggest_treatment(label):

    label = label.lower()

    if "rust" in label:
        return (
            "Rust disease detected.",
            "Spray copper fungicide every 7 days.",
            "Avoid wet leaves and improve air circulation."
        )

    elif "spot" in label:
        return (
            "Leaf spot disease detected.",
            "Apply neem oil or copper fungicide.",
            "Remove infected leaves immediately."
        )

    elif "blight" in label:
        return (
            "Blight disease detected.",
            "Use fungicides like chlorothalonil.",
            "Avoid excess watering."
        )

    elif "mildew" in label:
        return (
            "Powdery mildew detected.",
            "Use sulfur-based fungicide spray.",
            "Ensure proper sunlight and airflow."
        )

    else:
        return (
            "Plant condition detected.",
            "Use organic pesticide like neem oil.",
            "Monitor plant regularly."
        )

# -----------------------------
# TITLE
# -----------------------------

st.title("🌿 AgroDetect AI")
st.write("AI Based Plant Disease Detection System")

# -----------------------------
# LANGUAGE SELECTOR
# -----------------------------

language = st.selectbox(
    "Select Language / भाषा / భాష",
    ["English","Hindi","Telugu"]
)

# -----------------------------
# IMAGE INPUT
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg","png","jpeg"]
)

camera_photo = st.camera_input("Or Capture from Camera")

image = None

if uploaded_file:
    image = Image.open(uploaded_file)

elif camera_photo:
    image = Image.open(camera_photo)

# -----------------------------
# PREDICTION
# -----------------------------

if image:

    st.image(image, caption="Leaf Image", width=350)

    if st.button("Detect Disease"):

        with st.spinner("Analyzing plant..."):

            results = classifier(image)

        st.subheader("Top Predictions")

        for r in results[:3]:

            label = r["label"]
            confidence = round(r["score"]*100,2)

            st.write(f"{label} — {confidence}%")

        best_label = results[0]["label"]

        desc, treatment, precaution = suggest_treatment(best_label)

        desc = translate(desc, language)
        treatment = translate(treatment, language)
        precaution = translate(precaution, language)

        st.subheader("Disease Description")
        st.write(desc)

        st.subheader("Recommended Treatment")
        st.write(treatment)

        st.subheader("Precautions")
        st.write(precaution)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("About AgroDetect AI")

st.sidebar.write("""
AgroDetect AI helps farmers detect plant diseases using AI.

Features:

• AI leaf analysis  
• Camera capture  
• Multilingual support  
• Treatment recommendations  
• Farmer friendly interface
""")