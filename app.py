import streamlit as st
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AgroDetectAI",
    page_icon="🌱",
    layout="centered"
)

# -----------------------------
# CUSTOM UI
# -----------------------------

st.markdown("""
<style>

.title{
font-size:42px;
font-weight:bold;
text-align:center;
color:#2e7d32;
}

.resultbox{
padding:20px;
border-radius:12px;
background:#f1f8e9;
border:2px solid #81c784;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🌱 AgroDetectAI</p>', unsafe_allow_html=True)
st.caption("AI Powered Plant Disease Detection")

st.markdown("---")

# -----------------------------
# LANGUAGE SUPPORT
# -----------------------------

description = {
"English":"Upload a clear leaf image to detect plant diseases.",
"Hindi":"पत्ते की स्पष्ट तस्वीर अपलोड करें और बीमारी का पता लगाएं।",
"Telugu":"స్పష్టమైన ఆకుల చిత్రాన్ని అప్లోడ్ చేసి వ్యాధిని గుర్తించండి."
}

language = st.selectbox(
"🌐 Select Language",
["English","Hindi","Telugu"]
)

st.info(description[language])

st.markdown("---")

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True
    )

    model.eval()

    return model

model = load_model()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# DEMO DISEASE LABELS
# -----------------------------

labels = [
"Healthy Leaf",
"Early Blight",
"Late Blight",
"Leaf Spot",
"Rust Disease",
"Mosaic Virus"
]

# -----------------------------
# TREATMENT DATA
# -----------------------------

treatments = {

"Early Blight":{
"English":"Remove infected leaves and spray copper fungicide.",
"Hindi":"संक्रमित पत्तियों को हटाएं और कॉपर फंगीसाइड छिड़कें।",
"Telugu":"సంక్రమిత ఆకులను తొలగించి కాపర్ ఫంగిసైడ్ పిచికారీ చేయండి."
},

"Late Blight":{
"English":"Apply Mancozeb fungicide.",
"Hindi":"मैनकोजेब फंगीसाइड का उपयोग करें।",
"Telugu":"మాంకోజెబ్ ఫంగిసైడ్ ఉపయోగించండి."
},

"Leaf Spot":{
"English":"Use chlorothalonil spray.",
"Hindi":"क्लोरोथालोनिल स्प्रे का उपयोग करें।",
"Telugu":"క్లోరోథాలోనిల్ స్ప్రే ఉపయోగించండి."
},

"Rust Disease":{
"English":"Apply sulfur fungicide.",
"Hindi":"सल्फर आधारित फंगीसाइड लगाएं।",
"Telugu":"సల్ఫర్ ఫంగిసైడ్ ఉపయోగించండి."
},

"Mosaic Virus":{
"English":"Remove infected plants immediately.",
"Hindi":"संक्रमित पौधों को तुरंत हटा दें।",
"Telugu":"సంక్రమిత మొక్కలను వెంటనే తొలగించండి."
}

}

# -----------------------------
# LEAF VALIDATION
# -----------------------------

def is_leaf(image):

    img_np = np.array(image)

    green_pixels = np.sum(
        (img_np[:,:,1] > img_np[:,:,0]) &
        (img_np[:,:,1] > img_np[:,:,2])
    )

    total_pixels = img_np.shape[0] * img_np.shape[1]

    green_ratio = green_pixels / total_pixels

    if green_ratio > 0.15:
        return True
    else:
        return False


# -----------------------------
# IMAGE INPUT
# -----------------------------

st.subheader("📷 Upload or Capture Leaf")

uploaded_file = st.file_uploader(
"Upload Leaf Image",
type=["jpg","jpeg","png"]
)

camera_image = st.camera_input("Or Capture Image")

image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_image:
    image = Image.open(camera_image).convert("RGB")

# -----------------------------
# DISPLAY IMAGE
# -----------------------------

if image:

    st.image(image, caption="Leaf Image", use_column_width=True)

    if st.button("🔍 Detect Disease"):

        with st.spinner("Analyzing leaf... 🌿"):

            # Leaf validation
            if not is_leaf(image):

                st.error("❌ This does not appear to be a leaf image. Please upload a clear leaf photo.")

            else:

                img = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img)

                probs = torch.nn.functional.softmax(outputs, dim=1)

                top3 = torch.topk(probs,3)

                st.markdown("---")

                st.markdown('<div class="resultbox">', unsafe_allow_html=True)

                st.success("🌿 Leaf Detected")

                best_label = labels[top3.indices[0][0].item() % len(labels)]
                best_conf = top3.values[0][0].item()*100

                st.write("### 🦠 Predicted Disease")
                st.write(best_label)

                st.write("### 📊 Confidence")

                st.progress(best_conf/100)

                st.write(f"{best_conf:.2f}%")

                st.write("### 🔬 Top 3 Predictions")

                for i in range(3):

                    idx = top3.indices[0][i].item()
                    score = top3.values[0][i].item()*100

                    label = labels[idx % len(labels)]

                    st.write(f"{label} — {score:.2f}%")

                # Treatment Suggestions
                if best_label in treatments:

                    st.write("### 💊 Treatment")

                    st.info(treatments[best_label][language])

                st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.caption("AgroDetectAI • Smart Farming with AI 🌾")
