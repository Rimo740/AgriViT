import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageFilter
import json
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriViT",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING (Dark Glassmorphism) ---
st.markdown("""
    <style>
    /* 1. Main Background: Deep Black-Green Gradient */
    .stApp {
        background: radial-gradient(circle at top left, #052e16, #000000);
        color: #e2e8f0;
    }

    /* 2. Sidebar Glass Effect */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 20, 5, 0.6);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(74, 222, 128, 0.1);
    }
    
    /* 3. Headings with Neon Glow */
    h1, h2, h3 {
        color: #4ade80 !important; /* Neon Green */
        font-family: 'Helvetica Neue', sans-serif;
        text-shadow: 0 0 15px rgba(74, 222, 128, 0.3);
    }
    
    /* 4. Glass Cards for Streamlit Containers */
    div[data-testid="stVerticalBlock"] > div {
        /* Optional: Add spacing if needed */
    }

    /* 5. Custom Button Styling (Neon Green Gradient) */
    .stButton>button {
        color: #000;
        font-weight: bold;
        background: linear-gradient(90deg, #4ade80, #22c55e);
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(74, 222, 128, 0.6);
        color: #fff;
    }
    
    /* 6. Inputs & file uploaders (Dark Glass) */
    .stSelectbox > div > div, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px;
    }
    
    /* 7. Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(5px);
        color: #86efac;
        text-align: center;
        padding: 12px;
        border-top: 1px solid rgba(74, 222, 128, 0.2);
        font-size: 14px;
        z-index: 100;
    }
    
    /* 8. Custom Glass Card Class for Results */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIG & PATH FIX ---
DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOAD ASSETS ---
try:
    json_path = os.path.join(BASE_DIR, 'class_names.json')
    with open(json_path, 'r') as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    st.error(f"❌ Critical Error: 'class_names.json' not found at {json_path}. Did you extract the zip?")
    st.stop()

# --- MODEL LOADER ---
@st.cache_resource
def load_model(model_name):
    path = ""
    model = None
    
    try:
        if model_name == 'MobileNetV3':
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
            path = os.path.join(BASE_DIR, "mobilenet_v3.pth")
            
        elif model_name == 'EfficientNet-B0':
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
            path = os.path.join(BASE_DIR, "efficientnet_b0.pth")
            
        elif model_name == 'MobileViT':
            # Swin-T structure used as MobileViT proxy
            model = models.swin_t(weights=None)
            model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))
            path = os.path.join(BASE_DIR, "mobilevit.pth")

        # Check file existence
        if not os.path.exists(path):
            st.error(f"⚠️ File not found: {path}")
            return None
            
        # Load weights
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        # CRITICAL: Convert back to Float32 for CPU usage
        model.float() 
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

# --- PREPROCESSING ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- MAIN INTERFACE ---
st.markdown("# 🌿 AgriViT")
st.markdown("### Robust Crop Disease Detection System")
st.markdown("---")

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/plant-under-rain.png", width=80)
st.sidebar.markdown("### ⚙️ Configuration")
model_choice = st.sidebar.selectbox("Select Architecture:", ["MobileViT", "MobileNetV3", "EfficientNet-B0"])

# Load Model with Spinner
with st.spinner(f"Loading {model_choice}..."):
    model = load_model(model_choice)

if model is None:
    st.sidebar.error(f"⚠️ Model file for {model_choice} not found! Please ensure .pth files are in {BASE_DIR}")

st.sidebar.markdown("---")
st.sidebar.info("AgriViT uses Vision Transformers to detect diseases even in low light or blurry conditions.")

# Main Layout
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("#### 1. Upload Image")
    uploaded_file = st.file_uploader("Drop a leaf image here...", type=["jpg", "png", "jpeg"])

    st.markdown("#### 2. Simulation (Stress Test)")
    st.caption("Test the model's robustness by artificially degrading the image quality.")
    add_blur = st.checkbox("Apply Gaussian Blur")
    add_noise = st.checkbox("Add Sensor Noise")
    
with col2:
    st.markdown("#### 3. Analysis Results")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Apply Simulations
        if add_blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=4))
        if add_noise:
            img_np = np.array(image)
            noise = np.random.normal(0, 30, img_np.shape).astype(np.uint8)
            img_np = np.clip(img_np + noise, 0, 255)
            image = Image.fromarray(img_np)
            
        st.image(image, caption="Input Tensor", use_column_width=True, channels="RGB")
        
        if st.button("🔍 Run Diagnosis"):
            if model:
                # Progress Bar Animation
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                
                # Inference
                input_tensor = process_image(image)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                # Result Display
                class_name = CLASS_NAMES[predicted.item()]
                score = confidence.item() * 100
                
                # Color Logic for text glow
                if score > 85: glow_color = "#4ade80" # Green
                elif score > 60: glow_color = "#facc15" # Yellow/Orange
                else: glow_color = "#ef4444" # Red
                
                # HTML Card for Result
                st.markdown(f"""
                <div class="glass-card" style="border-color: {glow_color};">
                    <h2 style="color: {glow_color} !important; margin:0; text-shadow: 0 0 10px {glow_color};">{class_name}</h2>
                    <h4 style="color: #e2e8f0; margin-top: 10px;">Confidence: {score:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 3
                st.markdown("##### Alternative Diagnoses:")
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                for i in range(3):
                    name = CLASS_NAMES[top3_idx[0][i].item()]
                    prob = top3_prob[0][i].item() * 100
                    st.write(f"- **{name}**: {prob:.1f}%")
            else:
                st.error("Model not loaded correctly.")

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        <p>Developed by <b style="color: #fff;">Rimo Bhuiyan</b> | AI/ML Engineer | Co-Founder @CollabCircle</p>
    </div>
    """, unsafe_allow_html=True)