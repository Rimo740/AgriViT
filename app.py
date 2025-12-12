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

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    /* Card Styling */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #166534 !important; /* Dark Green */
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f0fdf4;
        border-right: 1px solid #bbf7d0;
    }
    
    /* Custom Button Styling */
    .stButton>button {
        color: white;
        background: linear-gradient(to right, #22c55e, #16a34a);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(34, 197, 94, 0.4);
    }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #16a34a;
    }
    
    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0fdf4;
        color: #166534;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #bbf7d0;
        font-size: 14px;
        z-index: 100;
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
        # (This handles the compressed weights correctly)
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
                
                # Color Logic
                if score > 85: status_color = "green"
                elif score > 60: status_color = "orange"
                else: status_color = "red"
                
                st.markdown(f"""
                <div style="background-color: #f0fdf4; border: 2px solid {status_color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: #166534; margin:0;">{class_name}</h2>
                    <h4 style="color: #555; margin:0;">Confidence: {score:.2f}%</h4>
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
    <div style="margin-top: 50px; text-align: center; color: #555; font-size: 0.9em;">
        <hr>
        <p>Developed by <b>Rimo Bhuiyan</b> | AI/ML Engineer | Co-Founder @CollabCircle</p>
    </div>
    """, unsafe_allow_html=True)