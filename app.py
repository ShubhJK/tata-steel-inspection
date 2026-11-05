import streamlit as st
import torch, cv2, numpy as np
from torchvision.models import efficientnet_b0
import torch.nn as nn
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

st.set_page_config(page_title="Tata Steel — Micro-Crack Inspection", layout="wide")

# --- PROFESSIONAL CSS & FONTS ---
st.markdown("""
<style>
/* Import Professional Font */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

/* Background Gradient */
.stApp { background: linear-gradient(135deg, #FFD700 0%, #FFC300 100%); }

/* Sidebar Styling */
[data-testid="stSidebar"] { background-color: #0A1F44; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; font-size: 1.05rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFD700 !important; font-weight: 800; }

/* Main Headings */
h1 { color: #0A1F44 !important; font-weight: 800 !important; font-size: 2.8rem !important; letter-spacing: -1px; }
h2, h3 { color: #0A1F44 !important; font-weight: 600 !important; }
p { color: #0A1F44; font-size: 1.1rem; font-weight: 500; }

/* File Uploader */
.stFileUploader { border: 2px dashed #008080; border-radius: 12px; padding: 15px; background-color: rgba(255, 255, 255, 0.1); }
.stFileUploader label { color: #FFD700 !important; font-weight: 600; }

/* Custom Metric Cards for Results */
.metric-card {
    background-color: #0A1F44;
    padding: 20px;
    border-radius: 12px;
    border-left: 8px solid #008080;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.metric-label {
    color: #FFD700 !important;
    font-size: 1rem !important;
    font-weight: 600;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    color: #FFFFFF !important;
    font-size: 1.8rem !important;
    font-weight: 800;
}

/* Footer */
.footer { color: #0A1F44 !important; font-weight: 600; text-align: center; margin-top: 60px; opacity: 0.8; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 6)
    m.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    m.eval()
    return m

model = load_model()
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
transform = Compose([Resize(height=224, width=224),
                     Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                     ToTensorV2()])

st.title("🔍 Tata Steel — Micro-Crack Inspection")
st.markdown("### Explainable AI for Surface Defect Detection")

with st.sidebar:
    st.header("Inspection Controls")
    st.write("Upload a steel plate image to detect micro-level surface defects.")
    f = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp"])

if f:
    img_pil = Image.open(f).convert("RGB")
    img_np = np.array(img_pil)
    tensor = transform(image=img_np)["image"].unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item(); conf = probs[idx].item()

    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    gcam = cam(input_tensor=tensor, targets=None)
    disp = cv2.resize(img_np, (224,224)).astype(np.float32)/255.0
    overlay = show_cam_on_image(disp, gcam[0], use_rgb=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original Image")
        # Fixed the deprecation warning here
        st.image(img_pil, use_container_width=True)
    with c2:
        st.subheader("Grad-CAM Explanation")
        st.image(overlay, use_container_width=True, caption="Heatmap highlights the pixels driving the prediction.")

    st.markdown("---")
    st.subheader("Inspection Result")
    
    # Custom HTML Cards to fix the blank navy boxes
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Defect</div><div class="metric-value">{CLASS_NAMES[idx].replace("_"," ").title()}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{conf*100:.2f}%</div></div>', unsafe_allow_html=True)
    with m3:
        decision = "⚠️ REVIEW" if conf < 0.80 else "✅ PASS"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Decision</div><div class="metric-value">{decision}</div></div>', unsafe_allow_html=True)

    st.subheader("Top-3 Predictions")
    top3 = torch.topk(probs, 3)
    for _, (i, p) in enumerate(zip(top3.indices, top3.values)):
        st.progress(float(p), text=f"{CLASS_NAMES[i].replace('_',' ').title()} — {p*100:.2f}%")

    st.info("💡 **Note:** A confidence score below 80% is automatically routed to human review. Verify with the Grad-CAM heatmap before accepting.")
else:
    st.info("👆 Please upload a steel plate image from the sidebar to begin inspection.")

# Styled Footer
st.markdown('<div class="footer">Tata Steel Ltd. | Automated Quality Inspection | EfficientNet-B0 + Grad-CAM</div>', unsafe_allow_html=True)