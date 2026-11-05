import streamlit as st
import torch, cv2, numpy as np, pandas as pd, os
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
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
.stApp { background: linear-gradient(135deg, #FFD700 0%, #FFC300 100%); }
[data-testid="stSidebar"] { background-color: #0A1F44; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; font-size: 1.05rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFD700 !important; font-weight: 800; }
h1 { color: #0A1F44 !important; font-weight: 800 !important; font-size: 2.8rem !important; letter-spacing: -1px; }
h2, h3 { color: #0A1F44 !important; font-weight: 600 !important; }
p, label, .stMarkdown { color: #0A1F44; font-size: 1.1rem; font-weight: 500; }

/* Tabs Styling - Teal Active Tab */
.stTabs [data-baseweb="tab-list"] { gap: 20px; }
.stTabs [data-baseweb="tab"] { background-color: rgba(255, 255, 255, 0.6); border-radius: 8px 8px 0 0; padding: 10px 20px; font-weight: 700; color: #0A1F44 !important; }
.stTabs [aria-selected="true"] { background-color: #008080 !important; }
.stTabs [aria-selected="true"] * { color: #FFFFFF !important; }

/* Custom Cards */
.metric-card { background-color: #0A1F44; padding: 20px; border-radius: 12px; border-left: 8px solid #008080; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px; }
.metric-label { color: #FFD700 !important; font-size: 1rem !important; font-weight: 600; margin-bottom: 5px; text-transform: uppercase; }
.metric-value { color: #FFFFFF !important; font-size: 1.8rem !important; font-weight: 800; }
.stDownloadButton>button { background-color: #008080 !important; color: #FFFFFF !important; font-weight: 700; border-radius: 8px; }
.stDownloadButton>button:hover { background-color: #00A8A8 !important; }
.footer { color: #0A1F44 !important; font-weight: 600; text-align: center; margin-top: 60px; opacity: 0.8; }
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

# --- CREATE TABS ---
tab1, tab2, tab3 = st.tabs(["📸 Standard Inspection", "🔬 High-Res Micro-Crack Scanner", "📊 Model Performance"])

# ==========================================
# TAB 1: STANDARD & BATCH PROCESSING
# ==========================================
with tab1:
    with st.sidebar:
        st.header("Inspection Controls")
        st.write("Standard mode for discrete 200x200 images.")
        uploaded_files = st.file_uploader("Choose images", type=["jpg","jpeg","png","bmp"], accept_multiple_files=True)
        st.markdown("---")
        threshold = st.slider("Confidence Threshold for PASS", 0.50, 0.99, 0.80, 0.01)

    if uploaded_files:
        results_data = []
        st.info(f"Processing {len(uploaded_files)} image(s)...")
        
        for uploaded_file in uploaded_files:
            img_pil = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img_pil)
            tensor = transform(image=img_np)["image"].unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            idx = probs.argmax().item(); conf = probs[idx].item()
            
            cam = GradCAM(model=model, target_layers=[model.features[-1]])
            gcam = cam(input_tensor=tensor, targets=None)
            disp = cv2.resize(img_np, (224,224)).astype(np.float32)/255.0
            overlay = show_cam_on_image(disp, gcam[0], use_rgb=True)
            
            decision = "⚠️ REVIEW" if conf < threshold else "✅ PASS"
            results_data.append({"Image Name": uploaded_file.name, "Predicted Defect": CLASS_NAMES[idx].replace("_"," ").title(), "Confidence": f"{conf*100:.2f}%", "Decision": decision})

            with st.expander(f"📄 {uploaded_file.name} — {decision}", expanded=True):
                c1, c2 = st.columns(2)
                c1.image(img_pil, caption="Original", use_container_width=True)
                c2.image(overlay, caption="Grad-CAM", use_container_width=True)
                
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="metric-card"><div class="metric-label">Defect</div><div class="metric-value">{CLASS_NAMES[idx].replace("_"," ").title()}</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{conf*100:.2f}%</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-card"><div class="metric-label">Decision</div><div class="metric-value">{decision}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        df = pd.DataFrame(results_data)
        st.download_button("📥 Download Inspection Report (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name='inspection_report.csv', mime='text/csv')
        st.dataframe(df, use_container_width=True)
    else:
        st.info("👆 Upload standard images from the sidebar.")

# ==========================================
# TAB 2: HIGH-RES MICRO-CRACK SCANNER
# ==========================================
with tab2:
    st.header("High-Resolution Micro-Crack Scanner")
    st.write("Upload a high-resolution steel strip (e.g., 1600x256). The AI uses a sliding window to scan for micro-level defects and maps them back to the original resolution.")
    
    highres_file = st.file_uploader("Upload High-Res Image", type=["jpg","jpeg","png","bmp"], key="highres")
    scan_threshold = st.slider("Micro-Crack Sensitivity Threshold", 0.50, 0.99, 0.70, 0.01, key="highres_slider")

    if highres_file:
        img_pil = Image.open(highres_file).convert("RGB")
        img_np = np.array(img_pil)
        h, w, _ = img_np.shape
        
        patch_size = 224
        stride = 112 # 50% overlap
        
        patches = []
        coords = []
        
        # Sliding Window Extraction
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img_np[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                coords.append((x, y))
                
        if not patches:
            st.warning("Image is smaller than 224x224. Please upload a larger high-res image.")
        else:
            with st.spinner(f"Scanning {len(patches)} micro-patches..."):
                defective_boxes = []
                batch_size = 32
                for i in range(0, len(patches), batch_size):
                    batch = patches[i:i+batch_size]
                    tensors = torch.stack([transform(image=p)["image"] for p in batch])
                    with torch.no_grad():
                        probs = torch.softmax(model(tensors), dim=1)
                    
                    for j, p in enumerate(probs):
                        max_conf, idx = torch.max(p, dim=0)
                        if max_conf.item() >= scan_threshold:
                            defective_boxes.append((coords[i+j], CLASS_NAMES[idx], max_conf.item()))
                
                # Draw boxes
                overlay_img = img_np.copy()
                for (box, cls_name, conf) in defective_boxes:
                    x, y = box
                    # Thicker Teal Box (thickness=5) for better visibility
                    cv2.rectangle(overlay_img, (x, y), (x+patch_size, y+patch_size), (0, 128, 128), 5)
                    # Larger Label
                    label = f"{cls_name} {conf*100:.1f}%"
                    cv2.putText(overlay_img, label, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if len(defective_boxes) > 0:
                    st.success(f"Scan Complete! Found {len(defective_boxes)} defective micro-regions.")
                else:
                    st.success("Scan Complete! No defects found above the threshold.")
                
                c1, c2 = st.columns(2)
                c1.image(img_pil, caption="High-Res Original", use_container_width=True)
                c2.image(overlay_img, caption="Micro-Crack Detection Overlay", use_container_width=True)

                # --- NEW: ZOOM INTO DEFECTS ---
                if len(defective_boxes) > 0:
                    st.markdown("---")
                    st.subheader("🔍 Zoomed-in Micro-Crack Details")
                    st.write("Here are the exact 224x224 patches that triggered the AI's alarm:")
                    
                    # Show up to 4 cropped defects
                    cols = st.columns(min(len(defective_boxes), 4))
                    for idx_loop, (box, cls_name, conf) in enumerate(defective_boxes[:4]):
                        x, y = box
                        crop = img_np[y:y+patch_size, x:x+patch_size]
                        cols[idx_loop].image(crop, caption=f"{cls_name.title()} ({conf*100:.1f}%)", use_container_width=True)

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab3:
    st.header("Model Evaluation")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model Architecture", "EfficientNet-B0")
    col_b.metric("Validation Accuracy", "~94.2%")
    col_c.metric("Validation F1-Score", "~93.8%")
    
    st.markdown("---")
    st.subheader("Confusion Matrix")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", use_container_width=True)
    else:
        st.warning("⚠️ `confusion_matrix.png` not found.")
        
    st.markdown("---")
    st.subheader("Grad-CAM Sanity Checks during Training")
    gcam_images = [f"gradcam_epoch_{i}.png" for i in [5, 10, 15, 20, 25] if os.path.exists(f"gradcam_epoch_{i}.png")]
    if gcam_images:
        cols = st.columns(len(gcam_images))
        for col, img_path in zip(cols, gcam_images):
            col.image(img_path, caption=img_path, use_container_width=True)

st.markdown('<div class="footer">Tata Steel Ltd. | Automated Quality Inspection | EfficientNet-B0 + Grad-CAM</div>', unsafe_allow_html=True)
