# 🔍 Tata Steel — Automated Micro-Crack Inspection System

An end-to-end, enterprise-grade Deep Learning pipeline for detecting micro-level surface defects on steel plates, utilizing **Transfer Learning**, **Explainable AI (Grad-CAM)**, and **High-Resolution Sliding-Window Inference**. 

🔗 **[View the Live Deployed Application Here](https://tata-steel-inspection-gyappgxsbxa8axr6mrv8xyw.streamlit.app/)** 

## 🚀 Key Features
* **High-Performance CNN**: Fine-tuned **EfficientNet-B0** for 6-class steel defect classification (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches).
* **High-Resolution Micro-Crack Scanner**: Real factory cameras produce massive images. This system uses a sliding-window algorithm to slice high-res images (e.g., 1600x256) into overlapping 224x224 patches, detecting micro-cracks without losing resolution.
* **Explainable AI (XAI)**: Integrated **Grad-CAM** heatmaps to provide visual interpretability, allowing quality inspectors to see exactly which pixels drove the AI's decision.
* **Production-Ready (ONNX)**: Includes an export script to convert the PyTorch model to **ONNX format** for 3x faster CPU inference on factory edge devices.
* **Enterprise Dashboard**: A custom Streamlit UI featuring batch image processing, dynamic confidence thresholding, CSV report generation, and training telemetry.

## 📊 Model Performance
* **Architecture**: EfficientNet-B0 (Pre-trained on ImageNet)
* **Validation Accuracy**: ~94.2%
* **Validation F1-Score**: ~93.8%
* **Data Augmentation**: Albumentations (CLAHE, RandomResizedCrop, Rotation, Noise, Sharpening) to simulate real factory lighting and sensor noise.

## 🛠️ Tech Stack
* **Deep Learning**: PyTorch, Torchvision, Timm
* **Computer Vision**: OpenCV, Albumentations, PyTorch-Grad-CAM
* **Deployment & MLOps**: Streamlit, ONNX, ONNX Runtime
* **Data Processing**: Pandas, NumPy, Scikit-Learn

## 📁 Repository Structure
* `app.py`: The Streamlit dashboard (Standard, High-Res Scanner, and Performance tabs).
* `train.py`: The PyTorch training pipeline with data augmentation and Grad-CAM sanity checks.
* `export_onnx.py`: Script to export the trained `.pth` model to `.onnx` for production edge deployment.
* `best_model.pth`: The trained PyTorch model weights.
* `best_model.onnx`: The optimized ONNX model for production.
* `requirements.txt`: All necessary Python dependencies.

## 💻 How to Run Locally
1. Clone the repository: `git clone https://github.com/ShubhJK/tata-steel-inspection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
