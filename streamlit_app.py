import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import os

# 1. Page Configuration
st.set_page_config(page_title="Leather Defect Detector", layout="centered")
st.title("🏭 Leather Defect AI Classifier")
st.write("Upload an image of a leather surface to identify manufacturing anomalies instantly.")

# 2. Look for model assets in the Root Directory
MODEL_WEIGHTS = "leather_model.pth"
CLASSES_FILE = "classes.txt"

@st.cache_resource
def load_app_model():
    """Loads classes and imports model mapping safely from the root path."""
    if not os.path.exists(MODEL_WEIGHTS) or not os.path.exists(CLASSES_FILE):
        st.error("🚨 Model weights or classes file not found in root directory! Please run `train.py` first.")
        return None, None
        
    with open(CLASSES_FILE, "r") as f:
        class_names = f.read().splitlines()
        
    from model import LeatherDefectClassifier
    model = LeatherDefectClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

model, class_names = load_app_model()

# 3. Image Preprocessing Transform
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. UI File Upload
uploaded_file = st.file_uploader("Choose a leather image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Surface Layer", use_column_width=True)
    
    st.write("🔍 Analyzing surface data...")
    input_tensor = img_transform(image).unsqueeze(0)
    
    # Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, class_idx = torch.max(probabilities, 0)
        
    predicted_class = class_names[class_idx.item()]
    confidence_score = confidence.item() * 100
    
    # 5. Render Output Results UI
    st.divider()
    if "good" in predicted_class.lower() or "perfect" in predicted_class.lower():
        st.success(f"🟢 **Status: Premium Grade / Flawless**")
    else:
        st.error(f"🔴 **Status Defect Detected: {predicted_class.title()}**")
        
    st.metric(label="Model Match Confidence", value=f"{confidence_score:.2f}%")