import streamlit as st
import torch
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
import os

# Page configuration
st.set_page_config(page_title="Leather Defect Detector", layout="centered")
st.title("🏭 Leather Defect AI Classifier")

MODEL_WEIGHTS = "leather_model.pth"
CLASSES_FILE = "classes.txt"

def run_inline_training():
    """Handles data downloading and training execution directly inside Streamlit."""
    from data_loader import get_dataloaders
    from model import LeatherDefectClassifier
    
    # 1. Load Data
    train_loader, class_names = get_dataloaders(batch_size=32)
    
    # Save the class tags text file
    with open(CLASSES_FILE, "w") as f:
        f.write("\n".join(class_names))
        
    # 2. Setup Network
    model = LeatherDefectClassifier(num_classes=len(class_names))
    
    # 3. Train Model (Keeping epochs minimal for Streamlit Cloud CPU limits)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader)
    
    # 4. Save Weights
    torch.save(model.state_dict(), MODEL_WEIGHTS)

@st.cache_resource
def load_app_model():
    """Checks for model assets. If missing, triggers training seamlessly."""
    if not os.path.exists(MODEL_WEIGHTS) or not os.path.exists(CLASSES_FILE):
        st.warning("⚠️ Weights not found! Initializing automated training on Streamlit Cloud...")
        with st.spinner("⏳ Downloading Kaggle dataset & training architecture (This can take 2-4 minutes)..."):
            run_inline_training()
        st.success("🎉 Model trained successfully!")
        
    with open(CLASSES_FILE, "r") as f:
        class_names = f.read().splitlines()
        
    from model import LeatherDefectClassifier
    model = LeatherDefectClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

# Run asset loader check
model, class_names = load_app_model()

st.write("Upload an image of a leather surface to identify manufacturing anomalies instantly.")

# Transform block
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# File Uploader component
uploaded_file = st.file_uploader("Choose a leather image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Surface Layer", use_column_width=True)
    
    st.write("🔍 Analyzing surface data...")
    input_tensor = img_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, class_idx = torch.max(probabilities, 0)
        
    predicted_class = class_names[class_idx.item()]
    confidence_score = confidence.item() * 100
    
    st.divider()
    if "good" in predicted_class.lower() or "perfect" in predicted_class.lower():
        st.success(f"🟢 **Status: Premium Grade / Flawless**")
    else:
        st.error(f"🔴 **Status: Defect Detected: **")
        
    