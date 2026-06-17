import os
import shutil
from pathlib import Path
import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path("./data/leather-defect-classification")

def download_and_setup_data():
    """Downloads dataset via kagglehub and saves it locally."""
    if not DATA_DIR.exists():
        print("📥 Downloading dataset from Kaggle...")
        downloaded_path = kagglehub.dataset_download("praveen2084/leather-defect-classification")
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for item in os.listdir(downloaded_path):
            s = os.path.join(downloaded_path, item)
            d = os.path.join(DATA_DIR, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print("✅ Dataset ready at:", DATA_DIR)
    else:
        print("📊 Dataset already exists locally.")

def get_dataloaders(batch_size=32):
    """Ensures dataset exists and returns train data loaders."""
    download_and_setup_data()
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_path = DATA_DIR / "train" if (DATA_DIR / "train").exists() else DATA_DIR
    train_dataset = datasets.ImageFolder(root=str(train_path), transform=train_transform)
    
    class_names = train_dataset.classes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # 0 for stability in cloud environments
    
    return train_loader, class_names