import torch
import pytorch_lightning as pl
from data_loader import get_dataloaders
from model import LeatherDefectClassifier

def main():
    # 1. Setup loaders and classes
    train_loader, class_names = get_dataloaders(batch_size=32)
    
    # Save classes right into the root folder
    with open("classes.txt", "w") as f:
        f.write("\n".join(class_names))
        
    print(f"Training on {len(class_names)} classes: {class_names}")

    # 2. Initialize Model
    model = LeatherDefectClassifier(num_classes=len(class_names))

    # 3. Train model
    trainer = pl.Trainer(max_epochs=3, accelerator="auto", devices=1)
    trainer.fit(model, train_loader)

    # 4. Save structural weights into the root folder
    torch.save(model.state_dict(), "leather_model.pth")
    print("🎉 Model successfully trained and saved to root directory as leather_model.pth!")

if __name__ == "__main__":
    main()