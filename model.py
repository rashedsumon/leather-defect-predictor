import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class LeatherDefectClassifier(pl.LightningModule):
    def __init__(self, num_classes=6, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Lightweight MobileNetV3 backbone for optimized training and inference
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)