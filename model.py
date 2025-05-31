import torch
import torch.nn as nn
from torchvision import models
from config import Config

class AnomalyDetectionModel(nn.Module):
    def __init__(self):
        super(AnomalyDetectionModel, self).__init__()
        # 使用預訓練 VGG16 並載入權重
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # 凍結特徵提取層
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # 修改分類器，使用 LayerNorm 替代 BatchNorm
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, Config.NUM_CLASSES)
        )
        
        # 移到指定設備
        self.model = self.model.to(Config.DEVICE)

    def forward(self, x):
        return self.model(x)

    def get_parameters(self):
        """獲取需要訓練的參數"""
        return [p for p in self.parameters() if p.requires_grad]

    def set_parameters(self, parameters):
        """設置模型參數（用於聯邦學習）"""
        for param, new_param in zip(self.parameters(), parameters):
            if param.shape == new_param.shape:
                param.data.copy_(new_param.data.to(Config.DEVICE))
            else:
                print(f"Warning: Parameter shape mismatch: {param.shape} vs {new_param.shape}")
