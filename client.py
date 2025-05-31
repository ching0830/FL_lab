import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import AnomalyDetectionModel
from config import Config
import torch.nn.utils as utils

class Client:
    def __init__(self, client_id, train_loader, test_loader, categories):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.categories = categories
        self.model = AnomalyDetectionModel().to(Config.DEVICE)
        self.criterion = nn.CrossEntropyLoss().to(Config.DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)

    def train_with_dp(self, epsilon, epochs):
        """使用差分隱私進行訓練"""
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-4  # 添加 L2 正則化
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=2,
            min_lr=1e-6
        )
        
        best_acc = 0.0
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向傳播
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 反向傳播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_NORM)
                
                # 添加差分隱私噪聲
                if epsilon < float('inf'):
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * (Config.CLIP_NORM / epsilon)
                            param.grad += noise
                
                # 更新參數
                optimizer.step()
                
                # 統計
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # 計算並打印每個epoch的結果
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Client {self.client_id} - Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%, LR={current_lr:.6f}')
            
            # 更新學習率
            scheduler.step(epoch_acc)
            
            # 保存最佳模型
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_params = [param.data.clone() for param in self.model.parameters()]
        
        # 使用最佳參數
        if best_acc > 0:
            for param, best_param in zip(self.model.parameters(), best_params):
                param.data.copy_(best_param)

    def get_parameters(self):
        """✅ 正確取得訓練後參數（直接從 model.parameters）"""
        return [param.data.clone() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        """✅ 正確更新參數（與 FedAvg 聚合結果同步）"""
        for param, new_param in zip(self.model.parameters(), parameters):
            if param.shape == new_param.shape:
                param.data.copy_(new_param.data.to(Config.DEVICE))
            else:
                print(f"⚠️ Shape mismatch: {param.shape} vs {new_param.shape}")

    def evaluate(self):
        """評估模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                try:
                    data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
                    
                    # 檢查數據形狀
                    if len(data.shape) != 4:
                        print(f"Warning: Unexpected data shape: {data.shape}")
                        continue
                    
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    
                except RuntimeError as e:
                    print(f"Error during evaluation:")
                    print(f"Data shape: {data.shape}")
                    print(f"Model device: {next(self.model.parameters()).device}")
                    print(f"Error message: {str(e)}")
                    raise e
        
        accuracy = correct / total if total > 0 else 0
        print(f'Client {self.client_id} - Test Accuracy: {accuracy:.2%}')
        return accuracy
