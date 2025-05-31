import torch
from model import AnomalyDetectionModel
from config import Config

class Aggregator:
    def __init__(self):
        self.global_model = AnomalyDetectionModel().to(Config.DEVICE)
        
    def aggregate(self, client_parameters_list):
        """FedAvg聚合算法"""
        with torch.no_grad():
            # 初始化聚合參數
            aggregated_parameters = []
            for param in self.global_model.parameters():
                aggregated_parameters.append(torch.zeros_like(param).to(Config.DEVICE))
            
            # 平均所有客戶端的參數
            for client_params in client_parameters_list:
                for i, param in enumerate(client_params):
                    aggregated_parameters[i].add_(param.to(Config.DEVICE) / len(client_parameters_list))
            
            # 印一下某層的參數平均值確認有變
            print(f"[Aggregator] Param 0 avg: {aggregated_parameters[0].mean().item():.6f}")
            print(f"[Aggregator] Param 0 std: {aggregated_parameters[0].std().item():.6f}")
            
            # 更新全局模型
            for param, new_param in zip(self.global_model.parameters(), aggregated_parameters):
                param.data.copy_(new_param)
                
    def get_parameters(self):
        """獲取全局模型參數"""
        return [param.data.clone() for param in self.global_model.parameters()] 