import os
import torch
import pandas as pd
from data_split import MVTecDataSplitter
from client import Client
from aggregator import Aggregator
from config import Config

def print_gpu_info():
    """打印GPU使用情況"""
    print("\n=== GPU 使用情況 ===")
    print(f"當前使用設備: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
        print(f"GPU 數量: {torch.cuda.device_count()}")
        print(f"當前 GPU 內存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"當前 GPU 內存緩存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("警告: 未檢測到 GPU，將使用 CPU 進行訓練")
    print("==================\n")

def train_federated(epsilon):
    """執行聯邦學習訓練"""
    # 初始化數據分割器
    data_splitter = MVTecDataSplitter()
    clients_data = data_splitter.get_all_clients_data()
    
    # 初始化客戶端
    clients = []
    for i, client_data in enumerate(clients_data):
        client = Client(
            client_id=i,
            train_loader=client_data['train'],
            test_loader=client_data['test'],
            categories=client_data['categories']
        )
        clients.append(client)
        print(f"Client {i} 模型設備: {next(client.model.parameters()).device}")
    
    # 初始化聚合器
    aggregator = Aggregator()
    print(f"全局模型設備: {next(aggregator.global_model.parameters()).device}")
    
    # 訓練記錄
    results = []
    
    # 聯邦學習訓練循環
    for round_idx in range(Config.NUM_ROUNDS):
        print(f"\n{'='*20} Round {round_idx + 1}/{Config.NUM_ROUNDS} {'='*20}")
        
        # 客戶端本地訓練
        client_parameters = []
        for client in clients:
            # print(f"\nTraining client {client.client_id} with categories: {client.categories}")
            # 使用新的訓練方法，包含差分隱私
            client.train_with_dp(epsilon, Config.EPOCHS_PER_ROUND)
            # 獲取訓練後的參數
            client_parameters.append(client.get_parameters())
        
        # 聚合模型參數
        aggregator.aggregate(client_parameters)
        
        # 更新客戶端模型
        for client in clients:
           client.set_parameters(aggregator.get_parameters())
        
        # 評估全局模型
        global_accuracy = sum(client.evaluate() for client in clients) / len(clients)
        results.append({
            'round': round_idx + 1,
            'epsilon': epsilon,
            'accuracy': global_accuracy
        })
        
        print(f"\nRound {round_idx + 1} - Global Accuracy: {global_accuracy:.2%}")
        print(f"{'='*50}")
        
        # 每輪結束後打印GPU使用情況
        if torch.cuda.is_available():
            print(f"當前 GPU 內存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    return results

def save_results(results):
    """保存訓練結果"""
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    all_results = []
    for privacy_level, round_results in results.items():
        for result in round_results:
            result['privacy_level'] = privacy_level
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(Config.RESULTS_DIR, 'federated_results.csv'), index=False)
    print("\nTraining completed. Results saved to federated_results.csv")

def main():
    # 設置內存優化
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    # 設置環境變量以優化內存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print_gpu_info()
    
    # 訓練不同隱私級別的模型
    results = {}
    for privacy_level, epsilon_value in Config.EPSILON_VALUES.items():
        print(f"\nTraining with epsilon = {epsilon_value}")
        results[privacy_level] = train_federated(epsilon_value)
        
        # 每完成一個隱私級別的訓練後清理內存
        torch.cuda.empty_cache()
        print_gpu_info()
    
    # 保存結果
    save_results(results)

if __name__ == "__main__":
    main() 