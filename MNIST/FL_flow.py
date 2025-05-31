import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import re

# LeNet模型定义
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一個卷積塊
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 第二個卷積塊
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全連接層
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        return x

# Local Differential Privacy: Laplace噪声
def add_laplace_noise(tensor, epsilon, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = torch.from_numpy(np.random.laplace(0, scale, tensor.size())).float().to(tensor.device)
    return tensor + noise

# 客户端本地训练
def local_train(model, train_loader, device, epochs, epsilon=None, learning_rate=0.001, clip_norm=1.0):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            if epsilon is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data = add_laplace_noise(param.grad.data, epsilon)
            
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_acc = 100. * correct / total
        scheduler.step(epoch_acc)
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            
    return model.state_dict()

# 聚合所有客户端的模型参数
def federated_aggregate(global_model, client_states):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_state[key].float() for client_state in client_states], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# KRUM 聚合方法
def krum_aggregate(global_model, client_states, malicious_clients=1):
    """
    KRUM 聚合方法
    malicious_clients: 最多可以容忍的惡意客戶端數量
    """
    global_dict = global_model.state_dict()
    n_clients = len(client_states)
    
    # 確保 malicious_clients 是整數
    malicious_clients = int(malicious_clients)
    
    # 計算每個客戶端之間的距離
    distances = torch.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(n_clients):
            if i != j:
                dist = 0
                for key in global_dict.keys():
                    # 確保參數是浮點數類型
                    param_i = client_states[i][key].float()
                    param_j = client_states[j][key].float()
                    dist += torch.norm(param_i - param_j)**2
                distances[i, j] = dist
    
    # 對每個客戶端，選擇最近的 n_clients - malicious_clients - 1 個客戶端
    selected_clients = []
    for i in range(n_clients):
        # 獲取最近的 n_clients - malicious_clients - 1 個客戶端的索引
        _, indices = torch.sort(distances[i])
        selected = indices[:n_clients - malicious_clients - 1]
        selected_clients.append(selected)
    
    # 聚合選中的客戶端
    for key in global_dict.keys():
        selected_params = []
        for i in range(n_clients):
            # 確保參數是浮點數類型
            selected_params.append(client_states[i][key].float())
        global_dict[key] = torch.stack(selected_params, 0).mean(0)
    
    global_model.load_state_dict(global_dict)
    return global_model

# 测试准确率
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100. * correct / total

# 主函数
if __name__ == "__main__":
    # 自動檢測版本號
    # 獲取所有現有的版本號
    existing_versions = []
    for filename in glob.glob('training_param_v*.txt'):
        match = re.search(r'v(\d+)', filename)
        if match:
            existing_versions.append(int(match.group(1)))
    
    # 確定下一個版本號
    version = max(existing_versions, default=0) + 1
    print(f"\n=== 開始訓練版本 v{version} ===")
    
    # 参数
    num_clients = 5
    num_rounds = 10
    local_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    epsilon_list = list(range(1, 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 選擇聚合方法
    aggregation_method = "fedavg"
    malicious_clients = 1

    # 打印 GPU 信息
    if torch.cuda.is_available():
        print(f"\n=== GPU 信息 ===")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"當前 GPU 記憶體使用: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        print(f"GPU 數量: {torch.cuda.device_count()}")
    else:
        print("\n=== 使用 CPU 進行訓練 ===")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # 划分数据到各个client
    client_loaders = []
    data_per_client = len(train_dataset) // num_clients
    for i in range(num_clients):
        indices = list(range(i * data_per_client, (i + 1) * data_per_client))
        client_subset = torch.utils.data.Subset(train_dataset, indices)
        loader = torch.utils.data.DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 记录每个epsilon下的准确率
    acc_list = []
    
    # 創建保存訓練參數的文件
    param_filename = f'training_param_v{version}.txt'
    with open(param_filename, 'w') as f:
        f.write("=== 訓練參數配置 ===\n")
        f.write(f"版本: v{version}\n")
        f.write(f"聚合方法: {aggregation_method}\n")
        if aggregation_method == "krum":
            f.write(f"可容忍惡意客戶端數量: {malicious_clients}\n")
        f.write(f"客戶端數量: {num_clients}\n")
        f.write(f"訓練輪數: {num_rounds}\n")
        f.write(f"本地訓練輪數: {local_epochs}\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"學習率: {learning_rate}\n")
        f.write(f"優化器: SGD with momentum=0.9\n")
        f.write(f"設備: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("\n=== 訓練結果 ===\n")
    
    for epsilon in epsilon_list:
        print(f"\n=== Privacy Budget ε={epsilon} ===")
        # 初始化全局模型
        global_model = LeNet().to(device)
        round_accuracies = []  # 記錄每輪的準確率
        
        for r in range(num_rounds):
            client_states = []
            for c in range(num_clients):
                local_model = copy.deepcopy(global_model).to(device)
                state = local_train(local_model, client_loaders[c], device, local_epochs, epsilon, learning_rate)
                client_states.append(state)
            
            # 根據選擇的方法進行聚合
            if aggregation_method == "krum":
                global_model = krum_aggregate(global_model, client_states, malicious_clients)
            else:
                global_model = federated_aggregate(global_model, client_states)
                
            acc = test(global_model, test_loader, device)
            round_accuracies.append(acc)
            print(f"Round {r+1}: Test Accuracy: {acc:.2f}%")
        
        final_acc = test(global_model, test_loader, device)
        acc_list.append(final_acc)
        print(f"Final Accuracy for ε={epsilon}: {final_acc:.2f}%")
        
        # 保存每個 epsilon 的訓練結果
        with open(param_filename, 'a') as f:
            f.write(f"\nε = {epsilon}:\n")
            f.write(f"最終準確率: {final_acc:.2f}%\n")
            f.write("每輪準確率: " + ", ".join([f"{acc:.2f}%" for acc in round_accuracies]) + "\n")
    
    # 保存模型參數
    model_filename = f'model_param_v{version}.pth'
    torch.save(global_model.state_dict(), model_filename)
    print(f"\n模型參數已保存為 '{model_filename}'")
    
    # 绘制直方图
    plt.figure(figsize=(8,6))
    plt.bar([str(e) for e in epsilon_list], acc_list)
    plt.xlabel('Privacy Budget ε')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Federated LeNet+MNIST+LDP ({aggregation_method.upper()}): Accuracy vs Privacy Budget')
    plt.ylim(0, 100)
    
    # 保存圖片
    plot_filename = f'eps-acc_v{version}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n圖片已保存為 '{plot_filename}'")
    
    plt.show() 