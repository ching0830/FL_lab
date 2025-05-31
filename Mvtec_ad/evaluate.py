import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from config import Config

def evaluate_model(model, test_loader):
    """評估模型性能，計算多個指標"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 異常類別的概率
    
    # 計算各項指標
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary'
    )
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5  # 如果只有一個類別，設置為0.5
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

def evaluate_all_clients(clients):
    """評估所有客戶端的模型性能"""
    all_metrics = []
    
    for client in clients:
        metrics = evaluate_model(client.model, client.test_loader)
        metrics['category'] = client.category
        all_metrics.append(metrics)
    
    # 計算平均指標
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'auc': np.mean([m['auc'] for m in all_metrics])
    }
    
    return all_metrics, avg_metrics 