# 聯邦學習實驗室

這個專案包含了兩個主要的聯邦學習實現：

## MNIST 數據集
- 實現了基本的聯邦學習流程
- 使用 MNIST 手寫數字數據集進行訓練和測試

## MVTec AD 數據集
- 基於聯邦學習的異常檢測
- 使用 MVTec AD 工業異常檢測數據集
- 包含數據分割、模型訓練和評估等功能

## 目錄結構
```
.
├── MNIST/
│   └── FL_flow.py
└── Mvtec_ad/
    ├── data/
    ├── results/
    ├── mvtec_anomaly_detection/
    ├── aggregator.py
    ├── client.py
    ├── config.py
    ├── data_split.py
    ├── evaluate.py
    ├── model.py
    └── train.py
```

## 使用說明
1. 安裝依賴：
```bash
pip install -r requirements.txt
```

2. 運行 MNIST 實驗：
```bash
python MNIST/FL_flow.py
```

3. 運行 MVTec AD 實驗：
```bash
python Mvtec_ad/train.py
```

## 依賴
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- scikit-learn 