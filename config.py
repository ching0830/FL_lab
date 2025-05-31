import torch

class Config:
    # 資料集設定
    DATASET_PATH = "data/mvtec_anomaly_detection"
    ALL_CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    
    # 客戶端類別分配 - 每個類別分成五等分
    CLIENT_CATEGORIES = {
        0: ["bottle_0", "cable_0", "capsule_0", "carpet_0", "grid_0",
            "hazelnut_0", "leather_0", "metal_nut_0", "pill_0", "screw_0",
            "tile_0", "toothbrush_0", "transistor_0", "wood_0", "zipper_0"],
        1: ["bottle_1", "cable_1", "capsule_1", "carpet_1", "grid_1",
            "hazelnut_1", "leather_1", "metal_nut_1", "pill_1", "screw_1",
            "tile_1", "toothbrush_1", "transistor_1", "wood_1", "zipper_1"],
        2: ["bottle_2", "cable_2", "capsule_2", "carpet_2", "grid_2",
            "hazelnut_2", "leather_2", "metal_nut_2", "pill_2", "screw_2",
            "tile_2", "toothbrush_2", "transistor_2", "wood_2", "zipper_2"],
        3: ["bottle_3", "cable_3", "capsule_3", "carpet_3", "grid_3",
            "hazelnut_3", "leather_3", "metal_nut_3", "pill_3", "screw_3",
            "tile_3", "toothbrush_3", "transistor_3", "wood_3", "zipper_3"],
        4: ["bottle_4", "cable_4", "capsule_4", "carpet_4", "grid_4",
            "hazelnut_4", "leather_4", "metal_nut_4", "pill_4", "screw_4",
            "tile_4", "toothbrush_4", "transistor_4", "wood_4", "zipper_4"]
    }
    
    # 模型設定
    MODEL_NAME = "vgg16"
    NUM_CLASSES = 2  # 正常 vs 異常
    INPUT_SIZE = (224, 224)
    
    # 聯邦學習設定
    NUM_CLIENTS = 5
    NUM_ROUNDS = 5
    
    EPOCHS_PER_ROUND = 3
    BATCH_SIZE = 16
    
    # 訓練參數
    LEARNING_RATE = 0.0001
    CLIP_NORM = 0.5
    
    # 差分隱私設定
    EPSILON_VALUES = {
        "baseline": float('inf'),  # 無差分隱私
        # "weak": 10.0,             # 弱隱私保護
        # "medium": 5.0,            # 中等隱私保護
        # "strong": 1.0             # 強隱私保護
    }
    
    # 設備設定
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 結果保存路徑
    RESULTS_DIR = "results"
    LOG_DIR = "logs" 