import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from PIL import Image
import random
from config import Config

class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True, split_idx=0, num_splits=5):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.split_idx = split_idx
        self.num_splits = num_splits

        self.images = []
        self.labels = []

        if is_train:
            # 僅載入 good 圖片
            self.image_dir = os.path.join(root_dir, 'train', 'good')
            all_images = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            split_size = len(all_images) // num_splits
            start_idx = split_idx * split_size
            end_idx = start_idx + split_size if split_idx < num_splits - 1 else len(all_images)
            self.images = all_images[start_idx:end_idx]
            self.labels = [0] * len(self.images)
        else:
            # 載入 test 資料，包括 good 和 defect
            good_dir = os.path.join(root_dir, 'test', 'good')
            if os.path.exists(good_dir):
                all_good_images = [f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                all_good_images.sort()
                split_size = len(all_good_images) // num_splits
                start_idx = split_idx * split_size
                end_idx = start_idx + split_size if split_idx < num_splits - 1 else len(all_good_images)
                good_images = all_good_images[start_idx:end_idx]
                self.images.extend([os.path.join('good', f) for f in good_images])
                self.labels.extend([0] * len(good_images))

            defect_dir = os.path.join(root_dir, 'test')
            defect_types = [d for d in os.listdir(defect_dir) if d != 'good' and os.path.isdir(os.path.join(defect_dir, d))]
            for defect_type in defect_types:
                defect_path = os.path.join(defect_dir, defect_type)
                all_defect_images = [f for f in os.listdir(defect_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                all_defect_images.sort()
                split_size = len(all_defect_images) // num_splits
                start_idx = split_idx * split_size
                end_idx = start_idx + split_size if split_idx < num_splits - 1 else len(all_defect_images)
                defect_images = all_defect_images[start_idx:end_idx]
                self.images.extend([os.path.join(defect_type, f) for f in defect_images])
                self.labels.extend([1] * len(defect_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.is_train:
            img_path = os.path.join(self.image_dir, self.images[idx])
        else:
            img_path = os.path.join(self.root_dir, 'test', self.images[idx])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class MVTecDataSplitter:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_all_clients_data(self):
        client_train_datasets = [[] for _ in range(Config.NUM_CLIENTS)]
        client_test_datasets = [[] for _ in range(Config.NUM_CLIENTS)]

        for category in Config.ALL_CATEGORIES:
            category_path = os.path.join(Config.DATASET_PATH, category)
            for cid in range(Config.NUM_CLIENTS):
                train_dataset = MVTecDataset(category_path, transform=self.transform, is_train=True, split_idx=cid)
                test_dataset = MVTecDataset(category_path, transform=self.transform, is_train=False, split_idx=cid)
                client_train_datasets[cid].append(train_dataset)
                client_test_datasets[cid].append(test_dataset)

        clients_data = []
        for cid in range(Config.NUM_CLIENTS):
            train_loader = DataLoader(
                ConcatDataset(client_train_datasets[cid]),
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            test_loader = DataLoader(
                ConcatDataset(client_test_datasets[cid]),
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            clients_data.append({
                'train': train_loader,
                'test': test_loader,
                'categories': Config.ALL_CATEGORIES
            })

        return clients_data

    def get_category_data(self, category):
        base_category = category.split('_')[0]
        split_idx = int(category.split('_')[1])
        category_path = os.path.join(Config.DATASET_PATH, base_category)

        train_dataset = MVTecDataset(
            root_dir=category_path,
            transform=self.transform,
            is_train=True,
            split_idx=split_idx,
            num_splits=5
        )

        test_dataset = MVTecDataset(
            root_dir=category_path,
            transform=self.transform,
            is_train=False,
            split_idx=split_idx,
            num_splits=5
        )

        return train_dataset, test_dataset
