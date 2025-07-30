import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances

# 修改为项目内路径
car_root = "/seu_nvme/home/fenglei/213240634/SMILE_20250604002666/SMILE/data/dataset/stanford_cars/cars_{}"
meta_default_path = "/seu_nvme/home/fenglei/213240634/SMILE_20250604002666/SMILE/data/dataset/stanford_cars/devkit/cars_{}.mat"
class CarsDataset(Dataset):
    """
    Stanford Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=car_root, transform=None, metas=meta_default_path):

        data_dir = car_root.format('train') if train else car_root.format('test')
        metas = meta_default_path.format('train_annos') if train else meta_default_path.format('test_annos_withlabels')
        print(data_dir)
        print(metas)
        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train
        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location!")
        
        # 检查文件是否存在
        if not os.path.exists(metas):
            raise FileNotFoundError(f"Metadata file not found: {metas}\n"
                                  f"请先下载Stanford Cars数据集！\n"
                                  f"下载地址: https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}\n"
                                  f"请确保数据集已正确解压到 data/dataset/stanford_cars/")
            
        labels_meta = mat_io.loadmat(metas)

        print(f"Loading {'train' if train else 'test'} data from {data_dir}")
        
        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit and idx >= limit:
                break

            # 构建图像路径
            img_path = os.path.join(data_dir, img_[5][0])
            
            if os.path.exists(img_path):
                self.data.append(img_path)
                self.target.append(img_[4][0][0])  # 类别标签从1开始
            else:
                print(f"Warning: Image not found: {img_path}")

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None
        
        print(f"Successfully loaded {len(self.data)} images for {'train' if train else 'test'} set")
        print(f"Classes range: {min(self.target)} to {max(self.target)}")

    def __getitem__(self, idx):
        image = self.loader(self.data[idx])
        target = self.target[idx] - 1  # 转换为0-195

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]
        return image, target, idx

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset, idxs):
    """子采样数据集"""
    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def subsample_classes(dataset, include_classes=range(160)):
    """子采样指定类别"""
    include_classes_cars = np.array(include_classes) + 1  # 转换为1-196
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):
    """获取训练和验证集索引"""
    train_classes = np.unique(train_dataset.target)

    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.target == cls)[0]
        
        if len(cls_idxs) > 1:
            val_size = max(1, int(val_split * len(cls_idxs)))
            v_ = np.random.choice(cls_idxs, replace=False, size=val_size)
            t_ = [x for x in cls_idxs if x not in v_]
        else:
            v_ = []
            t_ = cls_idxs.tolist()

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_scars_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0):
    """获取Stanford Cars数据集"""
    np.random.seed(seed)

    try:
        # 初始化完整训练集
        whole_training_set = CarsDataset(data_dir=car_root, transform=train_transform, 
                                       metas=meta_default_path, train=True)

        # 获取标记的训练集
        train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
        subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

        # 分割训练和验证集
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

        # 获取未标记数据
        unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
        train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

        # 获取测试集
        test_dataset = CarsDataset(data_dir=car_root, transform=test_transform, 
                                 metas=meta_default_path, train=False)

        # 选择是否分割训练集
        train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

        all_datasets = {
            'train_labelled': train_dataset_labelled,
            'train_unlabelled': train_dataset_unlabelled,
            'val': val_dataset_labelled,
            'test': test_dataset,
        }

        return all_datasets
    
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请按以下步骤下载Stanford Cars数据集:")
        print("1. 访问: https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset")
        print("2. 下载并解压到: data/dataset/stanford_cars/")
        print("3. 访问: https://www.kaggle.com/datasets/meaninglesslives/cars-devkit")
        print("4. 下载devkit并解压到: data/dataset/stanford_cars/devkit/")
        print("\n正确的目录结构:")
        print("data/dataset/stanford_cars/")
        print("├── cars_train/")
        print("├── cars_test/")
        print("└── devkit/")
        print("    ├── cars_train_annos.mat")
        print("    ├── cars_test_annos_withlabels.mat")
        print("    └── cars_meta.mat")
        raise


if __name__ == '__main__':
    try:
        x = get_scars_datasets(None, None, train_classes=range(98), prop_train_labels=0.5, split_train_val=False)

        print('\n=== 数据集统计 ===')
        for k, v in x.items():
            if v is not None:
                print(f'{k}: {len(v)}')

        print('\n=== 检查重叠 ===')
        overlap = set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs))
        print(f'Labelled and unlabelled overlap: {len(overlap)}')
        
        print(f'\n=== 总统计 ===')
        print(f'Total train instances: {len(set(x["train_labelled"].uq_idxs)) + len(set(x["train_unlabelled"].uq_idxs))}')
        print(f'Labelled classes: {len(set(x["train_labelled"].target))}')
        print(f'Unlabelled classes: {len(set(x["train_unlabelled"].target))}')
        print(f'Labelled set size: {len(x["train_labelled"])}')
        print(f'Unlabelled set size: {len(x["train_unlabelled"])}')
        
    except Exception as e:
        print(f"测试失败: {e}")
