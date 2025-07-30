import pickle
import os

# 创建目录
ssb_splits_dir = "/seu_nvme/home/fenglei/213240634/SMILE_20250604002666/SMILE/data/ssb_splits"
os.makedirs(ssb_splits_dir, exist_ok=True)

# Oxford Pet数据集有37个类别，创建合理的分割
# 前19个类别作为已知类别，后18个类别作为未知类别
pets_splits = {
    'known_classes': list(range(19)),  # 类别 0-18
    'unknown_classes': {
        'Hard': list(range(19, 25)),    # 类别 19-24 (6个)
        'Medium': list(range(25, 31)),  # 类别 25-30 (6个)  
        'Easy': list(range(31, 37))     # 类别 31-36 (6个)
    }
}

# 保存pkl文件
pets_pkl_path = os.path.join(ssb_splits_dir, 'pets_osr_splits.pkl')
with open(pets_pkl_path, 'wb') as f:
    pickle.dump(pets_splits, f)

print(f"Created {pets_pkl_path}")
print(f"Known classes: {pets_splits['known_classes']}")
print(f"Unknown classes: {sum(pets_splits['unknown_classes'].values(), [])}")

# import torchvision
# dataset = torchvision.datasets.OxfordIIITPet(root='./data/dataset', download=True)