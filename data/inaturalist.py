import os
from copy import deepcopy
import numpy as np
from typing import Any, Tuple
from PIL import Image
from torchvision.datasets import INaturalist
from torchvision.datasets.vision import VisionDataset
import json
from config import inaturalist_root

class INaturalist_SUB(VisionDataset):
    def __init__(self, root, version='2018', subclassname='', transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        # Store version for path construction
        self.version = version
        print(root)
        # Load JSON annotation file
        json_file = os.path.join(root, version, f'train{version}.json')
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Training annotation file not found: {json_file}")
            
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Build category mappings
        self._build_category_mapping()
        
        # Validate supercategory name
        valid_super_categories = [
            'Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida',
            'Aves', 'Chromista', 'Fungi', 'Insecta', 
            'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia'
        ]
        if subclassname and subclassname not in valid_super_categories:
            raise ValueError(f"Invalid subclassname '{subclassname}'. Please provide one of {valid_super_categories}.")
        
        # Filter dataset and create index
        self._filter_dataset(subclassname)
        self.uq_idxs = np.arange(len(self.index))
        
        print(f"Loaded {subclassname}: {self.category_num} categories, {len(self.index)} samples")

    # def _build_category_mapping(self):
    #     """Build category_id -> supercategory mapping from JSON data"""
    #     self.category_to_super = {}
    #     for cat in self.data['categories']:
    #         cat_id = cat['id']
    #         cat_name = cat['name']
    #         # Extract supercategory from the category name (format: "Supercategory/Species name")
    #         if '/' in cat_name:
    #             super_cat = cat_name.split('/')[0]
    #         else:
    #             # Fallback: use the category name itself
    #             super_cat = cat_name
    #         self.category_to_super[cat_id] = super_cat

    def _build_category_mapping(self):
        """从 JSON 标注构建 category_id -> super_category 映射"""
        self.category_to_super = {}
        supercategory_counts = {}
        
        for cat in self.data['categories']:
            cid = cat['id']
            # 首先尝试使用explicit的supercategory字段
            if 'supercategory' in cat and cat['supercategory']:
                super_cat = cat['supercategory']
            else:
                # 后备方案：从name字段解析
                name = cat['name']
                super_cat = name.split('/')[0] if '/' in name else name
            
            self.category_to_super[cid] = super_cat
            supercategory_counts[super_cat] = supercategory_counts.get(super_cat, 0) + 1
        
        print(f"Found supercategories: {supercategory_counts}")


    def _filter_dataset(self, subclassname):
        """Filter samples by supercategory and rebuild index"""
        filtered_index = []
        self.index_map = {}
        self.category_num = 0
        
        # Create image_id -> filename mapping
        image_map = {img['id']: img['file_name'] for img in self.data['images']}
        
        # Filter annotations by supercategory
        for ann in self.data['annotations']:
            cat_id = ann['category_id']
            image_id = ann['image_id']
            
            if cat_id in self.category_to_super:
                super_cat = self.category_to_super[cat_id]
                if super_cat == subclassname:
                    # Map original category_id to new sequential index
                    if cat_id not in self.index_map:
                        self.index_map[cat_id] = self.category_num
                        self.category_num += 1
                    
                    # Add sample to filtered index
                    if image_id in image_map:
                        filename = image_map[image_id]
                        filtered_index.append((self.index_map[cat_id], filename))
        
        self.index = filtered_index
        self.reverse_index_map = {v: k for k, v in self.index_map.items()}

    def __getitem__(self, index: int) -> Tuple[Any, Any, int, int]:
        cat_id, fname = self.index[index]
        
        # Construct image path: root/version/train_val2018/filename
        img_path = os.path.join(self.root, self.version, fname)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        target = cat_id
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        
        uq = int(self.uq_idxs[index])
        return img, target, uq, uq

    def __len__(self) -> int:
        return len(self.index)



def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices
    
def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed
    if len(idxs) == 0:
        print("[Warn] subsample_dataset received empty idxs; "
              "returning original dataset.")
        return dataset
    else:
        dataset.index = [dataset.index[i] for i in idxs]
        dataset.uq_idxs = dataset.uq_idxs[idxs]
        return dataset

    # if len(idxs) > 0:
    #     new_index = [dataset.index[i] for i in idxs]
    #     dataset.index = new_index
    #     dataset.uq_idxs = dataset.uq_idxs[idxs]

    #     return dataset

    # else:

    #     return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    # Extract all category indices from dataset.index
    class_indices = [index for index, _ in dataset.index]

    # Filter out the indices of category indices that are included in include_classes
    cls_idxs = [idx for idx, class_index in enumerate(class_indices) if class_index in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    # ---- 只改下面这 4 行 ----
    if len(cls_idxs) == 0:          # 一个都没匹配到
        print(f"[Warn] none of {include_classes} appear in dataset "
              f"(found {sorted(set(class_indices))}). "
              "Return the original dataset instead.")
        return dataset              # 直接把原数据集返回
    # ---- 修改到此结束 ----

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    targets = np.array([x for (x, _) in train_dataset.index])
    train_classes = np.unique(targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_inaturalist_datasets(train_transform, 
                          test_transform,
                          subclassname='',
                          train_classes=(0, 1, 8, 9),
                          prop_train_labels=0.8, 
                          split_train_val=False, 
                          seed=0,
                          data_root=inaturalist_root):

    np.random.seed(seed)

    # Init entire training set
    print('*'*80)
    print(data_root)
    whole_training_set = INaturalist_SUB(root=inaturalist_root, subclassname=subclassname, transform=train_transform)
    
    if train_classes is None:
        available_classes = sorted(set([cls for cls, _ in whole_training_set.index]))
        train_classes = range(len(available_classes))
        print(f"Auto-detected {len(available_classes)} classes for {subclassname}")
    
    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    whole_test_dataset = INaturalist_SUB(root=data_root, subclassname=subclassname, transform=test_transform)
    test_dataset = subsample_classes(deepcopy(whole_test_dataset), include_classes=train_classes)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    return {
        'train_labelled': train_dataset_labelled,
        'test': test_dataset,
        'train_unlabelled': train_dataset_unlabelled,
        'val_labelled': val_dataset_labelled
    }

if __name__ == '__main__':

    print("------------------Amphibia-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Amphibia', split_train_val=False,
                         train_classes=range(58), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Animalia-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Animalia', split_train_val=False,
                         train_classes=range(178), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print('------------------Arachnida-----------------')
    x = get_inaturalist_datasets(None, None, subclassname='Arachnida', split_train_val=False,
                         train_classes=range(114), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Fungi-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Fungi', split_train_val=False,
                         train_classes=range(321), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Mammalia-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Mammalia', split_train_val=False,
                         train_classes=range(234), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Mollusca-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Mollusca', split_train_val=False,
                         train_classes=range(262), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

    print("------------------Reptilia-----------------")
    x = get_inaturalist_datasets(None, None, subclassname='Reptilia', split_train_val=False,
                         train_classes=range(284), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    targets = np.array([x for (x, _) in x["train_labelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Labelled Classes: {len(train_classes)}')
    targets = np.array([x for (x, _) in x["train_unlabelled"].index])
    train_classes = np.unique(targets)
    print(f'Num Unabelled Classes: {len(train_classes)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')