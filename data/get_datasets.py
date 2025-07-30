from data.data_utils import MergedDataset
from torchvision import transforms
import torch

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.pets import get_oxford_pets_datasets
from data.inaturalist import get_inaturalist_datasets

from data.cifar import subsample_classes as subsample_dataset_cifar
from data.herbarium_19 import subsample_classes as subsample_dataset_herb
from data.stanford_cars import subsample_classes as subsample_dataset_scars
from data.imagenet import subsample_classes as subsample_dataset_imagenet
from data.cub import subsample_classes as subsample_dataset_cub
from data.fgvc_aircraft import subsample_classes as subsample_dataset_air
from data.pets import subsample_classes as subsample_dataset_pets
from data.inaturalist import subsample_classes as subsample_dataset_inaturalist
from copy import deepcopy
import pickle
import os

from config import osr_split_dir, inaturalist_root

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars,
    'pets': subsample_dataset_pets, 
    'inaturalist': subsample_dataset_inaturalist
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'pets': get_oxford_pets_datasets,
    'inaturalist': get_inaturalist_datasets
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)
                            # subclassname=getattr(args, 'subclassname', ''))

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform
    labelled_dataset = deepcopy(datasets['train_labelled'])
    labelled_dataset.transform = train_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_dataset


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # 定义图像预处理transforms（参照您的示例）
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'tinyimagenet':

        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'chinese_traffic_signs':

        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    elif args.dataset_name == 'pets':
        args.image_size = 224
        args.train_classes = range(18)
        args.unlabeled_classes = range(18, 37)
    
    elif args.dataset_name == 'inaturalist':

        # iNaturalist 各 super-category 中，默认使用 Animalia
        subclass = getattr(args, 'subclassname', 'Animalia')
        args.subclassname = subclass
        # 根据 super-category 映射到类别数
        cls_num_map = {
            'Amphibia': 144,
            'Animalia': 178,
            'Arachnida': 114,
            'Actinopterygii': 369,
            'Aves': 1258,
            'Chromista': 0,   # 若需支持，可自行补
            'Fungi': 321,
            'Insecta': 2031,
            'Mammalia': 234,
            'Mollusca': 262,
            'Plantae': 2917,
            'Protozoa': 0,    # 若需支持，可自行补
            'Reptilia': 284,
        }
        total = cls_num_map.get(subclass, None)
        if total is None or total == 0:
            raise ValueError(f"Unsupported subclassname {subclass}")
        split_point = int(total * 0.8)
        args.train_classes = list(range(split_point))
        args.unlabeled_classes = list(range(split_point, total))
        # args.train_classes = list(range(total))
        # args.unlabeled_classes = []
    #     train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Animalia',
    #                                train_classes=range(39), prop_train_labels=0.5, data_root=inaturalist_root)

    #     unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
    #     unlabelled_train_examples_test.transform = test_transform
    #     args.labeled_nums=39
    #     args.unlabeled_nums=77

    #     # return train_dataset, test_dataset, unlabelled_train_examples_test

    # elif args.data_set == 'Arachnida':

    #     train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Arachnida',
    #                                train_classes=range(28), prop_train_labels=0.5, data_root=inaturalist_root)

    #     unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
    #     unlabelled_train_examples_test.transform = test_transform
    #     args.labeled_nums=28
    #     args.unlabeled_nums=56

    #     # return train_dataset, test_dataset, unlabelled_train_examples_test

    # elif args.data_set == 'Fungi':

    #     train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Fungi',
    #                                train_classes=range(61), prop_train_labels=0.5, data_root=inaturalist_root)

    #     unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
    #     unlabelled_train_examples_test.transform = test_transform
    #     args.labeled_nums=61
    #     args.unlabeled_nums=121

    #     return train_dataset, test_dataset, unlabelled_train_examples_test

    # elif args.data_set == 'Mollusca':

    #     train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Mollusca',
    #                                train_classes=range(47), prop_train_labels=0.5, data_root=inaturalist_root)

    #     unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
    #     unlabelled_train_examples_test.transform = test_transform
    #     args.labeled_nums=47
    #     args.unlabeled_nums=93

    #     return train_dataset, test_dataset, unlabelled_train_examples_test

    else:

        raise NotImplementedError

    return args