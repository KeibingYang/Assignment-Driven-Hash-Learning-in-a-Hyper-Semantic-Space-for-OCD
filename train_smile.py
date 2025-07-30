import argparse
import os
import random
import time
from datetime import datetime
import sys
import logging
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm
import random
from torch.nn import functional as F
import torch.nn as nn

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path

from PALM import PALM
from fantasy import rot_color_perm24
from PHE import HASHHead, HashCenterManager
import itertools

class HashCenterManager:
    """简化版Hash Center管理类"""
    
    def __init__(self, num_classes, hash_dim, device, momentum=0.9):
        self.num_classes = num_classes
        self.hash_dim = hash_dim
        self.device = device
        self.momentum = momentum
        
        # 初始化hash centers
        self.hash_centers = torch.randn(num_classes, hash_dim, device=device)
        self.hash_centers = torch.sign(self.hash_centers)  # 二值化初始化
        
    def update_centers(self, hash_features, labels):
        """EMA更新hash centers"""
        with torch.no_grad():
            for cls in torch.unique(labels):
                if cls >= self.num_classes:
                    continue
                mask = (labels == cls)
                if mask.sum() > 0:
                    class_mean = torch.mean(hash_features[mask], dim=0)
                    self.hash_centers[cls] = (self.momentum * self.hash_centers[cls] + 
                                            (1 - self.momentum) * class_mean)

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

def get_prototypes(args, model, projection_head, labelled_dataset, device):
    rotation_transform, num_augmented = rot_color_perm24()
    args.num_augmented = num_augmented

    # 添加这两行：临时设置为eval模式避免BatchNorm错误
    model.eval()
    projection_head.eval()

    class_original_images = {}
    class_augmented_images = {}
    clses = set()
    
    # # 根据数据集名称确定解包策略
    # dataset_unpack_strategy = {
    #     'food101': 4,      # Food101返回4个值
    #     'cifar10': 3,      # CIFAR可能返回3个值
    #     'cifar100': 3,
    #     'scars': 3,
    #     'cub': 3,
    #     'aircraft': 3
    #     # 可根据需要添加其他数据集
    # }

    # expected_values = dataset_unpack_strategy.get(args.dataset_name, 3)  # 默认3个值
    # print(torch.cuda.memory_summary(0))

    for batch_data in labelled_dataset:
        # if expected_values == 4:
        #     img, cls, uq_idx1, uq_idx2 = batch_data
        # elif expected_values == 3:
        #     img, cls, uq_idx = batch_data
        # else:
        #     # 通用处理
        #     img, cls = batch_data[0], batch_data[1]

        img, cls = batch_data[0], batch_data[1]

        if isinstance(img, list):
            img = img[0]
        if img.dim() == 3:
            img = img.unsqueeze(0)  # [1, C, H, W]
        # 记录所有合法的cls
        if cls in args.train_classes:
            clses.add(cls)
            # 原始图片
            if cls not in class_original_images:
                class_original_images[cls] = []
            class_original_images[cls].append(img)  # [1, C, H, W]

            # 增强图片
            augmented = rotation_transform(img)  # [N, C, H, W]
            # print("-" * 80)
            # print("augmented shape")
            # print(augmented.shape)
            # print("-" * 80)
            if cls not in class_augmented_images:
                class_augmented_images[cls] = []
            class_augmented_images[cls].append(augmented)  # [N, C, H, W]
    clses = sorted(list(clses))

    prototypes = {}
    for cls in class_original_images:
        # 原始图片特征
        all_original = torch.cat(class_original_images[cls], dim=0).to(device)  # [num, C, H, W]
        with torch.no_grad():
            features = model(all_original)
            features, _, _ = projection_head(features)
            features = torch.nn.functional.normalize(features, dim=-1)
        original_proto = torch.mean(features, dim=0)  # [feat_dim]

        # 增强图片特征
        # 拼接所有增强图片，保证shape为 [num*N, C, H, W]
        N, C, H, W = augmented.shape
        num = len(class_augmented_images[cls])
        all_augmented = torch.cat(class_augmented_images[cls], dim=0)  # [num, N, C, H, W]
        all_augmented = all_augmented.view(num * N, C, H, W).to(device)  # [num*N, C, H, W]
        with torch.no_grad():
            aug_features = model(all_augmented)
            aug_features, _, _ = projection_head(aug_features)
            aug_features = torch.nn.functional.normalize(aug_features, dim=-1)
        augmented_proto = torch.mean(aug_features, dim=0)  # [feat_dim]

        prototypes[cls] = {
            'original': original_proto,
            'augmented': augmented_proto
        }
    # print(torch.cuda.memory_summary(0))

    # 打印检查
    # print("-" * 80)
    # print("known cls", clses)
    # print("Number of prototype pairs:", len(prototypes))
    first_cls = list(prototypes.keys())[0]
    # print(f"Class {first_cls}:")
    # print("- Original prototype shape:", prototypes[first_cls]['original'].shape)
    # print("- Augmented prototype shape:", prototypes[first_cls]['augmented'].shape)
    # print("-" * 80)

    return prototypes,clses

def generate_latent_ood_prototypes(prototypes, clses, device, num_ood=10):
    """
    生成潜在的OOD原型，通过在不同已知类原型间线性插值。
    
    Args:
        prototypes: dict, key为类别，value为该类别的原型tensor，形状为[feat_dim]
        clses: list，已知类别列表
        device: 设备
        num_ood: 生成的OOD原型数量
    
    Returns:
        ood_prototypes: tensor, 形状为[num_ood, feat_dim]
    """
    ood_protos = []
    # 取第一个类别的original prototype的维度
    feat_dim = next(iter(prototypes.values()))['original'].shape[0]
    for _ in range(num_ood):
        i, j = random.sample(clses, 2)
        proto_i = prototypes[i]['original'].to(device)
        proto_j = prototypes[j]['original'].to(device)
        lam = random.uniform(0, 1)
        ood_proto = lam * proto_i + (1 - lam) * proto_j
        ood_proto = F.normalize(ood_proto.unsqueeze(0), dim=-1)
        ood_protos.append(ood_proto)

    ood_prototypes = torch.cat(ood_protos, dim=0)  # [num_ood, feat_dim]
    return ood_prototypes

def get_id_prototype_matrix(prototypes, clses, device, use_augmented=False):
    """
    将所有已知类原型组织成矩阵形式
    """
    proto_list = []
    for cls in clses:
        if use_augmented and 'augmented' in prototypes[cls]:
            proto = prototypes[cls]['augmented'].to(device)
        else:
            proto = prototypes[cls]['original'].to(device)
        proto_list.append(proto.unsqueeze(0))
    
    return torch.cat(proto_list, dim=0)  # [num_classes, feat_dim]

def prototype_margin_ood_loss(features, id_prototypes, ood_prototypes, margin=0.5, similarity='cosine'):
    """
    多原型大间隔推离损失：确保所有样本远离OOD区域
    """
    if similarity == 'cosine':
        features = F.normalize(features, dim=-1)
        id_prototypes = F.normalize(id_prototypes, dim=-1)
        ood_prototypes = F.normalize(ood_prototypes, dim=-1)
        
        # 计算与已知类原型的相似度
        sim_id = torch.matmul(features, id_prototypes.T)  # [B, K]
        # 计算与OOD原型的相似度
        sim_ood = torch.matmul(features, ood_prototypes.T)  # [B, M]
    else:
        raise NotImplementedError("Only cosine similarity supported")
    
    # 每个样本与最相似的已知类原型的相似度
    max_sim_id, _ = torch.max(sim_id, dim=1)  # [B]
    # 每个样本与最不相似的OOD原型的相似度
    min_sim_ood, _ = torch.min(sim_ood, dim=1)  # [B]
    
    # 大间隔损失：要求与OOD的距离 > 与ID的距离 + margin
    loss = F.relu(margin + max_sim_id - min_sim_ood)
    return loss.mean()

def prototype_orthogonality_loss(id_prototypes, ood_prototypes):
    """
    原型正交约束损失：强制OOD原型与已知类原型正交
    """
    id_prototypes = F.normalize(id_prototypes, dim=-1)
    ood_prototypes = F.normalize(ood_prototypes, dim=-1)
    
    # 计算点积的绝对值
    dot_prod = torch.matmul(id_prototypes, ood_prototypes.T).abs()
    return dot_prod.mean()

def advanced_ood_loss(features, id_prototypes, ood_prototypes, margin=0.5):
    """
    高级OOD损失：结合多种约束
    """
    # 基础推离损失
    basic_loss = prototype_margin_ood_loss(features, id_prototypes, ood_prototypes, margin)
    
    # 正交约束损失
    orth_loss = prototype_orthogonality_loss(id_prototypes, ood_prototypes)
    
    # 分布分离损失：最大化样本与OOD原型的平均距离
    features_norm = F.normalize(features, dim=-1)
    ood_prototypes_norm = F.normalize(ood_prototypes, dim=-1)
    sim_matrix = torch.matmul(features_norm, ood_prototypes_norm.T)
    dist_loss = -torch.log(1 - sim_matrix.clamp(max=0.99)).mean()
    
    return basic_loss, orth_loss, dist_loss

def latent_ood_loss(features, ood_prototypes, similarity='cosine'):
    """
    计算潜在OOD原型辅助损失，鼓励已知类样本远离OOD原型。
    
    Args:
        features: tensor, 已知类样本特征，形状为[batch_size, feat_dim]
        ood_prototypes: tensor, 潜在OOD原型，形状为[num_ood, feat_dim]
        similarity: str， 相似度度量，支持'cosine'
    
    Returns:
        loss: tensor, 标量
    """
    if similarity == 'cosine':
        # 计算features与所有OOD原型的余弦相似度，形状 [batch_size, num_ood]
        sim_matrix = torch.matmul(features, ood_prototypes.T)
    else:
        raise NotImplementedError("Only cosine similarity is supported for now.")
    
    # 对每个样本，取与所有OOD原型的最大相似度
    max_sim, _ = torch.max(sim_matrix, dim=1)  # [batch_size]
    # 计算损失： -log(1 - max_sim)
    eps = 1e-6  # 防止log(0)
    loss = -torch.log(1 - max_sim.clamp(max=1 - eps))
    return loss.mean()

def hash_compact_separation_vectorized(hash_features, labels, alpha=1.0, beta=1.0, margin=2.0):
    """
    向量化实现，计算效率更高
    """
    device = hash_features.device
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    
    if num_classes < 2:
        return torch.tensor(0.0, device=device)
    
    # 创建类别mask矩阵
    batch_size = hash_features.size(0)
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
    
    # 计算所有样本对之间的距离矩阵
    hash_expanded_1 = hash_features.unsqueeze(1)  # [B, 1, D]
    hash_expanded_2 = hash_features.unsqueeze(0)  # [1, B, D]
    distance_matrix = torch.sum((hash_expanded_1 - hash_expanded_2) ** 2, dim=2)  # [B, B]
    
    # 类内紧凑损失：同类样本对的平均距离
    intra_mask = label_matrix & ~torch.eye(batch_size, dtype=torch.bool, device=device)
    if intra_mask.sum() > 0:
        intra_loss = (distance_matrix * intra_mask.float()).sum() / intra_mask.sum()
    else:
        intra_loss = torch.tensor(0.0, device=device)
    
    # 类间分离损失：计算类中心然后计算中心间距离
    class_centers = []
    for label in unique_labels:
        mask = (labels == label)
        center = torch.mean(hash_features[mask], dim=0)
        class_centers.append(center)
    
    class_centers = torch.stack(class_centers)  # [C, D]
    center_distances = torch.cdist(class_centers, class_centers, p=2)  # [C, C]
    
    # 去除对角线，计算类间分离损失
    center_mask = ~torch.eye(num_classes, dtype=torch.bool, device=device)
    inter_distances = center_distances[center_mask]
    inter_loss = F.relu(margin - inter_distances).mean()
    
    # 联合损失
    total_loss = alpha * intra_loss + beta * inter_loss
    
    return total_loss, intra_loss, inter_loss

def get_hash_center_loss(hash_features, labels, center_manager, 
                           alpha=1.0, beta=0.5, gamma=0.1):
    """
    统一的Hash Center损失函数
    
    Args:
        hash_features: [batch_size, hash_dim] 连续hash特征
        labels: [batch_size] 类别标签
        center_manager: HashCenterManager实例
        alpha: 类内紧凑权重
        beta: 类间分离权重  
        gamma: 二值化约束权重
    
    Returns:
        total_loss: 统一的hash center损失
    """
    centers = center_manager.hash_centers
    device = hash_features.device

    # 二值化约束损失 - 鼓励hash centers和features接近±1[1]
    binary_center_loss = torch.mean(torch.abs(torch.abs(centers) - 1.0))
    binary_feature_loss = torch.mean(torch.abs(torch.abs(hash_features) - 1.0))
    binary_loss = binary_center_loss + binary_feature_loss
    
    # 统一损失[1]
    total_loss = gamma * binary_loss
    
    return total_loss

def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args):
    logger.info("Start training...")
    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0

    prototypes, class_known = get_prototypes(args, model, projection_head, train_dataset.labelled_dataset, device)
    # 初始化PALM
    palm = PALM(
            args,
            num_classes=args.num_labeled_classes,
            prototypes=prototypes,  # 可根据需要暴露为参数
            proto_m=args.proto_m,
            temp=0.1,       # 可根据需要暴露为参数
            lambda_pcon=args.lambda_pcon,
            k=args.k,
            feat_dim=args.feat_dim,
            epsilon=0.05,    # 可根据需要暴露为参数
            class_known= class_known
            )
    palm.to(device)

    # 初始化Hash Center Manager
    hash_center_manager = HashCenterManager(
        num_classes=args.num_labeled_classes,
        hash_dim=args.code_dim,
        device=device,
        momentum=0.9
    )

    # 损失权重
    palm_loss_weight = 0.1  
    ood_margin_weight = 0.2
    hash_center_weight = 0.2  

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        projection_head.train()
        model.train()
        id_proto_matrix = get_id_prototype_matrix(prototypes, class_known, device)
        for batch_idx, batch in enumerate(tqdm(train_loader)):

            if len(batch) == 4:
                images, class_labels, uq_idxs, _ = batch  # 处理返回4个值的情况（如iNaturalist）
            elif len(batch) == 3:
                images, class_labels, uq_idxs = batch     # 处理返回3个值的情况

            class_labels = class_labels.to(device)
            images = torch.cat(images, dim=0).to(device)
            features = model(images)

            features, hash_features, variance_features = projection_head(features)
            features = torch.nn.functional.normalize(features, dim=-1)

            f1, f2 = [f for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # print(sup_con_feats.shape)
            sup_con_labels = class_labels
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            
            reg_loss = (1 - torch.abs(hash_features)).mean()
            
            # === PALM loss ===
            # PALM要求输入 [bsz, n_views, feat_dim]，sup_con_feats正好是这个shape
            palm_loss, palm_loss_dict = palm(sup_con_feats, sup_con_labels)
            
            # === ood loss===
            # 生成潜在OOD原型
            ood_prototypes = generate_latent_ood_prototypes(prototypes, class_known, device, num_ood=10)

            # 4. 高级OOD损失
            ood_margin_loss, ood_orth_loss, ood_dist_loss = advanced_ood_loss(
                features, id_proto_matrix, ood_prototypes, margin=0.5
            )

            # === Hash紧凑分离约束 ===
            h1, h2 = hash_features.chunk(2)
            combined_hash = torch.cat([h1, h2], dim=0)
            combined_labels = torch.cat([sup_con_labels, sup_con_labels], dim=0)
            
            # 更新hash centers
            hash_center_manager.update_centers(combined_hash, combined_labels)
            
            # === 统一的Hash Center损失 ===
            hash_center_loss = get_hash_center_loss(
                combined_hash, combined_labels, hash_center_manager,
                alpha=1.0, beta=0.5, gamma=0.1
            )

            # === 总loss ===
            loss = sup_con_loss * 1 + reg_loss * 3 
            loss = loss + palm_loss_weight * palm_loss 
            loss = loss + ood_margin_weight * ood_margin_loss
            # loss = loss + hash_center_loss * hash_center_weight

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 日志输出
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(labelled_train_loader)}] "
                    f"SupConLoss: {sup_con_loss.item():.4f} | RegLoss: {reg_loss.item():.4f} | "
                    f"PALMLoss: {palm_loss.item():.4f}  | OODOrth: {ood_orth_loss.item():.4f}"
                    f" | OODMargin: {ood_margin_loss.item():.4f}"
                    f" | hash_loss: {hash_center_loss.item():.4f} | "
                    f"TotalLoss: {loss.item():.4f}"
                )


        logger.info('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))


        with torch.no_grad():
            eval_results = test_on_the_fly(model, projection_head, unlabelled_train_loader,
                                         epoch=epoch, save_name='Train ACC Unlabelled', hamming_threshold=1,
                                         args=args)

        # 记录各个评估函数的结果
        for eval_func in args.eval_funcs:
            if eval_func in eval_results:
                result = eval_results[eval_func]
                logger.info(f'Epoch {epoch}, Train ACC Unlabelled_{eval_func}: '
                           f'All {result["all"]:.4f} | Old {result["old"]:.4f} | New {result["new"]:.4f}')

        # 使用v1的结果作为主要指标（向后兼容）
        main_result = eval_results.get('v1', list(eval_results.values())[0])
        all_acc = main_result['all']
        old_acc = main_result['old'] 
        new_acc = main_result['new']

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)
        args.writer.add_scalar('Loss/PALM', palm_loss.item(), epoch)
        
        # 记录所有评估函数的结果到TensorBoard
        for eval_func in args.eval_funcs:
            if eval_func in eval_results:
                result = eval_results[eval_func]
                args.writer.add_scalar(f'Test_Acc_{eval_func}/All', result['all'], epoch)
                args.writer.add_scalar(f'Test_Acc_{eval_func}/Old', result['old'], epoch)
                args.writer.add_scalar(f'Test_Acc_{eval_func}/New', result['new'], epoch)

        logger.info(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f} | Seen Class Acc: {train_acc_record.avg:.4f}')
        logger.info(f'Test Accuracies (v1): All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
        logger.info(f"model saved to {args.model_path}")
        
        exp_lr_scheduler.step()
        torch.save(model.state_dict(), args.model_path)
        logger.info(f"model saved to {args.model_path}")
        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
        logger.info(f"projection head saved to {args.model_path[:-3] + '_proj_head.pt'}")

        if old_acc > best_test_acc_lab:
            logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc:.4f}...')
            logger.info(f'Best Train Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
            torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
            logger.info(f"model saved to {args.model_path[:-3] + f'_best.pt'}")
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
            logger.info(f"projection head saved to {args.model_path[:-3] + f'_proj_head_best.pt'}")
            best_test_acc_lab = old_acc

def collect_features_and_labels(test_loader, model, device):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for images, labels, *_ in test_loader:
            images = images.to(device, non_blocking=True)
            features = model(images)    # 新：提取768维features！
            features = features.cpu()
            all_outputs.append(features)
            all_labels.append(labels.cpu())
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_outputs, all_labels

def tsne_all_proto_ood(all_outputs, all_labels, 
                    # proto_features, proto_labels, 
                    # ood_features, ood_labels, 
                    args, epoch, save_path="./TSNE/"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    os.makedirs(save_path, exist_ok=True)
    colors = [
        '#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
        '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
        '#e5c494', '#b3b3b3', '#1b9e77', '#d95f02', '#7570b3'
    ]
    class_names = list(args.class_to_idx.keys()) if hasattr(args, 'class_to_idx') else [str(i) for i in np.unique(all_labels)]
    print(f"all_outputs shape: {all_outputs.shape}")
    # print(f"proto_features shape: {proto_features.shape}")
    # print(f"ood_features shape: {ood_features.shape}")
    # stacked_features = np.concatenate([all_outputs, proto_features, ood_features], axis=0)
    stacked_features = all_outputs
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', 
                perplexity=25, early_exaggeration=10,
                random_state=epoch)
    emb = tsne.fit_transform(stacked_features)
    n_samples = all_outputs.shape[0]
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    print(f"unique_labelss shape: {unique_labels.shape}")
    total = 0
    for i, label in enumerate(unique_labels):
        color = colors[i % len(colors)]
        idxs = np.where(all_labels == label)[0]
        plt.scatter(emb[idxs,0], emb[idxs,1], c=color, label=str(label), s=24, alpha=1, edgecolor='none')
        total += len(idxs)
    print(f"Actually drawn {total} points on figure.")
    # 原型（略大+黑边）
    # for i, c in enumerate(colors):
    #     if i >= len(proto_labels): break
    #     idx = n_samples + i
    #     plt.scatter(emb[idx,0], emb[idx,1], c=c, s=22, marker='o', linewidths=1.0, alpha=1.0, zorder=10)
    # # OOD（略大+黑边+X）
    # for j in range(n_oods):
    #     idx = n_samples + n_protos + j
    #     plt.scatter(emb[idx,0], emb[idx,1], c='#e41a1c', s=24, marker='X', linewidths=1.2, alpha=0.98, label='OOD' if j==0 else None, zorder=20)
    # plt.legend(ncol=3, fontsize='small', frameon=False)
    plt.xticks([]); plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.title(f"t-SNE: All Samples (Epoch {epoch})", fontsize=15, pad=10)
    plt.tight_layout()
    file_prefix = f"{save_path}/Tsne_all_proto_ood_epoch_{epoch}"
    plt.savefig(file_prefix + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(file_prefix + ".svg", dpi=300, bbox_inches='tight')
    plt.close()
    np.save(file_prefix + "_embed.npy", stacked_features)


def test_on_the_fly(model, projection_head, test_loader,
                epoch, save_name, hamming_threshold,
                args):
    import numpy as np
    model.eval()
    projection_head.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    logger.info('Collating features...')
    # First extract all features
    for batch_idx, batch in enumerate(tqdm(test_loader)):
    # 通用解包处理，适应不同数据集返回值数量的差异
        if len(batch) == 4:
            images, label, uq_idx, _ = batch  # 处理返回4个值的情况（如iNaturalist）
        elif len(batch) == 3:
            images, label, uq_idx = batch     # 处理返回3个值的情况
        else:
            # 兜底处理
            images, label = batch[0], batch[1]


        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        _, feats, _ = projection_head(feats)

        feats = torch.nn.functional.normalize(feats, dim=-1)[:, :]
        print(feats.shape)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
    
    # ========== t-SNE 可视化部分 ==========
    all_feats = np.concatenate(all_feats)
    if epoch % 2 == 0 and epoch != 0:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        save_path = "./TSNE_on_the_fly/"
        os.makedirs(save_path, exist_ok=True)

        logger.info("Plotting t-SNE for on-the-fly features...")
        # t-SNE 降维
        all_feats_embedded = TSNE(
            n_components=2,
            init='pca',
            learning_rate='auto',
            perplexity=75,
            random_state=epoch
        ).fit_transform(all_feats)

        unique_labels = np.unique(targets)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))

        plt.figure(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            idxs = np.where(targets == label)[0]
            plt.scatter(
                all_feats_embedded[idxs, 0],
                all_feats_embedded[idxs, 1],
                c=[colors(i)],
                label=str(label),
                s=12
            )
        # plt.legend(ncol=4, fontsize='small')
        tsne_file = f"{save_path}TSNE_epoch_{epoch}.png"
        tsne_file_svg = f"{save_path}TSNE_epoch_{epoch}.svg"
        plt.savefig(tsne_file)
        plt.savefig(tsne_file_svg)
        plt.close()
        logger.info(f"t-SNE plot saved to {tsne_file}")
        # 保存嵌入
        np.save(f"{save_path}features_epoch_{epoch}.npy", all_feats)
        np.save(f"{save_path}labels_epoch_{epoch}.npy", targets)
        logger.info(f"Embeddings and labels saved to {save_path} for epoch {epoch}")
    # ========== t-SNE 可视化部分 ==========
    
    # -----------------------
    # On-The-Fly
    # -----------------------
    
    feats_hash = torch.Tensor(all_feats > 0).float().tolist()
    preds = []

    hash_dict = {}  # 字典用于存储feat_key -> index的映射
    hash_list = []  # 列表用于存储所有唯一的feat_key

    for feat in feats_hash:
        feat_key = tuple(int(x) for x in feat)  # 转成tuple方便判等和hash
        if feat_key not in hash_dict:
            hash_dict[feat_key] = len(hash_list)
            hash_list.append(feat_key)
        preds.append(hash_dict[feat_key])
    preds = np.array(preds)
    logger.info(f'Epoch {epoch} | Total samples: {len(preds)} | Unique hash codes: {len(hash_dict)}')

    # all_outputs, all_labels = collect_features_and_labels(test_loader, model, device)   # 已经768维
    # tsne_all_proto_ood(
    #     all_outputs, all_labels,
    #     # proto_features, proto_labels,
    #     # ood_features, ood_labels,
    #     args, epoch, save_path="./TSNE/"
    # )   
    # -----------------------
    # EVALUATE
    # -----------------------
    results = {}
    for eval_func in args.eval_funcs:
        # 为每个评估函数计算准确率
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=[eval_func], save_name=f'{save_name}_{eval_func}',
                                                        writer=args.writer)
        results[eval_func] = {
            'all': all_acc,
            'old': old_acc,
            'new': new_acc
        }

    return results

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    # seed = 1029
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_logger(seed=None, log_dir="output", data_name = None):
    """
    创建以日期+seed命名的日志文件，同时输出到控制台
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名：日期_seed格式
    date_str = datetime.now().strftime("%Y-%m-%d")
    if seed is not None:
        log_filename = f"Arachnida_{date_str}_{data_name}_seed{seed}.log"
    else:
        log_filename = f"Arachnida__{date_str}_{data_name}.log"
    
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 清理已有handler
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 控制台handler - INFO级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件handler - 所有级别，写入以日期+seed命名的文件
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')  # 'w'模式每次运行重新创建
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Log file created: {log_path}")
    logger.info(f"Seed: {seed}")
    
    return logger

def visualize_prototypes_and_ood(model, projection_head, labelled_dataset, device, epoch, save_dir):
    model.eval()
    projection_head.eval()

    from sklearn.manifold import TSNE

    # —— 1. 构建原始、增强和 OOD 原型 —— #
    # 从已有函数复用 get_prototypes 和 generate_latent_ood_prototypes
    prototypes, class_list = get_prototypes(args, model, projection_head, labelled_dataset, device)
    id_proto_orig = torch.stack([prototypes[c]['original'] for c in class_list])      # [C, D]
    id_proto_pair = torch.stack([torch.cat([
        prototypes[c]['original'].unsqueeze(0),
        prototypes[c]['augmented'].unsqueeze(0)
    ], dim=0).view(-1,)] for c in class_list).view(-1, projection_head.code_dim)  # [2C, D]
    ood_protos = generate_latent_ood_prototypes(prototypes, class_list, device, num_ood=10)  # [M, D]

    # 采样所有原型向量
    emb_orig = id_proto_orig.cpu().numpy()
    emb_pair = id_proto_pair.cpu().numpy()
    emb_ood  = ood_protos.cpu().numpy()

    # —— 2. 准备三组数据 for t-SNE —— #
    data_sets = {
        'Original Prototypes': emb_orig,
        'Prototype Pairs':     emb_pair,
        'After Adding OOD':    np.vstack([emb_pair, emb_ood])
    }

    # —— 3. 对每组做 t-SNE 并绘图 —— #
    for name, data in data_sets.items():
        tsne = TSNE(n_components=2, perplexity= max(5, min(30, len(data)//3)),
                    random_state=epoch)
        emb2d = tsne.fit_transform(data)

        plt.figure(figsize=(6,6))
        if name == 'Original Prototypes':
            plt.scatter(emb2d[:,0], emb2d[:,1],
                        c=range(len(class_list)), cmap='tab10', s=60)
        elif name == 'Prototype Pairs':
            # 前 C 点为 original，后 C 点为 augmented
            C = len(class_list)
            plt.scatter(emb2d[:C,0], emb2d[:C,1], marker='o',
                        c=range(C), cmap='tab10', s=60, label='original')
            plt.scatter(emb2d[C:,0], emb2d[C:,1], marker='^',
                        c=range(C), cmap='tab10', s=60, label='augmented')
        else:  # After Adding OOD
            C2 = emb_pair.shape[0]
            plt.scatter(emb2d[:C2,0], emb2d[:C2,1],
                        c='gray', marker='o', s=50, alpha=0.6, label='proto pairs')
            plt.scatter(emb2d[C2:,0], emb2d[C2:,1],
                        c='red', marker='x', s=80, label='OOD protos')

        plt.title(f"{name} t-SNE (Epoch {epoch})")
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/tsne_{name.replace(' ', '_')}_ep{epoch}.png")
        plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)
    
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    parser.add_argument('--n_protos', type=int, default=10, help='Number of prototypes per class')
    parser.add_argument('--palm_weight', type=float, default=1.0, help='Weight for PALM loss')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Regularization for Sinkhorn-Knopp')
    parser.add_argument('--top_k', type=int, default=5, help='Top-K pruning for soft assignments')

    parser.add_argument('--cache_size', type=int, default=6, help='PALM: cache size')
    parser.add_argument('--lambda_pcon', type=float, default=1.0, help='PALM: proto contrastive loss weight')
    parser.add_argument('--proto_m', type=float, default=0.999, help='PALM: EMA momentum')
    parser.add_argument('--k', type=int, default=5, help='PALM: top-k for prototype update')
    parser.add_argument('--save_path', type=str, default='checkpoints/model.pt')
    parser.add_argument('--method', type=str, default='topk-palm')
    parser.add_argument('--subclassname', type=str, default='Animalia', help='options: cifar10, cifar100, scars')

    logger = setup_logger()
        # ----------------------
        # Multiple Runs
        # ----------------------
    for run in range(0, 10):

        # ----------------------
        # INIT
        # ----------------------
        random_seed = random.randint(0, 10000)
        seed_torch(random_seed)
        args = parser.parse_args()
        device = torch.device('cuda:0')
        args = get_class_splits(args)
        print("seed:")
        print(random_seed)
        # 为每个run创建独立的日志
        logger = setup_logger(seed=random_seed, log_dir="output", data_name = args.dataset_name)

        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(args.unlabeled_classes)

        init_experiment(args, runner_name=['checkpoints'])
        logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

        # ----------------------
        # BASE MODEL
        # ----------------------
        if args.base_model == 'vit_dino':

            args.interpolation = 3
            args.crop_pct = 0.875
            pretrain_path = dino_pretrain_path

            model = vits.__dict__['vit_base']()

            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)

            if args.warmup_model_dir is not None:
                logger.info(f'Loading weights from {args.warmup_model_dir}')
                model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

            model.to(device)

            # NOTE: Hardcoded image size as we do not finetune the entire ViT model
            args.image_size = 224
            args.feat_dim = 768
            args.num_mlp_layers = 3
            args.code_dim = 12
            args.mlp_out_dim = None


            # ----------------------
            # HOW MUCH OF BASE MODEL TO FINETUNE
            # ----------------------
            for m in model.parameters():
                m.requires_grad = False

            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in model.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True

        else:

            raise NotImplementedError

        # --------------------
        # CONTRASTIVE TRANSFORM
        # --------------------
        train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
        train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

        # --------------------
        # DATASETS
        # --------------------
        train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_dataset = get_datasets(args.dataset_name,
                                                                                             train_transform,
                                                                                             test_transform,
                                                                                             args)



        # --------------------
        # DATALOADERS
        # --------------------
        labelled_train_loader = DataLoader(labelled_dataset, num_workers=args.num_workers, batch_size=args.batch_size, 
                                  shuffle=True, drop_last=True)
        unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                          batch_size=args.batch_size, shuffle=False)

        # ----------------------
        # PROJECTION HEAD
        # ----------------------
        projection_head = vits.__dict__['HASHHead'](in_dim=args.feat_dim,
                                   out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers, code_dim=args.code_dim, class_num=args.num_labeled_classes)
        projection_head.to(device)

        # 调用可视化
        # visualize_prototypes_and_ood(
        #     model, projection_head,
        #     labelled_dataset=train_dataset.labelled_dataset,
        #     device=device,
        #     epoch=20,
        #     save_dir='/tsne_visuals'
        # )

        # ----------------------
        # TRAIN
        # ----------------------
        train(projection_head, model, labelled_train_loader, test_loader, unlabelled_train_loader, args)

        
