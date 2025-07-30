import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from vision_transformer import vit_base

class HASHHead(nn.Module):
    def __init__(self, in_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, code_dim=12):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            layers.append(nn.BatchNorm1d(bottleneck_dim)) ## new
            layers.append(nn.GELU()) ## new
            self.mlp = nn.Sequential(*layers)
            
        self.hash = nn.Linear(bottleneck_dim, code_dim, bias=False)
        self.bn_h = nn.BatchNorm1d(code_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        h = self.hash(x)

        return h

class HashCenterManager:
    """管理Hash Centers的类"""
    
    def __init__(self, num_classes, hash_dim, device, momentum=0.9):
        """
        Args:
            num_classes: 类别数量
            hash_dim: hash特征维度
            device: 设备
            momentum: EMA更新动量
        """
        self.num_classes = num_classes
        self.hash_dim = hash_dim
        self.device = device
        self.momentum = momentum
        
        # 初始化hash centers - 使用Hadamard矩阵初始化
        self.hash_centers = self._initialize_centers()
        self.hash_centers = self.hash_centers.to(device)
        
        # 用于统计每个类别的样本数量
        self.class_counts = torch.zeros(num_classes, device=device)
        
    def _initialize_centers(self):
        """初始化hash centers"""
        # 使用随机初始化，然后正交化
        centers = torch.randn(self.num_classes, self.hash_dim)
        
        # 如果类别数小于hash维度，可以使用Hadamard矩阵
        if self.num_classes <= self.hash_dim:
            # 简单的正交初始化
            centers = torch.nn.functional.normalize(centers, dim=1)
            # 保证不同中心之间有足够的距离
            centers = torch.sign(centers)
        else:
            centers = torch.sign(torch.randn(self.num_classes, self.hash_dim))
            
        return centers
    
    def update_centers(self, hash_features, labels, update_mode='ema'):
        """更新hash centers
        
        Args:
            hash_features: [batch_size, hash_dim] hash特征
            labels: [batch_size] 标签
            update_mode: 更新模式 'ema' or 'mean'
        """
        with torch.no_grad():
            unique_labels = torch.unique(labels)
            
            for label in unique_labels:
                if label >= self.num_classes:
                    continue
                    
                mask = (labels == label)
                class_features = hash_features[mask]
                
                if len(class_features) == 0:
                    continue
                
                # 计算当前batch中该类别的平均特征
                batch_center = torch.mean(class_features, dim=0)
                
                if update_mode == 'ema':
                    # EMA更新
                    self.hash_centers[label] = (self.momentum * self.hash_centers[label] + 
                                              (1 - self.momentum) * batch_center)
                else:
                    # 简单平均更新
                    old_count = self.class_counts[label]
                    new_count = old_count + len(class_features)
                    
                    self.hash_centers[label] = ((old_count * self.hash_centers[label] + 
                                               len(class_features) * batch_center) / new_count)
                    self.class_counts[label] = new_count
    
    def get_centers(self):
        """获取当前的hash centers"""
        return self.hash_centers
    
    def get_center_distances(self):
        """计算centers之间的平均汉明距离"""
        centers_binary = torch.sign(self.hash_centers)
        distances = []
        
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                # 汉明距离
                hamming_dist = torch.sum(centers_binary[i] != centers_binary[j]).float()
                distances.append(hamming_dist)
        
        return torch.mean(torch.stack(distances)) if distances else torch.tensor(0.0)
