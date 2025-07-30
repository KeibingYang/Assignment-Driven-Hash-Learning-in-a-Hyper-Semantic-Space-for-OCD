import torch.nn as nn
import torch
import torch.nn.functional as F
   
class PALM(nn.Module):
    def __init__(self, args, num_classes=10, prototypes=None, proto_m=0.99, temp=0.1, lambda_pcon=1, k=0,feat_dim=128, epsilon=0.05, class_known=None):
        super(PALM, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  # temperature scaling
        self.n_views = args.n_views
        self.cache_size = args.cache_size
        
        self.lambda_pcon = lambda_pcon
        
        self.feat_dim = feat_dim
        
        self.epsilon = epsilon
        self.sinkhorn_iterations = 3
        self.k = min(k, self.cache_size)
        
        self.n_protos = 2
        self.proto_m = proto_m
        self.class_known = class_known
        
        # print("-" * 80)
        # print("n_protos in PALM")
        # print(self.n_protos)
        # print("feat_dim in PALM")
        # print(feat_dim)
        # print("-" * 80)
        # self.register_buffer("protos", torch.rand(self.n_protos,feat_dim))
        proto_list = []
        for cls in sorted(prototypes.keys()):
            proto_list.append(prototypes[cls]['original'].unsqueeze(0))
            proto_list.append(prototypes[cls]['augmented'].unsqueeze(0))
        self.protos = torch.cat(proto_list, dim=0)
        self.class_id_to_proto_idx = {
            class_id: i for i, class_id in enumerate(sorted(self.class_known))
        }

    
    def to(self, device):
        super().to(device)
        self.protos = self.protos.to(device)
        # self.proto_to_class = self.proto_to_class.to(device)
        return self
        
    def sinkhorn(self, features):
        # print("-" * 80)
        # print("sinkhorn")
        # # print(self.protos)
        # print(self.protos.shape)
        # print("-" * 80)
        # print("features")
        # print(features.shape)
        # print("-" * 80)
        out = torch.matmul(features, self.protos.detach().T)
            
        Q = torch.exp(out.detach() / self.epsilon).t()# Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0] # how many prototypes
        # print("-" * 80)
        # print(f"B= {B}")
        # print(f"K= {K}")
        # print("-" * 80)
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if torch.isinf(sum_Q):
            self.protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, self.protos.detach().T)
            Q = torch.exp(out.detach() / self.epsilon).t()# Q is K-by-B for consistency with notations from our paper
            sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q = F.normalize(Q, dim=1, p=1)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q = F.normalize(Q, dim=0, p=1)
            Q /= B

        Q *= B
        return Q.t()
        
    def mle_loss(self, features, targets):
        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz * n_views, feat_dim]
        device = features.device
        batch_size = features.shape[0]

        Q = self.sinkhorn(features)  # [B, n_protos]

        # 展平标签
        targets = targets.view(-1, 1).repeat(1, self.n_views).view(-1)  # [bsz * n_views]

        update_mask = torch.zeros_like(Q)  # [B, 2 * num_known_classes]
        for i, cls_id in enumerate(targets.tolist()):
            if cls_id in self.class_id_to_proto_idx:
                local_idx = self.class_id_to_proto_idx[cls_id]
                update_mask[i, local_idx * 2] = 1
                update_mask[i, local_idx * 2 + 1] = 1
            else:
                # 如果是未知类，跳过（不参与 EMA / MLE loss）
                continue

        Q_masked = Q * update_mask
        update_features = torch.matmul(Q_masked.T, features)
        self.protos = F.normalize(
            self.proto_m * self.protos + (1 - self.proto_m) * update_features,
            dim=1, p=2
        )

        logits = torch.matmul(features, self.protos.detach().T) / self.temp
        logits_exp = torch.exp(logits)

        mask = torch.zeros_like(logits_exp)
        for i, cls_id in enumerate(targets.tolist()):
            if cls_id in self.class_id_to_proto_idx:
                local_idx = self.class_id_to_proto_idx[cls_id]
                mask[i, local_idx * 2] = 1
                mask[i, local_idx * 2 + 1] = 1
            else:
                continue

        numerator = torch.sum(logits_exp * mask, dim=1) + 1e-12
        denominator = torch.sum(logits_exp, dim=1) + 1e-12
        loss = -torch.log(numerator / denominator).mean()

        return loss


    
    def proto_contra(self):
        device = self.protos.device
        protos = F.normalize(self.protos, dim=1)  # [2*C, feat_dim]

        # 构造 2 个视图下的原型标签（每两个原型来自同一类）
        proto_labels = torch.arange(len(self.class_known)).repeat_interleave(2).to(device)  # [2*C]
        mask = torch.eq(proto_labels.unsqueeze(0), proto_labels.unsqueeze(1)).float()  # [2*C, 2*C]

        # 计算对比损失
        anchor_dot_contrast = torch.div(torch.matmul(protos, protos.T), 0.5)  # [2*C, 2*C]

        # 避免数值爆炸
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 去掉对自己自身的比较
        logits_mask = 1 - torch.eye(len(proto_labels), device=device)
        mask *= logits_mask

        # 计算 log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()

        return loss

           
    def forward(self, features, targets):
        loss = 0
        loss_dict = {}

        g_con = self.mle_loss(features, targets)
        loss += g_con
        loss_dict['mle'] = g_con.cpu().item()
                    
        if self.lambda_pcon > 0:            
            g_dis = self.lambda_pcon * self.proto_contra()
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()
                                
        self.protos = self.protos.detach()
                
        return loss, loss_dict