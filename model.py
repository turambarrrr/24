import torch
import torch.nn as nn
from torchvision import models

class ReIDNet(nn.Module):
    def __init__(self, num_classes=585, pretrained=True, freeze_layers=2):
        """
        freeze_layers: 冻结前 N 个 stage (0~5)
        推荐 2：冻结 conv1, bn1, layer1, layer2，训练 layer3, layer4, fc
        """
        super(ReIDNet, self).__init__()
        backbone = models.resnet50(pretrained=pretrained)
        
        # 冻结指定层
        if freeze_layers > 0:
            # 各层名称映射
            stage_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
            for name, param in backbone.named_parameters():
                freeze = False
                if freeze_layers >= 1 and name.startswith('conv1'):
                    freeze = True
                elif freeze_layers >= 2 and name.startswith('bn1'):
                    freeze = True
                elif freeze_layers >= 3 and name.startswith('layer1'):
                    freeze = True
                elif freeze_layers >= 4 and name.startswith('layer2'):
                    freeze = True
                elif freeze_layers >= 5 and name.startswith('layer3'):
                    freeze = True
                elif freeze_layers >= 6 and name.startswith('layer4'):
                    freeze = True
                if freeze:
                    param.requires_grad = False

        # 去掉最后的 avgpool 和 fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 输出特征 2048
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        logits = self.fc(feat)
        return logits, feat


class TripletLoss(nn.Module):
    """带容错的三元组损失，当 batch 中某 ID 只有一个样本时自动跳过"""
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features, labels):
        # 欧氏距离矩阵
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(len(features), len(features))
        dist = dist + dist.t()
        dist.addmm_(features, features.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = labels.expand(len(labels), len(labels)).eq(labels.expand(len(labels), len(labels)).t())
        dist_ap, dist_an = [], []
        for i in range(len(labels)):
            if mask[i].sum() == 0:
                continue
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        if not dist_ap:
            return torch.tensor(0.0, device=features.device)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)