# siamese/model.py
import torch
import torch.nn as nn
from torchvision import models

class SiameseNet(nn.Module):
    def __init__(self, emb_size=128, backbone_name='resnet34', pretrained=True):
        super().__init__()
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            feat_dim = 512
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError("backbone_name not supported")
        modules = list(backbone.children())[:-1]  # remove fc
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, emb_size)
        )
    def forward(self, x):
        f = self.backbone(x)
        e = self.fc(f)
        e = nn.functional.normalize(e, p=2, dim=1)  # l2 normalize
        return e
