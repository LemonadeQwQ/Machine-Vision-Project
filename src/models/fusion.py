"""手册 E00–E09：独立 ResNet-50、layer4 微调、固定骨干 BN、简单拼接。"""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet50


def backbone(weights_path=None):
    model = resnet50(weights=None)
    if weights_path:
        model.load_state_dict(torch.load(Path(weights_path), map_location='cpu', weights_only=True), strict=True)
    model.fc = nn.Identity()
    model.requires_grad_(False)
    model.layer4.requires_grad_(True)
    model.eval()
    return model


class ResNetExperiment(nn.Module):
    def __init__(self, modality='derm', metadata_dim=0, weights_path=None):
        super().__init__()
        if modality not in ('derm', 'clinical', 'pair'):
            raise ValueError('Unknown modality')
        if metadata_dim and modality != 'pair':
            raise ValueError('Metadata experiment requires paired architecture')
        self.modality, self.metadata_dim = modality, metadata_dim
        self.encoders = nn.ModuleDict({name: backbone(weights_path) for name in ('derm', 'clinical')
                                      if modality in (name, 'pair')})
        self.metadata_encoder = nn.Sequential(nn.Linear(metadata_dim, 64), nn.ReLU()) if metadata_dim else None
        features = 4096 + 2 if modality == 'pair' else 2048
        features += 64 + 4 if metadata_dim else 0
        self.classifier = nn.Sequential(nn.Linear(features, 512), nn.ReLU(), nn.Dropout(.5), nn.Linear(512, 9))

    def train(self, mode=True):
        super().train(mode)
        # 与已有 layer4 调优策略一致：冻结所有骨干 BN 统计量，layer4 仿射参数仍可训练。
        for encoder in self.encoders.values():
            encoder.eval()
        return self

    def forward(self, batch):
        features = []
        for i, name in enumerate(('derm', 'clinical')):
            if name not in self.encoders:
                continue
            image = batch[name]
            present = batch['modality_mask'][:, i].bool()
            if self.modality != 'pair' and not present.all():
                raise ValueError('Required modality missing')
            if present.all():
                feature = self.encoders[name](image)
            elif present.any():
                encoded = self.encoders[name](image[present])
                feature = encoded.new_zeros((image.shape[0], 2048))
                feature[present] = encoded
            else:
                # 完全缺失时不调用该骨干，因此不会更新 BN 或制造伪图像特征。
                feature = image.new_zeros((image.shape[0], 2048))
            features.append(feature)
        if self.modality == 'pair':
            features.append(batch['modality_mask'].to(features[0].dtype))
        if self.metadata_encoder:
            features.extend([self.metadata_encoder(batch['metadata']), batch['metadata_mask']])
        return self.classifier(torch.cat(features, dim=1))

    def optimizer_groups(self, head_lr=1e-4, backbone_lr=1e-5):
        early, heads = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                (early if name.startswith('encoders.') else heads).append(param)
        return [{'params': early, 'lr': backbone_lr}, {'params': heads, 'lr': head_lr}]
