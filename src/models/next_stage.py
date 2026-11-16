"""E10 feature concatenation with independently controlled G / P / M components."""
import torch
from torch import nn

from .fusion import backbone
from .foundation import FrozenMONET, local_path, panderm_encoder


class ModalityGate(nn.Module):
    def __init__(self, derm_dim, clinical_dim=2048, hidden=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(derm_dim + clinical_dim + 2, hidden),
                                     nn.ReLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, derm, clinical, mask):
        scale = 2 * torch.sigmoid(self.network(torch.cat((derm, clinical, mask.to(derm.dtype)), 1)))
        complete = mask[:, 1:2].bool()
        # Missing clinical is a fixed derm-only path, including during training.
        scale = torch.where(complete, scale, torch.ones_like(scale))
        return derm * scale[:, :1], clinical * scale[:, 1:2] * mask[:, 1:2], scale


class NextStageExperiment(nn.Module):
    def __init__(self, config, metadata_dim, resnet_weights=None, *, initialize=True,
                 expected_assets=None):
        super().__init__()
        if metadata_dim < 1:
            raise ValueError('E11-E13 require train-fitted metadata')
        self.panderm, self.metadata_dim = config['panderm'], metadata_dim
        self.derm_dim = 768 if self.panderm else 2048
        derm = panderm_encoder(local_path(config['panderm_weights']) if initialize else None) if self.panderm else backbone(resnet_weights)
        self.encoders = nn.ModuleDict({'derm': derm, 'clinical': backbone(resnet_weights)})
        self.metadata_encoder = nn.Sequential(nn.Linear(metadata_dim, 64), nn.ReLU())
        # Construct the E10-compatible classifier before Gate so neutral Gate
        # initialization also preserves the baseline head's random sequence.
        self.classifier = nn.Sequential(nn.Linear(self.derm_dim + 2048 + 2 + 64 + 4 + (32 if config['monet'] else 0), 512),
                                        nn.ReLU(), nn.Dropout(.5), nn.Linear(512, 9))
        self.gate = ModalityGate(self.derm_dim, hidden=config['gate_hidden']) if config['gate'] else None
        self.monet = FrozenMONET(local_path(config['monet_dir']), (expected_assets or {}).get('monet')) if config['monet'] else None
        self.concept_project = nn.Sequential(nn.Linear(4, 32), nn.LayerNorm(32), nn.GELU()) if config['monet'] else None
        self.last_gate = None

    def train(self, mode=True):
        super().train(mode)
        for encoder in self.encoders.values():
            encoder.eval()
        if self.monet is not None:
            self.monet.eval()
        return self

    def forward(self, batch):
        mask = batch['modality_mask']
        if mask.ndim != 2 or mask.shape[1] != 2 or not torch.all((mask == 0) | (mask == 1)) or not mask[:, 0].bool().all():
            raise ValueError('E11-E13 require dermoscopy and a binary two-modality mask')
        derm_image = batch['derm'].contiguous() if self.panderm else batch['derm']
        derm = self.encoders['derm'](derm_image)
        present = mask[:, 1].bool()
        if present.all():
            clinical = self.encoders['clinical'](batch['clinical'])
        elif present.any():
            encoded = self.encoders['clinical'](batch['clinical'][present])
            clinical = encoded.new_zeros((len(derm), 2048))
            clinical[present] = encoded
        else:
            clinical = derm.new_zeros((len(derm), 2048))
        if self.gate is not None:
            derm, clinical, scale = self.gate(derm, clinical, mask)
            self.last_gate = scale.detach()
        parts = [derm, clinical, mask.to(derm.dtype), self.metadata_encoder(batch['metadata']), batch['metadata_mask']]
        if self.monet is not None:
            parts.append(self.concept_project(self.monet(batch['derm'])))
        return self.classifier(torch.cat(parts, 1))

    def optimizer_groups(self, head_lr=1e-4, backbone_lr=1e-5):
        encoders, heads = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (encoders if name.startswith('encoders.') else heads).append(parameter)
        return [{'params': encoders, 'lr': backbone_lr}, {'params': heads, 'lr': head_lr}]
