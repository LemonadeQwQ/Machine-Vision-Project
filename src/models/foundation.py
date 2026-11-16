"""Offline, pinned PanDerm and MONET adapters for the nine-class protocol.

MONET scoring adapts stages/monet_stage/monet.py; attribution and CC BY-NC-SA
license remain in stages/monet_stage/reference/. No legacy bootstrap is imported.
"""
from contextlib import nullcontext
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..data.common import ROOT, digest

PANDERM_SHA256 = 'be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1'
RESNET_SHA256 = '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
PANDERM_SOURCE = 'stages/panderm_stage/vendor/modeling_finetune.py'
MONET_REVISION = '1d1efd0b8d61bde82cde22d2e93c28d73441f29e'
MONET_SHA256 = 'f5b874fff0aa8119484a75a497c997da8b4ff300a8c81c838aab893f301390af'
MONET_FILES = ('config.json', 'merges.txt', 'model.safetensors', 'preprocessor_config.json',
               'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json')
CONCEPTS = (
    ('ulceration_crust', ('ulceration', 'crust')),
    ('vasculature_vessels', ('vasculature', 'vessels')),
    ('erythema', ('erythema',)),
    ('pigmented', ('pigmented',)),
)
TEMPLATES = ('This is skin image of {}', 'This is dermatology image of {}', 'This is image of {}')
REFERENCES = ('This is skin image', 'This is dermatology image', 'This is image')
SCALE = math.exp(4.5944)
MONET_DEPENDENCIES = {'transformers': '4.46.3', 'tokenizers': '0.20.3',
                      'huggingface-hub': '0.26.2', 'safetensors': '0.4.5'}


def require_version(package, expected):
    try:
        actual = version(package)
    except PackageNotFoundError as exc:
        raise RuntimeError(f'Missing optional dependency {package}=={expected}; see requirements-foundation.txt') from exc
    if actual != expected:
        raise RuntimeError(f'{package}=={expected} required for the pinned adapter, found {actual}')


def local_path(value):
    return (ROOT / value).resolve()


def foundation_environment():
    result = {}
    for name in ('timm', *MONET_DEPENDENCIES):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def require_monet_dependencies():
    for name, expected in MONET_DEPENDENCIES.items():
        require_version(name, expected)


def monet_assets(folder):
    """Require a pinned acquisition receipt and hash every model/tokenizer input."""
    folder = Path(folder)
    receipt_path = folder / 'download_manifest.json'
    if not receipt_path.is_file():
        raise FileNotFoundError(f'MONET pinned download receipt missing: {receipt_path}')
    allowed = {*MONET_FILES, 'download_manifest.json'}
    unexpected = sorted(path.name for path in folder.iterdir()
                        if path.name not in allowed and not (path.name == '.cache' and path.is_dir()))
    if unexpected:
        # HF tokenizers can consume optional added_tokens.json and adapter files.
        # Do not let an unrecorded input change the pinned local reconstruction.
        raise ValueError(f'Unexpected MONET input files/directories: {unexpected}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('repository') != 'chanwkim/monet' or receipt.get('revision') != MONET_REVISION:
        raise ValueError('MONET repository / revision mismatch')
    expected = receipt.get('sha256', {})
    if set(expected) != set(MONET_FILES) or expected.get('model.safetensors') != MONET_SHA256:
        raise ValueError('Incomplete / unexpected MONET acquisition manifest')
    actual = {name: digest(folder / name) for name in MONET_FILES}
    if actual != expected:
        raise ValueError('MONET model / tokenizer files differ from the acquisition manifest')
    return {'repository': 'chanwkim/monet', 'revision': MONET_REVISION, 'sha256': actual}


def component_assets(config):
    """Read-only prerequisite check; never fetch missing assets or install packages."""
    assets = {}
    if config['panderm']:
        require_version('timm', '0.9.16')
        path = local_path(config['panderm_weights'])
        if digest(path) != PANDERM_SHA256:
            raise ValueError('PanDerm_Base checkpoint SHA256 mismatch')
        assets['panderm'] = {'sha256': PANDERM_SHA256, 'source_sha256': digest(ROOT / PANDERM_SOURCE)}
    if config['monet']:
        require_monet_dependencies()
        assets['monet'] = monet_assets(local_path(config['monet_dir']))
    return assets


def freeze_panderm(encoder):
    if len(encoder.blocks) != 12 or not isinstance(encoder.norm, nn.LayerNorm):
        raise ValueError('Expected PanDerm_Base twelve blocks and final LayerNorm')
    encoder.requires_grad_(False)
    for block in encoder.blocks[-2:]:
        block.requires_grad_(True)
    encoder.norm.requires_grad_(True)
    encoder.eval()
    return encoder


def panderm_encoder(weights_path=None):
    require_version('timm', '0.9.16')
    from stages.panderm_stage.vendor.modeling_finetune import panderm_base_patch16_224
    encoder = panderm_base_patch16_224(drop_path_rate=0.0)
    encoder.head = nn.Identity()
    if weights_path is not None:
        if digest(weights_path) != PANDERM_SHA256:
            raise ValueError('PanDerm_Base checkpoint SHA256 mismatch')
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        encoder.load_state_dict(state, strict=True)
    return freeze_panderm(encoder)


def concept_contract():
    return {'version': 1, 'model': 'chanwkim/monet', 'revision': MONET_REVISION,
            'weights_sha256': MONET_SHA256,
            'concepts': [{'name': name, 'synonyms': list(terms)} for name, terms in CONCEPTS],
            'templates': list(TEMPLATES), 'references': list(REFERENCES),
            'inverse_temperature': SCALE, 'scoring': 'independent target-versus-reference sigmoid; mean templates',
            'synonyms': 'mean normalized text features without renormalizing the mean',
            'view': 'same augmented 224x224 derm tensor; ImageNet normalization',
            'normalization_mean': [.485, .456, .406], 'normalization_std': [.229, .224, .225],
            'preprocessor': 'project full-view transform; no CLIP processor resize/crop/renormalization',
            'vision_precision': 'FP16 on CUDA, FP32 on CPU; score arithmetic FP32',
            'frozen': True, 'live': True}


def concept_scores(images, targets, references):
    images = F.normalize(images.float(), dim=-1)
    target_similarity = torch.einsum('bd,ctd->bct', images, targets.float())
    reference_similarity = images @ references.float().T
    return torch.sigmoid((target_similarity - reference_similarity[:, None, :]) * SCALE).mean(-1)


class FrozenMONET(nn.Module):
    def __init__(self, folder, expected_assets=None):
        super().__init__()
        require_monet_dependencies()
        self.assets = monet_assets(folder)
        if expected_assets is not None and self.assets != expected_assets:
            raise ValueError('MONET assets differ from the checkpoint / preflight')
        from transformers import CLIPModel, CLIPTokenizer
        model, loading = CLIPModel.from_pretrained(str(folder), local_files_only=True, use_safetensors=True,
            attn_implementation='eager', output_loading_info=True)
        if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'MONET checkpoint loading was not exact: {loading}')
        vision = model.config.vision_config
        if (vision.image_size, vision.patch_size, vision.hidden_size, vision.num_hidden_layers, model.config.projection_dim) != (224, 14, 1024, 24, 768):
            raise ValueError('MONET must be the pinned CLIP ViT-L/14 with 224px input and 768-d projection')
        tokenizer = CLIPTokenizer.from_pretrained(str(folder), local_files_only=True)
        model.requires_grad_(False).eval()

        def encode(texts):
            tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                return F.normalize(model.get_text_features(**tokens).float(), dim=-1)

        targets = torch.stack([torch.stack([encode([template.format(term) for term in terms]).mean(0)
            for template in TEMPLATES]) for _, terms in CONCEPTS])
        self.vision, self.projection = model.vision_model, model.visual_projection
        self.register_buffer('targets', targets)
        self.register_buffer('references', encode(list(REFERENCES)))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def set_precision(self, device):
        dtype = torch.float16 if torch.device(device).type == 'cuda' else torch.float32
        self.vision.to(dtype=dtype)
        self.projection.to(dtype=dtype)

    @torch.no_grad()
    def forward(self, derm):
        if derm.ndim != 4 or tuple(derm.shape[1:]) != (3, 224, 224):
            raise ValueError('MONET requires the classifier\'s same normalized 224x224 view')
        context = torch.autocast('cuda', enabled=False) if derm.is_cuda else nullcontext()
        with context:
            values = derm.to(dtype=next(self.vision.parameters()).dtype).contiguous()
            embeddings = self.projection(self.vision(pixel_values=values).pooler_output)
            scores = concept_scores(embeddings, self.targets, self.references)
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite MONET scores')
        return scores.detach()
