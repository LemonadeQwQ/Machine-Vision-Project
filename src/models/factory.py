"""One model/checkpoint contract shared by training, resume, refit and inference."""
import hashlib

import torch

from ..class_schema import NINE_CLASSES
from ..data.common import ROOT, digest
from ..experiment_registry import is_new, validate_identity
from .foundation import (PANDERM_SHA256, PANDERM_SOURCE, RESNET_SHA256, MONET_SHA256,
                         MONET_REVISION, MONET_FILES, concept_contract, local_path)
from .fusion import ResNetExperiment
from .next_stage import NextStageExperiment

NEW_CHECKPOINT_FORMAT = 'multimodal9-checkpoint-v2'
EXTERNAL_PREFIXES = ('monet.vision.', 'monet.projection.')


def model_contract(config, metadata_dim, components):
    validate_identity(config)
    if not is_new(config):
        return None
    if type(metadata_dim) is not int or metadata_dim < 1:
        raise ValueError('New model metadata dimension must be a positive integer')
    required = {'resnet'} | ({'panderm'} if config['panderm'] else set()) | ({'monet'} if config['monet'] else set())
    if set(components) != required:
        raise ValueError('Checkpoint component set does not match configuration')
    if components['resnet'] != {'sha256': RESNET_SHA256}:
        raise ValueError('Unapproved ResNet initialization')
    if config['panderm'] and components['panderm'] != {'sha256': PANDERM_SHA256, 'source_sha256': digest(ROOT / PANDERM_SOURCE)}:
        raise ValueError('PanDerm weight / implementation identity mismatch')
    if config['monet']:
        assets = components['monet']
        if assets.get('repository') != 'chanwkim/monet' or assets.get('revision') != MONET_REVISION or set(assets.get('sha256', {})) != set(MONET_FILES) or assets['sha256']['model.safetensors'] != MONET_SHA256:
            raise ValueError('Unapproved / incomplete MONET component identity')
    return {'version': 2, 'configuration_id': config['experiment'], 'class_order': list(NINE_CLASSES),
            'derm': 'PanDerm_Base' if config['panderm'] else 'ResNet50_IMAGENET1K_V2',
            'clinical': 'ResNet50_IMAGENET1K_V2', 'derm_dim': 768 if config['panderm'] else 2048,
            'clinical_dim': 2048, 'metadata_dim': metadata_dim, 'metadata_features': 64, 'metadata_mask_dim': 4,
            'classifier': 'concat -> Linear512/ReLU/Dropout0.5/Linear9',
            'missing_clinical': 'skip encoder; zero features; derm gate scale forced to one',
            'gate': {'hidden': 128, 'activation': 'ReLU', 'scale': '2*sigmoid', 'neutral_final_layer': 'zeros'} if config['gate'] else None,
            'panderm': {'train_last_blocks': 2, 'final_norm': True, 'drop_path': 0.,
                        'weights_sha256': PANDERM_SHA256} if config['panderm'] else None,
            'monet': concept_contract() if config['monet'] else None,
            'concept_projection': 'Linear4to32/LayerNorm/GELU' if config['monet'] else None,
            'components': components,
            'external_checkpoint_components': ['monet.vision', 'monet.projection'] if config['monet'] else []}


def build_model(config, metadata_dim=0, *, initialize=False, components=None, monet_dir=None,
                legacy_constructor=None):
    validate_identity(config)
    weights = local_path(config['weights']) if initialize else None
    if not is_new(config):
        # The injectable legacy constructor also keeps existing engine fixtures valid.
        constructor = legacy_constructor or ResNetExperiment
        return constructor(config['modality'], metadata_dim, weights) if initialize else constructor(config['modality'], metadata_dim)
    local_config = dict(config)
    if monet_dir is not None:
        local_config['monet_dir'] = str(monet_dir)
    return NextStageExperiment(local_config, metadata_dim, weights, initialize=initialize,
                               expected_assets=components)


def prepare_model(model, config, device):
    model.to(device)
    if is_new(config):
        if config['channels_last']:
            for name, encoder in model.encoders.items():
                if name == 'clinical' or not config['panderm']:
                    encoder.to(memory_format=torch.channels_last)
        if model.monet is not None:
            model.monet.set_precision(device)
    elif config['channels_last']:
        model.to(memory_format=torch.channels_last)
    return model


def checkpoint_state(model, config):
    state = model.state_dict()
    if is_new(config) and config['monet']:
        return {key: value for key, value in state.items() if not key.startswith(EXTERNAL_PREFIXES)}
    return state


def text_features_digest(state):
    value = hashlib.sha256()
    for key in ('monet.targets', 'monet.references'):
        tensor = state[key].detach().cpu().float().contiguous()
        value.update(key.encode('ascii'))
        value.update(str(tuple(tensor.shape)).encode('ascii'))
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def restore_model(model, state, config):
    if is_new(config) and config['monet']:
        if any(key.startswith(EXTERNAL_PREFIXES) for key in state):
            raise ValueError('External frozen MONET weights must not be overridden by the checkpoint')
        expected_missing = {key for key in model.state_dict() if key.startswith(EXTERNAL_PREFIXES)}
        result = model.load_state_dict(state, strict=False)
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError(f'Incomplete multimodal checkpoint: {result}')
    else:
        model.load_state_dict(state, strict=True)
