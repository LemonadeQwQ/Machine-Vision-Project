"""Stable experiment identities; E14 combinations are deliberately not registered yet."""

LEGACY_EXPERIMENTS = tuple(f'E{i:02}' for i in range(11))
NEW_EXPERIMENTS = {
    'E11': {'gate': True, 'panderm': False, 'monet': False},
    'E12': {'gate': False, 'panderm': True, 'monet': False},
    'E13': {'gate': False, 'panderm': False, 'monet': True},
}
EXPERIMENTS = (*LEGACY_EXPERIMENTS, *NEW_EXPERIMENTS)
NEW_DEFAULTS = {
    'gate': False, 'panderm': False, 'monet': False,
    'gate_hidden': 128, 'panderm_last_blocks': 2,
    'panderm_weights': 'pretrained_weights/panderm_bb_data6_checkpoint-499.pth',
    'monet_dir': 'pretrained_weights/monet/1d1efd0b8d61bde82cde22d2e93c28d73441f29e',
}


def is_new(config):
    return config.get('experiment') in NEW_EXPERIMENTS


def comparison_contract(contract):
    # Fold-fitted categorical vocabulary sizes can differ legitimately.
    return {**contract, 'metadata_dim': 'fit_on_each_training_partition'}


def validate_identity(config):
    experiment = config.get('experiment')
    if experiment not in EXPERIMENTS or config.get('protocol') != 'resnet9-v1':
        raise ValueError('Unknown protocol / experiment')
    if experiment in LEGACY_EXPERIMENTS:
        if NEW_DEFAULTS.keys() & config.keys():
            raise ValueError('Foundation-model options are not valid for E00-E10')
        return
    for name, enabled in NEW_EXPERIMENTS[experiment].items():
        if type(config.get(name)) is not bool or config[name] != enabled:
            raise ValueError(f'{experiment} has a fixed {name}={enabled} definition')
    for name, expected in {'modality': 'pair', 'weighted': True, 'include_isic': True,
                           'metadata': True, 'augment': True, 'image_size': 224,
                           'gate_hidden': 128, 'panderm_last_blocks': 2}.items():
        if config.get(name) != expected or type(config.get(name)) is not type(expected):
            raise ValueError(f'{experiment} requires {name}={expected!r}')
    for name in ('panderm_weights', 'monet_dir'):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f'{name} must be an explicit local path')


def validate_run_identity(run):
    """Extra identity requirements apply only to new runs; old evidence is unchanged."""
    if run.get('experiment') not in EXPERIMENTS:
        raise ValueError('Unregistered result experiment')
    if run['experiment'] in NEW_EXPERIMENTS:
        validate_identity(run['config'])
        if run['config']['experiment'] != run['experiment'] or run.get('configuration_id') != run['experiment']:
            raise ValueError('Run configuration identity mismatch')
        if not run.get('model_contract') or not run.get('inputs', {}).get('components'):
            raise ValueError('Missing component / checkpoint provenance')
        if run['model_contract'].get('configuration_id') != run['experiment'] or run['model_contract'].get('components') != run['inputs']['components']:
            raise ValueError('Run model contract / component identity mismatch')
        from .models.factory import model_contract
        expected = model_contract(run['config'], run['model_contract'].get('metadata_dim'), run['inputs']['components'])
        if run['model_contract'] != expected or run.get('protocol', run['config']['protocol']) != run['config']['protocol']:
            raise ValueError('Run model recipe / protocol mismatch')
