"""只检查训练前输入、配置、权重和五折隔离；不创建模型或启动训练。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.common import ROOT, digest, read_csv, write_json
from src.experiment import configuration, preflight
from src.data.manifest import validate_rows
from src.data.manifest import fit_metadata
from src.losses import classification_loss
from src.experiment_registry import EXPERIMENTS, LEGACY_EXPERIMENTS


def check(data_root=None, weights=None, output=None, experiments=None, panderm_weights=None, monet_dir=None):
    overrides = {k: str(v) for k, v in {'data_root': data_root, 'weights': weights}.items() if v is not None}
    records = []
    # 同一次只读检查中图片存在性只检查一次，避免 50 份实验重复访问同一目录。
    base_config = configuration(ROOT / 'configs/experiments/E00_derm.yaml', overrides)
    from src.experiment import resolve
    for name in ('milk_dev', 'milk_holdout', 'isic_train'):
        validate_rows(read_csv(resolve(base_config['manifests']) / f'{name}.csv'), resolve(base_config['data_root']))
    selected = tuple(experiments) if experiments is not None else LEGACY_EXPERIMENTS
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(EXPERIMENTS):
        raise ValueError('Choose distinct registered experiments')
    for experiment in selected:
        paths = list((ROOT / 'configs/experiments').glob(experiment + '_*.yaml'))
        if len(paths) != 1:
            raise ValueError(f'Expected one configuration for {experiment}')
        extra = dict(overrides)
        if experiment == 'E12' and panderm_weights is not None:
            extra['panderm_weights'] = str(panderm_weights)
        if experiment == 'E13' and monet_dir is not None:
            extra['monet_dir'] = str(monet_dir)
        config = configuration(paths[0], extra)
        # 所有实验逐折检查。对同一原图仅检查存在性，下载时完整 SHA256 已另行记录。
        for fold in range(5):
            train, val, inputs = preflight(config, fold, verify_images=False)
            if config['modality'] == 'clinical' and any(int(r['has_clinical']) != 1 for r in train + val):
                raise ValueError('Clinical-only experiment has missing images')
            _, loss = classification_loss(train, config['weighted'], 'cpu')
            metadata_dim = fit_metadata(train).output_dim if config['metadata'] else 0
            records.append({'experiment': config['experiment'], 'fold': fold, 'train_samples': len(train), 'val_samples': len(val),
                'metadata_dim': metadata_dim, 'train_class_counts': loss['train_class_counts']})
            if 'components' in inputs:
                records[-1]['components'] = inputs['components']
        print(f"{config['experiment']}: all five folds preflight passed", flush=True)
    record = {'local_preparation': 'passed', 'protocol': 'resnet9-v1', 'runs_checked': records,
        'manifest_sha256': inputs['manifest_sha256'], 'weights_sha256': inputs['weights_sha256'],
        'real_training_started': False, 'new_files_uploaded_to_nibi': False,
        'cluster_verification': 'pending upload and real-data GPU smoke; not authorized in this preparation task',
        'data_counts': json.loads((ROOT / 'DATA/manifests/summary.json').read_text())['counts']}
    if output:
        write_json(output, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/preparation/readiness.json')
    parser.add_argument('--experiments', nargs='+', choices=EXPERIMENTS,
                        help='Default remains E00-E10; E11-E13 require explicit selection')
    parser.add_argument('--panderm-weights', type=Path)
    parser.add_argument('--monet-dir', type=Path)
    args = parser.parse_args()
    if args.experiments and any(item not in LEGACY_EXPERIMENTS for item in args.experiments) and args.output == ROOT / 'reports/preparation/readiness.json':
        parser.error('New experiments require a new explicit --output; preserve the historical readiness record')
    check(args.data_root, args.weights, args.output, args.experiments, args.panderm_weights, args.monet_dir)
