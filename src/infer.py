"""从本轮检查点离线重建模型，输出 sample_id + 九类概率。"""
import argparse
import json
from pathlib import Path
import torch

from .data.common import read_csv, digest
from .data.manifest import validate_rows
from .experiment import configure_runtime, epoch, image_root, load_checkpoint, loader, predictions, resolve
from .metadata_features import MetadataEncoder
from .models.fusion import ResNetExperiment


def predict(checkpoint_path, manifest, output, *, data_root=None, workers=None, device_name='cuda'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    checkpoint = load_checkpoint(checkpoint_path)
    config = dict(checkpoint['config'])
    if data_root:
        config['data_root'] = str(data_root)
    if workers is not None:
        config['workers'] = workers
    rows = validate_rows(read_csv(manifest), image_root(config))
    if any(r['split'] == 'holdout' for r in rows):
        if checkpoint.get('phase') != 'development_refit' or not checkpoint.get('refit_complete'):
            raise ValueError('Fixed holdout requires a completed development refit checkpoint')
        directory = resolve(config['manifests'])
        version = json.loads((directory / 'version.json').read_text())
        if digest(directory / 'version.json') != checkpoint['inputs']['manifest_sha256'] or digest(manifest) != version['files']['milk_holdout.csv']:
            raise ValueError('Fixed holdout manifest differs from frozen protocol')
    encoder = MetadataEncoder.from_dict(checkpoint['metadata']) if checkpoint['metadata'] else None
    model = ResNetExperiment(config['modality'], encoder.output_dim if encoder else 0)
    model.load_state_dict(checkpoint['model'], strict=True)
    device = torch.device(device_name)
    configure_runtime(config, device)
    model.to(device)
    if config['channels_last']:
        model.to(memory_format=torch.channels_last)
    data = loader(rows, config, encoder, False, device)
    result = epoch(model, data, torch.nn.CrossEntropyLoss(), config, device)
    if result['ids'] != [r['sample_id'] for r in rows]:
        raise ValueError('Inference ID order mismatch')
    predictions(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    result = predict(args.checkpoint, args.manifest, args.output, data_root=args.data_root, workers=args.workers)
    print(f"Exported {len(result['ids'])} samples to {args.output}")
