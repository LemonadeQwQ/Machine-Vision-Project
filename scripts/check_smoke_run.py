"""Check smoke coverage, epoch-boundary restart and predictions from a reloaded checkpoint."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.class_schema import NINE_CLASSES
from src.data.common import fingerprint, read_csv, write_json
from src.evaluate import load_evaluation_data
from src.experiment import coverage, load_checkpoint


def check(folder, expected):
    folder = Path(folder)
    if expected not in ('stopped', 'completed'):
        raise ValueError('Expected stopped or completed smoke')
    record = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    if record['purpose'] != 'smoke' or record['smoke_epochs'] != 2:
        raise ValueError('Requires a two-epoch smoke run')
    train, val = read_csv(folder / 'smoke_train.csv'), read_csv(folder / 'smoke_val.csv')
    if record['data_coverage'] != {'train': coverage(train), 'val': coverage(val)}:
        raise ValueError('Recorded coverage differs from smoke manifests')
    if set(r['label'] for r in train) != set(NINE_CLASSES) or set(r['label'] for r in val) != set(NINE_CLASSES):
        raise ValueError('Smoke must cover all nine classes')
    sources = Counter(r['source'] for r in train)
    expected_sources = {'MILK', 'ISIC2019'} if record['config']['include_isic'] else {'MILK'}
    if set(sources) != expected_sources or {r['source'] for r in val} != {'MILK'}:
        raise ValueError('Smoke source coverage is incomplete')
    if record['config']['include_isic'] and record['config']['modality'] == 'pair':
        if not any(r['source'] == 'ISIC2019' and int(r['has_clinical']) == 0 for r in train):
            raise ValueError('Paired ISIC smoke must exercise missing clinical images')
    last, best = load_checkpoint(folder / 'last.pt'), load_checkpoint(folder / 'best.pt')
    if last['signature'] != record['signature'] or best['signature'] != record['signature'] or best['epoch'] != last['best_epoch']:
        raise ValueError('Inconsistent smoke checkpoints')
    epochs = 1 if expected == 'stopped' else 2
    if record['epochs_completed'] != epochs or last['epoch'] != epochs:
        raise ValueError('Unexpected smoke epoch count')
    for history in last['history']:
        generator = torch.Generator().manual_seed(record['config']['seed'] + history['epoch'] - 1)
        order = torch.randperm(len(train), generator=generator).tolist()
        if history['train']['sample_order_sha256'] != fingerprint([train[i]['sample_id'] for i in order]):
            raise ValueError('Unexpected training sample order')
    if expected == 'stopped':
        if record['status'] != 'partial' or record['stop_reason'] != 'smoke_checkpoint_stop' or record['resume_from_epoch'] != 0:
            raise ValueError('Smoke did not stop at its first checkpoint')
    else:
        if record['status'] != 'completed' or record['resume_from_epoch'] != 1:
            raise ValueError('Smoke did not resume from epoch one')
        stopped = json.loads((folder / 'smoke_stopped_verification.json').read_text(encoding='utf-8'))
        if stopped['signature'] != record['signature'] or stopped['status'] != 'passed':
            raise ValueError('Missing matching stop verification')
        _, _, saved = load_evaluation_data(folder / 'validation_labels.csv', folder / 'predictions.csv',
                                          class_names=NINE_CLASSES, id_column='sample_id')
        _, _, reloaded = load_evaluation_data(folder / 'validation_labels.csv', folder / 'reloaded_predictions.csv',
                                             class_names=NINE_CLASSES, id_column='sample_id')
        np.testing.assert_allclose(reloaded, saved, rtol=1e-4, atol=1e-5)
    result = {'status': 'passed', 'stage': expected, 'signature': record['signature'],
              'epochs_completed': epochs, 'resume_from_epoch': record['resume_from_epoch'],
              'data_coverage': record['data_coverage'], 'purpose': 'smoke'}
    write_json(folder / f'smoke_{expected}_verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expect', choices=('stopped', 'completed'), required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.run, args.expect), indent=2))
