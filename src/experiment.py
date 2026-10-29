"""新 ResNet 协议的配置、训练、检查点与预测；默认要求 CUDA 计算节点。"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import torchvision
import yaml

from .class_schema import NINE_CLASSES
from .data.common import ROOT, digest, fingerprint, read_csv, write_csv, write_json
from .data.manifest import ManifestDataset, assert_disjoint, fit_metadata, validate_rows
from .evaluate import calculate_metrics, evaluate_files
from .losses import classification_loss
from .models.fusion import ResNetExperiment

PREPROCESS = {'version': 1, 'resize': 'aspect-preserving pad to square; bilinear; black padding',
              'normalization': 'ImageNet RGB mean/std', 'geometry': 'train: shared H/V flip p=.5 and rotation U(-20,20); val: none'}


def configuration(path, overrides=None):
    base = yaml.safe_load((ROOT / 'configs/experiments/base.yaml').read_text(encoding='utf-8'))
    supplied = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    unknown = supplied.keys() - (base.keys() | {'experiment'})
    if unknown:
        raise ValueError(f'Unknown configuration keys: {unknown}')
    base.update(supplied)
    if overrides:
        if overrides.keys() - base.keys():
            raise ValueError('Unknown override')
        base.update(overrides)
    if base['protocol'] != 'resnet9-v1' or base.get('experiment') not in [f'E{i:02}' for i in range(10)]:
        raise ValueError('Unknown protocol / experiment')
    if base['modality'] not in ('derm', 'clinical', 'pair') or (base['metadata'] and base['modality'] != 'pair'):
        raise ValueError('Invalid model configuration')
    for field in ('seed', 'max_epochs', 'patience', 'image_size', 'batch_size', 'cpu_threads', 'prefetch_factor'):
        if not isinstance(base[field], int) or base[field] < (0 if field == 'seed' else 1):
            raise ValueError(f'Invalid {field}')
    if base['workers'] < 0 or base['amp'] not in ('bf16', 'off'):
        raise ValueError('Invalid compute setting')
    for field in ('weighted', 'include_isic', 'metadata', 'augment', 'channels_last', 'tf32'):
        if not isinstance(base[field], bool):
            raise ValueError(f'{field} must be a boolean')
    return base


def resolve(value):
    return (ROOT / value).resolve()


def image_root(config):
    # SLURM_TMPDIR 随作业变化，作为运行时位置，不进入实验配置指纹。
    return Path(os.environ['CSC490_DATA_ROOT']).resolve() if os.getenv('CSC490_DATA_ROOT') else resolve(config['data_root'])


def code_record():
    files = {p.relative_to(ROOT).as_posix(): digest(p) for folder in ('src', 'scripts')
             for p in sorted((ROOT / folder).rglob('*.py'))}
    proc = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    commit = proc.stdout.strip() if proc.returncode == 0 else None
    if commit is None:
        source = ROOT / 'reports/preparation/source_version.json'
        if source.exists():
            commit = json.loads(source.read_text(encoding='utf-8')).get('git_base_commit')
    return {'git_commit': commit, 'commit_note': 'base commit; source_sha256 and per-file hashes identify actual working files',
            'source_sha256': fingerprint(files), 'files': files}


def preflight(config, fold, *, verify_images=True):
    if fold not in range(5):
        raise ValueError('Fold must be 0..4')
    directory = resolve(config['manifests'])
    version = json.loads((directory / 'version.json').read_text(encoding='utf-8'))
    if version['protocol'] != config['protocol']:
        raise ValueError('Manifest protocol mismatch')
    for file, expected in version['files'].items():
        if digest(directory / file) != expected:
            raise ValueError(f'Manifest modified: {file}')
    train = read_csv(directory / f'folds/fold_{fold}/train.csv')
    val = read_csv(directory / f'folds/fold_{fold}/val.csv')
    holdout = read_csv(directory / 'milk_holdout.csv')
    if any(r['source'] != 'MILK' for r in val + holdout):
        raise ValueError('Validation must be MILK only')
    if config['include_isic']:
        train += read_csv(directory / 'isic_train.csv')
    assert_disjoint(train, val, holdout)
    for rows in (train, val):
        validate_rows(rows, image_root(config) if verify_images else None)
    weights = json.loads((ROOT / 'configs/schema/pretrained_weights.json').read_text())
    if digest(resolve(config['weights'])) != weights['sha256']:
        raise ValueError('ImageNet weights mismatch')
    return train, val, {'manifest_sha256': digest(directory / 'version.json'),
                        'weights_sha256': weights['sha256'], 'fold': fold}


def configure_runtime(config, device):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: run this on an allocated GPU node')
        torch.cuda.manual_seed_all(config['seed'])
        if config['amp'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unsupported; make a separately recorded config with amp=off')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = config['tf32']
        torch.backends.cudnn.allow_tf32 = config['tf32']


def loader(rows, config, encoder, train, device):
    dataset = ManifestDataset(rows, image_root(config), modality=config['modality'], train=train,
        augment=config['augment'], size=config['image_size'], metadata_encoder=encoder, seed=config['seed'], verify=False)
    extra = {'persistent_workers': True, 'prefetch_factor': config['prefetch_factor']} if config['workers'] else {}
    # Worker base seeds consume DataLoader.generator only when workers are created.
    # Keep shuffle independent so persistent workers and a resumed process agree.
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(config['seed'])) if train else None
    return DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=config['workers'],
        pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(config['seed']), **extra)


def smoke_rows(rows, per_group):
    """Cover each available source / class / modality mask without changing splits."""
    counts, selected = Counter(), []
    for row in rows:
        key = (row['source'], row['label'], int(row['has_derm']), int(row['has_clinical']))
        if counts[key] < per_group:
            selected.append(row)
            counts[key] += 1
    return selected


def coverage(rows):
    return {'sources': dict(Counter(r['source'] for r in rows)),
            'classes': dict(Counter(r['label'] for r in rows)),
            'modality_masks': dict(Counter(f"{int(r['has_derm'])},{int(r['has_clinical'])}" for r in rows))}


def to_device(batch, device, channels_last):
    result = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    if channels_last:
        for k in ('derm', 'clinical'):
            result[k] = result[k].contiguous(memory_format=torch.channels_last)
    return result


def epoch(model, data, criterion, config, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    started = time.perf_counter()
    total_loss, total_weight, ids, truth, probabilities = 0., 0., [], [], []
    for step, batch in enumerate(data):
        if max_batches is not None and step >= max_batches:
            break
        batch = to_device(batch, device, config['channels_last'])
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and config['amp'] == 'bf16' else nullcontext()
        with torch.set_grad_enabled(training), autocast:
            logits = model(batch)
            loss = criterion(logits.float(), batch['label'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss')
        if training:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        weight = (criterion.weight[batch['label']].sum().item() if criterion.weight is not None else len(batch['label']))
        total_loss += loss.item() * weight
        total_weight += weight
        ids.extend(batch['sample_id'])
        truth.extend(batch['label'].detach().cpu().tolist())
        probabilities.extend(logits.detach().float().softmax(1).cpu().tolist())
    if not ids:
        raise ValueError('No batches processed')
    metrics, _, _, _, _ = calculate_metrics(np.asarray(truth), np.asarray(probabilities), class_names=NINE_CLASSES)
    seconds = time.perf_counter() - started
    return {'loss': total_loss / total_weight, **metrics, 'seconds': seconds, 'samples_per_second': len(ids) / seconds,
            'sample_order_sha256': fingerprint(ids),
            'ids': ids, 'labels': truth, 'probabilities': probabilities}


def predictions(path, result):
    write_csv(path, [dict(zip(['sample_id', *NINE_CLASSES], [key, *probs])) for key, probs in zip(result['ids'], result['probabilities'])])


def save_checkpoint(path, value):
    temp = Path(path).with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def load_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('format') != 'resnet9-checkpoint-v1' or value.get('class_order') != list(NINE_CLASSES) or value.get('preprocess') != PREPROCESS:
        raise ValueError('Checkpoint schema / preprocessing mismatch')
    return value


def compact(result):
    return {k: v for k, v in result.items() if k not in ('ids', 'labels', 'probabilities')}


def run(config, fold, output, *, smoke=False, resume=False, device_name='cuda', max_epochs_override=None,
        refit_summary=None, smoke_epochs=1, smoke_stop_after_epoch=None):
    if smoke_epochs not in (1, 2) or (not smoke and (smoke_epochs != 1 or smoke_stop_after_epoch is not None)):
        raise ValueError('Smoke epoch controls require --smoke and one or two epochs')
    if smoke_stop_after_epoch is not None and not 1 <= smoke_stop_after_epoch < smoke_epochs:
        raise ValueError('Smoke stop must precede the final smoke epoch')
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'Output exists; use a new directory or --resume: {output}')
    train, val, inputs = preflight(config, fold)
    phase = 'development_refit' if refit_summary else 'cross_validation'
    if refit_summary:
        if smoke or max_epochs_override is not None:
            raise ValueError('Refit cannot override the predeclared epoch rule')
        if refit_summary.get('status') != 'completed' or refit_summary.get('fold_count') != 5 or refit_summary.get('purpose') != 'formal':
            raise ValueError('Refit requires a completed formal five-fold summary')
        if refit_summary['config'] != config or refit_summary['manifest_sha256'] != inputs['manifest_sha256']:
            raise ValueError('Refit data / config differs from CV')
        if not 1 <= refit_summary['refit_epochs'] <= config['max_epochs']:
            raise ValueError('Invalid refit epoch count')
        train = read_csv(resolve(config['manifests']) / 'milk_dev.csv')
        if config['include_isic']:
            train += read_csv(resolve(config['manifests']) / 'isic_train.csv')
        validate_rows(train, image_root(config))
        assert_disjoint(train, [], read_csv(resolve(config['manifests']) / 'milk_holdout.csv'))
        val = []
        config = {**config, 'max_epochs': int(refit_summary['refit_epochs'])}
        inputs = {**inputs, 'fold': None, 'refit_summary_sha256': fingerprint(refit_summary)}
    if smoke:
        # 短检仍覆盖九类，不读取固定 holdout；结果明确标 smoke，不计正式实验。
        train = smoke_rows(train, 4)
        val = smoke_rows(val, 2)
    if max_epochs_override is not None:
        config = {**config, 'max_epochs': max_epochs_override}
    device = torch.device(device_name)
    configure_runtime(config, device)
    code = code_record()
    if refit_summary and refit_summary['code_sha256'] != code['source_sha256']:
        raise ValueError('Refit source differs from the CV source')
    signature = fingerprint({'config': config, 'inputs': inputs, 'code': code['source_sha256'],
                             'smoke': smoke, 'smoke_epochs': smoke_epochs if smoke else None})
    encoder = fit_metadata(train) if config['metadata'] else None
    model = ResNetExperiment(config['modality'], encoder.output_dim if encoder else 0, resolve(config['weights'])).to(device)
    if config['channels_last']:
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.optimizer_groups(config['head_lr'], config['backbone_lr']), weight_decay=config['weight_decay'])
    criterion, loss_record = classification_loss(train, config['weighted'], device)
    train_loader = loader(train, config, encoder, True, device)
    val_loader = loader(val, config, encoder, False, device) if val else None
    start_epoch, best_score, best_epoch, history = 0, -1., 0, []
    if resume:
        checkpoint = load_checkpoint(output / 'last.pt')
        if checkpoint['signature'] != signature:
            raise ValueError('Resume rejected: config / data / code differs')
        best_checkpoint = load_checkpoint(output / 'best.pt')
        if best_checkpoint['epoch'] != checkpoint['best_epoch'] or best_checkpoint['signature'] != signature:
            raise ValueError('Inconsistent last/best checkpoints; do not silently resume')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, best_score, best_epoch = checkpoint['epoch'], checkpoint['best_score'], checkpoint['best_epoch']
        history = checkpoint['history']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.running'
    # 作业内退出时删除；若节点被杀，使用者核对无作业后手动移除这个标记。
    with lock.open('x') as f:
        f.write(f'pid={os.getpid()} slurm={os.getenv("SLURM_JOB_ID", "")}\n')
    record = {'protocol': config['protocol'], 'experiment': config['experiment'], 'fold': None if refit_summary else fold, 'phase': phase,
        'purpose': 'smoke' if smoke else 'formal', 'status': 'running', 'signature': signature,
        'config': config, 'inputs': inputs, 'code': code, 'loss': loss_record, 'preprocess': PREPROCESS,
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'data_root_resolved': str(image_root(config))},
        'started_utc': datetime.now(timezone.utc).isoformat(), 'resume_from_epoch': start_epoch,
        'train_samples': len(train), 'val_samples': len(val), 'stop_reason': None,
        'smoke_epochs': smoke_epochs if smoke else None, 'data_coverage': {'train': coverage(train), 'val': coverage(val)}}
    write_json(output / 'run.json', record)
    write_json(output / 'config.json', config)
    write_json(output / 'metadata.json', encoder.to_dict() if encoder else None)
    if val:
        write_csv(output / 'validation_labels.csv', [{'sample_id': r['sample_id'], 'label': r['label']} for r in val])
    if smoke:
        write_csv(output / 'smoke_train.csv', train)
        write_csv(output / 'smoke_val.csv', val)
    try:
        total_epochs = smoke_epochs if smoke else config['max_epochs']
        for index in range(start_epoch, total_epochs):
            if not refit_summary and best_epoch and index - best_epoch >= config['patience']:
                record['stop_reason'] = 'early_stopping'
                break
            train_loader.dataset.epoch.fill_(index)
            train_loader.sampler.generator.manual_seed(config['seed'] + index)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            train_result = epoch(model, train_loader, criterion, config, device, optimizer)
            val_result = epoch(model, val_loader, criterion, config, device) if val_loader else None
            current = index + 1
            improved = True if refit_summary else val_result['macro_f1'] > best_score
            if improved:
                best_score, best_epoch = (-1. if refit_summary else val_result['macro_f1']), current
            history.append({'epoch': current, 'train': compact(train_result), 'val': compact(val_result) if val_result else None,
                'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None})
            value = {'format': 'resnet9-checkpoint-v1', 'signature': signature, 'config': config, 'inputs': inputs,
                'class_order': list(NINE_CLASSES), 'preprocess': PREPROCESS, 'metadata': encoder.to_dict() if encoder else None,
                'phase': phase, 'refit_complete': bool(refit_summary and current == total_epochs),
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current,
                'best_score': best_score, 'best_epoch': best_epoch, 'history': history,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
            save_checkpoint(output / 'last.pt', value)
            if improved:
                save_checkpoint(output / 'best.pt', value)
                if val_result:
                    predictions(output / 'predictions.csv', val_result)
            write_json(output / 'history.json', history)
            score_text = f"val_f1={val_result['macro_f1']:.4f} best={best_score:.4f}" if val_result else 'fixed-epoch development refit'
            print(f"{config['experiment']} fold={fold} epoch={current}/{total_epochs} {score_text}", flush=True)
            if smoke_stop_after_epoch is not None and current == smoke_stop_after_epoch:
                record['stop_reason'] = 'smoke_checkpoint_stop'
                break
        record['status'] = 'partial' if record['stop_reason'] == 'smoke_checkpoint_stop' else 'completed'
        record['stop_reason'] = record['stop_reason'] or ('smoke_limit' if smoke else 'max_epochs')
        record.update(best_epoch=best_epoch, best_macro_f1=None if refit_summary else best_score, epochs_completed=len(history))
        if val and record['status'] == 'completed':
            evaluate_files(output / 'validation_labels.csv', output / 'predictions.csv', output / 'evaluation',
                           model_name=f"{config['experiment']} fold {fold}", class_names=NINE_CLASSES, id_column='sample_id', overwrite=resume)
    except BaseException as exc:
        record.update(status='partial' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'run.json', record)
        lock.unlink(missing_ok=True)
        # Exception tracebacks can retain loaders after an interrupted run.
        # Release persistent worker processes before another fold/resume starts.
        for data in (train_loader, val_loader):
            iterator = getattr(data, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()
                data._iterator = None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--weights')
    parser.add_argument('--manifests')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-epochs', type=int, choices=(1, 2), default=1)
    parser.add_argument('--smoke-stop-after-epoch', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k in ('data_root', 'weights', 'manifests', 'workers') and v is not None}
    config = configuration(args.config, overrides)
    if args.preflight_only:
        train, val, info = preflight(config, args.fold)
        print(json.dumps({'train': len(train), 'val': len(val), **info}, indent=2))
    else:
        run(config, args.fold, args.output, smoke=args.smoke, resume=args.resume,
            smoke_epochs=args.smoke_epochs, smoke_stop_after_epoch=args.smoke_stop_after_epoch)


if __name__ == '__main__':
    main()
