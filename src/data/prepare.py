"""从官方 CSV 构建新协议清单；保留原始病灶编号，不猜旧 L 编号映射。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

from sklearn.model_selection import StratifiedGroupKFold

from ..class_schema import NINE_CLASSES
from .common import ROOT, digest, image_path, read_csv, unique, write_csv, write_json
from .deduplicate import UnionFind

FIELDS = ['sample_id', 'source', 'lesion_id', 'group_id', 'label', 'label_id', 'derm_path',
          'clinical_path', 'has_derm', 'has_clinical', 'metadata_key', 'split',
          'age_years', 'sex', 'anatomical_region', 'skin_tone_class', 'derm_image_id', 'clinical_image_id']


def onehot(row, columns):
    values = [float(row[c]) for c in columns]
    if any(x not in (0., 1.) for x in values) or sum(values) != 1:
        raise ValueError(f'Invalid one-hot label: {row}')
    return columns[values.index(1.)]


def mapping_table():
    schema = json.loads((ROOT / 'configs/schema/classes_9.json').read_text(encoding='utf-8'))
    if schema['class_names'] != list(NINE_CLASSES):
        raise ValueError('Schema differs from evaluator class order')
    rows = read_csv(ROOT / 'configs/schema/label_mapping.csv')
    result = {}
    for row in rows:
        key = (row['source'], row['source_label'])
        if key in result or row['action'] not in ('keep', 'merge', 'exclude') or not row['reference']:
            raise ValueError('Unresolved / duplicate label mapping')
        if row['action'] != 'exclude' and row['target_label'] not in NINE_CLASSES:
            raise ValueError('Invalid target label')
        result[key] = row['target_label'] or None
    return result


def group_folds(dev, count=5, seed=42):
    unique(dev, 'sample_id')
    for label in NINE_CLASSES:
        if len({r['group_id'] for r in dev if r['label'] == label}) < count:
            raise ValueError(f'Fewer than {count} groups for {label}')
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    folds, observed = [], []
    for train_idx, val_idx in splitter.split(dev, [r['label'] for r in dev], [r['group_id'] for r in dev]):
        train = [dict(dev[i], split='train') for i in train_idx]
        val = [dict(dev[i], split='val') for i in val_idx]
        if {r['group_id'] for r in train} & {r['group_id'] for r in val}:
            raise ValueError('Group leakage')
        for subset in (train, val):
            if {r['label'] for r in subset} != set(NINE_CLASSES):
                raise ValueError('A fold lacks one or more classes; revise split explicitly')
        observed.extend(r['sample_id'] for r in val)
        folds.append((train, val))
    if Counter(observed) != Counter(r['sample_id'] for r in dev):
        raise ValueError('OOF coverage failure')
    return folds


def prepare(data_root, output, cache):
    data_root, output, cache = Path(data_root), Path(output), Path(cache)
    mapping = mapping_table()
    milk_meta = read_csv(data_root / 'MILK10k/MILK10k_Training_Metadata.csv')
    milk_gt = read_csv(data_root / 'MILK10k/MILK10k_Training_GroundTruth.csv')
    milk_cols = list(milk_gt[0])[1:]
    isic_meta = unique(read_csv(data_root / 'ISIC2019/ISIC_2019_Training_Metadata.csv'), 'image')
    isic_gt = read_csv(data_root / 'ISIC2019/ISIC_2019_Training_GroundTruth.csv')
    isic_cols = list(isic_gt[0])[1:]
    inventory = []
    for source in ('MILK10k', 'ISIC2019'):
        inventory.extend(read_csv(data_root / '_audit' / f'{source}_Training_images.csv'))
    images = unique(inventory, 'image_id')
    # 相似性审计必须与当前官方清单相配，不能静默跳过或用别的数据缓存。
    audit = json.loads((cache / 'similarity_audit.json').read_text())
    from .common import fingerprint
    expected = fingerprint({'inputs': [digest(data_root / '_audit' / f'{s}_Training_images.csv') for s in ('MILK10k', 'ISIC2019')],
                            'method': 'phash64-d4-v1', 'radius': 4})
    if audit['signature'] != expected:
        raise ValueError('Stale similarity audit')
    uf = UnionFind(images)
    for edge in read_csv(cache / 'similarity_candidates.csv'):
        uf.union(edge['image_a'], edge['image_b'])
    hashes = defaultdict(list)
    for r in inventory:
        hashes[r['sha256']].append(r['image_id'])
    for members in hashes.values():
        for key in members[1:]:
            uf.union(members[0], key)
    # 同一真实病灶的两个模态／多张照片必须属于一个组。
    milk_pairs = defaultdict(dict)
    for r in milk_meta:
        modality = {'dermoscopic': 'derm', 'clinical: close-up': 'clinical'}[r['image_type']]
        if modality in milk_pairs[r['lesion_id']]:
            raise ValueError('Repeated MILK modality')
        milk_pairs[r['lesion_id']][modality] = r
    for pair in milk_pairs.values():
        if set(pair) != {'derm', 'clinical'}:
            raise ValueError('MILK pair missing')
        uf.union(pair['derm']['isic_id'], pair['clinical']['isic_id'])
    isic_lesions = defaultdict(list)
    for r in isic_meta.values():
        isic_lesions[r['lesion_id'] or r['image']].append(r['image'])
    for members in isic_lesions.values():
        for key in members[1:]:
            uf.union(members[0], key)
    source_splits = {}
    for name, part in [('train', 'dev'), ('val', 'holdout')]:
        for r in read_csv(ROOT / f'DATA/PROCESSED_DATA/{name}_split.csv'):
            if r['lesion_id'] in source_splits:
                raise ValueError('Original split overlap')
            source_splits[r['lesion_id']] = (part, r['label'])
    if set(source_splits) != {r['lesion_id'] for r in milk_gt}:
        raise ValueError('Original split coverage mismatch')
    excluded, milk = [], []
    def exclude(source, sample, reason):
        excluded.append({'source': source, 'sample_id': f'{source}:{sample}', 'reason': reason})
    for r in sorted(milk_gt, key=lambda r: r['lesion_id']):
        key = r['lesion_id']
        raw = onehot(r, milk_cols)
        part, old_label = source_splits[key]
        if raw != old_label:
            raise ValueError('Original split label mismatch')
        label = mapping[('MILK', raw)]
        if label is None:
            exclude('MILK', key, 'schema_exclusion')
            continue
        pair = milk_pairs[key]
        derm, clinical = pair['derm'], pair['clinical']
        for field in ('age_approx', 'sex', 'site', 'skin_tone_class'):
            if derm[field] != clinical[field]:
                raise ValueError('Conflicting pair metadata')
        sample = f'MILK:{key}'
        milk.append(dict(zip(FIELDS, [sample, 'MILK', key, uf.find(derm['isic_id']), label,
              NINE_CLASSES.index(label), images[derm['isic_id']]['relative_path'], images[clinical['isic_id']]['relative_path'],
              1, 1, sample, part, derm['age_approx'], derm['sex'], derm['site'], derm['skin_tone_class'], derm['isic_id'], clinical['isic_id']])))
    holdout = [r for r in milk if r['split'] == 'holdout']
    holdout_groups = {r['group_id'] for r in holdout}
    dev = []
    for r in milk:
        if r['split'] == 'dev':
            if r['group_id'] in holdout_groups:
                exclude('MILK', r['lesion_id'], 'similarity_group_touches_fixed_holdout')
            else:
                dev.append(r)
    # 对跨源候选做全局保守排除，使所有 fold 共用同一个 ISIC 训练池。
    all_milk_groups = {uf.find(r['isic_id']) for r in milk_meta}
    isic_labels = {r['image']: mapping[('ISIC2019', onehot(r, isic_cols))] for r in isic_gt}
    group_labels = defaultdict(set)
    for key, label in isic_labels.items():
        group_labels[uf.find(key)].add(label)
    isic = []
    for r in sorted(isic_gt, key=lambda r: r['image']):
        key, label = r['image'], isic_labels[r['image']]
        group = uf.find(key)
        if label is None:
            exclude('ISIC2019', key, 'schema_exclusion')
        elif group in all_milk_groups:
            exclude('ISIC2019', key, 'similarity_group_touches_milk')
        elif len(group_labels[group]) > 1:
            exclude('ISIC2019', key, 'conflicting_labels_in_lesion_or_similarity_group')
        elif key != min(hashes[images[key]['sha256']]):
            exclude('ISIC2019', key, 'redundant_exact_image')
        else:
            meta = isic_meta[key]
            sample = f'ISIC2019:{key}'
            isic.append(dict(zip(FIELDS, [sample, 'ISIC2019', meta['lesion_id'] or key, group, label,
                NINE_CLASSES.index(label), images[key]['relative_path'], '', 1, 0, sample, 'train',
                meta['age_approx'], meta['sex'], meta['anatom_site_general'], '', key, ''])))
    for r in dev + holdout + isic:
        for field in ('derm_path', 'clinical_path'):
            if r[field]:
                image_path(data_root, r[field])
    folds = group_folds(dev)
    summary = {'protocol': 'resnet9-v1', 'patient_id_available': False,
        'group_basis': 'lesion IDs + exact SHA256 + pHash64 D4 distance <= 4 candidate components',
        'similarity_candidates': audit, 'counts': {'milk_dev': len(dev), 'milk_holdout': len(holdout), 'isic_train': len(isic)},
        'exclusions': dict(Counter(r['reason'] for r in excluded)),
        'fixed_holdout': 'Original val membership retained after nine-class filtering; related dev candidates excluded',
        'splitter': 'StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)',
        'limitations': ['Patient-level independence cannot be claimed', 'Perceptual candidates are not confirmed duplicates',
                       'Historical use means holdout is not a previously unseen external benchmark']}
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp)
        for name, rows in [('milk_dev', dev), ('milk_holdout', holdout), ('isic_train', isic)]:
            write_csv(stage / f'{name}.csv', rows, FIELDS)
        write_csv(stage / 'exclusions.csv', excluded, ['source', 'sample_id', 'reason'])
        counts = []
        for name, rows in [('milk_dev', dev), ('milk_holdout', holdout), ('isic_train', isic)]:
            counts.extend({'split': name, 'class': c, 'count': sum(r['label'] == c for r in rows)} for c in NINE_CLASSES)
        for fold, (train, val) in enumerate(folds):
            for part, rows in [('train', train), ('val', val)]:
                write_csv(stage / f'folds/fold_{fold}/{part}.csv', rows, FIELDS)
                counts.extend({'split': f'fold_{fold}_{part}', 'class': c, 'count': sum(r['label'] == c for r in rows)} for c in NINE_CLASSES)
        write_csv(stage / 'class_counts.csv', counts)
        write_json(stage / 'summary.json', summary)
        files = {p.relative_to(stage).as_posix(): digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        source_files = [ROOT / 'configs/schema/classes_9.json', ROOT / 'configs/schema/label_mapping.csv',
            ROOT / 'DATA/PROCESSED_DATA/train_split.csv', ROOT / 'DATA/PROCESSED_DATA/val_split.csv',
            cache / 'similarity_candidates.csv', *data_root.glob('MILK10k/*Training*.csv'), *data_root.glob('ISIC2019/*Training*.csv')]
        write_json(stage / 'version.json', {'protocol': 'resnet9-v1', 'files': files,
            'sources': {p.name: digest(p) for p in source_files}})
        # 同版本重建应逐字节一致；出现差异时要求新目录，防止覆盖已有实验所引用的清单。
        for p in stage.rglob('*'):
            if p.is_file():
                target = output / p.relative_to(stage)
                if target.exists() and digest(target) != digest(p):
                    raise FileExistsError(f'Manifest changed; choose a new --output directory: {target}')
        for p in stage.rglob('*'):
            if p.is_file():
                target = output / p.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(p.read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'DATASETS')
    parser.add_argument('--output', type=Path, default=ROOT / 'DATA/manifests')
    parser.add_argument('--cache', type=Path, default=ROOT / 'DATA/cache')
    args = parser.parse_args()
    prepare(args.data_root, args.output, args.cache)
