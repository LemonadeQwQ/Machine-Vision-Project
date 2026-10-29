"""清单驱动的字典 batch；缺失模态不打开不存在的文件。"""
import random
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ..class_schema import NINE_CLASSES
from ..metadata_features import MetadataEncoder
from .common import image_path, read_csv, unique


def validate_rows(rows, root=None):
    if not rows:
        raise ValueError('Empty manifest')
    unique(rows, 'sample_id')
    for row in rows:
        if row['label'] not in NINE_CLASSES or int(row['label_id']) != NINE_CLASSES.index(row['label']):
            raise ValueError('Label / schema mismatch')
        if not row['group_id'] or not row['sample_id'].startswith(row['source'] + ':'):
            raise ValueError('Invalid group / source ID')
        available = []
        for name in ('derm', 'clinical'):
            flag = int(row[f'has_{name}'])
            if flag not in (0, 1) or bool(row[f'{name}_path']) != bool(flag):
                raise ValueError('Path / modality mask mismatch')
            if root is not None and flag:
                image_path(root, row[f'{name}_path'])
            available.append(flag)
        if not any(available):
            raise ValueError('Sample has no images')
    return rows


def assert_disjoint(train, val, holdout=()):
    for a, b in ((train, val), (train, holdout), (val, holdout)):
        for key in ('sample_id', 'group_id'):
            if {r[key] for r in a} & {r[key] for r in b}:
                raise ValueError(f'Train / validation / holdout overlap: {key}')


class ManifestDataset(Dataset):
    def __init__(self, rows, root, *, modality='pair', train=False, augment=True, size=224,
                 metadata_encoder=None, seed=42, epoch=0, verify=True):
        self.rows = validate_rows(rows, root if verify else None)
        self.root = Path(root)
        self.modality, self.train, self.augment = modality, train, augment
        self.size, self.encoder, self.seed = size, metadata_encoder, seed
        self.epoch = torch.tensor(epoch, dtype=torch.int64).share_memory_()
        if modality not in ('derm', 'clinical', 'pair'):
            raise ValueError('Unknown modality')
        if modality != 'pair' and any(int(r[f'has_{modality}']) != 1 for r in rows):
            raise ValueError(f'Missing required {modality} image')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # 按 epoch / 样本单独播种，不依赖 worker 的运行顺序；允许可靠的 epoch 边界恢复。
        rng = random.Random(self.seed + int(self.epoch.item()) * 1_000_003 + index)
        hflip, vflip, angle = rng.random() < .5, rng.random() < .5, rng.uniform(-20, 20)
        result = {'sample_id': row['sample_id'], 'label': torch.tensor(int(row['label_id']), dtype=torch.long),
                  'modality_mask': torch.tensor([int(row['has_derm']), int(row['has_clinical'])], dtype=torch.float32)}
        for modality in ('derm', 'clinical'):
            if self.modality not in ('pair', modality) or not int(row[f'has_{modality}']):
                result[modality] = torch.zeros(3, self.size, self.size)
                continue
            with Image.open(self.root / row[f'{modality}_path']) as im:
                im = ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'), (self.size, self.size),
                                  method=Image.Resampling.BILINEAR, color=(0, 0, 0))
                if self.train and self.augment:
                    if hflip:
                        im = ImageOps.mirror(im)
                    if vflip:
                        im = ImageOps.flip(im)
                    im = im.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                result[modality] = TF.normalize(TF.to_tensor(im), [.485, .456, .406], [.229, .224, .225])
        result['metadata'] = torch.tensor(self.encoder.transform(row) if self.encoder else [], dtype=torch.float32)
        result['metadata_mask'] = torch.tensor([bool(row[k]) for k in ('age_years', 'sex', 'anatomical_region', 'skin_tone_class')], dtype=torch.float32)
        return result


def fit_metadata(rows):
    return MetadataEncoder.fit(rows, include_skin_tone=True)
