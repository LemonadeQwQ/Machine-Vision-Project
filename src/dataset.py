"""读取统一 DATA；保持单张图片和整数标签的模型输入接口。"""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .Image_tensor import load_image_tensor
    from .class_schema import CLASSES, validate_class_order
else:
    from Image_tensor import load_image_tensor
    from class_schema import CLASSES, validate_class_order

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "DATA" / "splits" / "train.csv"
VAL_CSV = ROOT / "DATA" / "splits" / "val.csv"
METADATA_CSV = ROOT / "DATA" / "images.csv"
CLASS_NAMES = CLASSES
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
MODALITIES = ("clinical", "dermoscopic")


def read_csv(path, required_columns):
    """读取 CSV；格式错误在训练前报出，避免静默读错列。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path} 包含重复列名")
        missing = set(required_columns) - set(columns)
        if missing:
            raise ValueError(f"{path} 缺少必要列: {sorted(missing)}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: CSV 列数与表头不一致")
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV 中没有样本: {path}")
    return rows


def read_image_pairs(metadata_csv):
    """从 images.csv 建立病灶配对表；路径相对于该 CSV 所在的 DATA。"""
    metadata_csv = Path(metadata_csv).resolve()
    rows = read_csv(metadata_csv, ["lesion_id", "modality", "image_id", "image_path"])
    pairs = {}
    seen_images = set()
    for row in rows:
        lesion_id, image_id, modality = (row[k] for k in ("lesion_id", "image_id", "modality"))
        if not lesion_id.startswith("L") or not lesion_id[1:].isascii() or not lesion_id[1:].isdigit():
            raise ValueError(f"无效统一病灶编号: {lesion_id!r}")
        if modality not in MODALITIES:
            raise ValueError(f"未知图片模态: {modality!r} ({lesion_id})")
        if image_id != f"{lesion_id}_{modality}":
            raise ValueError(f"图片编号与病灶/模态不一致: {image_id!r}")
        expected_path = f"images/{lesion_id}/{image_id}.jpg"
        if row["image_path"] != expected_path:
            raise ValueError(f"图片路径与病灶/模态不一致: {row['image_path']!r}")
        pair = pairs.setdefault(lesion_id, {})
        if modality in pair or image_id in seen_images:
            raise ValueError(f"图片编号或模态重复: {image_id}")
        pair[modality] = {
            "image_id": image_id,
            "image_path": metadata_csv.parent / row["image_path"],
            "source_dataset": row.get("source_dataset", ""),
        }
        seen_images.add(image_id)
    for lesion_id, pair in pairs.items():
        if set(pair) != set(MODALITIES):
            raise ValueError(f"临床图/皮肤镜配对不完整: {lesion_id}")
        if len({image["source_dataset"] for image in pair.values()}) > 1:
            raise ValueError(f"两张配对图片来源不一致: {lesion_id}")
    return pairs


class Milk10kDataset(Dataset):
    """一个样本对应一个病灶的所选模态；保留原类名以兼容已有调用。"""

    def __init__(self, split_csv, training=False,
                 image_type="dermoscopic", metadata_csv=METADATA_CSV, *,
                 _image_pairs=None, class_names=CLASSES):
        # 旧临床图参数仍可用，但统一目录使用 clinical 名称。
        if image_type == "clinical: close-up":
            image_type = "clinical"
        if image_type not in MODALITIES:
            raise ValueError(f"不支持的 image_type: {image_type}")
        self.class_names = validate_class_order(class_names)
        class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.training = training
        self.image_type = image_type
        split_rows = read_csv(split_csv, ["lesion_id", "label"])
        pairs = read_image_pairs(metadata_csv) if _image_pairs is None else _image_pairs

        one_hot_columns = set(CLASS_NAMES).intersection(split_rows[0])
        if one_hot_columns and one_hot_columns != set(self.class_names):
            raise ValueError(f"{split_csv} 的 one-hot 类别列不完整")
        self.samples = []
        seen_lesions = set()
        for row in split_rows:
            lesion_id, label = row["lesion_id"], row["label"]
            if not lesion_id or lesion_id != lesion_id.strip() or lesion_id in seen_lesions:
                raise ValueError(f"split 中 lesion_id 为空、有多余空格或重复: {lesion_id!r}")
            if label not in class_to_idx:
                raise ValueError(f"未知类别标签: {label!r}")
            if lesion_id not in pairs:
                raise ValueError(f"images.csv 中找不到病灶: {lesion_id}")
            pair = pairs[lesion_id]
            # 即使本轮只读一种模态，也确认该病例的两张图片都存在。
            for image in pair.values():
                if not image["image_path"].is_file():
                    raise FileNotFoundError(f"配对图片不存在: {image['image_path']}")
            if one_hot_columns:
                values = [float(row[name]) for name in self.class_names]
                expected = [float(name == label) for name in self.class_names]
                if values != expected:
                    raise ValueError(f"one-hot 与 label 不一致: {lesion_id}")
            image = pair[image_type]
            self.samples.append({
                "lesion_id": lesion_id,
                "image_id": image["image_id"],
                "image_path": image["image_path"],
                "label": label,
                "label_id": class_to_idx[label],
            })
            seen_lesions.add(lesion_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tensor = load_image_tensor(sample["image_path"], training=self.training)
        return tensor, sample["label_id"]


def seed_worker(worker_id):
    """使用多个读取进程时，为每个进程设置 Python/NumPy 随机种子。"""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloaders(batch_size=32, num_workers=0, seed=42,
                       train_csv=TRAIN_CSV, val_csv=VAL_CSV,
                       image_type="dermoscopic", metadata_csv=METADATA_CSV):
    """创建训练和验证 loader；训练开始前调用一次，后续每个 epoch 复用。"""
    # 每一个batch 默认有32个样本，设置随机种子，目前默认42，目前只用主进程所以num-worker = 0

    # saftty check：由于参数输入可调节，所以需要看每一个参数是否在安全范围内
    if batch_size < 1 or num_workers < 0 or not 0 <= seed < 2**32:
        raise ValueError("batch_size 必须为正数，num_workers 非负，seed 在 [0, 2**32)")

    # 配对表只读取一次；两个 Dataset 共享它来构造各自的样本列表。
    image_pairs = read_image_pairs(metadata_csv)
    train_dataset = Milk10kDataset(
        train_csv, training=True, image_type=image_type, metadata_csv=metadata_csv, _image_pairs=image_pairs)
    val_dataset = Milk10kDataset(
        val_csv, training=False, image_type=image_type, metadata_csv=metadata_csv, _image_pairs=image_pairs)

    # safty check，防止training 和 val set的data出现相同的，也就是数据泄漏
    train_ids = {sample["lesion_id"] for sample in train_dataset.samples}
    val_ids = {sample["lesion_id"] for sample in val_dataset.samples}
    if train_ids & val_ids:
        raise ValueError("train/val 存在相同 lesion_id")

    # 将python，np，torch的三个随机操作通过我们设定的seed进行设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #当前图片旋转、翻转的随机性

    # 训练和验证使用独立 generator，其中种子得不一样
    train_generator = torch.Generator().manual_seed(seed) #训练图片排列顺序的随机性
    val_generator = torch.Generator().manual_seed(seed + 1) #验证 loader 自己的随机状态


    # 创建pytorch的 DATALOADER类，training和val分开创建
    # 这里就是batch ！！！
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # 每一次顺序打乱
        num_workers=num_workers, # 目前默认为主程序
        drop_last=False, # 保留 剩余minibatch的data
        worker_init_fn=seed_worker,
        generator=train_generator,  # 使用我们创建的训练随机generator
    )
    # 和training 同理
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    return train_loader, val_loader
