import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .Image_tensor import get_tensor
else:
    from Image_tensor import get_tensor


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "DATA" / "PROCESSED_DATA" / "train_split.csv"
VAL_CSV = ROOT / "DATA" / "PROCESSED_DATA" / "val_split.csv"
METADATA_CSV = ROOT / "DATA" / "RAW_DATA" / "MILK10k_Training_Metadata.csv"

# 需要分类的11个类别
CLASS_NAMES = (
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
)

# 将class name 映射为可以使用的INDEX
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}


def read_csv(path, required_columns):
    """读取 CSV，并检查必要的列。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} 缺少必要列: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV 中没有样本: {path}")
    return rows


class Milk10kDataset(Dataset):
    """每个样本是一个 lesion 的所选模态图片，以及对应的整数类别标签。"""
    # 数据图的CLASS


    # init，默认皮肤镜照片，后续使用可以改为临床图
    def __init__(self, split_csv, training=False,
                 image_type="dermoscopic", metadata_csv=METADATA_CSV):

        # 检测输入splict类型，如果不是皮肤镜或者临床则报错
        if image_type not in ("dermoscopic", "clinical: close-up"):
            raise ValueError(f"不支持的 image_type: {image_type}")
        # 将对应class对像是否是训练集，是哪一个splict data
        self.training = training
        self.image_type = image_type

        # 读取两个表：
        # split_rows 拥有每一个需要的病灶的id以及他的label
        # metadata_rows 拥有图片id，病灶id(用于连接两个表格)，还有其是皮肤镜还是临床图片的type
        split_rows = read_csv(split_csv, ["lesion_id", "label"])
        metadata_rows = read_csv(metadata_csv, ["lesion_id", "image_type", "isic_id"])

        # 建立病灶id -> 图片id 的字典
        lesion_to_image = {}
        seen_images = set()
        for row in metadata_rows:
            if row["image_type"].strip() != image_type:
                continue
            lesion_id = row["lesion_id"].strip()
            image_id = row["isic_id"].strip()
            if not lesion_id or not image_id:
                raise ValueError("metadata 中 lesion_id 或 isic_id 为空")
            if lesion_id in lesion_to_image:
                raise ValueError(f"同一 lesion 有多张 {image_type} 图片: {lesion_id}")
            if image_id in seen_images:
                raise ValueError(f"metadata 中所选模态的 image_id 重复: {image_id}")
            lesion_to_image[lesion_id] = image_id
            seen_images.add(image_id)

        # 将每一个样本的信息：{image_id，lesion_id，label，label_id}出存在samples里
        self.samples = []
        seen_lesions = set()
        for row in split_rows:
            lesion_id = row["lesion_id"].strip()
            label = row["label"].strip()
            if not lesion_id or lesion_id in seen_lesions:
                raise ValueError(f"split 中 lesion_id 为空或重复: {lesion_id!r}")
            if label not in CLASS_TO_IDX:
                raise ValueError(f"未知类别标签: {label!r}")
            if lesion_id not in lesion_to_image:
                raise ValueError(f"metadata 中找不到 lesion 的 {image_type} 图片: {lesion_id}")

            # 如果 split 有 one-hot 列，确认它们与 label 一致。
            if all(name in row for name in CLASS_NAMES):
                values = [float(row[name]) for name in CLASS_NAMES]
                expected = [1.0 if name == label else 0.0 for name in CLASS_NAMES]
                if values != expected:
                    raise ValueError(f"one-hot 与 label 不一致: {lesion_id}")

            self.samples.append({
                "lesion_id": lesion_id,
                "image_id": lesion_to_image[lesion_id],
                "label": label,
                "label_id": CLASS_TO_IDX[label],
            })
            seen_lesions.add(lesion_id)

    def __len__(self):
        # len(dataset) 返回该 split 中的样本数量。
        return len(self.samples)

    def __getitem__(self, index):
        # dataset[index] 只读取这一条样本对应的图片。
        sample = self.samples[index]
        image_tensor = get_tensor(sample["image_id"], training=self.training)
        return image_tensor, sample["label_id"]


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

    # 将训练集和验证集的对象class进行创建
    train_dataset = Milk10kDataset(
        train_csv, training=True, image_type=image_type, metadata_csv=metadata_csv)
    val_dataset = Milk10kDataset(
        val_csv, training=False, image_type=image_type, metadata_csv=metadata_csv)

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
