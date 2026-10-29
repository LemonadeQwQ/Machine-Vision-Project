"""Train one unified-data ResNet-50 baseline and export unified evaluation files.

Example (from repository root):
    python src/train.py --epochs 5 --batch-size 32 --output-dir runs/resnet50_derm_01

This is SINGLE-MODAL: dataset.py presently returns one image per
lesion. Clinical and dermoscopic files are paired in DATA; each run selects
one modality and keeps the existing model input interface.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import SequentialSampler

if __package__:
    from .class_schema import CLASSES, validate_class_order
    from .dataset import (
        CLASS_NAMES, METADATA_CSV, TRAIN_CSV, VAL_CSV, create_dataloaders,
    )
    from .engine import train_one_epoch, validate_one_epoch
    from .metrics import compute_metrics
    from .evaluate import evaluate_files
    from .models.resnet50 import build_resnet50
else:
    from class_schema import CLASSES, validate_class_order
    from dataset import CLASS_NAMES, METADATA_CSV, TRAIN_CSV, VAL_CSV, create_dataloaders
    from engine import train_one_epoch, validate_one_epoch
    from metrics import compute_metrics
    from evaluate import evaluate_files
    from models.resnet50 import build_resnet50

ROOT = Path(__file__).resolve().parents[1]


# get_parser()
# 定义命令行训练参数，包括图片模态、epochs、batch size、优化器参数和输入输出路径。
# 实现：通过 argparse 注册参数、默认值和允许的选项，返回 parser 供 main() 解析。

def get_parser():
    parser = argparse.ArgumentParser(description="Train a single-modality ResNet-50 baseline on unified DATA")
    parser.add_argument("--image-type", choices=("dermoscopic", "clinical", "clinical: close-up"),
                        default="dermoscopic", help="Use one image modality (default: dermoscopic)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weighting", choices=("inverse", "none"), default="inverse",
                        help="CE class weights from training counts only (default: inverse)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=VAL_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Separate directory for this run's checkpoints and evaluation")
    return parser


# set_seed() 
# 设置随机种子，使不同训练实验更容易复现和公平比较。
# 实现：分别设置 Python、NumPy、PyTorch 的随机种子；有 CUDA 时也设置 CUDA 种子。
def set_seed(seed):
    """Seed stochastic operations once, BEFORE creating loaders and the model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# 根据 --device 参数选择 GPU (CUDA) 或 CPU。
# 实现：auto 模式优先选可用 CUDA，否则选 CPU；若强制 CUDA 但不可用则报错。
def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    return torch.device("cuda" if requested == "cuda" or
                        (requested == "auto" and torch.cuda.is_available()) else "cpu")


def build_classification_loss(train_samples, device, weighting="inverse", *, class_names=CLASSES, beta=0.99):
    """按训练病灶计数构造 CE；N/(C*n_c) 与 1/n_c 保持相同的类别比例。

    权重顺序严格对应 class_names。只读取训练 Dataset 的标签，不加载图片，
    不使用验证或测试集的类别分布。训练样本的平均权重为 1。
    """
    class_names = validate_class_order(class_names)
    if weighting not in ("inverse", "none", "effective"):
        raise ValueError(f"Unknown class weighting: {weighting}")
    if weighting == "effective" and not (np.isfinite(beta) and 0 <= beta < 1):
        raise ValueError("beta must be finite and in [0, 1)")
    counts = [0] * len(class_names)
    for sample in train_samples:
        label = sample["label_id"]
        if not isinstance(label, (int, np.integer)) or not 0 <= label < len(class_names):
            raise ValueError(f"Invalid training label_id: {label!r}")
        counts[label] += 1
    total = sum(counts)
    if not total:
        raise ValueError("Cannot build classification loss from an empty training set")

    weights = None
    if weighting == "inverse":
        missing = [name for name, count in zip(class_names, counts) if count == 0]
        if missing:
            raise ValueError(f"Inverse class weights require training samples for every class: {missing}")
        weights = torch.tensor([total / (len(class_names) * count) for count in counts],
                               dtype=torch.float32, device=device)
    if weighting == "effective":
        if any(count == 0 for count in counts):
            raise ValueError("Effective class weights require training samples for every class")
        effective = (np.ones(len(counts)) if beta == 0 else
                     -np.expm1(np.asarray(counts, dtype=np.float64) * np.log(beta)) / (1 - beta))
        raw = 1 / effective
        raw /= np.dot(raw, counts) / total
        weights = torch.tensor(raw, dtype=torch.float32, device=device)
    config = {
        "name": "CrossEntropyLoss",
        "weighting": weighting,
        "formula": "N / (C * n_c)" if weighting == "inverse" else None,
        "reduction": "mean",
        "class_order": list(class_names),
        "train_class_counts": counts,
        "class_weights": weights.detach().cpu().tolist() if weights is not None else None,
    }
    if weighting == "effective":
        config.update(beta=beta, effective_class_counts=effective.tolist(),
                      formula="(1-beta)/(1-beta**n_c), normalized to training-sample mean 1")
    return nn.CrossEntropyLoss(weight=weights), config


# 保存当前模型及训练状态，用于保留最优模型或最后一轮模型。
# 实现：通过 torch.save 写入模型权重、optimizer 状态、epoch、最佳 Macro-F1、
#       验证 loss、类别顺序、图片模态和实验参数。这里只保存信息，并不执行断点恢复。
def save_checkpoint(path, model, optimizer, epoch, best_f1, args, *, val_loss, loss_config):
    """Store enough state and settings to inspect or resume training later."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_macro_f1": float(best_f1),
        "val_loss": float(val_loss),
        "class_order": list(CLASSES),
        "image_type": args.image_type,
        "loss_config": loss_config,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }, path)


# 把验证结果导出为按 lesion_id 对齐的指定类别概率 CSV（默认 11 类），供统一评估程序读取。
# 实现：检查验证 loader 为顺序采样，确认数据集样本、标签和预测数量一致；
#       再按统一类别顺序写出 lesion_id 和每一类预测概率。
def export_predictions(path, loader, val_result, *, class_names=CLASSES):
    """Export lesion-level probabilities compatible with src/evaluate.py.

    The current dataset returns (images, label) without IDs. Therefore take IDs
    from dataset.samples ONLY when the validation sampler preserves row order.
    """
    class_names = validate_class_order(class_names)
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Validation loader must use sequential sampling to align lesion IDs")
    if not hasattr(loader.dataset, "samples"):
        raise ValueError("Validation dataset must expose .samples with lesion_id")
    samples = loader.dataset.samples
    probabilities = val_result["probabilities"]
    labels = val_result["true_labels"]
    if len(samples) != len(probabilities) or len(samples) != len(labels):
        raise ValueError("Validation dataset size and model predictions disagree")
    for sample, label in zip(samples, labels):
        if sample["label_id"] != label:
            raise ValueError("Validation labels are not aligned with lesion IDs")

    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("lesion_id", *class_names))
        for sample, probs in zip(samples, probabilities):
            if len(probs) != len(class_names):
                raise ValueError(f"Expected exactly {len(class_names)} class probabilities")
            writer.writerow((sample["lesion_id"], *[f"{p:.10f}" for p in probs]))


# 单模态 ResNet-50 baseline 的完整训练入口，连接数据、模型、训练、验证及结果输出。
# 实现：检查参数与路径，固定随机种子并选择 device；创建 DataLoaders、预训练 ResNet-50、
#       CrossEntropyLoss 和 AdamW；逐 epoch 训练和验证，根据验证 Macro-F1 保存 best.pt，
#       同时保存 last.pt 与 history.csv；最后重新加载 best.pt，导出预测并调用统一评估程序。
#       当前仅支持 dermoscopic-only 或 clinical-only；不包含双模态或多 backbone 融合。
# Configure -> train/validate -> select best Macro-F1 checkpoint -> export final results.
def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("epochs, batch_size, lr must be positive; weight_decay nonnegative")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("num_workers must be nonnegative and seed in [0, 2**32)")
    if tuple(CLASS_NAMES) != tuple(CLASSES):
        raise ValueError("dataset.py and class_schema.py use different class orders")
    output_dir = args.output_dir.resolve()
    reserved = ("best.pt", "last.pt", "history.csv", "predictions.csv", "evaluation", "loss_config.json")
    if any((output_dir / name).exists() for name in reserved):
        raise FileExistsError(f"Run outputs already exist in {output_dir}; choose a new --output-dir")

    set_seed(args.seed)
    device = resolve_device(args.device)
    # Create loaders once; reshuffling and randomized transforms happen each epoch.
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        train_csv=args.train_csv, val_csv=args.val_csv,
        image_type=args.image_type, metadata_csv=args.metadata_csv,
    )
    if not isinstance(val_loader.sampler, SequentialSampler):
        raise ValueError("Validation loader must not shuffle")

    # 固定使用 train 的计数；val 复用同一 criterion，不重新估计权重。
    criterion, loss_config = build_classification_loss(
        train_loader.dataset.samples, device, args.class_weighting)
    loss_config["train_csv"] = str(args.train_csv.resolve())

    # torchvision downloads ImageNet weights on first use if not yet cached.
    model = build_resnet50(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "loss_config.json").open("w", encoding="utf-8") as stream:
        json.dump(loss_config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"CrossEntropyLoss class weighting: {args.class_weighting} (training counts only)", flush=True)
    for name, count, weight in zip(CLASSES, loss_config["train_class_counts"],
                                   loss_config["class_weights"] or [1.0] * len(CLASSES)):
        print(f"  {name}: train count={count}, weight={weight:.6f}", flush=True)
    best_f1 = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = validate_one_epoch(model, val_loader, criterion, device)
        scores = compute_metrics(val_result["true_labels"], val_result["pred_labels"])
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "val_macro_f1": scores["macro_f1"],
            "val_accuracy": scores["accuracy"],
        }
        history.append(row)
        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_f1,
                            args, val_loss=val_result["loss"], loss_config=loss_config)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_f1,
                        args, val_loss=val_result["loss"], loss_config=loss_config)
        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"train loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f} | "
              f"val Macro-F1 {row['val_macro_f1']:.4f} | accuracy {row['val_accuracy']:.4f}",
              flush=True)

    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Important: evaluate the *best* model, not whichever weights remained at last epoch.
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_result = validate_one_epoch(model, val_loader, criterion, device)
    export_predictions(output_dir / "predictions.csv", val_loader, best_result)
    # Teammate's unified evaluator computes authoritative full-validation metrics.
    evaluate_files(
        labels_path=args.val_csv,
        predictions_path=output_dir / "predictions.csv",
        output_dir=output_dir / "evaluation",
        model_name=f"ResNet50 {args.image_type} (best epoch {checkpoint['epoch']})",
    )
    print(f"Best epoch: {checkpoint['epoch']}; "
          f"Macro-F1: {best_f1:.4f}. Saved outputs in {output_dir}")
    return output_dir


def main():
    args = get_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
