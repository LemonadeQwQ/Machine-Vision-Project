"""复用原 CE 检查；新协议将逆频率权重归一到类别均值为 1。"""
from .class_schema import NINE_CLASSES
from .train import build_classification_loss


def classification_loss(rows, weighted, device):
    samples = [dict(r, label_id=int(r['label_id'])) for r in rows]
    criterion, record = build_classification_loss(samples, device=device,
        weighting='inverse' if weighted else 'none', class_names=NINE_CLASSES)
    if weighted:
        criterion.weight.div_(criterion.weight.mean())
        record['class_weights'] = criterion.weight.cpu().tolist()
        record['formula'] = '(1/n_c) / mean_c(1/n_c)'
    return criterion, record
