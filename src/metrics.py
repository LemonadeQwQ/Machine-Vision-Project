"""Metrics adapter: share the evaluator's class order and metric definitions.

evaluate.py implements fixed-schema Macro F1, per-class F1 and confusion
matrices. Reuse it here so training and exported evaluation use the same rules.
"""

import numpy as np

if __package__:
    from .class_schema import CLASSES, validate_class_order
    from .evaluate import calculate_metrics
else:
    from class_schema import CLASSES, validate_class_order
    from evaluate import calculate_metrics


# compute_metrics() 
# 根据整数标签和显式类别顺序计算指标；默认仍为 11 类。
# 实现：先检查标签维度、长度和类别范围；把预测类别转换为 one-hot 概率矩阵；
#       调用队友已实现的 calculate_metrics，保持 Macro-F1 和混淆矩阵定义一致。
#       返回汇总指标、每类别指标及混淆矩阵；此函数不重新实现指标公式。
def compute_metrics(true_labels, pred_labels, *, class_names=CLASSES):
    """Compute fixed-schema metrics from integer ground truth and predictions.

    The existing evaluator accepts a probability matrix. One-hot predictions
    preserve exactly the same argmax labels without needing real probabilities.
    Zero-support classes remain in Macro F1 with undefined F1 treated as 0.
    """
    class_names = validate_class_order(class_names)
    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(pred_labels, dtype=int)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.size == 0:
        raise ValueError("true_labels and pred_labels must be nonempty 1D arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("true_labels and pred_labels must have the same length")
    if np.any(y_true < 0) or np.any(y_true >= len(class_names)):
        raise ValueError("true_labels contains an invalid class index")
    if np.any(y_pred < 0) or np.any(y_pred >= len(class_names)):
        raise ValueError("pred_labels contains an invalid class index")

    # calculate_metrics uses all selected classes, zero_division=0, and rows=true.
    probs_as_one_hot = np.eye(len(class_names), dtype=float)[y_pred]
    aggregate, per_class, matrix, _, _ = calculate_metrics(y_true, probs_as_one_hot, class_names=class_names)
    return {
        **aggregate,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
