"""Evaluate lesion-level probability CSVs against labels, matched by lesion_id."""
# 用途：对比每个病变的真实标签与模型预测，统一计算分类指标并生成混淆矩阵。
# 输入：标签 CSV + 预测概率 CSV；默认标签是现有验证集，预测文件必须由参数指定。
# 输出：4 项整体指标、逐类指标、逐条预测记录，以及数量/比例两种混淆矩阵。
# 评估对象是“每个病变的最终预测”，图片读取、模型训练和多骨干融合在上游完成。
# 阅读顺序：main -> evaluate_files -> load_evaluation_data -> calculate_metrics -> 保存结果。

# 标准库处理参数、CSV/JSON、文件路径、输入指纹和临时结果目录。
import argparse
import csv
import hashlib
import json
import platform
import tempfile
import textwrap
from pathlib import Path

import matplotlib
# 使用无界面的绘图后端，在 Windows 或没有桌面的集群上都能直接保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
# NumPy 处理 N×11 概率矩阵；scikit-learn 提供分类指标和混淆矩阵计算。
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# 兼容 python -m src.evaluate 和直接执行脚本两种导入方式。
if __package__:
    from .class_schema import CLASSES
else:
    from class_schema import CLASSES

# 默认验证标签相对于脚本所在仓库定位；显式传入的相对路径仍相对于终端目录。
ROOT = Path(__file__).resolve().parents[1]
# 每行概率之和允许偏离 1 的绝对误差为 0.00001，以容纳浮点数/导出精度误差。
PROBABILITY_SUM_ATOL = 1e-5
# 一次评估固定生成这七个文件，覆盖检查和最终保存共用同一份清单。
OUTPUT_FILES = (
    "metrics.json", "per_class_metrics.csv", "predictions_aligned.csv",
    "confusion_matrix.csv", "confusion_matrix_normalized.csv",
    "confusion_matrix.png", "confusion_matrix_normalized.png",
)


# CSV 的通用检查入口，返回（每行记录组成的列表，列名列表）。
# required 指定必需列；exact=True 还会拒绝额外列，适用于严格的预测格式。
def read_table(path, required, *, exact=False):
    """Reject malformed CSVs before any output is written; IDs remain strings."""
    # 兼容带 BOM 的 UTF-8 文件；用列名读取，且保留病变编号的字符串形式。
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames or []
        # 重复列名会让字典中的值相互覆盖，因此必须在读取记录前拒绝。
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path}: duplicate column names")
        missing = set(required) - set(columns)
        extra = set(columns) - set(required)
        # 标签表允许附带其他元数据；预测表只能包含 lesion_id 和 11 个类别概率列。
        if missing or (exact and extra):
            raise ValueError(
                f"{path}: missing columns={sorted(missing)}, "
                f"unexpected columns={sorted(extra) if exact else []}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    seen = set()
    # 从第 2 行开始定位错误，因为第 1 行是表头。seen 用于检查重复病变编号。
    for line, row in enumerate(rows, 2):
        # DictReader 把多出的字段放在 None 键下，缺少字段则对应 None 值。
        # 空字符串不属于字段数量错误，会由后续编号/概率检查处理。
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}:{line}: row length does not match header")
        lesion = row["lesion_id"]
        # 不自动删去编号中的空格，避免悄悄把原本不同或错误的编号合并。
        if not lesion or lesion != lesion.strip():
            raise ValueError(f"{path}:{line}: empty lesion_id or surrounding whitespace")
        if lesion in seen:
            raise ValueError(f"{path}:{line}: duplicate lesion_id {lesion!r}")
        seen.add(lesion)
    return rows, columns


# 加载并对齐两份输入，是计算指标之前必须经过的数据检查层。
# 返回 ids（N 个编号）、y_true（N 个真实类别索引）、probabilities（N×11 概率矩阵）。
def load_evaluation_data(labels_path, predictions_path):
    """Return IDs, integer true labels and probabilities in label-file row order."""
    # 最简真实标签只需编号和 label；现有验证集附带的 one-hot 列也能直接使用。
    labels, columns = read_table(labels_path, ("lesion_id", "label"))
    # one-hot 可完全省略，但只要出现任意一列，就必须包含全部 11 列并与 label 一致。
    one_hot = set(columns) & set(CLASSES)
    if one_hot and one_hot != set(CLASSES):
        raise ValueError(f"{labels_path}: supply all 11 one-hot columns or none")
    for row in labels:
        label = row["label"]
        if label not in CLASSES:
            raise ValueError(f"{labels_path}: unknown label {label!r}")
        if one_hot:
            try:
                values = [float(row[c]) for c in CLASSES]
            except ValueError as exc:
                raise ValueError(f"{labels_path}: invalid one-hot values") from exc
            # 根据 label 构造期望的 one-hot，例如 BCC 应为 [0, 1, 0, ...]。
            if values != [float(c == label) for c in CLASSES]:
                raise ValueError(
                    f"{labels_path}: one-hot columns disagree with label for {row['lesion_id']}"
                )

    # 预测表必须严格遵守统一格式，不能直接传类别编号、额外 label 列或不完整概率。
    predictions, _ = read_table(predictions_path, ("lesion_id", *CLASSES), exact=True)
    # 先建立“编号 -> 预测记录”字典，再按标签表的编号顺序取预测。
    # 因此两个 CSV 的行顺序可以不同，不会把不同病变的结果配在一起。
    by_id = {row["lesion_id"]: row for row in predictions}
    ids = [row["lesion_id"] for row in labels]
    # 必须评估标签文件中的全部病变；不允许悄悄丢弃缺失预测或额外预测。
    missing = set(ids) - set(by_id)
    extra = set(by_id) - set(ids)
    if missing or extra:
        raise ValueError(
            "Prediction IDs must exactly match label IDs: "
            f"{len(missing)} missing (examples: {sorted(missing)[:3]}), "
            f"{len(extra)} extra (examples: {sorted(extra)[:3]})"
        )
    # 同时按编号重排行、按 CLASSES 重排列，得到固定顺序的浮点概率矩阵。
    # 只能核对列名；模型导出时若把某类概率误写到另一列，数字本身无法揭示这种错误。
    try:
        probabilities = np.array(
            [[float(by_id[lesion][c]) for c in CLASSES] for lesion in ids], dtype=float
        )
    except ValueError as exc:
        raise ValueError("Probabilities must be numeric, with no empty cells") from exc
    # isfinite 拒绝 NaN 和无穷大；后两项拒绝负概率或大于 1 的概率。
    # any(axis=1) 汇总每行是否存在任意错误，报出第一个出错病变。
    invalid = (~np.isfinite(probabilities)
               | (probabilities < 0) | (probabilities > 1)).any(axis=1)
    if invalid.any():
        raise ValueError(
            f"Probabilities must be finite and within [0, 1]; invalid lesion: "
            f"{ids[int(np.flatnonzero(invalid)[0])]}"
        )
    # axis=1 对每个病变的 11 类概率求和；rtol=0 表示只使用绝对误差。
    # 这里不做 softmax 或自动归一化；上游应导出概率，而不是原始 logits。
    bad_sums = ~np.isclose(probabilities.sum(axis=1), 1.0, atol=PROBABILITY_SUM_ATOL, rtol=0)
    if bad_sums.any():
        raise ValueError(
            f"Probability rows must sum to 1 within {PROBABILITY_SUM_ATOL}; invalid lesion: "
            f"{ids[int(np.flatnonzero(bad_sums)[0])]}. Export probabilities, not logits."
        )
    # 把真实类别名映射成 0～10 的索引，供 scikit-learn 与预测索引进行比较。
    y_true = np.array([CLASSES.index(row["label"]) for row in labels], dtype=int)
    return ids, y_true, probabilities


# 输入必须已经过 load_evaluation_data 验证，且真实标签与概率行已对齐。
# 返回：整体指标、逐类指标、数量矩阵、归一化矩阵和预测类别索引。
def calculate_metrics(y_true, probabilities):
    """Calculate metrics from validated arrays returned by load_evaluation_data."""
    # 每行取最大概率所在的列作为预测类别；最高值完全并列时取最靠前的类别。
    y_pred = probabilities.argmax(axis=1)
    # 显式指定全部 11 类，使某类即使没有样本，也保留在指标和矩阵中。
    indices = np.arange(len(CLASSES))
    # 对每类分别计算：Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2TP/(2TP+FP+FN)。
    # TP 是该类预测正确数；FP 是其他类被误判为该类的数量；FN 是该类被错分的数量。
    # support 是真实样本数；zero_division=0 规定分母为零时相应指标记作 0。
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    # matrix[i, j] 表示真实类别 i 被预测为类别 j 的数量，对角线是正确预测。
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    # support[:, None] 把长度 11 的数组变成 11×1，以便每行除以对应真实样本数。
    # 零样本行保留为 0，避免除零；绘图时会把该行标成 n/a。
    normalized = np.divide(
        matrix, support[:, None], out=np.zeros_like(matrix, dtype=float),
        where=support[:, None] != 0,
    )
    # 一类对应一条记录；矩阵第 i 列之和是“被预测成该类”的数量。
    # 转成 Python 的 float/int，方便后面写入 CSV 和 JSON。
    per_class = [
        {"class": c, "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(support[i]),
         "predicted_count": int(matrix[:, i].sum())}
        for i, c in enumerate(CLASSES)
    ]
    # 所有指标均为 0～1 的数值。Accuracy 对样本计数，Macro 对 11 个类别等权平均。
    # Macro F1 是各类 F1 的均值，即使某类缺失，也保留该类的 0 分参与平均。
    aggregate = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
    }
    return aggregate, per_class, matrix, normalized, y_pred


# 公共 CSV 写出函数：先写表头，再写数据；newline="" 避免 Windows 多出空行。
def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


# 把 11×11 矩阵画成热力图。normalized=False 显示数量，True 显示真实类内的百分比。
# model_name 仅用于图上的说明文字，不会加载模型或影响计算。
def plot_confusion(matrix, support, path, model_name, *, normalized=False):
    # 设置画布和留白，为标题、类别名、色条及底部说明留出空间。
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    fig.subplots_adjust(left=0.13, right=0.89, top=0.83, bottom=0.17)
    title = "Confusion matrix | true-class percentages" if normalized else "Confusion matrix | lesion counts"
    fig.suptitle(title, x=0.13, y=0.975, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.13, 0.91, textwrap.fill(model_name, width=85), fontsize=10, color="#526373")
    # 比例图的颜色范围固定为 0～1；数量图按本次最大格子数量设置范围。
    upper = 1.0 if normalized else max(1, int(matrix.max()))
    rendered = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=upper)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class", labelpad=12)
    ax.set_ylabel("True class", labelpad=12)
    # 每格直接写出数字：数量保留整数，比例转成百分比；深色格使用白字便于阅读。
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = matrix[i, j]
            if normalized:
                label = f"{100 * value:.1f}%" if value not in (0, 1) else f"{100 * value:.0f}%"
                if support[i] == 0:
                    label = "n/a"
            else:
                label = str(int(value))
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if value > upper / 2 else "#172B3A")
    # 右侧色条解释“颜色深浅对应的数值”。比例图用百分比刻度，数量图只用整数。
    # 数量图的 Lesions 是“病变数量”的单位，不是额外的类别名称。
    colorbar = fig.colorbar(rendered, ax=ax, fraction=0.04, pad=0.025)
    if normalized:
        colorbar.formatter = PercentFormatter(xmax=1, decimals=0)
        colorbar.set_label("Share within true class")
    else:
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.set_label("Lesions")
    colorbar.update_ticks()
    note = ("Each row uses its true-class support; n/a means no true examples."
            if normalized else "Rows are true labels; columns are predicted labels. Each lesion counts once.")
    fig.text(0.13, 0.035, note, fontsize=9, color="#526373")
    fig.savefig(path, dpi=160)
    # 保存后释放图形资源，方便以后批量评估多个模型。
    plt.close(fig)


# 保存输入文件的路径和 SHA-256 内容指纹，用于追溯某次评估使用的数据版本。
def source_record(path):
    path = Path(path).resolve()
    try:
        displayed = path.relative_to(ROOT).as_posix()
    except ValueError:
        displayed = path.as_posix()
    return {"path": displayed, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# 对外的完整评估入口：串起数据检查、指标计算、元信息整理和七个结果文件的生成。
# 也可从其他 Python 代码中调用；返回写入 metrics.json 的整体结果字典。
def evaluate_files(labels_path, predictions_path, output_dir, *, model_name=None, overwrite=False):
    labels_path, predictions_path, output_dir = map(Path, (labels_path, predictions_path, output_dir))
    ids, y_true, probabilities = load_evaluation_data(labels_path, predictions_path)
    aggregate, per_class, matrix, normalized, y_pred = calculate_metrics(y_true, probabilities)
    model_name = model_name or predictions_path.stem
    # 先比较实际绝对路径，防止输出文件恰好覆盖输入；此保护不受 overwrite 开关影响。
    input_paths = {labels_path.resolve(), predictions_path.resolve()}
    targets = [output_dir / name for name in OUTPUT_FILES]
    if any(path.resolve() in input_paths for path in targets):
        raise ValueError("Output files would overwrite an input CSV; choose another output directory")
    # 默认保护已有实验结果；只有显式开启 --overwrite 才替换同名输出。
    existing = [path.name for path in targets if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"Output files already exist: {', '.join(existing)}. "
            "Choose another directory or use --overwrite."
        )
    # 明确记录缺少真实样本的类别，提醒使用者：宏平均仍包含全部 11 类。
    absent = [row["class"] for row in per_class if row["support"] == 0]
    warnings = []
    if absent:
        warnings.append(
            f"No true examples for {', '.join(absent)}. Undefined per-class metrics are 0; "
            "Macro F1 still averages all 11 classes. Compare the same label split across runs."
        )
    # 统计最高概率完全相同的并列预测；分类仍使用前面 argmax 的固定规则。
    maxima = probabilities.max(axis=1, keepdims=True)
    ties = int(np.count_nonzero((probabilities == maxima).sum(axis=1) > 1))
    # 结果除了分数，也保存计算规则、数据指纹和环境版本，便于复查与模型间比较。
    summary = {
        "schema_version": 1, "model_name": model_name, "unit": "lesion", "n_lesions": len(ids),
        "class_order": list(CLASSES), "primary_metric": "macro_f1",
        "metric_scale": "0 to 1", **aggregate,
        "metric_policy": {
            "macro_average_classes": "all 11, including classes absent from labels",
            "zero_division": 0, "prediction": "argmax probability",
            "tie_break": "first class in class_order",
            "probability_sum_atol": PROBABILITY_SUM_ATOL,
            "normalized_confusion_matrix": "rows divided by true-class support; absent rows stored as 0",
        },
        "tied_prediction_rows": ties, "classes_without_true_examples": absent,
        "warnings": warnings,
        "sources": {"labels": source_record(labels_path), "predictions": source_record(predictions_path)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # 先在临时目录生成完整表格和图片，生成完成后再逐个移到正式目录。
    # 这样输入/绘图失败时不会先发布半套新结果；此处不改变原始标签或预测 CSV。
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output_dir) as temp:
        stage = Path(temp)
        (stage / "metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # 每类指标表便于比较不同类别；两个矩阵 CSV 便于后续重新画图或做统计。
        write_csv(stage / "per_class_metrics.csv", list(per_class[0]),
                  [list(row.values()) for row in per_class])
        for name, values in (("confusion_matrix.csv", matrix),
                             ("confusion_matrix_normalized.csv", normalized)):
            write_csv(stage / name, ("true_class", *CLASSES),
                      [[c, *values[i].tolist()] for i, c in enumerate(CLASSES)])
        # 逐条结果按标签文件顺序保存，correct=1 表示预测正确，0 表示错误。
        # 保留全部概率，后续可按病变编号查找误判并回看图片。
        write_csv(stage / "predictions_aligned.csv",
                  ("lesion_id", "true_label", "predicted_label", "correct", *CLASSES),
                  [[lesion, CLASSES[y_true[i]], CLASSES[y_pred[i]], int(y_true[i] == y_pred[i]),
                    *probabilities[i].tolist()] for i, lesion in enumerate(ids)])
        # 矩阵每行总数就是真实类别的样本量；用同一批结果生成两张图。
        support = matrix.sum(axis=1)
        plot_confusion(matrix, support, stage / "confusion_matrix.png", model_name)
        plot_confusion(normalized, support, stage / "confusion_matrix_normalized.png",
                       model_name, normalized=True)
        # 所有文件生成成功后逐个替换目标文件；离开 with 后临时目录自动清理。
        for name in OUTPUT_FILES:
            (stage / name).replace(output_dir / name)
    return summary


# 命令行入口。argv=None 时读取终端参数，也可以传入参数列表供其他代码调用。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # 必需参数：模型概率 CSV 和结果目录。可选标签路径默认指向仓库现有验证集。
    parser.add_argument("--predictions", required=True, type=Path, help="CSV with lesion_id and 11 probability columns")
    parser.add_argument("--labels", type=Path, default=ROOT / "DATA/PROCESSED_DATA/val_split.csv",
                        help="True-label CSV; default is the repository validation split")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for metrics, tables and figures")
    # model-name 仅标记实验名称；overwrite 表示允许替换已有结果文件。
    parser.add_argument("--model-name", help="Display name stored in metrics and figures; defaults to prediction filename")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing evaluation output files")
    args = parser.parse_args(argv)
    try:
        summary = evaluate_files(args.labels, args.predictions, args.output_dir,
                                 model_name=args.model_name, overwrite=args.overwrite)
    # 输入、文件或编码错误统一输出简短说明，并用非零退出码告知运行失败。
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(f"Evaluated {summary['n_lesions']:,} lesions: {summary['model_name']}")
    print(f"Accuracy: {summary['accuracy']:.6f} | Macro F1: {summary['macro_f1']:.6f}")
    for warning in summary["warnings"]:
        print(f"Warning: {warning}")
    print(f"Outputs: {args.output_dir.resolve()}")


# 直接运行文件时执行 main；导入此模块时只提供函数供调用。
if __name__ == "__main__":
    main()
