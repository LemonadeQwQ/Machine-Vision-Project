"""Analyze MILK10k class counts at lesion level; no images or GPU required."""
# 用途：统计已有训练、验证标签中的类别分布，以一个病变为一个样本。
# 输入：默认读取 DATA/PROCESSED_DATA 下的 train_split.csv 和 val_split.csv。
# 输出：统计 CSV、两张分布图、JSON 摘要和英文 Markdown 报告。
# 执行顺序：main -> read_labels -> summarize -> save_figures / write_report。
# 这一步直接使用真实标签，不需要图片、模型预测或 GPU。

# 标准库分别负责命令行参数、CSV/JSON 读写、文件指纹、版本信息和计数。
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import matplotlib
# Agg 后端直接把图保存为文件，在没有图形桌面的计算集群上也能运行。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 同时支持以模块方式运行，以及直接运行这个 Python 文件。
if __package__:
    from .class_schema import CLASSES
else:
    from class_schema import CLASSES

# 根据脚本的位置找到仓库根目录，默认数据路径不受终端当前目录影响。
ROOT = Path(__file__).resolve().parents[1]


# 读取一份标签表，返回“每行一个字典”的列表；验证失败时抛出 ValueError。
def read_labels(path):
    # utf-8-sig 兼容带 BOM 的 UTF-8 CSV；DictReader 用列名访问每个字段。
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        # 分布分析使用现有完整标签格式：病变编号、类别名和全部 11 个 one-hot 列。
        if not {"lesion_id", "label", *CLASSES}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing required label columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: no label rows")
    # 用集合记录已经出现的病变编号，避免同一个病变被重复计数。
    seen = set()
    # 表头占第 1 行，因此从 2 开始编号，便于报错时定位数据记录。
    for line, row in enumerate(rows, 2):
        lesion, label = row["lesion_id"], row["label"]
        if not lesion or not lesion.strip() or lesion in seen:
            raise ValueError(f"{path.name}:{line}: missing or duplicate lesion_id")
        seen.add(lesion)
        if label not in CLASSES:
            raise ValueError(f"{path.name}:{line}: unknown label {label!r}")
        # CSV 中的 0、1、0.0、1.0 都转换为数值；one-hot 应恰好有一个 1。
        try:
            values = [float(row[name]) for name in CLASSES]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}:{line}: invalid one-hot values") from exc
        # 三项条件同时满足：仅含 0/1、只有一个正类、正类位置与 label 一致。
        if (any(value not in (0, 1) for value in values)
                or sum(values) != 1 or values[CLASSES.index(label)] != 1):
            raise ValueError(f"{path.name}:{line}: one-hot columns disagree with label")
    return rows


# 输入两份已验证的记录，返回按 CLASSES 排列的逐类统计结果。
def summarize(train, validation):
    # 同一病变不能同时出现在训练集和验证集；这里只检查病变编号交集。
    overlap = {row["lesion_id"] for row in train} & {row["lesion_id"] for row in validation}
    if overlap:
        raise ValueError(f"Training and validation overlap by {len(overlap)} lesions")
    # tc、vc 分别是训练和验证的类别计数；Counter 对未出现的类别返回 0。
    tc = Counter(row["label"] for row in train)
    vc = Counter(row["label"] for row in validation)
    result = []
    # 遍历固定的 11 类，使样本数为 0 的类别也保留在输出表里。
    for label in CLASSES:
        tp, vp = 100 * tc[label] / len(train), 100 * vc[label] / len(validation)
        result.append({
            "class": label, "train_count": tc[label], "validation_count": vc[label],
            "total_count": tc[label] + vc[label],
            "train_percent": tp, "validation_percent": vp,
            "total_percent": 100 * (tc[label] + vc[label]) / (len(train) + len(validation)),
            # tp、vp 的单位是百分数（0～100），相减得到“百分点差”，不是相对增长率。
            "validation_minus_train_pp": vp - tp,
        })
    return result


# 两张图共用的外观设置：网格放在条形后方，隐藏边框并统一刻度样式。
def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E2E8EF", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8, colors="#344354")


# rows 是统计结果；nt、nv 是训练/验证样本总数；out 是结果目录。
# 两张图都统计病变，不会把同一病变的两张图片重复计算。
def save_figures(rows, nt, nv, out):
    # 按总数量从大到小展示，便于看出类别不均衡；不改变公共类别编号。
    ordered = sorted(rows, key=lambda row: -row["total_count"])
    labels = [r["class"] for r in ordered]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#344354", "text.color": "#172B3A",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    # 第一张图：训练与验证合并后的数量。右侧留空间标注数量和总占比。
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.14)
    fig.suptitle("MILK10k class distribution", x=0.13, y=0.965,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(0.13, 0.905, f"{nt + nv:,} lesions across 11 classes | Training + validation",
             fontsize=11, color="#526373")
    counts = [r["total_count"] for r in ordered]
    bars = ax.barh(labels, counts, color="#24688D", height=0.66)
    # 横向条形图默认从下往上排列，翻转纵轴后最大类别显示在最上方。
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.30)
    ax.set_xlabel("Number of lesions", labelpad=10)
    # 在每根条形的末端标注实际数量和百分比，而不只依赖横轴估读。
    for bar, row in zip(bars, ordered):
        ax.text(bar.get_width() + max(counts) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{row['total_count']:,}  ({row['total_percent']:.2f}%)",
                va="center", fontsize=10)
    style_axes(ax)
    fig.text(0.13, 0.035, "Each lesion is counted once. Image pairs are not double-counted.",
             fontsize=9, color="#526373")
    fig.savefig(out / "class_counts.png", dpi=180)
    plt.close(fig)

    # 第二张图：比较每类在训练/验证各自集合中的占比，避免总样本量不同造成误读。
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.13)
    fig.suptitle("Class proportions in training and validation", x=0.13, y=0.965,
                 ha="left", fontsize=18, fontweight="bold")
    gap = max(abs(r["validation_minus_train_pp"]) for r in rows)
    fig.text(0.13, 0.91,
             f"Percentages within each split | Largest absolute gap: {gap:.4f} percentage points",
             fontsize=10, color="#526373")
    positions = list(range(len(labels)))
    # 同一类别画两根略微错开的条形：蓝色是训练集，橙色是验证集。
    for key, offset, color, title in (
        ("train_percent", -0.19, "#24688D", f"Training (n = {nt:,})"),
        ("validation_percent", 0.19, "#CB7A32", f"Validation (n = {nv:,})"),
    ):
        bars = ax.barh([y + offset for y in positions], [r[key] for r in ordered],
                       height=0.32, color=color, label=title)
        for bar, row in zip(bars, ordered):
            value = row[key]
            ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}%", va="center", fontsize=9)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(r["train_percent"], r["validation_percent"]) for r in rows) * 1.19)
    ax.set_xlabel("Share of lesions within the split (%)", labelpad=10)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2, frameon=False, borderaxespad=0)
    style_axes(ax)
    fig.text(0.13, 0.03,
             "Denominators: all training lesions for training; all validation lesions for validation.",
             fontsize=9, color="#526373")
    fig.savefig(out / "split_proportions.png", dpi=180)
    plt.close(fig)


# 记录输入文件来源和 SHA-256 字节指纹，便于确认一次分析使用了哪份数据。
def source_record(path):
    resolved = path.resolve()
    # 仓库内文件保存相对路径，方便队友使用；外部输入保存绝对路径。
    try:
        displayed = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        displayed = resolved.as_posix()
    return {"path": displayed, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# 把统计结果转成机器可读的 JSON 摘要和便于团队阅读的英文报告。
def write_report(rows, nt, nv, sources, out):
    largest = max(rows, key=lambda r: r["total_count"])
    # “最小类别”仅在实际出现的类别中选择，避免用 0 作为不均衡比值的分母。
    smallest = min((r for r in rows if r["total_count"] > 0), key=lambda r: r["total_count"])
    gap = max(rows, key=lambda r: abs(r["validation_minus_train_pp"]))
    ratio = largest["total_count"] / smallest["total_count"]
    # 用训练集选出多数类，再读取该类在验证集的比例作为“恒预测该类”的理论准确率。
    # 这里只计算参照值，不运行或训练任何分类模型。
    majority = max(rows, key=lambda r: r["train_count"])
    environment = {"python": platform.python_version(), "matplotlib": matplotlib.__version__}
    # 摘要同时保存统计值、已通过的检查、输入指纹以及 Python/Matplotlib 版本。
    summary = {
        "unit": "lesion", "class_order": list(CLASSES),
        "train_lesions": nt, "validation_lesions": nv, "total_lesions": nt + nv,
        "train_fraction": nt / (nt + nv), "validation_fraction": nv / (nt + nv),
        "largest_class": largest["class"], "smallest_present_class": smallest["class"],
        "largest_to_smallest_present_ratio": ratio,
        "maximum_absolute_proportion_gap_pp": abs(gap["validation_minus_train_pp"]),
        "training_majority_class": majority["class"],
        "majority_only_validation_accuracy_percent": majority["validation_percent"],
        "checks": {"unique_lesions_within_splits": True, "splits_disjoint": True,
                   "labels_match_one_hot": True},
        "sources": sources, "environment": environment,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # 报告里的数字按展示需要四舍五入；单独输出的 CSV 保留更多精度。
    table = [
        "| Class | Training | Validation | Total | Total share | Training share | Validation share | Difference (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        table.append(
            f"| {r['class']} | {r['train_count']:,} | {r['validation_count']:,} "
            f"| {r['total_count']:,} | {r['total_percent']:.2f}% | {r['train_percent']:.2f}% "
            f"| {r['validation_percent']:.2f}% | {r['validation_minus_train_pp']:+.4f} |")
    # 根据当前统计结果拼出英文报告，包含表格、图片引用、发现和评估建议。
    lines = [
        "# MILK10k class distribution analysis",
        "",
        f"The supplied label files contain {nt + nv:,} lesions: {nt:,} training "
        f"({100 * nt / (nt + nv):.0f}%) and {nv:,} validation ({100 * nv / (nt + nv):.0f}%).",
        f"The largest-to-smallest present class count ratio is {ratio:.1f}:1; the maximum "
        f"split proportion difference is {abs(gap['validation_minus_train_pp']):.4f} percentage "
        "points. Each row represents one lesion, not one image.",
        "",
        "## Class counts and proportions", "", *table, "",
        "Shares use the corresponding split size as denominator. Total share uses all "
        "lesions. Difference is validation share minus training share in percentage "
        "points. Displayed values are rounded; the CSV preserves more precision.",
        "",
        "![Lesion counts by class](class_counts.png)", "",
        "![Training and validation class proportions](split_proportions.png)", "",
        "## Findings", "",
        f"- {largest['class']} is the largest class: {largest['total_count']:,} lesions "
        f"({largest['total_percent']:.2f}% of all labels).",
        f"- {smallest['class']} is the smallest present class: {smallest['total_count']:,} lesions "
        f"({smallest['total_percent']:.2f}%). The largest class is {ratio:.1f} times as frequent.",
        f"- The largest absolute split proportion difference is "
        f"{abs(gap['validation_minus_train_pp']):.4f} percentage points ({gap['class']}). "
        "Similar proportions do not establish the splitting algorithm or random seed.",
        f"- {smallest['class']} has only {smallest['validation_count']} validation lesions. "
        "Its per-class scores can be sensitive to individual predictions.",
        "", "## Implications for evaluation", "",
        f"A classifier that always predicts the training majority class, {majority['class']}, would obtain "
        f"{majority['validation_percent']:.2f}% validation accuracy from label counts alone. "
        "This is a theoretical constant-prediction reference, not a trained-model result. "
        "Accuracy alone does not describe performance across all classes.",
        "",
        "Use Macro F1 as the primary metric: it gives each of the 11 class F1 scores equal "
        "weight. Also report accuracy, per-class F1, support and a confusion matrix. "
        "Keep the same splits and class order across models. Macro F1 does not remove "
        "uncertainty from the very small validation classes.",
        "", "## Method and reproducibility", "",
        "Count the label column after checking for unique lesion IDs, disjoint splits, "
        "known class names and matching one-hot labels. Include zero-count classes. "
        "No resampling or label changes are made. This describes the supplied labels; "
        "it does not verify image availability, image quality or diagnostic correctness.",
        "",
        "Sources are relative to the repository unless external inputs were specified:", "",
    ]
    for name, record in sources.items():
        lines.append(f"- {name}: {record['path']} (SHA-256: {record['sha256']}).")
    lines += [
        "", "Run from the repository root: python src/analyze_class_distribution.py", "",
        "The script writes class_distribution.csv, summary.json, this report and two PNG "
        "figures. Existing outputs with these names are replaced on rerun. Default paths "
        "are anchored to the script location, so running from another directory also works. "
        "Optional arguments: --train-csv, --val-csv and --output-dir.",
        "", f"Runtime versions: Python {environment['python']}, Matplotlib {environment['matplotlib']}.", "",
    ]
    (out / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


# 命令行入口：设置默认路径，然后依次读取、统计并写出完整分析结果。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "DATA/PROCESSED_DATA/train_split.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT / "DATA/PROCESSED_DATA/val_split.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/class_distribution")
    args = parser.parse_args()
    try:
        train, val = read_labels(args.train_csv), read_labels(args.val_csv)
        rows = summarize(train, val)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot analyze label files: {exc}\n")
    # 所有输入检查通过后再创建结果目录；重新运行会替换同名分析文件。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "class_distribution.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_figures(rows, len(train), len(val), args.output_dir)
    sources = {"training": source_record(args.train_csv), "validation": source_record(args.val_csv)}
    summary = write_report(rows, len(train), len(val), sources, args.output_dir)
    print(f"Analyzed {len(train) + len(val):,} lesions across {len(CLASSES)} classes.")
    print(f"Training: {len(train):,}; validation: {len(val):,}.")
    print(f"Largest: {summary['largest_class']}; smallest: {summary['smallest_present_class']}.")
    print(f"Largest absolute split gap: {summary['maximum_absolute_proportion_gap_pp']:.4f} pp.")
    print(f"Outputs: {args.output_dir.resolve()}")


# 直接执行文件时启动流程；被其他代码导入时不自动运行 main。
if __name__ == "__main__":
    main()
