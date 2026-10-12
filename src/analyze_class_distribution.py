"""Analyze MILK10k class counts at lesion level; no images or GPU required."""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
           "MAL_OTH", "MEL", "NV", "SCCKA", "VASC")


def read_labels(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"lesion_id", "label", *CLASSES}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing required label columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: no label rows")
    seen = set()
    for line, row in enumerate(rows, 2):
        lesion, label = row["lesion_id"], row["label"]
        if not lesion or not lesion.strip() or lesion in seen:
            raise ValueError(f"{path.name}:{line}: missing or duplicate lesion_id")
        seen.add(lesion)
        if label not in CLASSES:
            raise ValueError(f"{path.name}:{line}: unknown label {label!r}")
        try:
            values = [float(row[name]) for name in CLASSES]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}:{line}: invalid one-hot values") from exc
        if (any(value not in (0, 1) for value in values)
                or sum(values) != 1 or values[CLASSES.index(label)] != 1):
            raise ValueError(f"{path.name}:{line}: one-hot columns disagree with label")
    return rows


def summarize(train, validation):
    overlap = {row["lesion_id"] for row in train} & {row["lesion_id"] for row in validation}
    if overlap:
        raise ValueError(f"Training and validation overlap by {len(overlap)} lesions")
    tc = Counter(row["label"] for row in train)
    vc = Counter(row["label"] for row in validation)
    result = []
    for label in CLASSES:
        tp, vp = 100 * tc[label] / len(train), 100 * vc[label] / len(validation)
        result.append({
            "class": label, "train_count": tc[label], "validation_count": vc[label],
            "total_count": tc[label] + vc[label],
            "train_percent": tp, "validation_percent": vp,
            "total_percent": 100 * (tc[label] + vc[label]) / (len(train) + len(validation)),
            "validation_minus_train_pp": vp - tp,
        })
    return result


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E2E8EF", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8, colors="#344354")


def save_figures(rows, nt, nv, out):
    ordered = sorted(rows, key=lambda row: -row["total_count"])
    labels = [r["class"] for r in ordered]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#344354", "text.color": "#172B3A",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.14)
    fig.suptitle("MILK10k class distribution", x=0.13, y=0.965,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(0.13, 0.905, f"{nt + nv:,} lesions across 11 classes | Training + validation",
             fontsize=11, color="#526373")
    counts = [r["total_count"] for r in ordered]
    bars = ax.barh(labels, counts, color="#24688D", height=0.66)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.30)
    ax.set_xlabel("Number of lesions", labelpad=10)
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

    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.13)
    fig.suptitle("Class proportions in training and validation", x=0.13, y=0.965,
                 ha="left", fontsize=18, fontweight="bold")
    gap = max(abs(r["validation_minus_train_pp"]) for r in rows)
    fig.text(0.13, 0.91,
             f"Percentages within each split | Largest absolute gap: {gap:.4f} percentage points",
             fontsize=10, color="#526373")
    positions = list(range(len(labels)))
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


def source_record(path):
    resolved = path.resolve()
    try:
        displayed = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        displayed = resolved.as_posix()
    return {"path": displayed, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_report(rows, nt, nv, sources, out):
    largest = max(rows, key=lambda r: r["total_count"])
    smallest = min((r for r in rows if r["total_count"] > 0), key=lambda r: r["total_count"])
    gap = max(rows, key=lambda r: abs(r["validation_minus_train_pp"]))
    ratio = largest["total_count"] / smallest["total_count"]
    majority = max(rows, key=lambda r: r["train_count"])
    environment = {"python": platform.python_version(), "matplotlib": matplotlib.__version__}
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
    table = [
        "| Class | Training | Validation | Total | Total share | Training share | Validation share | Difference (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        table.append(
            f"| {r['class']} | {r['train_count']:,} | {r['validation_count']:,} "
            f"| {r['total_count']:,} | {r['total_percent']:.2f}% | {r['train_percent']:.2f}% "
            f"| {r['validation_percent']:.2f}% | {r['validation_minus_train_pp']:+.4f} |")
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


if __name__ == "__main__":
    main()
