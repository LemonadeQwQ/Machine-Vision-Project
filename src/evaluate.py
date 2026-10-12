"""Evaluate lesion-level probability CSVs against labels, matched by lesion_id."""
import argparse
import csv
import hashlib
import json
import platform
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

if __package__:
    from .class_schema import CLASSES
else:
    from class_schema import CLASSES

ROOT = Path(__file__).resolve().parents[1]
PROBABILITY_SUM_ATOL = 1e-5
OUTPUT_FILES = (
    "metrics.json", "per_class_metrics.csv", "predictions_aligned.csv",
    "confusion_matrix.csv", "confusion_matrix_normalized.csv",
    "confusion_matrix.png", "confusion_matrix_normalized.png",
)


def read_table(path, required, *, exact=False):
    """Reject malformed CSVs before any output is written; IDs remain strings."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path}: duplicate column names")
        missing = set(required) - set(columns)
        extra = set(columns) - set(required)
        if missing or (exact and extra):
            raise ValueError(
                f"{path}: missing columns={sorted(missing)}, "
                f"unexpected columns={sorted(extra) if exact else []}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    seen = set()
    for line, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}:{line}: row length does not match header")
        lesion = row["lesion_id"]
        if not lesion or lesion != lesion.strip():
            raise ValueError(f"{path}:{line}: empty lesion_id or surrounding whitespace")
        if lesion in seen:
            raise ValueError(f"{path}:{line}: duplicate lesion_id {lesion!r}")
        seen.add(lesion)
    return rows, columns


def load_evaluation_data(labels_path, predictions_path):
    """Return IDs, integer true labels and probabilities in label-file row order."""
    labels, columns = read_table(labels_path, ("lesion_id", "label"))
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
            if values != [float(c == label) for c in CLASSES]:
                raise ValueError(
                    f"{labels_path}: one-hot columns disagree with label for {row['lesion_id']}"
                )

    predictions, _ = read_table(predictions_path, ("lesion_id", *CLASSES), exact=True)
    by_id = {row["lesion_id"]: row for row in predictions}
    ids = [row["lesion_id"] for row in labels]
    missing = set(ids) - set(by_id)
    extra = set(by_id) - set(ids)
    if missing or extra:
        raise ValueError(
            "Prediction IDs must exactly match label IDs: "
            f"{len(missing)} missing (examples: {sorted(missing)[:3]}), "
            f"{len(extra)} extra (examples: {sorted(extra)[:3]})"
        )
    try:
        probabilities = np.array(
            [[float(by_id[lesion][c]) for c in CLASSES] for lesion in ids], dtype=float
        )
    except ValueError as exc:
        raise ValueError("Probabilities must be numeric, with no empty cells") from exc
    invalid = (~np.isfinite(probabilities)
               | (probabilities < 0) | (probabilities > 1)).any(axis=1)
    if invalid.any():
        raise ValueError(
            f"Probabilities must be finite and within [0, 1]; invalid lesion: "
            f"{ids[int(np.flatnonzero(invalid)[0])]}"
        )
    bad_sums = ~np.isclose(probabilities.sum(axis=1), 1.0, atol=PROBABILITY_SUM_ATOL, rtol=0)
    if bad_sums.any():
        raise ValueError(
            f"Probability rows must sum to 1 within {PROBABILITY_SUM_ATOL}; invalid lesion: "
            f"{ids[int(np.flatnonzero(bad_sums)[0])]}. Export probabilities, not logits."
        )
    y_true = np.array([CLASSES.index(row["label"]) for row in labels], dtype=int)
    return ids, y_true, probabilities


def calculate_metrics(y_true, probabilities):
    """Calculate metrics from validated arrays returned by load_evaluation_data."""
    y_pred = probabilities.argmax(axis=1)
    indices = np.arange(len(CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    normalized = np.divide(
        matrix, support[:, None], out=np.zeros_like(matrix, dtype=float),
        where=support[:, None] != 0,
    )
    per_class = [
        {"class": c, "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(support[i]),
         "predicted_count": int(matrix[:, i].sum())}
        for i, c in enumerate(CLASSES)
    ]
    aggregate = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
    }
    return aggregate, per_class, matrix, normalized, y_pred


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def plot_confusion(matrix, support, path, model_name, *, normalized=False):
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    fig.subplots_adjust(left=0.13, right=0.89, top=0.83, bottom=0.17)
    title = "Confusion matrix | true-class percentages" if normalized else "Confusion matrix | lesion counts"
    fig.suptitle(title, x=0.13, y=0.975, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.13, 0.91, textwrap.fill(model_name, width=85), fontsize=10, color="#526373")
    upper = 1.0 if normalized else max(1, int(matrix.max()))
    rendered = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=upper)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class", labelpad=12)
    ax.set_ylabel("True class", labelpad=12)
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
    plt.close(fig)


def source_record(path):
    path = Path(path).resolve()
    try:
        displayed = path.relative_to(ROOT).as_posix()
    except ValueError:
        displayed = path.as_posix()
    return {"path": displayed, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def evaluate_files(labels_path, predictions_path, output_dir, *, model_name=None, overwrite=False):
    labels_path, predictions_path, output_dir = map(Path, (labels_path, predictions_path, output_dir))
    ids, y_true, probabilities = load_evaluation_data(labels_path, predictions_path)
    aggregate, per_class, matrix, normalized, y_pred = calculate_metrics(y_true, probabilities)
    model_name = model_name or predictions_path.stem
    input_paths = {labels_path.resolve(), predictions_path.resolve()}
    targets = [output_dir / name for name in OUTPUT_FILES]
    if any(path.resolve() in input_paths for path in targets):
        raise ValueError("Output files would overwrite an input CSV; choose another output directory")
    existing = [path.name for path in targets if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"Output files already exist: {', '.join(existing)}. "
            "Choose another directory or use --overwrite."
        )
    absent = [row["class"] for row in per_class if row["support"] == 0]
    warnings = []
    if absent:
        warnings.append(
            f"No true examples for {', '.join(absent)}. Undefined per-class metrics are 0; "
            "Macro F1 still averages all 11 classes. Compare the same label split across runs."
        )
    maxima = probabilities.max(axis=1, keepdims=True)
    ties = int(np.count_nonzero((probabilities == maxima).sum(axis=1) > 1))
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
    # Finish rendering in a temporary directory before publishing any result files.
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output_dir) as temp:
        stage = Path(temp)
        (stage / "metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        write_csv(stage / "per_class_metrics.csv", list(per_class[0]),
                  [list(row.values()) for row in per_class])
        for name, values in (("confusion_matrix.csv", matrix),
                             ("confusion_matrix_normalized.csv", normalized)):
            write_csv(stage / name, ("true_class", *CLASSES),
                      [[c, *values[i].tolist()] for i, c in enumerate(CLASSES)])
        write_csv(stage / "predictions_aligned.csv",
                  ("lesion_id", "true_label", "predicted_label", "correct", *CLASSES),
                  [[lesion, CLASSES[y_true[i]], CLASSES[y_pred[i]], int(y_true[i] == y_pred[i]),
                    *probabilities[i].tolist()] for i, lesion in enumerate(ids)])
        support = matrix.sum(axis=1)
        plot_confusion(matrix, support, stage / "confusion_matrix.png", model_name)
        plot_confusion(normalized, support, stage / "confusion_matrix_normalized.png",
                       model_name, normalized=True)
        for name in OUTPUT_FILES:
            (stage / name).replace(output_dir / name)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="CSV with lesion_id and 11 probability columns")
    parser.add_argument("--labels", type=Path, default=ROOT / "DATA/PROCESSED_DATA/val_split.csv",
                        help="True-label CSV; default is the repository validation split")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for metrics, tables and figures")
    parser.add_argument("--model-name", help="Display name stored in metrics and figures; defaults to prediction filename")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing evaluation output files")
    args = parser.parse_args(argv)
    try:
        summary = evaluate_files(args.labels, args.predictions, args.output_dir,
                                 model_name=args.model_name, overwrite=args.overwrite)
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(f"Evaluated {summary['n_lesions']:,} lesions: {summary['model_name']}")
    print(f"Accuracy: {summary['accuracy']:.6f} | Macro F1: {summary['macro_f1']:.6f}")
    for warning in summary["warnings"]:
        print(f"Warning: {warning}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
