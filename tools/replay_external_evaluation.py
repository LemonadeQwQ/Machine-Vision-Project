"""Recheck archived external predictions with the shared evaluator, without images or weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.class_schema import NINE_CLASSES
from src.evaluate import evaluate_files, load_evaluation_data

def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"Duplicate CSV columns: {path}")
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed CSV: {path}")
    return fields, rows


def canonical_predictions(predictions_path, labels_path):
    """Validate archive annotations before stripping them from the evaluator input."""
    fields, rows = read_rows(predictions_path)
    expected = {"lesion_id", "source", "truth", "prediction", *NINE_CLASSES}
    if set(fields) != expected:
        raise ValueError("Expected the archived nine-class prediction schema")
    _, labels = read_rows(labels_path)
    by_id = {row["lesion_id"]: row for row in labels}
    if len(by_id) != len(labels):
        raise ValueError("Duplicate label lesion_id")
    ids = [row["lesion_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise ValueError("Prediction/label lesion IDs do not match exactly")
    for row in rows:
        label = by_id[row["lesion_id"]]
        if row["truth"] != label["label"] or row["source"] != label["source_dataset"]:
            raise ValueError(f"Archived truth/source disagrees with labels: {row['lesion_id']}")
        probabilities = [float(row[c]) for c in NINE_CLASSES]
        predicted = NINE_CLASSES[max(range(len(NINE_CLASSES)), key=probabilities.__getitem__)]
        if row["prediction"] != predicted:
            raise ValueError(f"Archived prediction disagrees with argmax: {row['lesion_id']}")
    return [{key: row[key] for key in ("lesion_id", *NINE_CLASSES)} for row in rows]


def replay(archive, labels, output):
    archive, labels, output = Path(archive), Path(labels), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    # Validate both models before creating outputs. Original predictions stay immutable.
    prepared = {name: canonical_predictions(archive / f"{name}_predictions.csv", labels)
                for name in ("nine_derm", "nine_fusion")}
    output.mkdir(parents=True)
    results = {}
    for name, rows in prepared.items():
        target = output / f"{name}_canonical_predictions.csv"
        with target.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("lesion_id", *NINE_CLASSES))
            writer.writeheader()
            writer.writerows(rows)
        load_evaluation_data(labels, target, class_names=NINE_CLASSES)
        summary = evaluate_files(labels, target, output / name, class_names=NINE_CLASSES,
                                 model_name=f"Archived external predictions: {name}")
        recorded = json.loads((archive / f"{name}_metrics.json").read_text(encoding="utf-8-sig"))
        # The stage reports torch.float32 scores; the evaluator uses float64.
        for metric in ("accuracy", "macro_f1"):
            if abs(summary[metric] - recorded[metric]) > 2e-6:
                raise ValueError(f"Archived {name} {metric} does not reproduce")
        if abs(summary["macro_recall"] - recorded["balanced_accuracy"]) > 2e-6:
            raise ValueError(f"Archived {name} balanced accuracy does not reproduce")
        with (output / name / "confusion_matrix.csv").open(encoding="utf-8", newline="") as stream:
            matrix = [[int(row[c]) for c in NINE_CLASSES] for row in csv.DictReader(stream)]
        if matrix != recorded["confusion_matrix"]:
            raise ValueError(f"Archived {name} confusion matrix does not reproduce")
        results[name] = {key: summary[key] for key in ("n_lesions", "accuracy", "macro_f1", "macro_recall")}
    result = {
        "verification": "Recomputed supplied predictions; no training or model inference was run.",
        "class_order": list(NINE_CLASSES), "models": results,
        "input_sha256": {name: hashlib.sha256((archive / name).read_bytes()).hexdigest()
                         for name in ("nine_derm_predictions.csv", "nine_fusion_predictions.csv")},
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.archive, args.labels, args.output_dir), indent=2))
