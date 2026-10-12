"""Behavioral tests: known answers, ID alignment, invalid exports and CLI output."""
import csv
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
evaluation = importlib.import_module("evaluate")
CLASSES = evaluation.CLASSES


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vision-eval-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.labels = self.directory / "labels.csv"
        self.predictions = self.directory / "predictions.csv"
        self.label_rows = [{"lesion_id": f"synthetic_{i:02d}", "label": c}
                           for i, c in enumerate(CLASSES)]
        self.prediction_rows = [
            {"lesion_id": row["lesion_id"], **{c: int(c == row["label"]) for c in CLASSES}}
            for row in self.label_rows
        ]
        self.write(self.labels, ("lesion_id", "label"), self.label_rows)
        self.write(self.predictions, ("lesion_id", *CLASSES), self.prediction_rows)

    @staticmethod
    def write(path, columns, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def load(self):
        return evaluation.load_evaluation_data(self.labels, self.predictions)

    def test_perfect_predictions_for_all_classes(self):
        _, truth, probabilities = self.load()
        scores, per_class, matrix, normalized, _ = evaluation.calculate_metrics(truth, probabilities)
        self.assertEqual(scores, dict(accuracy=1.0, macro_f1=1.0, macro_precision=1.0, macro_recall=1.0))
        np.testing.assert_array_equal(matrix, np.eye(11))
        np.testing.assert_array_equal(normalized, np.eye(11))
        self.assertTrue(all(row["support"] == 1 for row in per_class))

    def test_rows_and_columns_match_by_name_not_position(self):
        _, expected_truth, expected_probabilities = self.load()
        columns = (*reversed(CLASSES), "lesion_id")
        self.write(self.predictions, columns, list(reversed(self.prediction_rows)))
        ids, truth, probabilities = self.load()
        self.assertEqual(ids, [r["lesion_id"] for r in self.label_rows])
        np.testing.assert_array_equal(truth, expected_truth)
        np.testing.assert_array_equal(probabilities, expected_probabilities)

    def test_hand_calculated_errors_and_absent_classes(self):
        bcc, nv = CLASSES.index("BCC"), CLASSES.index("NV")
        truth = np.array([bcc, bcc, nv, nv])
        probabilities = np.eye(11)[[bcc, nv, nv, nv]]
        scores, rows, matrix, normalized, _ = evaluation.calculate_metrics(truth, probabilities)
        self.assertEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(rows[bcc]["precision"], 1.0)
        self.assertAlmostEqual(rows[bcc]["recall"], 0.5)
        self.assertAlmostEqual(rows[bcc]["f1"], 2 / 3)
        self.assertAlmostEqual(rows[nv]["precision"], 2 / 3)
        self.assertAlmostEqual(rows[nv]["recall"], 1.0)
        self.assertAlmostEqual(scores["macro_f1"], (2 / 3 + 0.8) / 11)
        self.assertEqual(matrix[bcc, nv], 1)
        self.assertEqual(normalized[bcc, nv], 0.5)
        self.assertEqual(rows[0]["support"], 0)
        np.testing.assert_array_equal(normalized[0], 0)

    def test_real_validation_constant_bcc_reference(self):
        label_file = ROOT / "DATA/PROCESSED_DATA/val_split.csv"
        with label_file.open(encoding="utf-8-sig", newline="") as stream:
            labels = list(csv.DictReader(stream))
        predictions = [
            {"lesion_id": r["lesion_id"], **{c: int(c == "BCC") for c in CLASSES}}
            for r in reversed(labels)
        ]
        self.write(self.predictions, ("lesion_id", *CLASSES), predictions)
        _, truth, probabilities = evaluation.load_evaluation_data(label_file, self.predictions)
        scores, _, matrix, _, _ = evaluation.calculate_metrics(truth, probabilities)
        bcc_count = sum(row["label"] == "BCC" for row in labels)
        self.assertEqual(len(labels), 1048)
        self.assertEqual(bcc_count, 504)
        self.assertAlmostEqual(scores["accuracy"], 504 / 1048)
        self.assertAlmostEqual(scores["macro_f1"], (2 * 504 / (1048 + 504)) / 11)
        self.assertEqual(matrix[:, CLASSES.index("BCC")].sum(), 1048)

    def test_invalid_probability_values_and_sums_rejected(self):
        for value in ("nan", "inf", "-0.1", "1.1", "", "text", "0.5"):
            with self.subTest(value=value):
                rows = [dict(r) for r in self.prediction_rows]
                rows[0][CLASSES[0]] = value
                self.write(self.predictions, ("lesion_id", *CLASSES), rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_missing_extra_duplicate_and_whitespace_ids_rejected(self):
        variants = [
            self.prediction_rows[:-1],
            self.prediction_rows + [{**self.prediction_rows[0], "lesion_id": "extra"}],
            self.prediction_rows + [self.prediction_rows[0]],
            [{**self.prediction_rows[0], "lesion_id": " synthetic_00"}, *self.prediction_rows[1:]],
        ]
        for rows in variants:
            with self.subTest(length=len(rows)):
                self.write(self.predictions, ("lesion_id", *CLASSES), rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_duplicate_true_ids_unknown_label_and_partial_one_hot_rejected(self):
        variants = [
            (("lesion_id", "label"), self.label_rows + [self.label_rows[0]]),
            (("lesion_id", "label"), [{**self.label_rows[0], "label": "UNKNOWN"}]),
            (("lesion_id", "label", "BCC"), [{**self.label_rows[0], "BCC": 0}]),
        ]
        for columns, rows in variants:
            with self.subTest(columns=columns):
                self.write(self.labels, columns, rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_mismatched_one_hot_rejected(self):
        rows = [{**self.label_rows[0], **{c: int(c == "BCC") for c in CLASSES}}]
        self.write(self.labels, ("lesion_id", *CLASSES, "label"), rows)
        with self.assertRaisesRegex(ValueError, "one-hot"):
            self.load()

    def test_malformed_prediction_csv_rejected(self):
        for text in (
            "lesion_id,lesion_id\na,b\n",
            "lesion_id," + ",".join(CLASSES) + "\n",
            "lesion_id," + ",".join(CLASSES) + "\nx,1\n",
            "lesion_id," + ",".join(CLASSES) + "\nx," + ",".join(["0"] * 12) + "\n",
        ):
            with self.subTest(text=text[:30]):
                self.predictions.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.load()

    def test_probability_tolerance_and_tie_order(self):
        rows = [dict(r) for r in self.prediction_rows]
        rows[0][CLASSES[0]] = 0.999999
        rows[1][CLASSES[0]], rows[1][CLASSES[1]] = 0.5, 0.5
        self.write(self.predictions, ("lesion_id", *CLASSES), rows)
        _, truth, probabilities = self.load()
        *_, predicted = evaluation.calculate_metrics(truth, probabilities)
        self.assertEqual(predicted[1], 0)

    def test_cli_from_another_directory_and_safe_rerun(self):
        out = self.directory / "results"
        command = [sys.executable, str(ROOT / "src/evaluate.py"),
                   "--labels", str(self.labels), "--predictions", str(self.predictions),
                   "--output-dir", str(out), "--model-name", "Synthetic perfect test"]
        first = subprocess.run(command, cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual({p.name for p in out.iterdir()}, set(evaluation.OUTPUT_FILES))
        summary = json.loads((out / "metrics.json").read_text())
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["macro_f1"], 1.0)
        self.assertEqual(summary["n_lesions"], 11)
        for filename in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            self.assertEqual((out / filename).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        second = subprocess.run(command, cwd=self.directory, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("--overwrite", second.stderr)
        self.assertEqual(summary, json.loads((out / "metrics.json").read_text()))
        third = subprocess.run([*command, "--overwrite"], cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(third.returncode, 0, third.stderr)

    def test_output_cannot_replace_input_even_with_overwrite(self):
        renamed = self.directory / "predictions_aligned.csv"
        self.predictions.rename(renamed)
        original = renamed.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            evaluation.evaluate_files(self.labels, renamed, self.directory, overwrite=True)
        self.assertEqual(renamed.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
