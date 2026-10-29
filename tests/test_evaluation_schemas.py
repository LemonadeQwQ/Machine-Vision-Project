"""Regression checks for explicit schemas and the archived external CSV adapter."""
import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluate as evaluation
from src.class_schema import CLASSES, EIGHT_CLASSES, NINE_CLASSES, validate_class_order
from src.evaluate import calculate_metrics, load_evaluation_data
from tools.replay_external_evaluation import canonical_predictions


class SchemaTests(unittest.TestCase):
    def test_legacy_cli_defaults_to_the_existing_public_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root / 'DATA/PROCESSED_DATA/val_split.csv'
            labels.parent.mkdir(parents=True)
            labels.write_text('lesion_id,label\nsynthetic-1,BCC\n', encoding='utf-8')
            predictions = root / 'predictions.csv'
            with predictions.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('lesion_id', *CLASSES))
                writer.writerow(('synthetic-1', *(float(c == 'BCC') for c in CLASSES)))
            output = root / 'evaluation'
            with patch.object(evaluation, 'ROOT', root), redirect_stdout(io.StringIO()):
                evaluation.main(['--predictions', str(predictions), '--output-dir', str(output)])
            metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['n_lesions'], 1)
            self.assertEqual(metrics['accuracy'], 1.0)
            self.assertAlmostEqual(metrics['macro_f1'], 1 / 11)

    def test_new_schemas_and_sample_ids_require_explicit_labels(self):
        for schema, identifier in [('9', 'sample_id'), ('9', 'lesion_id'),
                                   ('8', 'lesion_id'), ('11', 'sample_id')]:
            with self.subTest(schema=schema, identifier=identifier):
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as failure:
                    evaluation.main(['--class-schema', schema, '--id-column', identifier,
                                     '--predictions', 'unused.csv', '--output-dir', 'unused'])
                self.assertEqual(failure.exception.code, 2)
                self.assertIn('--labels is required', errors.getvalue())

    def test_eight_and_nine_schemas_keep_absent_classes_in_macro_f1(self):
        for classes in (EIGHT_CLASSES, NINE_CLASSES):
            with self.subTest(classes=classes):
                truth = np.array([0, 0, 1, 1])
                probs = np.eye(len(classes))[[0, 1, 1, 1]]
                scores, _, matrix, _, _ = calculate_metrics(truth, probs, class_names=classes)
                self.assertEqual(scores["accuracy"], .75)
                self.assertAlmostEqual(scores["macro_f1"], (2 / 3 + .8) / len(classes))
                self.assertEqual(matrix.shape, (len(classes), len(classes)))
                with self.assertRaises(ValueError):
                    validate_class_order(tuple(reversed(classes)))

    def test_archived_annotations_must_agree_with_canonical_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            labels, predictions, canonical = (Path(folder) / n for n in ("labels.csv", "predictions.csv", "canonical.csv"))
            labels.write_text("lesion_id,label,source_dataset\nL1,BCC,HIBA\n", encoding="utf-8")
            row = dict(lesion_id="L1", truth="BCC", prediction="BCC", source="HIBA",
                       **{c: float(c == "BCC") for c in NINE_CLASSES})
            def write(path, rows):
                with path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            write(predictions, [row])
            write(canonical, canonical_predictions(predictions, labels))
            _, truth, probs = load_evaluation_data(labels, canonical, class_names=NINE_CLASSES)
            self.assertEqual(probs.argmax(1).tolist(), truth.tolist())
            for key, bad in (("truth", "NV"), ("source", "MIDAS"), ("prediction", "NV")):
                with self.subTest(key=key):
                    write(predictions, [{**row, key: bad}])
                    with self.assertRaises(ValueError):
                        canonical_predictions(predictions, labels)


if __name__ == "__main__":
    unittest.main()
