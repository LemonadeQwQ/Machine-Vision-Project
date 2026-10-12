# Unified model evaluation

Evaluate one prediction per lesion against the existing labels. Every model uses the
same 11 classes, ID matching and metric rules. Images, model weights and PyTorch are
not needed to run the evaluator once a probability CSV is available.

## Install and run

The dependency versions were tested with Python 3.13. In your selected Python
environment, run these commands from the repository root:

```powershell
python -m pip install -r requirements-evaluation.txt
python src/evaluate.py --predictions path/to/predictions.csv --output-dir reports/evaluation/run_01 --model-name model_01
```

Replace `path/to/predictions.csv` with the model's exported prediction file.
`--labels` defaults to `DATA/PROCESSED_DATA/val_split.csv`, resolved from the repository
location. Explicit relative paths are resolved from the terminal's current directory.
`--output-dir` is required so each experiment has a deliberate results location.
`--model-name` is a display label; it does not select or load a model.
On Windows, you can replace `python` with `.\.venv\Scripts\python.exe` to use the
environment described in the repository README.

Existing result filenames are protected. Choose a new output directory for a new
experiment. To replace a previous evaluation deliberately, add `--overwrite`.
Inputs cannot be overwritten by output files, even with that flag.

## Prediction file contract

The CSV must have exactly these column names:

```text
lesion_id,AKIEC,BCC,BEN_OTH,BKL,DF,INF,MAL_OTH,MEL,NV,SCCKA,VASC
```

- Use one row per lesion, after combining paired-image or multiple-backbone outputs.
- Use the original `lesion_id`, without whitespace, file extensions or renamed IDs.
- Include every lesion in the label file exactly once, with no additional lesions.
- Export probabilities between 0 and 1, not logits, class indices or percentages.
  Each row must sum to 1 with absolute tolerance `1e-5`; export sufficient precision.
  Values are checked but are not automatically renormalized.
- Both CSV row order and probability-column order may vary: the evaluator matches
  IDs and class column names. Headers are case-sensitive. Missing, duplicate or
  extra probability columns are rejected.
- Each named column must actually contain that class's probability. A model with a
  different internal class order must reorder its outputs before naming the columns;
  this semantic mismatch cannot be inferred from numeric probabilities.

The canonical class order is defined once in [class_schema.py](../src/class_schema.py).
The data-distribution analysis and evaluator both import it.

For example, export existing probabilities from code running at the repository root:

```python
import csv
from src.class_schema import CLASSES

# lesion_ids: the original IDs from the inference batches.
# probabilities: N rows, each with 11 probabilities in CLASSES order.
assert len(lesion_ids) == len(probabilities)
assert all(len(row) == len(CLASSES) for row in probabilities)
with open("predictions.csv", "w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("lesion_id", *CLASSES))
    writer.writerows((lesion, *row) for lesion, row in zip(lesion_ids, probabilities))
```

## True labels

The label CSV requires `lesion_id,label`; labels must use the class names above.
The existing validation CSV is supported directly. Its 11 one-hot columns, when
present, must all be present and agree with `label`. Unrelated metadata columns are
allowed in the label file. Empty or duplicate IDs and unknown labels are errors.

Use the same held-out label file for model comparisons. The evaluator cannot check
whether a model trained on those lesions or whether different lesions belong to the
same patient. Those split and training decisions remain the training pipeline's job.

## Metric definitions

The predicted class is the largest probability. Exact ties choose the first class
in the canonical order; the number of tied rows is recorded in `metrics.json`.

- **Accuracy:** correct predictions divided by all evaluated lesions.
- **Macro F1:** arithmetic mean of the 11 per-class F1 scores, the primary metric.
- **Macro precision and recall:** arithmetic means across the same 11 classes.
- **Per-class metrics:** precision, recall, F1, true support and predicted count.
- **Confusion matrices:** rows are true classes; columns are predictions. The
  normalized matrix divides each row by its true-class support.

JSON and CSV metric values are fractions from 0 to 1, not percentages. Undefined
precision, recall or F1 is set to 0. All 11 classes remain in every table and macro
average, including absent classes. Consequently, perfect predictions on a subset
that omits classes can have Macro F1 below 1. The JSON records an absent-class warning.
An unsupported normalized matrix row is stored as zeros and displayed as `n/a`.

The supplied validation split contains only two MAL_OTH lesions. A single changed
prediction can substantially affect that class's recall and F1; report support
alongside the score.

## Output files

| File | Contents |
| --- | --- |
| `metrics.json` | Aggregate scores, class order, policies, model name, input hashes and runtime versions |
| `per_class_metrics.csv` | Precision, recall, F1, support and predicted count for all 11 classes |
| `predictions_aligned.csv` | IDs, true and predicted labels, correctness and probabilities in label-file order |
| `confusion_matrix.csv` | Integer counts with class names on both axes |
| `confusion_matrix_normalized.csv` | Fractions within each true class |
| `confusion_matrix.png` | Count heatmap |
| `confusion_matrix_normalized.png` | Percentage heatmap |

## Run without a trained model

The included example contains 22 artificial IDs, two per class, with four deliberate
errors. Its scores verify the evaluator and are **not model performance**.
The prediction rows are reversed to demonstrate matching by ID.

```powershell
python src/evaluate.py --labels examples/evaluation/synthetic_labels.csv --predictions examples/evaluation/synthetic_predictions.csv --output-dir reports/evaluation_demo --model-name "Synthetic demo - not trained model performance" --overwrite
python -m unittest discover -s tests -v
```

The demonstration's expected accuracy is `18 / 22 = 0.818181...`.
Tests also cover a perfect 11-class prediction, hand-calculated errors, missing
classes, reordered rows and columns, invalid inputs, and a theoretical constant-BCC
prediction on the real validation labels: accuracy `504 / 1048 = 0.480916...`.
The latter uses constructed predictions and is not a trained model result.
