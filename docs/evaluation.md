# Unified model evaluation

The current ResNet protocol uses nine classes and `sample_id`. Select both
explicitly and supply the actual fold manifest, generated locally and excluded
from the public case-level data delivery. The updated class schema, data and
training dependencies are included in the integrated A/B/C public code.

## Install and run

Evaluation alone needs `requirements-evaluation.txt`; it does not load images,
PyTorch or model weights. Run from the repository root:

```powershell
python -m pip install -r requirements-evaluation.txt
python src/evaluate.py --class-schema 9 --id-column sample_id --labels DATA/manifests/folds/fold_0/val.csv --predictions path/to/predictions.csv --output-dir reports/evaluation/E00_fold_0 --model-name E00_fold_0
```

Explicit relative paths resolve from the terminal's current directory.
`--model-name` is a display label, not a model selector. Existing result filenames
are protected; choose a new directory or deliberately pass `--overwrite`.
Input files cannot be overwritten by outputs even with that flag.

## Explicit schemas and prediction contract

| Schema | Probability columns in canonical order | Use |
| --- | --- | --- |
| 9 | AKIEC,BCC,BKL,DF,INF,MEL,NV,SCCKA,VASC | Current ResNet protocol |
| 11 | AKIEC,BCC,BEN_OTH,BKL,DF,INF,MAL_OTH,MEL,NV,SCCKA,VASC | Existing legacy evaluator |
| 8 | AKIEC,BCC,BKL,DF,MEL,NV,SCCKA,VASC | Historical comparison excluding INF |

The schema is never inferred from the CSV. Legacy defaults remain
`--class-schema 11 --id-column lesion_id`; with labels omitted in this mode,
the evaluator uses `DATA/PROCESSED_DATA/val_split.csv` relative to the repository.
Every other schema/identifier combination requires an explicit `--labels` path,
avoiding silent selection of a historical split for a new experiment.

The new protocol's exact prediction header is:

```text
sample_id,AKIEC,BCC,BKL,DF,INF,MEL,NV,SCCKA,VASC
```

- Export one row per case, using the manifest's unchanged `sample_id`.
- Include every label ID once and no extras. Empty, duplicate or whitespace-padded
  IDs are rejected.
- Export finite probabilities in `[0, 1]`, not logits or percentages. Each row
  must sum to 1 within absolute tolerance `1e-5`; values are not renormalized.
- Row order and probability-column order may vary. Matching uses ID and column
  names; missing, duplicate or extra prediction columns are rejected.
- Each column must contain the named class's actual probability. Reorder model
  outputs before naming columns; numeric validation cannot detect a semantic swap.

Labels require `sample_id,label` (or `lesion_id,label` for legacy evaluation),
using the selected class names. If any one-hot class column is present, all
selected class columns must be present and agree with `label`. Other label
metadata is allowed.

```python
import csv
from src.class_schema import NINE_CLASSES

# sample_ids and probabilities come from the same inference batches.
# Each probability row must already use NINE_CLASSES order.
assert len(sample_ids) == len(probabilities)
assert all(len(row) == len(NINE_CLASSES) for row in probabilities)
with open("predictions.csv", "w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("sample_id", *NINE_CLASSES))
    writer.writerows((sample, *row) for sample, row in zip(sample_ids, probabilities))
```

Use the same population, schema and selection protocol for model comparisons.
The evaluator cannot determine whether a model trained on a case or whether
different lesions belong to the same patient.

## Metric definitions

The predicted class has the largest probability. Exact ties choose the first
class in canonical order; tied-row counts are recorded in `metrics.json`.

- **Macro F1:** arithmetic mean of all selected per-class F1 values; primary metric.
- **Accuracy:** correct predictions divided by the evaluated case count.
- **Macro precision / recall:** arithmetic means across the same fixed class set.
- **Per-class metrics:** precision, recall, F1, true support and predicted count.
- **Confusion matrices:** rows are truth, columns are predictions; normalized rows
  divide by true-class support.

Scores are fractions in `[0, 1]`. Undefined precision, recall and F1 are zero.
Absent classes remain in macro averages and trigger a warning; perfect predictions
on a subset missing classes can therefore have Macro F1 below 1. An unsupported
normalized matrix row is zero and displayed as `n/a`.

## Output files

| File | Contents |
| --- | --- |
| `metrics.json` | Aggregate scores, class order, policies, model name, input hashes and runtime versions |
| `per_class_metrics.csv` | Precision, recall, F1, support and predicted count for the selected schema |
| `predictions_aligned.csv` | IDs, truth, predictions, correctness and probabilities in label-file order |
| `confusion_matrix.csv` | Integer counts |
| `confusion_matrix_normalized.csv` | Fractions within each true class |
| `confusion_matrix.png` | Count heatmap |
| `confusion_matrix_normalized.png` | Percentage heatmap |

## Five-fold summaries and final evaluation

After all five formal folds complete:

```powershell
python scripts/summarize_cv.py --runs runs/E00 --manifests DATA/manifests --output reports/evaluation/E00_cv
python scripts/collect_results.py --runs runs --output reports/experiments/experiment_status.csv
```

Summaries require five completed formal CV runs with matching experiment,
configuration, protocol, source fingerprint and manifest version. The tool
recomputes fold scores from probabilities and checks each development case occurs
exactly once in OOF. Fold means, sample standard deviations and pooled OOF are
reported separately. Partial runs, smoke runs and mixed versions are rejected.

Each fold selects the highest validation Macro F1, retaining the earliest tied
epoch. Final refit uses the median of the five best epochs on the full development
set. Only then evaluate the fixed holdout; do not tune settings or select epochs
on it. See the [experiment plan](experiment_plan.md). Real results remain pending;
artificial smoke results never populate the formal results table.

## Synthetic checks and historical compatibility

The existing 11-class demonstration has 22 artificial IDs and four deliberate
errors, giving expected accuracy `18 / 22`. It verifies ID alignment and metrics,
not model performance:

```powershell
python src/evaluate.py --class-schema 11 --labels examples/evaluation/synthetic_labels.csv --predictions examples/evaluation/synthetic_predictions.csv --output-dir reports/evaluation/demo_local --model-name "Synthetic demo - not trained model performance"
python -B -m unittest discover -s tests -p test_evaluation_schemas.py -v
python -B -m unittest discover -s tests -p test_evaluate.py -v
```

`tools/replay_external_evaluation.py` retains a historical nine-class CSV adapter
used by synthetic schema tests. Its archived model inputs and old external
populations are not this round's results and are not required for CI. Historical
archives are not distributed here; manual replay requires explicit user-supplied
archive and label paths. Their metrics are not new ResNet results. Comparisons
across different evaluated populations do not establish a model improvement.
