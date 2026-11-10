# Machine-Vision-Project CSC490H5

Nine-class skin-lesion classification with ResNet-50, dermoscopic and clinical
images, and controlled comparisons of class weighting, ISIC expansion, metadata
and augmentation.

## A/B/C integration — 2026-11-09

All E00–E10 experiments have completed five independent folds: 55 folds in total,
with 3,968 development cases in each pooled OOF set. E10 (primary) and E07
(control) completed their fixed 13- and 14-epoch refits, followed by evaluation on
the same 1,037-case holdout. This version includes the data/metadata acceptance,
E10 configuration, refit documentation, aggregate results and protocol regressions.

A `29caf7a` (4 paths), B `ed741cb` (7 paths; PR #5), and C `89e2bc0` (76 paths)
are integrated with their original commits and authors preserved. Integration
resolves older documentation conflicts, retains the public evaluation interfaces,
and updates all current completion statements. Data details are in the
[metadata inventory](docs/metadata_inventory.md) and
[final aggregate acceptance](reports/data_audit/20261031_final_acceptance.json).

The October 28/29 deliveries and their original authors remain intact, including
A `0bacf4a`, B `ccc10bd` and C `df34478`, integrated through
[PR #4](https://github.com/LemonadeQwQ/Machine-Vision-Project/pull/4).
Human cross-review is recorded separately in the
[contribution record](docs/contributions.md).

## Development cross-validation

| Experiment | Configuration | Five-fold Macro F1 mean +/- sample SD | Pooled OOF Macro F1 |
| --- | --- | ---: | ---: |
| E00 | Dermoscopy | 48.07% +/- 4.23 pp | 48.81% |
| E01 | Clinical | 38.35% +/- 3.81 pp | 39.23% |
| E02 | Paired fusion | 49.63% +/- 2.39 pp | 49.93% |
| E03 | Dermoscopy + weighted CE | 48.75% +/- 1.72 pp | 48.81% |
| E04 | Dermoscopy + ISIC | 53.59% +/- 2.58 pp | 53.77% |
| E05 | Dermoscopy + ISIC + weighted CE | 54.39% +/- 2.69 pp | 54.76% |
| E06 | Paired + weighted CE | 49.06% +/- 2.84 pp | 48.99% |
| E07 | Paired + ISIC + weighted CE | 54.70% +/- 2.51 pp | 54.89% |
| E08 | Paired + metadata | 48.73% +/- 2.81 pp | 49.12% |
| E09 | Paired, augmentation off | 48.40% +/- 4.05 pp | 48.49% |
| E10 | Paired + ISIC + weighted CE + metadata | 57.00% +/- 2.07 pp | 56.92% |

See the [eleven-experiment CV summary](reports/experiments/20261031_e10/SUMMARY.md)
for exact tables, source fingerprints and aggregate plots. Sample SD is not a
confidence interval. Validation folds selected epochs, so these are development
selection estimates. Historical PanDerm, MONET and Gate results are separate.

## Final refit and fixed holdout

| Model | Role | Refit epochs | Holdout Macro F1 | Accuracy | Cases |
| --- | --- | ---: | ---: | ---: | ---: |
| E10 | Primary | 13 | 56.62% | 70.49% | 1,037 |
| E07 | Control | 14 | 53.28% | 70.01% | 1,037 |

Roles were fixed before final evaluation. E10 was added after inspecting
E00–E09 and selected by highest five-fold mean Macro F1; this was not a
preregistered cross-configuration choice. Both models were newly initialized
and refitted on 3,968 MILK + 20,279 ISIC training rows, with no validation early
stopping. See the [holdout summary](reports/holdout/20261031_holdout_e10_e07_v1/SUMMARY.md)
for class metrics, confusion matrices and limitations. Online training F1 is not
a holdout score. Differences are descriptive, not significance claims.

## Data and protocol

Classes are `AKIEC,BCC,BKL,DF,INF,MEL,NV,SCCKA,VASC`. Five grouped folds use
3,968 MILK development lesions; the fixed holdout has 1,037 lesions. ISIC's
20,279-image pool enters training only for E04/E05/E07/E10. Missing clinical
images use a mask. E10 adds training-fitted age, sex, site and skin-tone metadata
to E07's paired ResNet, ISIC and weighted-CE configuration.

Conservative similarity-group isolation excluded 182 development lesions
associated with holdout groups and isolated related/conflicting ISIC groups.
Candidates are not manually confirmed duplicates; original images were not
deleted. Official tables lack patient IDs, so patient-level separation is not
established. The inherited holdout has historical exposure and is not a
never-used external blind test.

Each fold starts independently from ImageNet V2, runs at most 100 epochs,
selects validation Macro F1 with earliest-epoch tie handling, and stops after
15 epochs without improvement. Refit uses the median of the five best epochs.
See the [CV protocol](docs/cross_validation.md),
[experiment matrix](docs/experiment_plan.md), [data protocol](docs/data_protocol.md)
and [evaluation contract](docs/evaluation.md).

## Repository guide

| Path | Role / delivery owner |
| --- | --- |
| `src/data/`, `src/class_schema.py`, `configs/schema/` | Data, grouping and class contract — A |
| `src/experiment.py`, `src/models/fusion.py`, `configs/experiments/` | Training, models and E00–E10 configurations — B |
| `src/refit.py`, `src/infer.py` | Final refit and checkpoint inference — B |
| `src/evaluate.py`, `src/metrics.py` | Fixed-schema evaluation and metric adapter — C |
| `scripts/summarize_cv.py`, `scripts/collect_results.py` | OOF summary and formal result collection — C |
| `tools/plot_cv_summary.py` | Plot published aggregates without case data — C |
| `reports/experiments/`, `reports/holdout/` | Completed aggregate experimental results — C |
| `tests/test_resnet_protocol.py`, `tests/test_evaluation_schemas.py` | Artificial-input protocol and evaluation regressions — C |
| `DATA/PROCESSED_DATA/`, `src/train.py` | Historical 11-class labels and compatible legacy training entry point |

Images, new case-level manifests/probabilities, weights, complete logs and
cluster deployment records are not distributed. The
[result-format guide](docs/results_format.md) distinguishes public aggregates
from the inputs needed to reproduce training or recompute OOF predictions.

## Offline setup and integration checks

Use Python 3.13. From the repository root, the existing setup is:

```powershell
python -m venv .venv
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
```

Run the three public offline suites from the repository root:

```powershell
python -B -m unittest discover -s tests -p test_resnet_protocol.py -v
python -B -m unittest discover -s tests -p test_evaluation_schemas.py -v
python -B -m unittest discover -s tests -p test_evaluate.py -v
```

These checks use artificial inputs and preserved legacy labels; they do not
establish GPU readiness. The public-specific label defaults, two additional CLI
regressions and existing CI remain intact. Current integration validation is
recorded in the [contribution record](docs/contributions.md); historical acceptance
counts are not substituted for the new candidate's results.

## Unified evaluation

Select the schema and identifier explicitly and provide actual local labels:

```powershell
python src/evaluate.py --class-schema 9 --id-column sample_id --labels DATA/manifests/folds/fold_0/val.csv --predictions path/to/predictions.csv --output-dir reports/evaluation/E00_fold_0
```

Every label ID must occur exactly once with nine named probability columns.
Outputs include Macro F1, Accuracy, macro precision/recall, class metrics and
confusion matrices. Case-level outputs from a local evaluation are not public
delivery files. See the [evaluation guide](docs/evaluation.md).

The existing [distribution analysis](reports/class_distribution/analysis.md)
and [synthetic demonstration](reports/evaluation_demo/README.md) are retained.
Their historical 11-class split (4,192 training / 1,048 validation lesions) is
not the new nine-class protocol. Final report/reproduction work, human review,
and release approval remain separate completion items.
