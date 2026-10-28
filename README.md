# Machine-Vision-Project CSC490H5

Skin-lesion classification with ResNet-50 using dermoscopic and clinical images.
The current plan uses nine diagnostic classes: E00 (dermoscopic), E01 (clinical),
and E02 (simple two-image feature fusion), each with five independent folds.
E03–E09 are later controlled comparisons.

## Delivery status — 2026-10-28

This C-owned branch supplies evaluation, CV summaries, protocol tests, CI and
public documentation. **A's data/schema delivery and B's training/configuration
delivery are still pending in public.** The complete nine-class implementation
has been prepared internally; this branch alone cannot run it. CI reports missing
dependencies explicitly. Merge the three deliveries and run the checks before
treating the public integration as complete.

Real-data GPU smoke checks and formal training have not started. There are no
new E00–E09 model-performance results. Historical PanDerm, MONET and Gate results
are separate work. Existing label-distribution reports and the synthetic
evaluation demonstration are not evidence of trained-model performance.

## Experiment and data protocol

| Item | New nine-class protocol |
| --- | --- |
| Classes | AKIEC, BCC, BKL, DF, INF, MEL, NV, SCCKA, VASC |
| MILK development set | 3,968 lesions, five grouped folds |
| Fixed MILK holdout | 1,037 lesions; evaluated after final refit |
| ISIC expansion pool | 20,279 images; training side only in selected experiments |
| Core models | E00 dermoscopic ResNet-50; E01 clinical ResNet-50; E02 two independent ResNet-50 feature encoders with concatenation |
| Selection | At most 100 epochs per fold; best validation Macro F1; stop after 15 epochs without improvement |
| Reporting | Five-fold mean, sample standard deviation and pooled out-of-fold predictions |
| Final refit | Median of the five best epochs; full development set training, then fixed holdout evaluation |

Both MILK sets currently have both modalities. ISIC has no clinical image; the
fusion interface uses a modality mask. Conservative similarity-group isolation
excluded 182 development lesions associated with holdout groups and isolated
related/conflicting ISIC groups. Candidates are not manually confirmed duplicates,
and original images were not deleted. The official tables contain no patient
identifier, so patient-level separation is not established. The inherited holdout
is not a never-used external blind test.

Images, per-case protocol manifests, similarity caches and weights are obtained
or generated locally. A's delivery provides aggregate counts and manifest hashes.
Do not substitute historical splits for the new manifests or create empty files
to bypass input checks.

## Repository guide

| Path | Role / delivery owner |
| --- | --- |
| `src/evaluate.py`, `src/metrics.py` | Fixed-schema evaluation and training metric adapter — C |
| `scripts/summarize_cv.py`, `scripts/collect_results.py` | Verified OOF summaries and formal-result collection — C |
| `scripts/check_training_ready.py`, `scripts/check_smoke_run.py` | Input checks and smoke/restart verification — C; require A/B interfaces |
| `tests/test_resnet_protocol.py`, `tests/test_evaluation_schemas.py` | Artificial-input regression tests — C; require A/B dependencies |
| `src/data/`, `src/class_schema.py`, `configs/schema/label_mapping.csv` | Data, grouping and class contract — A, pending public delivery |
| `src/experiment.py`, `src/models/fusion.py`, `configs/experiments/` | New training entry point, models and configurations — B, pending public delivery |
| `reports/experiments/` | E00–E09 status table; all 50 fold rows remain planned |
| `DATA/PROCESSED_DATA/`, `src/train.py` | Existing 11-class labels and legacy training entry point |
| `examples/evaluation/`, `tests/test_evaluate.py` | Existing synthetic demonstration and evaluator compatibility tests |

Read the [experiment plan](docs/experiment_plan.md),
[contribution record](docs/contributions.md), and
[evaluation contract](docs/evaluation.md). A's `docs/data_protocol.md` and B's
`docs/training.md` are pending delivery; link them here when those files land.
Cluster environment setup and job submission are maintained internally.

## Offline setup and checks

Use a Python 3.13 virtual environment, with commands run from the repository root.
On Windows, replace `python` below with `.\.venv\Scripts\python.exe` after creating
the environment; alternatively activate it first.

```powershell
python -m venv .venv
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
```

After A/B dependencies have been integrated, run:

```powershell
python -B -m unittest discover -s tests -p test_resnet_protocol.py -v
python -B -m unittest discover -s tests -p test_evaluation_schemas.py -v
python -B -m unittest discover -s tests -p test_evaluate.py -v
```

The first two suites test the new protocol and schemas using artificial inputs.
The third preserves the existing evaluator checks, including constructed
constant-class predictions against committed historical labels. No test downloads
weights or trains on real images. CPU tests do not validate GPU memory, throughput,
batch size or cluster execution.

## Unified evaluation

Select the schema and identifier explicitly and supply the actual fold labels:

```powershell
python src/evaluate.py --class-schema 9 --id-column sample_id --labels DATA/manifests/folds/fold_0/val.csv --predictions path/to/predictions.csv --output-dir reports/evaluation/E00_fold_0
```

Predictions cover every label ID exactly once, with one probability column per
selected class. Outputs include Macro F1, Accuracy, macro precision/recall,
per-class metrics, aligned predictions and confusion matrices. Zero-support
classes remain in macro averages. See the [evaluation guide](docs/evaluation.md).

The existing 11-class artificial demonstration is retained:

```powershell
python src/evaluate.py --class-schema 11 --labels examples/evaluation/synthetic_labels.csv --predictions examples/evaluation/synthetic_predictions.csv --output-dir reports/evaluation/demo_local --model-name "Synthetic demo - not trained model performance"
```

## Historical baseline and reports

The historical split contains 4,192 training and 1,048 validation lesions;
`DATA/RAW_DATA/MILK10k_Training_Metadata.csv` maps 10,480 image records to 5,240
lesions. These are the earlier 11-class data contract, separate from the new
nine-class manifests. The legacy `src/train.py` entry point remains for
compatibility; follow B's new training guide for formal experiments.

The [distribution analysis](reports/class_distribution/analysis.md),
[class counts](reports/class_distribution/class_distribution.csv), and
[synthetic report](reports/evaluation_demo/README.md) are preserved.
Reproduce the historical label analysis without images or a GPU:

```powershell
python src/analyze_class_distribution.py
```

Next comes deployment and real-data GPU smoke verification on Nibi, freezing
runtime settings and running the core 15 folds. This public delivery does not
perform those actions.
