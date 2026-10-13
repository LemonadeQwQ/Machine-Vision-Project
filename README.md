# Machine-Vision-Project CSC490H5

This project aims to classify skin lesions into 11 diagnostic categories using
paired clinical close-up and dermoscopic images from MILK10k. The current
implementation provides a **single-modality ResNet-50 baseline**: each run uses
either dermoscopic images or clinical close-up images. Paired-image and
multi-backbone feature fusion remain planned work.

## Current implementation

| Component | Available functionality |
| --- | --- |
| Label analysis | Lesion-level class counts, split checks, distribution plots and an analysis report |
| Dataset and DataLoaders | Join split labels to image metadata by `lesion_id`; select one image modality; shuffle training samples and preserve validation order |
| Image preprocessing | RGB conversion, resize, crop and ImageNet normalization; random rotation and flips during training |
| Model | ResNet-50 initialized with `IMAGENET1K_V2` pretrained weights and an 11-class linear output layer |
| Training and validation | Cross-entropy loss, AdamW, per-epoch validation, Accuracy and Macro F1, and checkpoint selection by validation Macro F1 |
| Prediction export and evaluation | Export one probability row per lesion, then generate unified metrics and confusion matrices from the best checkpoint |

The committed reports contain real **label-distribution analysis** and a
**synthetic evaluation demonstration**. They do not establish trained-model
performance. Original JPG images and trained checkpoints are not included in the
repository; running the training pipeline requires the images and training
dependencies described below.

## Repository guide

| File or directory | Role |
| --- | --- |
| `DATA/PROCESSED_DATA/` | Existing lesion-level training and validation label CSVs |
| `DATA/RAW_DATA/MILK10k_Training_Metadata.csv` | Maps lesion IDs to image IDs and image modalities |
| `src/class_schema.py` | Canonical class order used by analysis and evaluation |
| `src/analyze_class_distribution.py` | Validates the label splits and generates distribution reports |
| `src/Image_tensor.py` | Indexes JPG files and applies training/validation image transforms |
| `src/dataset.py` | Builds single-modality datasets and DataLoaders from the existing splits |
| `src/models/resnet50.py` | Builds the pretrained ResNet-50 classifier |
| `src/engine.py` | Implements one training epoch and one validation epoch |
| `src/metrics.py` | Reuses the unified evaluator's metric definitions during training |
| `src/train.py` | Connects data loading, training, checkpointing, prediction export and evaluation |
| `src/evaluate.py` | Evaluates a probability CSV independently of the model implementation |
| `tests/test_evaluate.py` | Tests evaluation calculations, input validation and output protection |
| `docs/evaluation.md` | Prediction CSV contract and detailed evaluation instructions |
| `examples/evaluation/` | Artificial labels and predictions for checking the evaluator |
| `reports/` | Committed distribution reports and synthetic evaluation outputs |

## Data preparation

The repository includes these CSV files:

| File | Contents |
| --- | --- |
| `DATA/PROCESSED_DATA/train_split.csv` | 4,192 training lesions, with labels and one-hot class columns |
| `DATA/PROCESSED_DATA/val_split.csv` | 1,048 validation lesions, with labels and one-hot class columns |
| `DATA/RAW_DATA/MILK10k_Training_Metadata.csv` | 10,480 image records for 5,240 lesions, linking clinical and dermoscopic images |

The supplied split is 80% training and 20% validation, with no shared lesion IDs.
Keep the same split for model comparisons. Splits are checked at the lesion level;
this does not by itself establish patient-level separation.

For training, obtain the original MILK10k images separately and extract them under:

```text
DATA/
  RAW_DATA/
    MILK10k_Training_Metadata.csv
    MILK10k_DATA/
      .../
        ISIC_<image_id>.jpg
  PROCESSED_DATA/
    train_split.csv
    val_split.csv
```

`src/Image_tensor.py` searches `DATA/RAW_DATA/MILK10k_DATA/` recursively for
`*.jpg`. Each filename without its extension must exactly match an `isic_id` in
the metadata, and image IDs must be unique across subfolders. This image root is
currently defined in `Image_tensor.py`; there is no `--image-root` training flag.
The original images are excluded by `.gitignore`.

The current loader returns one image and one integer class label per lesion.
The two supported modality values are `dermoscopic` and `clinical: close-up`.

## Environment setup on Windows

Run all commands below from the repository root in PowerShell. The analysis and
evaluation dependencies have been tested with Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
```

This installs NumPy, scikit-learn and Matplotlib for evaluation, and also covers
class-distribution analysis. If only the distribution analysis is needed, install
`requirements-analysis.txt` instead.

In VS Code, use **Python: Select Interpreter** to choose
`.venv\Scripts\python.exe`. Calling the environment's Python executable directly,
as shown here, does not require activating the environment.

### Additional training dependencies

Training also requires **PyTorch, torchvision and Pillow**. The evaluation
requirements file does not install PyTorch or torchvision.

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
to select Windows, Pip, Python, and the CPU or CUDA build appropriate for your
machine. Install the matching `torch` and `torchvision` packages into the same
`.venv`: replace `pip` or `pip3` in the generated command with
`.\.venv\Scripts\python.exe -m pip`. Audio packages are not used by this project.

Install Pillow and verify the training imports:

```powershell
.\.venv\Scripts\python.exe -m pip install Pillow
.\.venv\Scripts\python.exe -c "import torch, torchvision; from PIL import Image; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The training dependency versions are not yet pinned in a shared requirements
file. Record the versions used for each experiment. The first model construction
downloads the ImageNet pretrained weights unless they are already cached.

## Train a single-modality baseline

After installing the training dependencies and placing the images in the expected
folder, start a dermoscopic run:

```powershell
.\.venv\Scripts\python.exe src/train.py --image-type dermoscopic --epochs 5 --batch-size 32 --output-dir runs/resnet50_derm_01
```

Run a separate clinical close-up baseline with its own output directory:

```powershell
.\.venv\Scripts\python.exe src/train.py --image-type "clinical: close-up" --epochs 5 --batch-size 32 --output-dir runs/resnet50_clinical_01
```

Each run fine-tunes the ResNet-50 parameters, including the new classifier, using
cross-entropy loss on raw logits and the AdamW optimizer. Training images are
resized to a short edge of 320, randomly rotated by up to 180 degrees and flipped
horizontally/vertically, then center-cropped to 224 x 224 and normalized with
ImageNet statistics. Validation uses the same resize, crop and normalization
without the random augmentation.

### Training options

| Argument | Default | Purpose |
| --- | --- | --- |
| `--image-type` | `dermoscopic` | Select `dermoscopic` or `clinical: close-up` |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `32` | Lesions per batch |
| `--lr` | `0.0001` | AdamW learning rate |
| `--weight-decay` | `0.0001` | AdamW weight decay |
| `--seed` | `42` | Seed for Python, NumPy and PyTorch random operations |
| `--num-workers` | `0` | DataLoader worker processes |
| `--device` | `auto` | Use CUDA if available, otherwise CPU; can explicitly select `cpu` or `cuda` |
| `--output-dir` | Required | Separate directory for this experiment |

`--train-csv`, `--val-csv` and `--metadata-csv` can override the three default CSV
paths listed above. Default input paths are resolved from the repository location;
explicit relative paths and `--output-dir` are resolved from the current directory.
Run `src/train.py --help` with the same Python executable for all options.

These are starting defaults, not tuned hyperparameters. Seeds help repeat an
experiment but do not guarantee identical results across hardware and software
configurations.

### Training outputs

| Output inside `--output-dir` | Contents |
| --- | --- |
| `best.pt` | Checkpoint with the highest validation Macro F1 |
| `last.pt` | Checkpoint from the final completed epoch |
| `history.csv` | Per-epoch train loss, validation loss, validation Macro F1 and Accuracy; written after the training loop |
| `predictions.csv` | Validation probabilities from the reloaded best checkpoint: `lesion_id` plus 11 class columns |
| `evaluation/` | Unified metrics, per-class scores, aligned predictions and count/normalized confusion matrices |

Checkpoints include model and optimizer state, epoch, class order, modality and
run configuration. Automatic resume is not implemented. Use a fresh output
directory for each run: training refuses to replace these existing outputs.
The script selects the best epoch on the validation split and reports results on
that same split; these are validation scores, not independent test-set results.

## Class distribution analysis

Analyze the labels without images, PyTorch or a GPU. Counts are per lesion.

```powershell
.\.venv\Scripts\python.exe src/analyze_class_distribution.py
```

Inputs are the existing training and validation label CSVs. Outputs in
`reports/class_distribution/` include:

- [Analysis and interpretation](reports/class_distribution/analysis.md)
- [Counts and percentages by class](reports/class_distribution/class_distribution.csv)
- [Lesion counts chart](reports/class_distribution/class_counts.png)
- [Training and validation proportions chart](reports/class_distribution/split_proportions.png)
- [Summary, input checksums and runtime versions](reports/class_distribution/summary.json)

The script checks label consistency and split overlap before counting. Optional
arguments are `--train-csv`, `--val-csv` and `--output-dir`; run with `--help`
for details. Default paths are relative to the repository. Rerunning replaces
generated outputs with the same names.

The labels are strongly imbalanced: BCC has 2,522 lesions across both splits,
whereas MAL_OTH has 9, including only 2 in validation. Use Macro F1 alongside
Accuracy, per-class scores and support when comparing models.

## Unified model evaluation

Training calls the evaluator automatically after exporting the best checkpoint's
predictions. It can also evaluate probability CSVs from other models without
loading images, PyTorch or model weights:

```powershell
.\.venv\Scripts\python.exe src/evaluate.py --predictions path/to/predictions.csv --output-dir reports/evaluation/run_01 --model-name model_01
```

Replace the prediction path with a real model export. `--labels` defaults to the
existing validation label CSV. The tool matches by `lesion_id`, validates all 11
class probabilities and saves Accuracy, Macro F1, macro precision/recall,
per-class metrics and count/normalized confusion matrices. Macro F1 averages
across all 11 classes, with undefined scores set to zero.

`--output-dir` specifies where the evaluation results are saved; `--model-name`
is a display label. Choose a new output directory or explicitly use `--overwrite`
to replace existing evaluation outputs. See the
[evaluation guide](docs/evaluation.md) for the exact CSV format and metric rules.

### Demonstration without images or a trained model

Use the included artificial data to check the evaluation workflow:

```powershell
.\.venv\Scripts\python.exe src/evaluate.py --labels examples/evaluation/synthetic_labels.csv --predictions examples/evaluation/synthetic_predictions.csv --output-dir reports/evaluation/demo_local --model-name "Synthetic demo - not trained model performance"
```

The [committed synthetic demonstration](reports/evaluation_demo/README.md) uses
22 artificial lesions with four deliberate errors. Its figures and scores
illustrate the evaluator and must not be presented as trained-model results.

Run the evaluator tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests cover metric calculations, ID/column alignment, invalid inputs and
output protection. They do not verify real-image training or GPU execution.

## Next steps

- Run and document both single-modality baselines on the real images.
- Identify additional suitable datasets and check label compatibility and split design.
- Define hyperparameter candidates and compare experiments using validation Macro F1.
- Compare other pretrained backbones and investigate training from scratch.
- Implement paired-image loading and multi-backbone feature fusion.
- Record reproducible training environments and consolidate experiment results,
  performance analysis and visualizations for the final report.
