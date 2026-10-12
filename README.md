# Machine-Vision-Project CSC490h5
Given paired clinical close-up and dermoscopic images of the same skin lesion, our system aims to classify the lesion into one of 11 diagnostic categories.

## Class distribution analysis

Analyze the existing training and validation labels without images, PyTorch, or
a GPU. Counts are per lesion, not per image.

From the repository root on Windows, create an isolated environment and install
the analysis dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe src/analyze_class_distribution.py
```

In VS Code, use **Python: Select Interpreter** to choose
`.venv\Scripts\python.exe`. If Matplotlib is already installed in the selected
environment, `python src/analyze_class_distribution.py` is sufficient.

Inputs:

- `DATA/PROCESSED_DATA/train_split.csv`
- `DATA/PROCESSED_DATA/val_split.csv`

Outputs in `reports/class_distribution/`:

- [Analysis and interpretation](reports/class_distribution/analysis.md)
- [Counts and percentages by class](reports/class_distribution/class_distribution.csv)
- [Lesion counts chart](reports/class_distribution/class_counts.png)
- [Training and validation proportions chart](reports/class_distribution/split_proportions.png)
- [Summary, input checksums and runtime versions](reports/class_distribution/summary.json)

The script checks label consistency and split overlap before counting. Optional
arguments are `--train-csv`, `--val-csv` and `--output-dir`; run with `--help`
for details. Default paths are relative to the repository, not the terminal's
current directory. Rerunning replaces generated outputs with the same names.

This analysis describes the label distribution. It does not report trained-model
performance or establish that image files are present. Model training still
requires the original MILK10k images and a separate training environment.
