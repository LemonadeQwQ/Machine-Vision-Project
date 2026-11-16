# MONET reference and changes

Source: Kim et al., MONET, https://github.com/suinleelab/MONET

Paper: https://doi.org/10.1038/s41591-024-02887-x

Official annotation notebook source blob: `b55468e84ed03e20089deb5f72b5918e94ba30be`.
The notebook is an upstream reference; its extracted code cells are not included
in this public delivery. Upstream license text is preserved in
`MONET_LICENSE.txt` (CC BY-NC-SA 4.0).

The current E13 adapter is implemented in `src/models/foundation.py`. It uses
pinned `chanwkim/monet` weights at revision
`1d1efd0b8d61bde82cde22d2e93c28d73441f29e`.

Changes: four selected concept groups, each scored independently using
fixed target/reference text similarities; text embeddings computed once and
the text encoder then released; frozen image encoder runs in FP16 on CUDA,
while score arithmetic remains FP32.

The adapter consumes the classifier's same augmented 224x224 dermoscopic
view: aspect-preserving square padding, shared training flips/rotation, and
ImageNet normalization. It performs no separate CLIP processor resize, crop,
or renormalization. This is the current nine-class protocol; historical
short-side-320/center-crop experiments and previously generated concept CSV
files are not inputs to E13. Scores are not equated with those historical CSVs.

The model and weights retain their upstream attribution and licensing
requirements. No diagnosis calibration or segmentation is implied.
