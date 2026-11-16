# MONET reference and changes

Source: Kim et al., MONET, https://github.com/suinleelab/MONET

Paper: https://doi.org/10.1038/s41591-024-02887-x

Official annotation notebook source blob: `b55468e84ed03e20089deb5f72b5918e94ba30be`.
The saved `.py.txt` contains its code cells only, with notebook output omitted.
Upstream license text is preserved in `MONET_LICENSE.txt` (CC BY-NC-SA 4.0).




The current E13 adapter is implemented in `src/models/foundation.py`.

It uses the pinned MONET weights from `chanwkim/monet`, revision `1d1efd0b8d61bde82cde22d2e93c28d73441f29e`.

The adapter computes four groups of concept scores using target/reference text similarities and fixed scoring parameters.

Text features are calculated during initialization. The MONET image encoder remains frozen, and concept scores are computed from the same augmented 224x224 dermoscopic image used by the classifier.

The current implementation does not use the historical MONET crop pipeline or previously generated concept CSV files.

The model and its weights are subject to their upstream attribution and licensing requirements. No diagnosis calibration or segmentation is implied.





Changes: four selected concept groups; text embeddings computed once; text encoder
then released; frozen image encoder runs in FP16 on CUDA; scores remain FP32.
The current project's augmented 224x224 view is shared with PanDerm (resize short
side 320, random rotation in training, center crop), using ImageNet normalization
as in the author's example. This crop/resize differs from the example, so these
scores are regenerated consistently for both datasets rather than equated with
the original MILK CSV scores. No diagnosis calibration or segmentation is implied.
