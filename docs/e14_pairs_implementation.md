# E14 pair implementation and public dependency contract

The 2026-11-16 A/B/C integration includes the input protocol, all four combination configurations, grouped registry, neutral Gate initialization fix, readiness support and synthetic tests. The public candidate is validated as one tree; earlier C-only dependency failures describe a historical intermediate branch.

| Identity | Enabled components | B configuration |
|---|---|---|
| E14-GP | Gate + PanDerm | `configs/experiments/E14-GP_pair_gate_panderm.yaml` |
| E14-GM | Gate + MONET | `configs/experiments/E14-GM_pair_gate_monet.yaml` |
| E14-PM | PanDerm + MONET | `configs/experiments/E14-PM_pair_panderm_monet.yaml` |

B also owns `configs/experiments/E14-GPM_pair_gate_panderm_monet.yaml`, `src/experiment_registry.py`, and `src/models/next_stage.py`. The registry fixes each unique identity and exposes singles/pairs/triple groups; bare E14, swapped identifiers and inconsistent component flags are rejected. B's initialization-order correction constructs the MONET projection before the neutral Gate, preserving common initial parameters in GM versus E13 and GPM versus PM. E11–E13 behavior remains covered by the existing regression suite. A owns the accompanying `docs/data_protocol.md` update, and B owns `docs/training.md`.

All configurations preserve the existing nine-class data protocol, five folds, independent initialization, augmentation, metadata fitting and optimizer settings. PanDerm replaces the dermoscopic encoder (768 features); the clinical encoder remains ResNet-50 (2048 features). Gate starts neutral and missing clinical images bypass the clinical encoder. MONET uses the same augmented image view, remains frozen, and stores its reusable text features; the external vision/projection weights retain their fixed content checks.

The readiness script applies optional weight-directory overrides based on the selected configuration's enabled components. Its default remains E00–E10. Select combinations explicitly and use a new output path:

```bash
python scripts/check_training_ready.py --experiments E14-GP E14-GM E14-PM --data-root /path/to/data --weights /path/to/resnet50.pth --panderm-weights /path/to/panderm.pth --monet-dir /path/to/monet --output /path/to/new-readiness.json
```

The registry-driven collector expands to 18 configurations and 90 planned fold positions. Planned positions do not assert 90 new completed runs. Collector and CV summarizer implementations already support registered identities and do not need another copy; refit runs must not enter CV summaries.

The public pair tests cover configuration identity, neutral initialization, grouped collection, artificial two-epoch training, uninterrupted/interrupted/resumed state equality, checkpoint replay, and external MONET component handling. They use tiny encoders and artificial images, not official weights or real cases. Cluster submission, budget and scheduler tests remain outside this public contribution. No tests are permanently skipped to conceal missing B dependencies.

The three pair configurations completed 15 formal folds under their recorded training snapshots; [the aggregate results](../reports/experiments/20261115_e11_e14_summary/SUMMARY.md) distinguish that evidence from the public code-integration status. No GPU validation of this exact public tree is claimed.
