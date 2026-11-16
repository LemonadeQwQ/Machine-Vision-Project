# E14-GPM and fixed-epoch refit public contract

C provides synthetic three-component and refit tests, evaluation/readiness adaptations and aggregate evidence. The branch still requires B's four E14 configurations, grouped registry and neutral Gate initialization-order fix; the exact paths are in the [pair contract](e14_pairs_implementation.md). The group is `triple = ('E14-GPM',)` with all three component flags enabled. C does not submit B's files.

GPM retains the pair protocol, PanDerm dermoscopic features, clinical ResNet encoder, neutral Gate and frozen MONET concepts. The common parameters and initial output match E14-PM before Gate learning. Missing clinical images bypass that encoder and keep neutral scaling. The checkpoint has the same v2/external MONET contract as the single and pair configurations.

The existing generic `src/refit.py` / `src/experiment.py` interface requires a completed formal five-fold summary, identical full configuration, data manifest, pretrained components and training source fingerprint. It initializes from official pretrained assets, fits metadata and class weights on the complete training pool, and uses the selected median best epoch. It does not warm-start from a fold checkpoint. Validation and holdout inference are absent; only the final declared epoch marks `refit_complete`. Holdout membership metadata may be read for isolation checks, while holdout images and metrics are not evaluated.

For a future reproducible run, use the exact source snapshot matching its CV summary:

```bash
python -m src.refit --cv-summary /path/to/completed/cv_summary.json --output /path/to/new-refit
```

Do not relax the source guard to run an old summary against this adapted public tree. The public refit test uses a two-epoch artificial example to verify full-pool inclusion, no validation/holdout image access, independent initialization, interrupted resume, the final completion flag, and rejection of source/component drift. It does not claim a 46-epoch model has completed.

The five formal GPM folds are complete. Based on the eight-candidate primary ranking, GPM was selected for a 46-epoch refit on 24,247 training entries. The last saved observation on 2026-11-15 was queued, not confirmed running; no final model or new holdout score is released. The user chose to retain the queued job and stop manual waiting. Resource-specific scripts, job identifiers, accounts and raw logs remain outside the public delivery. See [results and limitations](../reports/experiments/20261115_e11_e14_summary/SUMMARY.md) and [model selection](model_selection.md).
