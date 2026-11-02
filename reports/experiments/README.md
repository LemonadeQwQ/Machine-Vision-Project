# Formal nine-class ResNet experiments

E00–E10 completed all 55 folds. [experiment_status.csv](experiment_status.csv)
records the real completed runs; the [latest comparison](20261031_e10/SUMMARY.md)
contains all eleven groups. Every group's pooled OOF covers 3,968 development
cases. Historical models, synthetic tests and resource measurements are excluded.

The [October 29 core snapshot](20261029_core/SUMMARY.md) remains unchanged.
E03–E09 pooled OOF class tables and confusion matrices are in
`20261030_expansion/E03/summary/pooled_oof/` through `E09`.
`20261031_e10/` contains the eleven-group aggregate tables, E10's pooled OOF
class results, E10/E07 comparisons and the current aggregate JSON.

Redraw the current comparison from public aggregate numbers only:

```bash
python tools/plot_cv_summary.py --summary reports/experiments/20261031_e10/public_summary.json --output reports/experiments/20261031_e10/cv_comparison.png
```

The plotting tool also accepts the three-group `20261029_core/public_summary.json`.
Historical snapshots keep their historical holdout status. The latest plot
states that holdout results are reported separately.

E10 primary and E07 control completed 13/14-epoch refits and final holdout
assessment; see the [holdout summary](../holdout/20261031_holdout_e10_e07_v1/SUMMARY.md).
CV means, sample SD, pooled OOF and final holdout metrics are distinct quantities.

`collect_results.py` enumerates E00–E10 but still marks absent runs planned;
the committed completed status is copied from accepted real runs.
`summarize_cv.py` recomputes from local probabilities and manifests, which are
not distributed here. Public aggregates cannot substitute for case-level inputs.

This C branch defers tests to A/B/C integration; B's E10 configuration and
registration are pending. See [contributions](../../docs/contributions.md).
