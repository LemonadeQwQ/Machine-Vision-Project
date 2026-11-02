# E10 / E07 fixed-holdout aggregate results

Read [SUMMARY.md](SUMMARY.md) for both models, their fixed refit procedure,
per-class results and limitations. E10 primary / E07 control were selected before
final evaluation. Both cover the same 1,037 cases; Macro F1 is 56.62% / 53.28%,
and Accuracy is 70.49% / 70.01%.

[summary.csv](summary.csv) and [per_class_comparison.csv](per_class_comparison.csv)
provide exact aggregate numbers. Each model's `evaluation/` directory contains
sanitized metrics, per-class metrics and count/normalized confusion matrices
as CSV and PNG. No new case-level data, images or weights are included.

The original inference was completed on 2026-10-31. This public export removes
source paths from metrics JSON while retaining source hashes and metric values.
It does not rerun inference. The inherited holdout has historical exposure;
results are descriptive comparisons, not a new external blind benchmark or
statistical significance claim.

CV results are reported [separately](../../experiments/20261031_e10/SUMMARY.md).
B's E10 dependency and refit documentation remain separate pending deliveries;
C's integration checks are deferred until A/B/C merge.
