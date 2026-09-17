# amlworld_xgb — results

Mean ± half-width of the 95% across-seed t-interval (`ci_kind = seed_t`).
Bootstrap bands are per-timestep only and never appear here (PR-E3).

## amlworld — temporal

| config | seeds | test_f1 | test_pr_auc | test_roc_auc | test_p_at_r50 | test_p_at_r80 |
|---|---|---|---|---|---|---|
| xgb.base | 3 | 0.2089 ± 0.0000 | 0.1088 ± 0.0000 | 0.9380 ± 0.0000 | 0.0898 ± 0.0000 | 0.0336 ± 0.0000 |
| xgb.base_gfp | 3 | 0.5395 ± 0.0000 | 0.5210 ± 0.0000 | 0.9831 ± 0.0000 | 0.5700 ± 0.0000 | 0.0990 ± 0.0000 |
