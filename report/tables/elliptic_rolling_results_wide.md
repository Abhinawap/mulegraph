# elliptic_rolling — results

Mean ± half-width of the 95% across-seed t-interval (`ci_kind = seed_t`).
Bootstrap bands are per-timestep only and never appear here (PR-E3).

## elliptic_pp — temporal

| config | seeds | test_f1 | test_pr_auc | test_roc_auc | test_p_at_r50 | test_p_at_r80 |
|---|---|---|---|---|---|---|
| sage.base | 5 | 0.5888 ± 0.0178 | 0.5674 ± 0.0651 | 0.8833 ± 0.0066 | 0.7214 ± 0.0634 | 0.1955 ± 0.0109 |
| sage.base_gfp | 5 | 0.5537 ± 0.0165 | 0.5705 ± 0.0297 | 0.8772 ± 0.0061 | 0.6279 ± 0.0587 | 0.1848 ± 0.0128 |
| xgb.base | 5 | 0.7215 ± 0.0000 | 0.7264 ± 0.0000 | 0.9107 ± 0.0000 | 0.9834 ± 0.0000 | 0.1959 ± 0.0000 |
| xgb.base_gfp | 5 | 0.7183 ± 0.0000 | 0.7148 ± 0.0000 | 0.8749 ± 0.0000 | 0.9976 ± 0.0000 | 0.1589 ± 0.0000 |

## elliptic_pp — temporal_rolling

| config | seeds | test_f1 | test_pr_auc | test_roc_auc | test_p_at_r50 | test_p_at_r80 |
|---|---|---|---|---|---|---|
| sage.base | 5 | 0.6014 ± 0.0266 | — | — | — | — |
| sage.base_gfp | 5 | 0.5964 ± 0.0367 | — | — | — | — |
| xgb.base | 5 | 0.7336 ± 0.0000 | — | — | — | — |
| xgb.base_gfp | 5 | 0.7386 ± 0.0000 | — | — | — | — |
