# Is mulegraph worth continuing?

**Decision memo · 15 Sep 2026.** Prompted by the first real Elliptic++ run (`ec4501f`): temporal F1 0.57–0.78, GFP features not helping either model. Sources: the repo, MLflow, and the 2019–2026 literature listed at the end. Published copy: https://claude.ai/artifact/7r8imhqjeXYSXAw8hgDvJY

## Verdict

**Yes. Continue, and stop treating Elliptic as the headline.**

The numbers are not bad results. They are the known shape of this dataset and they match two independent published baselines to within a few F1 points. The dissertation question was never "do GNNs win"; it was "how much comes from graph features vs graph models under leakage-free evaluation, and can you detect when a model dies without labels". The t43 collapse just measured is the second half of that question arriving on schedule.

What changes: Elliptic becomes a replication-with-extension of a paper that appeared in April 2026; AMLworld's inductive two-by-two becomes the non-negotiable floor for v1b; v2 (drift + retraining under label lag) is where the distinct contribution lives.

## 1. Our numbers against the literature

Train t1–34, test from t35 (ours starts at t38 after a validation block at t35–37, which is stricter). Illicit-class F1, mean over seeds where reported.

| Model | Ours (`ec4501f`) | Weber 2019 | Maganti 2026 |
|---|---|---|---|
| Trees on all 165 features | 0.778 | 0.788 (RF) | 0.821 ± 0.003 (RF, 10 seeds) |
| Trees on 93 local features | 0.722 | 0.694 (RF) | — |
| Trees on local + causal GFP | 0.718 | — | — (not tried) |
| GraphSAGE | 0.593 ± 0.020 | — | 0.689 ± 0.017 |
| GCN / Skip-GCN | — | 0.628 / 0.705 | 0.503 (GCN) |
| MLP | — | 0.653 | 0.549 |

Our SAGE is about 0.1 below Maganti's. Two things explain most of it and neither is a bug: he early-stops on *test* F1 (PR-E4 forbids exactly that), and he uses the 165-feature block that already contains one hop of neighbour aggregation; our SAGE gets the 93 local features and has to learn the hop itself. Our XGBoost rows are within 0.04 of both papers.

Trees beating GNNs on Elliptic under temporal evaluation is the 2019 result and the 2026 result. We reproduced it in nine minutes on a laptop with a cleaner protocol.

## 2. Why the table looks awful: t43

Refitting the deterministic XGBoost configs (`a64a16e`, seed 0) and scoring each test window separately:

| Config | F1 t38–42 | F1 t43–49 |
|---|---|---|
| xgb.raw165 | 0.894 | 0.032 |
| xgb.base | 0.848 | 0.019 |
| xgb.base_gfp | 0.829 | 0.033 |

169 of the 828 test illicit nodes fall after the t43 dark-market shutdown; no model finds them. The headline 0.72 is a weighted average of "fine" and "dead". Maganti reports the same cliff (≈ 0.38 early window under his stricter setup, ≈ 0.03 collapsed); Weber's Figure 2 shows it for every model in 2019, including a Random Forest retrained after every test step. P@R0.8 ≈ 0.2 for all five configs for the same reason.

Split at t43, the temporal result is two clean results: a benchmark on t38–42 and a natural drift event on t43–49. This is why the per-timestep curves (#12) and v2 exist.

## 3. What others have already done

| Paper | Showed | Did not do |
|---|---|---|
| Weber et al. 2019, KDD-ADF | RF beats GCN on Elliptic temporal split; t43 collapse first documented | No CIs, no engineered graph features, no drift detection |
| Altman, Egressy et al. 2023, NeurIPS D&B (AMLworld) | Synthetic bank data; GBT+GFP ≈ PNA (HI-Medium 59.5 vs 59.7) | No inductive regime as a contrast, no drift, no per-typology shift |
| Egressy et al. 2024, AAAI (Multi-GNN) | GIN+EU / PNA "closely match or outperform" tree baselines on AMLworld | Disagrees with the GFP paper; nobody has run both under one protocol |
| Blanuša et al. 2024, ICAIF (GFP) | GFP+XGBoost beats PNA on every AMLworld set: HI-Small 63.2 vs 56.8, LI-Small 27.3 vs 16.5 | Temporal 60/20/20 only; no inductive test; IBM evaluating IBM |
| Maganti, Apr 2026, arXiv | Strict inductive Elliptic: RF 0.821 beats every GNN; shuffled edges beat real edges | Trees never get graph features; early-stops on test; single dataset; no drift or retraining |
| Šafář et al. 2026, FSI:DI / DFRWS | Elliptic feature construction is opaque; leakage across standard splits inflates results | Paywalled; abstract only read so far |
| Heidrich et al. Mar 2026, arXiv | Significance-tested protocol for graph-derived signals in tabular ML on a crypto fraud set | GBT only, no GNN, no temporal framing |
| 2026 survey, delayed-label drift | Supervised detectors "inapplicable in their original form" with 30–180-day label delay | No fraud-graph retraining-policy comparison exists |

## 4. Threats

- **Real — Elliptic has been pre-empted.** Maganti published the Elliptic answer five months before the MVP. It is not sunk because he compared architectures, not feature sets: his trees never receive engineered graph features, so the question the two-by-two isolates is still open, and he lists retraining under drift as future work.
- **Real, unread — Elliptic itself may be leaky.** Šafář et al. say the standard splits leak and overestimate performance. The design already avoids the most likely culprit: `base` is the 93 local features (PR-M7); the 72 neighbour aggregates only appear in `raw165`. The paper has not been read. Get it through the library before the methods chapter. If the leak touches the local block, Elliptic becomes the drift dataset only and AMLworld carries the benchmark.
- **Known — AMLworld is synthetic.** NFR-4 already forbids claiming real-world generalisation. The two IBM groups disagree on it, so a third party running both families under one budget on one protocol is useful even on synthetic data.
- **Live — schedule, not compute.** The full Elliptic grid took nine minutes on the laptop GPU. Gates 3–5 (AMLworld timings via IBM's code) are untouched with 46 days to the MVP date, and v1b scope is unagreed (#9). The AMLworld loader and the Multi-GNN adaptation are where time goes.

## 5. What is genuinely ours

1. **Method.** The feature × model two-by-two with *causal* GFP features. Maganti compares models; IBM compares GBT+GFP to GNNs but never GNN+GFP. We have all four cells, on both datasets.
2. **Protocol.** Validation-only thresholding, paired-by-seed significance, and an explicit rule for deterministic pairs (#27). Maganti early-stops on test. That difference is citeable.
3. **Regime.** The inductive contrast on AMLworld, where accounts persist. Nobody in the table runs random vs temporal vs temporal-inductive on the same data.
4. **Drift.** Label-free lead time on a real event (t43), and injected typology shifts on AMLworld. Weber and Maganti both name the collapse and stop.
5. **Policy.** Retraining policies under label lag with an oracle comparator. The 2026 survey confirms this is open; no fraud-graph version exists.

The early GFP result (local + GFP ≤ local for XGBoost) is coherent with the field: on Elliptic the timestep components are disconnected, so a one-timestep GFP window has little to see, and Maganti finds the real edges carry less signal than shuffled ones under shift. On AMLworld, where GFP is IBM's own headline, we find out whether that reverses. Either answer is S2.

## 6. Recommendation

Continue, with three changes:

1. **Reposition Elliptic.** In the methods and related-work draft (#8), frame it as replication + decomposition of Maganti 2026, cite Šafář 2026 for the local-vs-aggregated split, and lead with the per-window numbers, not the mean.
2. **Make AMLworld the floor in #9.** "AMLworld two-by-two under temporal + inductive is the minimum v1b, not a negotiable extra." It is the only place the inductive regime is defined and the only place the literature disagrees.
3. **Weight v2.** The t43 event is now a measured fact in our own pipeline (F1 0.85 → 0.02). S3 and S4 are the parts of the spec no published work covers. Protect the v2 window.

Pivot triggers (neither means stop):

- AMLworld cannot be loaded and timed by gate 5 → the existing §1.6 risk ladder: drop random, GNN seeds to 3, PNA trial-0 only, PNA to Later. Worst case is "Elliptic replication + drift monitor", still a complete dissertation with a weaker benchmark chapter.
- Šafář's leakage affects the 93 local features → Elliptic becomes drift-only; AMLworld carries S1/S2.

This week: obtain Šafář et al. via the library; add Maganti and Šafář to the related-work notes; open #9 with the AMLworld floor.

## Sources

- Weber, Domeniconi et al. 2019. [Anti-Money Laundering in Bitcoin: Experimenting with GCNs for Financial Forensics](https://arxiv.org/abs/1908.02591) (KDD-ADF). Table 1, Figure 2.
- Maganti 2026. [When Graph Structure Becomes a Liability: A Critical Re-Evaluation of GNNs for Bitcoin Fraud Detection under Temporal Distribution Shift](https://arxiv.org/abs/2604.19514) (arXiv, 21 Apr 2026).
- Altman, Blanuša, von Niederhäusern, Egressy, Anghel, Atasu 2023. [Realistic Synthetic Financial Transactions for Anti-Money Laundering Models](https://arxiv.org/abs/2306.16424) (NeurIPS D&B).
- Egressy et al. 2024. [Provably Powerful Graph Neural Networks for Directed Multigraphs](https://arxiv.org/abs/2306.11586) (AAAI).
- Blanuša et al. 2024. [Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection](https://arxiv.org/abs/2402.08593) (ICAIF). Table 4.
- Šafář, Pluskal, Veselý, Ryšavý 2026. [The enemy of reproducibility is opacity: What's inside the Elliptic bitcoin dataset (and why it is wrong)](https://www.sciencedirect.com/science/article/pii/S2666281726000818) (FSI: Digital Investigation; [DFRWS USA 2026](https://dfrws.org/presentation/the-enemy-of-reproducibility-is-opacity-whats-inside-the-elliptic-bitcoin-dataset-and-why-it-is-wrong/)). Abstract only.
- Elmougy, Liu 2023. [Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network](https://dl.acm.org/doi/10.1145/3580305.3599803) (KDD; Elliptic++).
- Heidrich et al. 2026. [A Systematic Evaluation Protocol of Graph-Derived Signals for Tabular Machine Learning](https://arxiv.org/abs/2603.13998) (arXiv).
- Malik 2026. [Do Transaction-Level and Actor-Level AML Queues Agree?](https://arxiv.org/abs/2604.23494) (arXiv; Elliptic++, RF only).
- 2026 survey. [Concept drift detection in delayed and partially labeled data streams: an experimental survey](https://www.sciencedirect.com/science/article/pii/S1051200426004434).
- Repo: `report/tables/elliptic_mvp_results.csv` at `ec4501f`; MLflow parents `341e0f93…` (`ec4501f`) and `281caef1…` (`a64a16e`); `docs/project_spec.md` §1.5, §1.6; `docs/project_status.md` Verified method notes.
