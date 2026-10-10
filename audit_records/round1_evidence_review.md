# Independent review and repair record

Reviewed package: bilingual report sources dated 15 September 2026 (at that time, Chinese 14 pages and English 16 pages, 6 figures, 11 tables, 19 equations).

Review date: 15 September 2026.

Stance: recompute key numbers from the frozen CSV, JSON, and Parquet artifacts and from the implementation. Do not treat a previous review's conclusion as evidence.

The public repository keeps the English report. This note preserves what the review found. It is not a request to rebuild the Chinese edition.

## 0. Summary

No P0 numerical defect was found. Every printed headline number, derivation, and scatter-plot pairing that was checked could be recomputed from the frozen artifacts. An earlier RMSE-source mismatch was confirmed fixed: the score 6214.36 belongs to the curve with RMSE 0.682788, not 0.630545.

Three P1 gaps and five P2 issues were fixed.

1. Equation (1) listed three model families. The R1 scatter uses ten family labels.
2. The v1 and v2 candidate panels were fit under different rules, and the report had not said so.
3. The report gave the mean and an interval, not the realized downside. On this window the headline's worst day is −5922.66, against −2340.48 for the reference.

After the repairs, both editions of that package were recompiled, page-rendered, recompiled from the zip, and checked by redrawing all six figures and comparing SHA-256. 206 raw experiment-artifact hashes were unchanged.

## 1. What was recomputed

| Check | Method | Result |
|---|---|---|
| Headline 6693.8750 and reference 6214.3618 | Paired daily recomputation from `dispatch_daily.csv`, without reading `summary.json` | Match |
| Difference 479.5132 and 7.7162% | Same | Match |
| Single-seed mean 6375.3549 and ensemble increment 318.5201 | Five seed rows in `holdout_seed_bag_v1/leaderboard.csv` | Match |
| Scatter-plot RMSE | Recomputed from `predictions.csv` | All six drifts match in sign |
| Curve behind 6214.36 | Replay through `optimize_day` | 6214.3618, RMSE 0.682788 |
| R1/R2 correlations | Separate Pearson and Spearman implementation | v1 +0.351, v2 +0.005; centered +0.252 / −0.108; window rank +0.849 / +0.777 |
| 3321 windows, index bounds, power codes | `contracts.py` and `dispatch.py` | Match equation (2) |
| Losses | `holdout_dispatch_loss_v1/loss.py` and `holdout_dfl_ltr_v1/loss.py` | Match equations (13) and (14), including both temperatures |
| Architecture | `sequence_models.py` | Two encoder layers, width 32, four heads, feed-forward 64, dropout 0.1 |
| Control epochs 29 and 13 | `selection_frozen.json` | `final_epochs: 29` matches |
| Five bibliography entries | Authors, venue, volume, pages, year | All five are real and correctly cited |

Equations (1)–(19) were checked for dimensions, indexes, assumptions, and algebra, including the RMSE/MSE equivalence on a fixed equal-weighted sample, information constraints, scale-and-shift invariance, the projected regret bound, the block-sum bound, the conditional-mean boundary, and the ensemble variance formula. No derivation error was found.

## 2. P1 repairs

### P1-1. Model space

Equation (1) had defined `H` as trees, LSTM, and Transformer, with ridge and a seasonal mean described as extra controls. The candidate `family` field has ten labels: blend, catboost, climatology, dlinear, hgb, lgb, lstm, ridge, transformer, xgboost. `ridge_contextual_a1` (realized scores 6202.84 and 6184.39) and `climatology_weekday_slot` are prominent in figure 2.

`H` was rewritten as the families that actually appear: gradient-boosted trees, ridge, a weekday-and-slot mean, DLinear, LSTM, Transformer, and a fixed blend. The blend is post-processing, not a new base family.

### P1-2. Two fitting processes

v1 records `tree_early_stopping: false`, `tree_num_boost_round: 1500`, and `ridge_alpha: 1.0`, with the protocol "train all 301 complete days, then RMSE on the last 59." v2 records a 271/30 selection split, tree early stopping, and sequence patience 12. The panels are not two samples from one process. The report already refused to pool them. It now says why, and `candidate_panel_rules.json` stores the manifest fields. A later review found that "all v1 trees use 1500 rounds" was still too coarse, because HistGB uses 300. That correction is in the second review.

### P1-3. Downside

Recomputed from `paired_daily.csv`:

| Quantity | Headline | Reference | Matched baseline |
|---|---:|---:|---:|
| Worst day | −5922.66 | −2340.48 | −12064.47 |
| Negative-score days | 1 | 2 | 3 |
| Daily median | 6984.89 | 6683.35 | 6878.29 |

Winning-day differences total +46151.09. Losing-day differences total −17859.81. The largest single-day shortfall is −3582.18 on 12 December 2025.

The first write-up of this paragraph said those two signed totals *differ* by the net gain 28291.28. They are already signed, so they *sum* to the net gain. Subtracting them gives 64010.89. The second review caught that wording error. The generator now asserts the three-way identity. The numbers themselves were not wrong.

The report text, the paired table, the limitations, and the daily figure were updated so the mean is not shown without the worst day. The objective remains risk-neutral. The worst day is a description of this window, not a tail-risk model.

## 3. P2 repairs

| Id | Issue | Repair |
|---|---|---|
| P2-1 | Displayed subtraction of equation (18) differed by 0.0001 from the rounded terms | State that the difference uses unrounded values |
| P2-2 | "86.0%" was hardcoded in a figure | Read `dec30_share_total_gain_pct` |
| P2-3 | ASCII hyphens for negative table entries | `f()` and `rng()` emit a real minus |
| P2-4 | Vertical jitter in a panel looked like a second axis | The caption says the jitter only avoids overplotting |
| P2-5 | A nearly empty last page in the English edition | Appendix B states the checks that were actually run, without a stronger claim than the second review later allowed |

A delivery script that appended a hardcoded page count on every run was changed to rewrite that block from the audit file. Two consecutive runs left one block.

## 4. Untouched artifacts

206 files were hashed before and after and matched, including `reports/phase_b/final_metrics.json`, the v1/v2 RMSE and MAE columns, notebooks 11–14's prediction and dispatch outputs, and the soft-regret control run. No model was trained, no search was extended, and the headline and dates were not changed.

## 5. Left open

| Question | Status in the report |
|---|---|
| Does +479.51 transfer? | No. The interval [−227.87, 1626.28] crosses zero, and 86.02% of the net gain is one day |
| Can the training surrogate select models? | No. The unchanged proxy correlates +0.320 on v1 and −0.165 on v2. Refitting a checkpoint on the validation dates and scoring them again does not answer this |
| Is the loss causally responsible? | No. +160.99 and +318.52 are an arithmetic split. Selection-only gaps are about +4 / −29 / 0. A same-initialization window-CE Transformer at 6570.71 is above the listwise single-seed mean 6375.35 |
| Physical battery | Not implemented. Section 3.6 separates the 8+8 contract from state of charge, efficiency, and network constraints |
| Market identity and publication time | The anonymized files do not establish a publication contract or physical units |

A further data property, recorded in appendix A: 42.09% of adjacent slots share a price. The median day has 35 distinct values out of 96, and the flattest day has 15. That is why 11 of 14 tie days can choose different windows and settle at the same value, and why zero exact oracle-index hits can coexist with five days that match the oracle value.

## 6. Acceptance of that package

Tectonic 0.17.0 compiled both editions. The log had no overfull boxes, missing characters, unresolved references, or LaTeX warnings. Underfull hbox counts matched the pre-repair log (5 Chinese, 4 English).

All 32 pages then in the package (15 Chinese + 17 English) were viewed at 2×. No clipping, overlap, missing glyph, or overflow was found. The English isolated last page was the P2-5 item.

Independent acceptance from the final zip: page text and content streams matched the delivered PDFs byte for byte; six figure PDFs matched SHA-256; no absolute paths, `..`, parquet, pt, npz, or exe entries. Ten of thirteen existing derived records were byte-identical to the previous package. `summary.json`, `paired_daily.csv`, and `source_manifest.json` changed because fields were added, and `candidate_panel_rules.json` was new. Both languages had the same 81 decimal tokens and the same 36 equation, figure, and table labels.

The review was local. Overleaf online was not used. `overleaf_online_tested` is false.

Points that are easy to skip in a verbal summary, and that the report states: the 59 days were already exposed; +479.51 is an observed score on this window, not a transferable effect; the implementation has no state of charge, efficiency, or network constraint.
