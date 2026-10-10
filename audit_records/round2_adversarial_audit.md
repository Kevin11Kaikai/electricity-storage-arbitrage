# Second review: adversarial audit and one repair pass

Date: 15 September 2026. The previous round's conclusions were treated as unverified claims. The delivered package at that time had both a Chinese and an English edition. This repository publishes the English edition. The findings below are about the numbers and the wording, not a request to restore the Chinese build.

## 1. Conclusion

After the repairs, the package is a checkable retrospective technical report. It is not a guarantee of a future score and not a proof of a deployable battery policy.

Independent replay confirms the headline settlement values: 6693.875025, reference 6214.361782, difference 479.513243. Added tail, tie, and price-flatness statistics also match. The headline, dates, and candidate set were not changed. No training, checkpoint inference, or new model search was run.

Local wording was wrong even where the settlement table was right. Those sentences were repaired. The evidence still does not establish that the new loss will select better models in the future, that listwise training has an isolated causal advantage, that the mean gain is stable across periods, that the ensemble works by variance reduction, or that a physical battery was modeled.

## 2. Method

- SHA-256 of 121 files in the reviewed directory, saved before edits.
- SHA-256 of 2,090 protected files: reports, notebooks, extensionless sources, and the four zip/PDF pairs then on disk. That count is the protected list, not "every file in the repository."
- `independent_audit.py` does not import the report generator. It rebuilds block sums with `np.convolve` and replays 62 curves from five sources, 3,658 actions, through `optimize_day`. No saved action disagreed. The largest settlement gap is floating-point scale.
- Twenty control curves were replayed separately. Three saved window-vote action sets were resettled with the true price `A`. A vote is not treated as a model with its own price curve.
- Every record is indexed by date. The window is 3 November–31 December 2025, 96 slots each day, unique and complete. The 62 curves share one label vector.
- The bootstrap resamples a date index directly. It does not use the report generator's frequency-weight matrix. Block lengths 7, 1, and 3, 10,000 replicates, seed `20260915 + block length`.
- Redrawing the six figures from the same records reproduced the PDF SHA-256 values. That supports rebuilding the figures in the same environment. It is not a proof of an earlier process.

## 3. Confirmed defects and repairs

### P0-01. Signed totals were described as a difference

The downside paragraph said the winning-day total and the losing-day total *differ* by 28291.28. The two numbers are already signed. +46151.087216 + (−17859.805861) = +28291.281355. Subtracting them gives +64010.893077. The settlement table was not wrong. Both editions were changed to addition. The generator asserts the identity at the same 1e−6 win/tie/loss tolerance and stores a tie rounding residual near −3.64e−12.

### P1-01. v1/v2 fitting rules were over-generalized

HistGB in v1 uses `max_iter=300`, and `V1_HGB_MAX_ITER=300` is explicit in `recipes.py`. The sentence "every v1 tree uses 1500 rounds" is false. In v2, `lgb_baseline` is a fixed fit. HistGB checks validation every 20 iterations up to 300 and stops selection after three unimproved checks. It is not the same 1500-round, 100-round-patience rule as the other trees. The text and `candidate_panel_rules.json` now state the exceptions. v1 sequence models fit 287 days with a 14-day early-stopping tail. Tuned v2 candidates use 271/30 and then refit on 301 days.

### P1-02. Two validation aggregates are not one locked model

From `grid_runs.json`: seed 42's best capture and best realized score are both at epoch 5. Seed 43's best capture is epoch 20 and its best realized score is epoch 17. The value 0.599645634 is the mean of the two seeds' best captures. 8290.240007 is the mean of their best realized scores, not the realized score at the capture-best epochs. At the capture-best epochs the mean realized score is 8134.137357. These are not the five-seed ensemble's validation result. `validation_selection_detail.json` rebuilds them from the per-epoch log. The validation curves are missing, so daily settlement was not replayed. A log summary is not a forecast replay.

The median capture epoch 12.5 is rounded to 12 by Python and then scaled to 13 for the refit.

### P1-03. Training loss is not the outer objective

The added downside paragraph said every training and evaluation objective is a risk-neutral expected score. Equations (1), (13), and (14), and both `loss.py` files, use squared error, block sums, and cross-entropy. Only the outer problem is expected realized score. The text now says that. The weak association of the training proxy is still reported.

### P1-04. Resampling conditions and algebraic coupling

The bootstrap redraws the 59 dates. It does not redraw the 16 or 19 candidates and does not retrain. Window rank `T` and realized score are functions of the same predicted and true window vectors. High correlation can be partly algebraic. The two statistics are not identical. The correlation is not evidence for future model selection or for a causal effect.

### P1-05. Amplitude control was applied to only one diagnostic

The same frozen curves and the same day-block sample were applied to the actual training proxy. Partial associations are v1 +0.058542 and v2 −0.360822, with conditional intervals about [−0.443, 0.335] and [−0.664, 0.023]. The main figure still shows the original loss. Temperatures were not retuned to chase a sign.

### P1-06. Verification language claimed more than was reproduced

An appendix sentence said every number was re-derived rather than copied. The generator copied `comparison_zh.csv` directly. Validation totals and epoch counts were not replayed from curves. The appendix now separates curve replay, control-curve replay, and per-epoch logs. The useful five-item reproduction list remains. It is not a universal pass certificate.

### P1-07. Prediction-rule space, blend weights, and checkpoints

Thirty-five candidates fit in seven rule classes. The blend is post-processing, fixed at 0.75 Transformer + 0.25 LightGBM after a validation RMSE grid, not a weight chosen before any fit. The weekday-slot mean is the sample mean by weekday and slot. Saved metadata do not list fit dates. v1 sequence models take gradient steps on 287 days. Not every checkpoint is a 301-day gradient refit. No checkpoint was reloaded for inference in this review.

### P1-08. Exact argmax and numerical ties

`_stable_first_argmax` uses a tolerance of 64 times float64 epsilon times the maximum absolute score. Training hard labels use a tensor argmax and do not use that tolerance. Exact algebraic invariance, the deterministic tie break, and floating-point near-ties are different statements. Regret bounds assume an exact maximizer.

### P2 layout and tooling

A Chinese table had floated into the appendix checklist, and the English checklist occupied a last page alone. Float barriers were restored. The English reproduction note and the verification scope share page 18 of that edition. No filler text was added to pad a page. All 34 pages of the revised package were viewed.

`comparison.csv` is joined on the stable candidate key, not on a Chinese delimiter plus column position. The verifier no longer rewrites a work-plan file. A Windows console encoding failure on the first Chinese page log was not treated as a passed process. After writing UTF-8 records and an ASCII console summary, two consecutive runs exited 0 with the same audit-file hash.

## 4. Ruling on the previous round

Confirmed wrong, and repaired: "all v1 trees use 1500 rounds"; "all v2 trees share one early-stopping rule"; the signed totals "differ by" the net gain; "every training objective is the outer expected score"; "every number was recomputed and asserted."

Tightened rather than rejected: validation metrics come from different best epochs; the blend was selected and then frozen; checkpoint dates are not in the metadata alone; correlation intervals hold the candidate set fixed; `T` is partly algebraic; amplitude control must cover the training proxy.

Not treated as an error: +479.51 is a real score on a finite window. An interval that crosses zero does not erase the observation. The abstract may bold the observed number if it also states the one-day concentration and the interval.

Not adopted: declaring the interval invalid because the bootstrap is skewed, or inserting an ordinary IID BCa interval as a repair. For v1 `T`, the bootstrap median is about 0.771 and the mean about 0.749, and about 89.8% of resamples fall below the point estimate. The interval is a conditional sensitivity range. Block lengths 1, 3, and 7 stay in the report.

Kept from the previous round: same-curve RMSE on both axes of the six drifts; the hard-label limit stated for a unique maximizer; the distinction between oracle-index hits and oracle-value hits; the tail numbers; the figure reading 86% from data; the real minus sign in tables.

## 5. Numeric comparison

The row-by-row comparison is `all_required_numeric_comparisons.csv`. Reported values use the precision requested for the report, so last-digit gaps versus the unrounded replay are expected. 6375.354942 is a six-digit truncation of 6375.354942863501 and prints as 6375.35 at two decimals. No raw metric was edited to close a rounding gap.

The largest shortfall date recomputes as **2025-12-12**, and it is the only date inside the tie tolerance. Win/tie/loss counts are 25 + 14 + 20 = 59. Same-window ties plus other-window ties are 3 + 11 = 14. The signed day totals add to 28291.281355 up to floating-point rounding.

Six RMSE source drifts, also in `rmse_drift.json`. The drift is same-curve RMSE minus the historical leaderboard value. It is not a change introduced by this review. The other 29 point candidates agree within 1e−5. The v2 Transformer ranks 15th of 19 on the recomputed RMSE.

| Source / model | Leaderboard RMSE | Saved-curve RMSE | Drift |
|---|---:|---:|---:|
| v1 `transformer_contextual` | 0.514318 | 0.519558 | +0.005240 |
| v1 `lstm_contextual` | 0.573845 | 0.552808 | −0.021037 |
| v1 `dlinear_contextual` | 0.655333 | 0.600021 | −0.055312 |
| v2 `transformer_contextual` | 0.630545 | 0.682788 | +0.052243 |
| v2 `lstm_contextual` | 0.528645 | 0.517490 | −0.011155 |
| v2 `blend_lgb_transformer` | 0.584431 | 0.617157 | +0.032726 |

Independent convolutions and sliced sums disagree on a few machine-precision ties for `T`. Sliced `np.sum` reproduces 0.849365753 / 0.776688344 and partial correlations 0.824054364 / 0.712879890. Three-decimal statements do not change. Details are in `rank_numeric_sensitivity.json`.

## 6. Equation check

| Eq. | Scope that was checked |
|---|---|
| 1 | 96-dimensional `q` and `A`. Equal-weighted RMSE and MSE share a minimizer on a fixed sample. That is not a claim about expected RMSE. Seven rule classes cover all 35 candidates. The blend is not a base family. A population squared-error objective is not a claim that every regularized fit solves the same unregularized problem |
| 2 | `sum(81−c) = 3321`. Charge −1000 for eight slots, then discharge +1000, with the index bounds in `build_day_power`. Idle is the zero vector |
| 3 | The two saved curves reproduce both RMSEs and both settlements. Different recipes, so not a one-parameter causal experiment |
| 4 | Every active action and idle has `1'u = 0`. A positive scale preserves order, including idle, except for the documented numerical tolerance |
| 5 | `q'(u_A − u_q) ≤ 0` gives the regret bound. Projection and Cauchy–Schwarz need an exact maximizer. The bound is not a training-convergence certificate |
| 6 | Realized value is outside the forecast. The forecast uses only pre-decision information. Parameters are not rechosen with evaluation-period `A` |
| 7 | The oracle does not depend on the predictor, so expected regret and expected realized score are equivalent for a fixed feasible set when the expectations exist |
| 8 | The conditional mean is optimal for population squared error under finite second moments, among decisions measurable with respect to `X`. The information set is not enlarged in the proof |
| 9 | State of charge is indexed 0…96 and power 0…95. The 4 MWh illustration reaches a state of charge of 1.0 against a 0.9 cap |
| 10 | Strategy constraints and battery constraints are an intersection. Equal energy blocks do not restore state of charge when efficiency is below one |
| 11 | `C` is 96×96, `B` is 89×96, `g` has 3321 rows. Block means must be multiplied by 8 to become block sums. The solver score multiplies by 1000 |
| 12 | One active action's value error is at most `2P‖BCe‖∞`, so the gap between two actions is at most `4P‖BCe‖∞`. Idle error is 0. Block MSE is not that infinity-norm bound |
| 13 | Block MSE plus `λ` times hard cross-entropy, on window scores that omit `P`. Labels are detached. The hard argmax is not day-centered; the maximizing set is unchanged in exact arithmetic. Idle is not in the training distribution |
| 14 | Temperatures 0.05 and 0.10, `λ = 2`, plus block MSE. As `τ_tgt → 0` the target becomes uniform on the maximizing set, and one-hot only if the maximizer is unique. Fixed-temperature cross-entropy is not scale-invariant |
| 15 | Averaging commutes with a fixed action's value and does not commute with argmax. It is not an average of actions or a window vote. The saved headline curve matches the five-seed mean within 1e−12 |
| 16 | Daily capture weights days equally. Aggregate capture weights days by oracle opportunity. Every recorded oracle value is positive. Validation selection and the reported metric are different |
| 17 | Within-day Spearman, then a 59-day average. Across-model Pearson is separate. Average ranks are a diagnostic, not the solver's tie break |
| 18 | Unrounded identity: 479.513243 = 160.993161 + 318.520082. Arithmetic only |
| 19 | The ensemble variance formula needs equal variances, a common pairwise correlation, and a valid correlation matrix. `σ²` and `ρ` were not estimated. The formula is not a mechanism proof |

Both loss implementations were read against the formulas. A NumPy proxy was recomputed on every saved curve. The audit environment did not have PyTorch, so an extra direct Torch forward pass was not run. That absence is recorded in `pure_loss_formula_checks.json` and is not a passed check. No package was installed and no model was constructed for the missing check.

## 7. Other rulings

- 6693.88 is a mean realized score. 479.51 is the paired mean gap versus the reference. 318.52 is the curve-mean minus the mean of the five single-seed scores. 160.99 is the remainder of that arithmetic split, not a pure loss effect.
- Seed dispersion 118.80 is the spread of five trainings, not day-level uncertainty.
- Initialization hashes were checked within each family of the soft-regret run: DLinear 10, LSTM 11, Transformer 11, consistent inside each family (`initialization_hash_consistency.json`). The review did not reinitialize models to re-create those hashes. Different losses have different selected epochs and budgets. 6570.71 does not prove the older loss is causally better.
- `analysis_plan.json` describes a post-hoc analysis, not a prospective registration. The panels have 16 and 19 candidates, with no exclusions and no pooling.
- Figure code reads performance from the packaged records. Axis styling numbers are not synthetic performance.

The English and Chinese texts of that package were compared paragraph by paragraph, including the abstract's claim boundary, the six-step argument, idle, the two fitting rules, the signed totals, and the appendix's check scope. Neither edition made a stronger profit or deployment claim than the other.

Author fields in the report source stay a neutral technical-report label. No name, affiliation, publication, or deployment record was added by the review.

## 8. References checked on 15 September 2026

| Work | Check |
|---|---|
| Lago et al. 2021 | Authors, Applied Energy 293, 116983, and the DOI match. Cited as a forecasting review, not as the origin of this series |
| Elmachtoub and Grigas | Management Science 68(1):9–26, volume year 2022. Online publication in 2021 is not a volume-year error. SPO+ is attributed to this paper |
| Mandi et al. 2022 | ICML, PMLR 162:14935–14947. Equation 16 is listwise cross-entropy. Two temperatures, the full active set, and block MSE are adaptations |
| Lakshminarayanan et al. 2017 | Three authors, NeurIPS 30. Cited only as background for independently initialized ensembles, not for their calibration results |
| Ke et al. 2017 | Eight authors, LightGBM, NeurIPS 30. Used as the tree-baseline citation |

The two NeurIPS bib entries do not invent page numbers. A direct ScienceDirect fetch returned 403 once. The Lago record was cross-checked through the authors' institutional archive and the publisher search record.

## 9. Evidence that is still missing

1. Validation-window curves. Only the per-epoch log can be summed. Daily actions and scores on those 30 days were not replayed.
2. A new time period and a matched loss experiment. One-day concentration, an interval through zero, and simultaneous changes of seed count, ensembling, and epoch budget block a pure loss effect and a transferable gain.
3. State of charge, efficiency, capacity, network limits, costs, and a publication-time contract for the anonymized forecast inputs.
4. Coverage of the day-block percentile interval at 59 days and a small fixed candidate set. It was not established by simulation or theory.
5. A pure Torch forward check. The audit environment had no PyTorch.

## 10. Build and protection

All 2,090 protected SHA-256 values were unchanged. Counts: 19 equations, 6 figures, 11 tables.

The English edition of the revised package was 18 pages. Recompilation from its zip matched page text and PDF content streams. Six figure SHA-256 values matched, and seven numeric tables regenerated. Zip SHA-256 `d1ddf9016f868d1b4aecfca84f3400aa451ffe1b94139bd82e7b29bf7276ed61`. PDF SHA-256 `4f0be368b514fde730de55a946a06248c8df209e549496c0606f35b8c255ff27`.

The Chinese edition of that same package was 16 pages. It is not shipped here. Its zip SHA-256 was `00fe9b7c870066271162eda738d8200dd13319ce50c5a033959a6101e3d8d443` and its PDF SHA-256 was `d794e3ac96af24af613b9258b38d29d1f98682cdfdf2f95dad6d847bdffd701c`.

Page notes and render hashes are in `visual_page_review.json`. Overleaf online was not used. Fontconfig messages appeared locally. TeX Gyre and Fandol glyphs rendered. A few underfull-box messages are not clipping, missing glyphs, or unresolved references. The log is not claimed to be warning-free.

Suggested reading order for the machine records: `all_required_numeric_comparisons.csv`, `solver_replay.json`, `controls_replay.json`, `independent_summary.json`, `independent_correlations.csv`, `rank_numeric_sensitivity.json`, `validation_epoch_reconstruction.json`, `rmse_drift.json`, `pure_loss_formula_checks.json`.
