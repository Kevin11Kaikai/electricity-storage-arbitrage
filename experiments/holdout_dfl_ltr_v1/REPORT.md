# holdout_dfl_ltr_v1 close-out

The 59-day window had already appeared in notebook 11. This note answers whether six decision-focused routes beat notebook 12's dispatch-score incumbent. It does not name a new champion. `realized_profit_mean` is the raw dispatch score, not currency.

**Pass rule, as specified:** passed on the point estimate. The highest new recipe is `r1_listwise_transformer_B` at **6698.87**, capture **0.7442**, above the incumbent notebook-12 `lstm_dispatch` at 6301.33 / 0.7001. **10 of 26** new recipes clear the line.

That maximum is not the result that survives the uncertainty checks. The same seed, the same data, and the same code, rerunning `r1_listwise/transformer` eight times, produced:

```
6293.4  6617.4  6299.2  6645.0  6319.1  6297.1  6309.9  6297.1     spread 351.6
```

The largest curve difference across those runs is **5.4×10⁻²**, from CPU reduction order, with the seed unchanged. Five of 59 days changed window. **2025-12-30 contributed 91% of the gap**: the discharge block jumped from `td=67` (16:45) to `td=81` (20:15), and that day's score jumped from 9,099 to 27,996.

`realized_profit` is a step function of the forecast. `optimize_day` takes an argmax over 3,321 windows. A perturbation of order 10⁻² flips a near tie, and on a spike day (oracle 31,559 on 12-30) the 59-day mean jumps by more than 300. A lead of a few hundred score units on this window sits inside that noise floor, before sampling noise is counted.

| Recipe | Spread across 8 reruns | Curve spread | Days with a flipped window |
|---|---:|---:|---:|
| `r1_listwise/transformer` | **351.6** | 5.4e-2 | 5/59 |
| `r2_pairdiff/lstm` | **0.0** | 1.5e-2 | 0/59 |
| `r6_cheap/dlinear` | **0.0** | 0.0 | 0/59 |
| Five-seed mean Transformer curve (3 reruns) | **0.1** | — | — |

The two results that clear a paired-by-day bootstrap are both LSTMs and are stable at the bit level. Averaging five seeds cuts the Transformer spread from 352 to 0.1. That is a better reason to average than "the average score is higher."

The later frozen headline in `experiments/holdout_seed_bag_v1` is a different, pre-registered object: the five-seed listwise Transformer curve mean at **6693.88**. Do not treat 6698.87 and 6693.88 as the same number.

## 1. Clearing the line is not evidence for the maximum

| Recipe | Realized | Capture | RMSE | Versus 6301 | Versus fixed window 5946 |
|---|---:|---:|---:|---|---|
| `r1_listwise_transformer_B` | **6698.9** | **0.744** | 0.77 | +398, CI [−368, +1344], P=0.181 | +753, CI [−97, +1851], P=0.049 |
| `r1_listwise_transformer_A` | 6592.3 | 0.732 | 2.04 | +291, P=0.260 | +646, P=0.072 |
| `r2_pairdiff_lstm_B` | 6581.6 | 0.731 | 0.87 | +280, CI [−4, +615], P=0.027 | **+636, CI [+247, +1077]** |
| `r3_spo_lstm_A` | 6514.5 | 0.724 | 0.86 | +213, P=0.085 | **+569, CI [+236, +979]** |
| `r3_spo_lstm_B` | 6476.0 | 0.719 | 0.84 | | |
| `r3_spo_transformer_B` | 6435.4 | 0.715 | 1.92 | | |
| `r5_ensemble` | 6402.2 | 0.711 | — | | |
| `r2_pairdiff_transformer_B` | 6392.4 | 0.710 | 1.74 | | |
| `r6_cheap_dlinear_B` | 6367.8 | 0.707 | 0.51 | | +422, CI [+29, +866] |
| `r2_pairdiff_transformer_A` | 6315.7 | 0.702 | 2.51 | | |

Under a null that every new recipe's true level equals the incumbent (within-day sd 4261, between-recipe daily correlation 0.86):

- P(at least one of 26 exceeds 6301) = **0.793**
- P(the best exceeds the incumbent by at least +398) = **0.517**

The maximum of 26 recipes happens about half the time under that null. The paired interval for `r1_listwise_transformer_B` versus the incumbent contains zero (P=0.181). The pass rule is met. It does not show that R1 is a better loss than notebook 12.

## 2. Results that do stand

Pre-specified **per-route** comparisons, each route represented by its own best realized score, against the no-model fixed window selected on the training span (5945.9 on the 59 days):

| Route | Representative | Realized | Versus fixed window | 95% CI | Bonferroni p×6 |
|---|---|---:|---:|---|---:|
| **r2_pairdiff** | `r2_pairdiff_lstm_B` | 6581.6 | **+636** | [+247, +1077] | **0.001** |
| **r3_spo** | `r3_spo_lstm_A` | 6514.5 | **+569** | [+236, +979] | **0.001** |
| r6_cheap | `r6_cheap_dlinear_B` | 6367.8 | +422 | [+29, +866] | 0.104 (not cleared) |
| r1_listwise | `r1_listwise_transformer_B` | 6698.9 | +753 | [−97, +1851] | 0.294 |
| r5_ensemble | `r5_ensemble` | 6402.2 | +456 | contains 0 | — |
| r4_rerank | reranking declined | 5363.9 | negative | — | — |

Pairwise differences (Mandi et al., Eq. 13) and SPO+ still exceed the fixed window after a six-way Bonferroni correction. Earlier recomputed recipes (16 from v1, 19 from v2, and 3 from notebook 12) did not: the notebook-12 incumbent was +355 with P=0.074, and the notebook-11 score leader was +268 with P=0.124. The pre-registered stop was not triggered.

## 3. Route-level reading

- **R1 listwise.** Highest point estimate (6699) and the widest day-to-day variance. The selected `τ_tgt = 0.05` is the bottom of the grid, so the search did not show that a softer target dominated the notebook-12 one-hot on this sample.
- **R2 pairwise differences.** Strongest evidence and smallest variance. It regresses window-score differences instead of fitting a distribution.
- **R3 SPO+.** Also distinguishable from the fixed window. This is a sequence-model training loss on 271 new days, not the earlier linear residual calibration. Neither result overturns the other.
- **R4 reranker.** Validation selected `blend = 0`, which declines to rerank. The D2 top-50 ceiling of +781 was not captured. That is a negative result of the method, not a crashed run. The `blend = 0` option was added so a second stage worse than its own first stage would not be deployed.
- **R5 ensemble.** 6402, above the line, interval contains zero. Weights: `lstm_r1` 0.49 / `transformer_r1` 0.245 / `dlinear_r1` 0.245 / `hgb` 0.02.
- **R6 cheap control.** Pinball at q=0.5, DLinear, five-seed average = 6367.8, RMSE 0.514. The single curve is 6071.5, about 230 below the incumbent. The pass is the average, not the single curve. The single-curve advantages of R2 and R3 (+154 and +130 versus the incumbent) are not an artifact of a weak baseline.

## 4. Protocol notes

Deviations P1–P4 are in [PLAN.md](PLAN.md). The median track-B minus track-A gap is **+92** (range −504 to +747). That is the value of changing the selection statistic and averaging seeds, separated from the change of loss.

Track B scores the pointwise mean of five curves. The incumbent is a single curve. The decomposition (single-curve mean, then the averaged curve) is the right comparison:

| Route / head | Single-curve mean | After averaging | Gain from averaging | Single curve versus 6301 |
|---|---:|---:|---:|---:|
| **r2_pairdiff / lstm** | **6455.1** | 6581.6 | +126.5 | **+153.8** |
| **r3_spo / lstm** | **6431.3** | 6476.0 | +44.8 | **+129.9** |
| r1_listwise / transformer | 6430.1 † | 6698.9 | +268.8 | +128.7 † |
| r3_spo / transformer | 6232.3 | 6435.4 | +203.1 | −69.0 |
| r1_listwise / lstm | 6219.0 | 6087.6 | −131.4 | −82.3 |
| r2_pairdiff / transformer | 6205.8 | 6381.9 | +176.1 | −95.5 |
| r6_cheap / transformer | 6134.2 | 6116.7 | −17.5 | −167.2 |
| r6_cheap / dlinear | 6071.5 | 6367.8 | +296.4 | **−229.9** |
| r6_cheap / lstm | 5522.4 | 5078.2 | −444.2 | −778.9 |
| r1_listwise / dlinear | 5379.0 | 5363.9 | −15.1 | −922.4 |
| r3_spo / dlinear | 5320.1 | 5402.7 | +82.6 | −981.2 |
| r2_pairdiff / dlinear | 4905.4 | 5509.1 | +603.7 | −1395.9 |

† The Transformer single-curve number moves by about ±350 across processes (6365, 6366, 6370, 6430, and 6433 were all observed). That row is not stable.

On the single-curve ranking, R2 is first and R3 is second. About +269 of R1's lead is the average. Averaging is not uniformly helpful: 4 of 12 rows are negative, the worst by −444. It reduces variance. It does not guarantee a higher mean.

Val30 can mislead. R4's selected first stage was `dlinear_r1` (val30 capture 0.5716, the highest) and 5364 on the 59-day window, among the worst. R5's base learners were refit under this protocol. The frozen v1/v2 prediction files contain only the 59 evaluation days, so they cannot set validation weights.

D1 correction: hit@1 is 6 of 2,242 recipe-days, not 0/59. The direction is unchanged.

## 5. Boundary

Oracle 9000.89 is not approached. The best capture in this round is 0.744. Phase B metrics and the v1/v2 RMSE/MAE columns were not edited.

## 6. What the evidence supports doing next

1. Repair the evaluation before repairing the model. One day can decide the ranking, and that day's window can flip under 10⁻² numerical noise. A rolling-origin comparison inside the development span would be the primary comparison; the 59-day window would be a last confirmation. Every "not significant" and every "significant" claim here is a statement about n=59 paired days.
2. Report single-curve and ensemble columns separately. Comparing a five-seed average with a single-curve incumbent is not a matched contrast. R6 was misread that way.
3. Report an interval for the Transformer, or turn on deterministic algorithms. A single point estimate is not a result for that head.
4. R2 and R3 are the living directions. R1's single-curve advantage is inside the ±350 jump. If `τ_tgt` is tried again, the grid should extend below 0.05, to 0.01 and 0.02.
5. Do not repeat R4 in the same form. A reranker, if tried again, should model the 89 block sums rather than featurize 3,321 windows.
