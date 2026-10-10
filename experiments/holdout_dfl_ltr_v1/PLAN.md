# holdout_dfl_ltr_v1 plan (written before training)

The 59-day window below is the same historically exposed evaluation period used in earlier notebooks. The recorded flag `holdout_not_used_for_selection` means this protocol did not use those days to pick the recipe. It does not mean the window was untouched.

**Sole pass rule:** under this protocol, 59-day `realized_profit_mean` **> 6301.33** (capture > 0.700079). The field name is the raw dispatch score, not currency.

**Incumbent:** notebook 12 `lstm_dispatch` = 6301.33 / 0.700079.

**Left unchanged:** `reports/phase_b/final_metrics.json`, the v1/v2 leaderboard RMSE/MAE columns, the `optimize_day` rule, and `holdout_dispatch_loss_v1/`.

## 0. Four measured facts that set the routes

The routes were chosen from the four measurements below, not from a paper ranking. The recomputation lives in `diagnostics.py`.

### D1. A one-hot window label is both hard to learn and the wrong target

The true score surface is flat at the top (59-day means, official score):

| True rank K | 1 | 2 | 5 | 10 | 50 |
|---|---:|---:|---:|---:|---:|
| Window value | 9001 | 8990 | 8937 | 8888 | 8356 |
| Share of oracle | 100% | **99.9%** | **99.3%** | 98.7% | **92.8%** |

**Hit@1 is about 0.** Across 38 recipes × 59 days = 2,242 recipe-days, the true argmax is hit **6** times. **33 of 38** recipes never hit it. The best recipe hits 2/59. An early draft said 0/59 after checking only 4 recipes; the full count changes the number, not the conclusion.

Notebook 12's `CE(s/τ, oracle_window)` penalizes the second-best window (99.9% of oracle) as harshly as a window with negative value. The one-hot target it is asked to learn is empirically almost never the unique argmax.

### D2. The curve already ranks good windows highly. The error is inside that shortlist

Perfect reranking of the incumbent's own top-K windows:

| K | 1 (status quo) | 5 | 20 | 50 | 200 |
|---|---:|---:|---:|---:|---:|
| Upper bound | 6301 | 6463 | 6841 | **7082** | 7872 |
| Versus incumbent | — | +161 | +540 | **+781** | +1571 |

The measured direction toward the oracle mean of 9001 is reranking inside the model's own top-K, not a more accurate curve. That is why route R4 exists.

### D3. The action set has 3,321 members, so the listwise distribution is exact

Solution caches, differentiable black-box estimators, and perturbed optimizers approximate a feasible set that cannot be enumerated. Here `|W| = 3321` and the Gibbs distribution over a `(batch, 3321)` tensor is exact. Route R1 is an entropy-regularized exact Fenchel–Young loss. DBB and perturbation were excluded on that ground.

### D4. The selection statistic has to change, or sampling noise swamps the loss

- Standard error of a 59-day mean is about **549**. For 30 days it is about **764**.
- Notebook 12's per-epoch `val_realized` standard deviation is **383–764**, the same size as 30-day sampling noise. Those epoch-to-epoch moves do not carry model information.
- The first 3 days of val30 (4 Oct–2 Nov) hold 17.9% of that window's oracle mass. The best constant window selected on val30 scores 5360 on the 59-day window, below the weekday-slot mean of 5644.
- Switching the selection statistic to **daily-mean capture** (each day's realized/oracle, then an equal-weight mean) raises the signal-to-noise ratio of the same comparison from 0.203 to **0.260**.

## 1. Routes

Notation: true prices `p`, forecast `p̂`, block sums `S(p) = conv(p, 1_8)`, window score `s_w(p) = S[td] − S[tc]` in `R^3321`. The notebook-12 convention is kept: the loss does not multiply by 1000.

Measured invariance on 59/59 days: `s(a·p + b) = a·s(p)` for `a > 0`, and the window-versus-idle decision is unchanged. Every loss is built on `s`.

| # | Route | External source | Why it might beat window CE |
|---|---|---|---|
| **R1** | Listwise soft labels | Mandi et al., ICML 2022, Eq. 16 | D1. Notebook 12 is the `τ_tgt → 0` limit. Near-optimal windows (92.8%–99.9% of oracle) stop being treated as fully wrong |
| **R2** | Pairwise score differences | Mandi et al., Eq. 13 | Regresses decision-relevant differences. Translation invariant |
| **R3** | SPO+ as a training loss | Elmachtoub and Grigas, Management Science 2022 | Convex regret upper bound. Distinct from an earlier linear residual calibration that had no idle action |
| **R4** | Strong curve plus a top-K reranker | Ranking view plus D2 | Measured ceiling +781 at K=50 |
| **R5** | Decision-weighted convex combination | Task-metric weight selection on frozen curves | No new training |
| **R6** | Cheap control | Pinball and peak/trough weighted MSE | If R1–R4 cannot beat R6, the more elaborate losses are in doubt on this sample |

Explicitly excluded, and recorded as exclusions: DBB and perturbed optimizers (D3); learning a local surrogate only to avoid a solver call (one `optimize_day` is a convolution plus an argmax); continuous state-of-charge methods that do not match this contract; reinforcement learning; editing `optimize_day`; tuning on the 59 days; nested 5-fold selection.

Literature depth recorded with the plan: Mandi et al. was checked equation by equation. Elmachtoub and Grigas 2022, the decision-focused-learning survey, and the black-box and perturbation papers were checked at citation and abstract level, not read page by page. Nearby 2025–2026 value-oriented forecasting papers were noted as design context, not as same-task rankings.

## 2. Loss definitions

### R1 listwise, two temperatures

```
q  = softmax( s(p)  / τ_tgt )      # detached target on the true window values
π̂  = softmax( s(p̂) / τ_pred )
L1 = CE(π̂, q) = − Σ_w q_w · log π̂_w
L  = MSE( S(p̂_centered), S(p_centered) ) + λ · L1
```

`τ_tgt` is set from D1 so the target mass sits near the top 20–50 windows. Mandi et al. use one temperature. The split is deliberate: target flatness is a property of the labels, and prediction sharpness is an optimization hyperparameter.

### R2 pairwise difference

Pairs are the true best window, the model's current argmax, and random windows:

```
L2 = mean[ ( (s_u(p̂) − s_v(p̂)) − (s_u(p) − s_v(p)) )² ]
```

### R3 SPO+, maximization form

```
L3 = max_w (2p̂ − p)' z_w  −  2p̂' z_{w*(p)}  +  p' z_{w*(p)}
subgradient in p̂ = 2 ( z_{w*(2p̂−p)} − z_{w*(p)} )
```

`z_w` is the coded power vector from `build_day_power`. Each example calls `optimize_day` twice.

### R4 reranker

A frozen first-stage curve proposes K windows. A shallow second stage scores those K candidates with the R1 listwise loss. `K ∈ {20, 50}`.

## 3. Protocol

**Frozen split:** 360 complete days, then 301 development days (2 Jan–2 Nov 2025: 271 fit, last 30 select), then the 59-day window (3 Nov–31 Dec) scored once. `holdout_not_used_for_selection: true` is written into the manifest. The oracle window on a training day comes only from that day's training label `A`.

Three guards were added because of D4:

1. Selection uses 30-day daily-mean capture. It is still selection by dispatch score, not by RMSE.
2. Each final recipe reports the mean of seeds 42–46. A single 59-day seed has a standard error near 549.
3. Every 59-day result is reported with a paired-by-day bootstrap interval against 6301.33, against a no-model fixed window (5946, selected on the training span), and against a same-architecture MSE run.

### Deviations from the notebook-12 plan

Kept: the 271/30/301/59 schedule, scaled epoch counts, one 59-day score, max 80 epochs, patience 12, oracle windows from training labels only, selection by score rather than RMSE, three compact heads, day-centered eight-slot sums without the factor 1000, no idle term inside the loss, and no edits to `optimize_day` or the frozen RMSE tables.

| # | Notebook 12 | This round | Why |
|---|---|---|---|
| P1 | Seed 42 once | Grid on 2 seeds; selected config retrained on seeds 42–46 | A single-seed 6214 is not a property of the method |
| P2 | Select by 30-day realized mean | Also select by 30-day daily-mean capture | Signal-to-noise 0.203 → 0.260 |
| P3 | Three `(λ, τ)` pairs | Larger grids, because `τ_tgt` is the new hyperparameter | |
| P4 | Pass line `> 6214` | Pass line `> 6301.33`, plus the paired interval and the fixed-window control | The pass line is unchanged; the extras are required reporting |

Track A keeps the notebook-12 selection rule (seed 42, 30-day realized mean) so it is comparable to 6301.33. Track B is the five-seed mean selected by daily-mean capture. The primary call uses track A. A large A-versus-B gap is itself a result.

The grid stage trains 2 seeds (42 and 43) on 271 days. The selected configuration of each route is then retrained on 301 days with seeds 42–46 and scored once. Approximate training counts: R6 18+15, R1 144+15, R2 54+15, R3 36+15, R4 and R5 add no further sequence-model training. A within-development rolling-origin variant was logged as a secondary row and was not the primary table.

## 4. Pre-registered reading of the outcome

Pass means 59-day `realized_profit_mean` > 6301.33.

Required alongside that call, without changing it:

| Comparison | Why it is reported |
|---|---|
| Versus 6301.33, paired interval and P(≤0) | A gap on the order of +87 has P(≤0) = 0.39 at n = 59 |
| Versus the fixed window 5946 | No earlier recipe had cleared it with an interval excluding zero |
| Versus a same-architecture MSE model | Avoids anchoring the gain on the weak v1 LSTM at 4554 |
| Multiplicity when more than three new recipes are reported | Under the null, P(best of 3 exceeds the line) = 0.645 |

Reading rules: an interval whose lower end is above zero is "distinguishable from the incumbent"; a pass whose interval contains zero is "above the line, not distinguishable from the incumbent"; a miss is reported as a miss. This plan does not refresh the 35-recipe leaderboard, does not name a new champion, and does not edit the Phase B frozen metrics.

## 5. Deliverables

`loss.py`, `rerank.py`, `ensemble.py`, `diagnostics.py`, and `run.py` in this directory; `PLAN.md` (this file); `REPORT.md` (the close-out, including the miss wording). Experiment artifacts were written under `reports/holdout_dfl_ltr_v1/` and are not redistributed here because the prediction files contain the evaluation labels.

Planned checks in `tests/test_dfl_ltr.py` were specified (soft-target limit, translation invariance, SPO+ subgradient, reranker limit, block-value agreement with `optimize_day`, and recomputation of D1–D4). That test module is not in this repository.

## 6. Risks written down in advance

1. The likely outcome was a pass that is not distinguishable from the incumbent. If all six five-seed means had intervals against the fixed window that contained zero, the plan was to stop and write the negative result.
2. R4 is the only route with a measured path toward 9001, and the easiest to overfit. Hit@50 was 0.15. K is a hyperparameter selected on the 30-day capture, not on the 59 days.
3. A success or failure of this SPO+ training loss does not confirm or overturn the earlier residual-calibration attempt. The two setups differ, and the report keeps them separate.
