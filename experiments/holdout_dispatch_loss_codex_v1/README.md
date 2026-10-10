# Matched-initialization dispatch-loss control

The preselected soft-regret DLinear did not win. The LSTM branch improved on this window. A same-initialization window-CE Transformer control scored higher still. Do not assign the control's score to the new loss, and do not rename the best branch after the fact as the preselected primary.

Mean raw dispatch score on the 59 days from 3 Nov through 31 Dec 2025:

- Matched LightGBM baseline: 5,786.10.
- Prior best LSTM (window cross-entropy): 6,301.33.
- Soft-regret LSTM: 6,485.73. This is a declared branch, not the preselected primary.
- Soft-regret DLinear: 5,135.06. This is the primary selected on the development window.
- Same-initialization window-CE Transformer: 6,570.71. This is not the new loss.

Parameters for this round were not fit on those 59 days. Loss weights and epoch counts were selected on the 30 development days from 4 Oct through 2 Nov 2025. The raw score is `sum(price × power)`, not currency. These are single-seed results. They are not a claim of a stable advantage.

The 59-day window had already been used in earlier comparisons. It is not an untouched holdout. This round did not use it to choose parameters.

## Protocol

Train on 271 complete development days, select on the next 30, refit the frozen epoch count on all 301, then replay the 59 days once. The three sequence models and the 67 contextual features match the earlier window-CE run. The LightGBM baseline still uses its original 11 features, so a gap against that baseline is not attributed to the new loss alone. Preprocessing is fit on the corresponding training days only. The optimizer and the 3,321 legal windows, including idle, are unchanged.

The new objective is day-centered block-sum MSE plus a soft expected score that includes idle. Controls are: the same initialization with the earlier window-CE loss; point MSE with selection by RMSE; and the same MSE trajectory with selection by dispatch score. Each selection rule stops on its own. Epochs after that stop are not reused. Three loss-parameter configurations were frozen in `protocol.json` before the 59-day replay.

The earlier window-CE run set the seed after the model was constructed, and the CPU environment did not keep a full matching record. This run fixes seed 42 before construction. The same-initialization control is there to limit causal language. The new loss and the idle term were not ablated separately. MSE did not receive the three-configuration search. A matching reload is not the same thing as stable training.

## Recorded commands

From the repository root, with the project interpreter. Training for this round is finished. Do not retrain in order to redraw figures.

```bash
python -m unittest discover -s tests -p test_codex_dispatch_loss.py -v
python -m experiments.holdout_dispatch_loss_codex_v1.run prepare
python -m experiments.holdout_dispatch_loss_codex_v1.run train
python -m experiments.holdout_dispatch_loss_codex_v1.evaluate
python -m experiments.holdout_dispatch_loss_codex_v1.report
python -m unittest discover -s tests -p test_codex_dispatch_artifacts.py -v
python -m experiments.holdout_dispatch_loss_codex_v1.execute_notebook
python -m experiments.holdout_dispatch_loss_codex_v1.run check-protected
```

`train` exits when the frozen candidates are already present. `evaluate` reuses a complete result. The unit-test modules named above are not in this repository. The recorded run reported 13 loss and budget tests and 6 read-only artifact checks passed. The notebook those checks executed is not in this tree.

Run directory, when the raw artifacts are present: `reports/holdout_dispatch_loss_codex_v1/runs/20260914T221004Z/`.

- `protocol_frozen.json`: method, budget, data, and source hashes before training.
- `selection_frozen.json`: parameters and epochs chosen only on the development window.
- `candidate_freeze.json`: model hashes and the primary, frozen before the test replay.
- `comparison.csv`, `dispatch_daily.csv`, `predictions.parquet`: scores, daily decisions, and point forecasts. The prediction file contains evaluation labels and is not redistributed.
- `analysis.json`: the preselected primary, each branch, the controls, time splits, and the negative result.
- `evaluation_checks.json`: independent scoring and action checks.

Recorded worker time for the training code was 163.39 seconds of wall clock and 636.41 seconds of CPU, about 256.09 seconds of which is process startup. That is not the whole development session. The run started 2026-09-14 22:10:04 UTC, with a 120-minute cap, 34 training launches, a 360-second per-launch timeout, and at most one retry in total. No extra training was started because budget remained.

Recorded environment: Python 3.11.15, PyTorch 2.14.0+cpu, NumPy 2.4.6, pandas 3.0.5, Matplotlib 3.11.1, PyArrow 25.0.1. No new packages were installed for that run. The environment was not an operating-system isolation boundary.

## What this run supports

The soft-regret LSTM is above the matched MSE run and above the prior window-CE LSTM on this window. The development ranking did not survive. On 4 Dec the primary lost mainly on the charge window. On 23 Nov the LSTM branch gained mainly on the discharge window.

Not supported: a clear gain from changing only the selection metric; a new loss that is better for every architecture; the preselected primary beating the prior window-CE run; a single seed as a reliability claim. How much of the gap is season, sample size, soft-versus-hard targets, or model capacity is not identified.

The design references a public forecasting competition entry and the ICML 2022 decision-ranking paper. They are design context, not a same-task ranking.
