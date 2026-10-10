# Matched-initialization control: recorded state

- Stage: complete. This round is finished. No training job is running, and the 59-day scores are not a reason to launch another search.
- Run, when artifacts are present: `reports/holdout_dispatch_loss_codex_v1/runs/20260914T221004Z/`.
- Recorded checks: 40 protected files unchanged; 13 loss and budget tests and 6 read-only artifact checks passed. The notebook that displayed the figures is not in this repository.
- Budget of that round: start 2026-09-14 22:10:04 UTC, deadline 2026-09-15 00:10:04 UTC, 120 minutes, 34 training launches, 360 seconds each, at most one retry. It does not renew itself.
- Close-out 2026-09-14 22:40:49 UTC, about 30.8 minutes of development wall clock; 32 training launches, 0 failures, 0 retries, 1 cache hit on an identical DLinear MSE refit. Worker wall clock 163.39 seconds, CPU 636.41 seconds, including about 256.09 seconds of process startup. Unused budget was not used to keep training.
- Protocol: 271 days of training, then 30 days to choose parameters and epochs, then a frozen refit on 301 days, then one 59-day replay. Candidates were frozen before that replay. Nothing was trained after it.
- Score: 3 Nov–31 Dec 2025, 59 days. Loss weights were not fit on those days. The daily mean is `sum(price × power)`, not currency. The window is not an untouched holdout.
- Result: preselected soft-regret DLinear 5,135.06, behind the prior window-CE LSTM 6,301.33. Soft-regret LSTM branch 6,485.73, an improvement and not the preselected primary. The highest control is the same-initialization window-CE Transformer at 6,570.71, which is not credited to the new loss.
- Mechanism notes: changing only the selection metric did not clearly help. The soft-regret gap depends on the architecture. The LSTM advantage versus the prior window-CE run is larger in the first 30 days and nearly flat over the last 29. DLinear on 4 Dec lost mainly on the charge window. The LSTM on 23 Nov gained mainly on the discharge window.
- Limits: one seed; soft regret and idle changed together; the MSE comparison does not have the same three-configuration search; the earlier run did not fix the seed before construction, which is why the matched window-CE control exists. No operating-system isolation was claimed.
- Evidence files, when the run directory is present: `comparison.csv`, `dispatch_daily.csv`, `predictions.parquet`, `analysis.json`, `ledger.jsonl`, plus the freeze files. `predictions.parquet` contains evaluation labels and is not redistributed.
- Do not retrain, and do not tune on the 59-day scores. Read [README.md](README.md) for the table and the command list.
