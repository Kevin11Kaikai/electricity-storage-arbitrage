"""Read saved evidence and draw the notebook figures. Never train or retune."""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .common import PROTOCOL_PATH, Ledger, read_json, write_json, sha256
from .run import run_directory


LABELS = {
    "lgb_baseline": "Matched LightGBM baseline",
    "nb11_v2_transformer": "Notebook 11 / v2 Transformer",
    "nb11_v2_ridge": "Notebook 11 / v2 Ridge",
    "nb11_v1_transformer": "Notebook 11 / v1 Transformer (historical reference)",
    "nb11_v1_lstm": "Notebook 11 / v1 LSTM (historical reference)",
    "cursor_original_transformer": "Prior window-CE, Transformer",
    "cursor_original_lstm": "Prior window-CE, LSTM",
    "cursor_original_dlinear": "Prior window-CE, DLinear",
}
for _family in ["transformer", "lstm", "dlinear"]:
    _name = {"transformer": "Transformer", "lstm": "LSTM", "dlinear": "DLinear"}[_family]
    LABELS[f"codex_soft_regret_{_family}_realized"] = f"Soft regret, {_name}"
    LABELS[f"codex_cursor_ce_{_family}_realized"] = f"Same-initialization window CE, {_name}"
    LABELS[f"codex_mse_{_family}_realized"] = f"MSE / select by dispatch score, {_name}"
    LABELS[f"codex_mse_{_family}_rmse"] = f"MSE / select by RMSE, {_name}"

TEST_CAPTION = "59 days, 2025-11-03 to 2025-12-31. Parameters were not fit on these days. The raw score is not currency."
BLUE, ORANGE, RED, GRAY, GREEN = "#256b9b", "#d08324", "#b34849", "#818b96", "#387f6a"


def load_evidence():
    protocol = read_json(PROTOCOL_PATH)
    run_dir = run_directory(protocol)
    marker = read_json(run_dir / "evaluation_complete.json")
    for name in ["analysis", "comparison"]:
        suffix = ".json" if name == "analysis" else ".csv"
        if sha256(run_dir / (name + suffix)) != marker[name + "_sha256"]:
            raise RuntimeError(f"evaluated {name} changed; do not render silently")
    comparison = pd.read_csv(run_dir / "comparison.csv")
    daily = pd.read_csv(run_dir / "dispatch_daily.csv", parse_dates=["date"])
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    predictions["times"] = pd.to_datetime(predictions.times)
    return run_dir, protocol, read_json(run_dir / "analysis.json"), comparison, daily, predictions


def public_table(comparison):
    table = comparison.copy()
    table["configuration"] = table.candidate.map(LABELS)
    table["role"] = table.group.map({
        "baseline": "Matched baseline", "nb11_v2": "Notebook 11 reference", "nb11_v1_context_only": "Non-matched historical reference",
        "cursor_original": "Prior window-CE result", "codex_main": "Predeclared soft-regret branch",
        "matched_cursor_control": "Window-CE control, not the soft-regret loss",
        "mse_realized_control": "Selection-metric control", "mse_rmse_control": "MSE reference",
    })
    table.loc[table.preselected_primary.fillna(False).astype(bool), "role"] = "Preselected soft-regret primary"
    cols = ["configuration", "role", "realized_profit_mean", "difference_vs_cursor_best", "negative_days", "idle_days", "RMSE"]
    return table[cols].rename(columns={"realized_profit_mean": "mean_raw_dispatch_score", "difference_vs_cursor_best": "difference_vs_prior_best_lstm", "negative_days": "negative_score_days", "idle_days": "idle_days", "RMSE": "point_rmse_auxiliary"})


def control_table(analysis):
    rows = []
    for control in analysis["controls"]:
        rows.append({
            "family": control["family"],
            "selection_only_score_minus_rmse": control["selection_only"]["mean_difference"],
            "soft_regret_minus_mse_both_by_score": control["soft_vs_mse_money"]["mean_difference"],
            "soft_regret_minus_matched_window_ce": control["soft_vs_matched_ce"]["mean_difference"],
        })
    return pd.DataFrame(rows)


def cost_table(run_dir):
    rows = [read_json_line(line) for line in (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line]
    complete = pd.DataFrame([r for r in rows if r["event"] == "training_end" and r.get("status") == "ok"])
    table = complete.groupby("method", sort=False).agg(
        launches=("job_id", "size"), worker_seconds=("seconds", "sum"), cpu_seconds=("cpu_seconds", "sum"),
        process_wall_seconds=("subprocess_wall_seconds", "sum"), optimizer_steps=("optimizer_steps", "sum"),
    ).reset_index()
    return table.rename(columns={"method": "method", "launches": "training_launches", "worker_seconds": "worker_wall_seconds", "cpu_seconds": "cpu_seconds", "process_wall_seconds": "process_wall_seconds_including_startup", "optimizer_steps": "optimizer_steps"})


def read_json_line(line):
    import json
    return json.loads(line)


def style():
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig, figures, name):
    path = figures / (name + ".png")
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_overview(figures, analysis, comparison):
    ids = ["lgb_baseline", "nb11_v2_transformer", "cursor_original_transformer", "cursor_original_lstm", "cursor_original_dlinear", "codex_soft_regret_transformer_realized", "codex_soft_regret_lstm_realized", "codex_soft_regret_dlinear_realized", "codex_cursor_ce_transformer_realized"]
    data = comparison.set_index("candidate").loc[ids]
    labels = [LABELS[name] for name in ids]
    labels[7] += "  <- preselected primary"
    labels[8] += "  <- control, not the new loss"
    colors = [GRAY, GRAY, ORANGE, ORANGE, ORANGE, BLUE, BLUE, RED, GREEN]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.barh(np.arange(len(ids)), data.realized_profit_mean, color=colors, height=.65)
    ax.set_yticks(np.arange(len(ids)), labels)
    ax.invert_yaxis()
    for i, value in enumerate(data.realized_profit_mean):
        ax.text(value + 55, i, f"{value:,.2f}", va="center", fontsize=11)
    ax.axvline(analysis["cursor_original_best_score"], color=ORANGE, ls="--", lw=1, label="Prior window-CE best")
    ax.set_xlim(0, float(data.realized_profit_mean.max()) * 1.18)
    ax.set_xlabel("Mean raw dispatch score: sum(realized price x locked action), higher is better")
    ax.set_ylabel("Frozen candidate")
    ax.set_title("No across-the-board win: the LSTM branch improves, and the preselected DLinear primary fails", loc="left", pad=18)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    fig.text(.02, -.015, TEST_CAPTION + " Source: comparison.csv; every branch is kept.", fontsize=9)
    fig.tight_layout()
    return save(fig, figures, "01_holdout_scores")


def figure_controls(figures, comparison):
    families = ["transformer", "lstm", "dlinear"]
    methods = [("mse", "rmse", "Point MSE, select by RMSE", GRAY), ("mse", "realized", "Point MSE, select by dispatch score", BLUE), ("cursor_ce", "realized", "Hard-window CE, same initialization", ORANGE), ("soft_regret", "realized", "Soft regret, including idle", GREEN)]
    indexed = comparison.set_index("candidate")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5), sharey=True)
    for ax, family in zip(axes, families):
        for j, (method, policy, label, color) in enumerate(methods):
            value = indexed.loc[f"codex_{method}_{family}_{policy}", "realized_profit_mean"]
            ax.bar(j, value, color=color, width=.66, label=label)
            ax.text(j, value + 65, f"{value:.1f}", ha="center", fontsize=10)
        ax.set_xticks(range(4), ["MSE\nby RMSE", "MSE\nby score", "Hard\nwindow", "Soft\nregret"])
        ax.set_title(family.upper() if family == "lstm" else family.capitalize())
        ax.set_xlabel("Training objective / selection rule")
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("59-day mean raw dispatch score, higher is better")
    axes[0].set_ylim(0, 7500)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.07), frameon=False)
    fig.text(.02, -.02, TEST_CAPTION + " Initialization matches inside each model. MSE has no three-point loss search.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return save(fig, figures, "02_selection_and_loss_controls")


def figure_selection(figures, comparison):
    main = comparison.loc[comparison.group == "codex_main"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for row in main.itertuples():
        color = RED if row.family == "dlinear" else (BLUE if row.family == "lstm" else GRAY)
        for ax, field in zip(axes, ["validation_realized", "realized_profit_mean"]):
            value = getattr(row, field)
            ax.scatter(row.family, value, color=color, s=80, zorder=3)
            ax.annotate(f"{value:,.2f}", (row.family, value), xytext=(0, 12), textcoords="offset points", ha="center")
    axes[0].set_title("Selection window: 30 development days (10-04 to 11-02)")
    axes[1].set_title("Evaluation window: 59 days (11-03 to 12-31)")
    for ax in axes:
        ax.set_ylabel("Mean raw dispatch score in that window")
        ax.set_xlabel("Three predeclared model branches")
        ax.set_ylim(4600, 9000)
        ax.margins(x=.2)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("The development winner is not the test winner. The primary was not switched to LSTM after the test.", fontsize=13, y=1.05)
    fig.text(.02, -.04, "The two windows are not equally difficult, so the drop in level is not all overfitting. The model ranking does reverse. The test window does not select parameters.", fontsize=9)
    fig.tight_layout()
    return save(fig, figures, "03_validation_selection_reversal")


def figure_daily(figures, analysis, daily):
    wide = daily.pivot(index="date", columns="candidate", values="realized_profit").sort_index()
    ref = wide[analysis["cursor_original_best_candidate"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    names = [(analysis["primary_candidate"], "Preselected primary, DLinear"), (analysis["observed_best_codex_candidate"], "LSTM branch, not the primary")]
    for row_i, (candidate, title) in enumerate(names):
        delta = wide[candidate] - ref
        axes[row_i, 0].bar(delta.index, delta, color=np.where(delta >= 0, BLUE, RED), width=.8)
        axes[row_i, 1].plot(delta.index, delta.cumsum(), color=BLUE, lw=1.7)
        axes[row_i, 0].set_title(f"{title}: daily gap versus prior window-CE LSTM")
        axes[row_i, 1].set_title(f"Cumulative gap, end {delta.sum():+,.0f}")
        axes[row_i, 0].set_ylabel("Daily raw-score gap")
        axes[row_i, 1].set_ylabel("Cumulative raw-score gap")
        for ax in axes[row_i]:
            ax.axhline(0, color=GRAY, lw=1)
            ax.axvline(pd.Timestamp("2025-12-03"), color=GRAY, ls=":", lw=1)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.set_xlabel("Date (dotted line splits the first 30 days from the last 29)")
            ax.grid(alpha=.12)
    fig.text(.015, -.02, TEST_CAPTION + " Positive and negative days are both shown. Source: dispatch_daily.csv.", fontsize=9)
    fig.tight_layout()
    return save(fig, figures, "04_daily_differences")


def case_records(analysis, daily, predictions):
    cases = [
        (analysis["primary_vs_cursor_daily"]["worst_day"], analysis["primary_candidate"], "Primary versus prior LSTM, largest loss day"),
        (next(c for c in analysis["controls"] if c["family"] == "lstm")["soft_vs_cursor_original"]["best_day"], "codex_soft_regret_lstm_realized", "LSTM branch versus prior LSTM, largest gain day"),
    ]
    rows = []
    for date, candidate, reason in cases:
        point = predictions.loc[predictions.times.dt.normalize() == pd.Timestamp(date)]
        for name in [candidate, "cursor_original_lstm"]:
            row = daily.loc[(daily.date == pd.Timestamp(date)) & (daily.candidate == name)].iloc[0]
            c, d = int(row.tc), int(row.td)
            actual = point.A.to_numpy()
            rows.append({"date": date, "candidate": name, "selection_reason": reason,
                         "charge_start": f"{c // 4:02d}:{(c % 4) * 15:02d}", "discharge_start": f"{d // 4:02d}:{(d % 4) * 15:02d}",
                         "actual_charge_mean": float(actual[c:c+8].mean()), "actual_discharge_mean": float(actual[d:d+8].mean()),
                         "realized_score": float(row.realized_profit), "predicted_score": float(row.predicted_profit), "oracle_score": float(row.oracle_profit)})
    return rows


def figure_cases(figures, analysis, daily, predictions, cases):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True)
    for i in range(2):
        pair = cases[2*i:2*i+2]
        point = predictions.loc[predictions.times.dt.normalize() == pd.Timestamp(pair[0]["date"])]
        t = np.arange(96) / 4
        all_values = [point.A.to_numpy()] + [point[r["candidate"]].to_numpy() for r in pair]
        lo, hi = min(v.min() for v in all_values), max(v.max() for v in all_values)
        span = max(hi-lo, .1)
        for j, record in enumerate(pair):
            ax = axes[i, j]
            row = daily.loc[(daily.date == pd.Timestamp(record["date"])) & (daily.candidate == record["candidate"])].iloc[0]
            c, d = int(row.tc), int(row.td)
            ax.plot(t, point.A, color="#252a30", lw=1.4, label="Realized price (scoring only)")
            ax.plot(t, point[record["candidate"]], color=BLUE, lw=1.2, ls="--", label="Frozen model forecast")
            ax.axvspan(c/4, (c+8)/4, color=BLUE, alpha=.16, label="Charge window chosen from the forecast")
            ax.axvspan(d/4, (d+8)/4, color=ORANGE, alpha=.2, label="Discharge window chosen from the forecast")
            ax.set_title(f"{record['date']} · {LABELS[record['candidate']]}\nRealized score {record['realized_score']:+,.0f}; predicted score {record['predicted_score']:+,.0f}", fontsize=11)
            ax.set_ylim(lo-.1*span, hi+.1*span)
            ax.set_ylabel("Price A (source scale)")
            ax.set_xlabel("Hour of day")
            ax.set_xticks(np.arange(0, 25, 4))
            ax.grid(alpha=.13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.075), frameon=False)
    fig.text(.015, -.02, "Cases are the largest and smallest paired gaps, for mechanism, not as the average effect. Forecasts were not shifted by the realized price." + TEST_CAPTION, fontsize=9)
    fig.tight_layout()
    return save(fig, figures, "05_actual_decision_examples")


def build():
    started = time.perf_counter()
    run_dir, protocol, analysis, comparison, daily, predictions = load_evidence()
    style()
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    cases = case_records(analysis, daily, predictions)
    outputs = [figure_overview(figures, analysis, comparison), figure_controls(figures, comparison),
               figure_selection(figures, comparison), figure_daily(figures, analysis, daily),
               figure_cases(figures, analysis, daily, predictions, cases)]
    public_table(comparison).to_csv(run_dir / "comparison_public.csv", index=False, encoding="utf-8-sig")
    control_table(analysis).to_csv(run_dir / "controls_public.csv", index=False, encoding="utf-8-sig")
    cost_table(run_dir).to_csv(run_dir / "costs_public.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cases).to_csv(run_dir / "case_details.csv", index=False)
    write_json(run_dir / "figure_manifest.json", {
        "sources": ["comparison.csv", "dispatch_daily.csv", "predictions.parquet", "analysis.json", "ledger.jsonl"],
        "figures": [{"path": str(p.relative_to(run_dir)), "sha256": sha256(p)} for p in outputs],
        "generator": "experiments/holdout_dispatch_loss_codex_v1/report.py", "training_performed": False,
        "case_selection": "Primary worst paired day and LSTM branch best paired day versus the prior window-CE LSTM, selected after evaluation for illustration only.",
    })
    Ledger(run_dir, protocol).add("report_figures_built", seconds=time.perf_counter()-started, count=len(outputs), training=False)
    print("\n".join(str(p) for p in outputs))
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == "__main__":
    build()
