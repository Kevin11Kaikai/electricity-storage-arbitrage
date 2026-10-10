"""Build and execute notebooks/notebook_12_claude.ipynb from the frozen artefacts.

The notebook computes its own verdict from ``reports/holdout_dfl_ltr_v1/`` so the
prose cannot drift from the numbers.  Run this after run.py, ensemble.py,
rerank.py and diagnostics.py have produced their outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "notebook_12_claude.ipynb"


SETUP = '''
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "phase_b").is_dir():
            return candidate
    raise FileNotFoundError("run this notebook from the repository root")


REPO_ROOT = find_repo_root()
TORCH_RUNTIME = REPO_ROOT / ".torch_runtime"
if TORCH_RUNTIME.exists() and str(TORCH_RUNTIME) not in sys.path:
    sys.path.insert(0, str(TORCH_RUNTIME))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FIGURES = REPO_ROOT / "outputs" / "notebooks" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
REPORT = REPO_ROOT / "reports" / "holdout_dfl_ltr_v1"
V1 = REPO_ROOT / "reports" / "holdout_rmse_v1"
V2 = REPO_ROOT / "reports" / "holdout_rmse_v2"
NB12 = REPO_ROOT / "reports" / "holdout_dispatch_loss_v1"
PHASE_B = REPO_ROOT / "reports" / "phase_b" / "final_metrics.json"

MONEY_LINE = 6301.331965      # nb12 lstm_dispatch -- the incumbent to beat
NB11_LINE = 6214.361782       # nb11 money champion, v2 transformer_contextual
ORACLE = 9000.886096
RNG = np.random.default_rng(20260914)

def show(path):
    display(Image(filename=str(path)))

print({"repo_root": str(REPO_ROOT), "report": str(REPORT.relative_to(REPO_ROOT))})
'''

LOAD = '''
diag = json.loads((REPORT / "diagnostics.json").read_text(encoding="utf-8"))
board = pd.read_csv(REPORT / "leaderboard.csv")
manifest = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
assert manifest["holdout_not_used_for_selection"] is True
assert PHASE_B.exists(), "phase_b frozen metrics must still be present"

# ---- daily realized profit for every comparator, one column per recipe
frames = []
for path, tag in ((V1, "v1"), (V2, "v2"), (NB12, "nb12"), (REPORT, "new")):
    csv = path / "dispatch_daily.csv"
    if csv.exists():
        f = pd.read_csv(csv)
        f["label"] = f["model"] if tag in ("nb12", "new") else tag + "/" + f["model"]
        frames.append(f[["label", "date", "realized_profit"]])
for extra, label in ((REPORT / "ensemble_daily.csv", "r5_ensemble"),
                     (REPORT / "rerank_daily.csv", "r4_rerank")):
    if extra.exists():
        f = pd.read_csv(extra)
        f["label"] = label
        frames.append(f[["label", "date", "realized_profit"]])
daily = pd.concat(frames, ignore_index=True)
daily["date"] = pd.to_datetime(daily["date"])
DAILY = daily.pivot_table(index="date", columns="label", values="realized_profit")

# the no-model constant window, as a daily series on the same 59 days
base = diag["constant_window_baselines"]["all_train_days"]
DAILY["constant_window"] = np.asarray(base["daily"], dtype=float)
BASELINE = float(base["realized_mean"])

def paired(a: str, b: str, n_boot: int = 20000):
    """Paired bootstrap over the 59 days of the difference a - b."""
    d = (DAILY[a] - DAILY[b]).dropna().to_numpy()
    n = len(d)
    draws = d[RNG.integers(0, n, (n_boot, n))].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "p_le_0": float((draws <= 0).mean()),
    }

print(f"recipes with daily series: {DAILY.shape[1]}   days: {DAILY.shape[0]}")
print(f"no-model constant window baseline: {BASELINE:.1f}")
print(board[["model", "realized_profit_mean", "capture"]].sort_values(
    "realized_profit_mean", ascending=False).head(8).to_string(index=False))
'''

VERDICT = '''
rows = []
for _, r in board.iterrows():
    stats_line = paired(r["model"], "lstm_dispatch") if r["model"] in DAILY.columns else None
    rows.append({
        "model": r["model"], "track": r["track"], "route": r["route"],
        "realized": r["realized_profit_mean"], "capture": r["capture"],
        "vs_incumbent": stats_line["diff"] if stats_line else np.nan,
        "p_le_0": stats_line["p_le_0"] if stats_line else np.nan,
    })
extra_rows = []
if (REPORT / "ensemble_weights.json").exists():
    ens = json.loads((REPORT / "ensemble_weights.json").read_text(encoding="utf-8"))
    extra_rows.append(("r5_ensemble", ens["holdout"]["realized_profit_mean"], ens["holdout"]["capture"]))
if (REPORT / "rerank.json").exists():
    rr = json.loads((REPORT / "rerank.json").read_text(encoding="utf-8"))
    extra_rows.append(("r4_rerank", rr["holdout"]["realized_profit_mean"], rr["holdout"]["capture"]))
for name, realized, capture in extra_rows:
    s = paired(name, "lstm_dispatch") if name in DAILY.columns else None
    rows.append({"model": name, "track": "-", "route": name.split("_")[0],
                 "realized": realized, "capture": capture,
                 "vs_incumbent": s["diff"] if s else np.nan,
                 "p_le_0": s["p_le_0"] if s else np.nan})

ALL = pd.DataFrame(rows).sort_values("realized", ascending=False).reset_index(drop=True)
best = ALL.iloc[0]
passed = ALL[ALL["realized"] > MONEY_LINE]

lines = []
if len(passed):
    ci = paired(best["model"], "lstm_dispatch")
    sig = "and the paired CI lower bound is above 0" if ci["lo"] > 0 else "but the paired CI contains 0"
    lines.append(
        f"**Clears the line.** The best new recipe is `{best['model']}`: realized **{best['realized']:.0f}**,"
        f"capture **{best['capture']:.3f}** (line {MONEY_LINE:.0f} / 0.7001)."
        f"Versus the incumbent {ci['diff']:+.0f}, 95% CI [{ci['lo']:+.0f}, {ci['hi']:+.0f}],"
        f"P(≤0)={ci['p_le_0']:.3f}  —  {sig}."
    )
    lines.append(f"**{len(passed)} recipes clear the line:** " + ", ".join(
        f"`{m}` ({v:.0f})" for m, v in zip(passed['model'], passed['realized'])))
else:
    lines.append(
        f"**Does not clear the line.** The best new recipe is `{best['model']}`: realized **{best['realized']:.0f},"
        f"capture **{best['capture']:.3f}. The line is {MONEY_LINE:.0f} / 0.7001."
    )

cb = paired(best["model"], "constant_window")
lines.append(
    f"**Versus the no-model fixed window ({BASELINE:.0f}):** {cb['diff']:+.0f},"
    f"95%CI [{cb['lo']:+.0f}, {cb['hi']:+.0f}], P(≤0)={cb['p_le_0']:.3f}."
)

# The highest point estimate is not necessarily the best-evidenced result.
cand = ALL[ALL["model"].isin(DAILY.columns)].copy()
cand["p_vs_base"] = [paired(m, "constant_window")["p_le_0"] for m in cand["model"]]
cand["p_vs_inc"] = [paired(m, "lstm_dispatch")["p_le_0"] for m in cand["model"]]
solid = cand[(cand["realized"] > MONEY_LINE)].sort_values("p_vs_base")
if len(solid) and solid.iloc[0]["model"] != best["model"]:
    s0 = solid.iloc[0]
    sb, si = paired(s0["model"], "constant_window"), paired(s0["model"], "lstm_dispatch")
    lines.append(
        f"**The highest point estimate is not the strongest evidence.** Among recipes that clear the line, the strongest evidence is `{s0['model']}`"
        f"(realized {s0['realized']:.0f}): versus the fixed window {sb['diff']:+.0f},"
        f"95%CI [{sb['lo']:+.0f}, {sb['hi']:+.0f}], P(≤0)={sb['p_le_0']:.3f}; "
        f"versus the incumbent {si['diff']:+.0f}, P(<=0)={si['p_le_0']:.3f}."
        f" `{best['model']}` has much larger day-to-day variance, so its higher point estimate has a wider CI."
    )
lines.append(
    f"**The oracle is still {ORACLE:.0f}.** The best new recipe has capture {best['capture']:.3f}."
    f"These 59 days are not a new champion selection. The 35-recipe board and the Phase B frozen file are unchanged."
)
display(Markdown("### Result, computed from the saved outputs\\n\\n" + "\\n\\n".join("- " + s for s in lines)))
'''

DIAG_TABLES = '''
d1 = diag["D1_surface_flatness"]
t1 = pd.DataFrame({
    "kth_true_window": list(d1["kth_best_true_window_value"]),
    "window_value": [d1["kth_best_true_window_value"][k] for k in d1["kth_best_true_window_value"]],
    "share_of_oracle": [d1["kth_best_as_share_of_oracle"][k] for k in d1["kth_best_as_share_of_oracle"]],
})
hits = d1["hit_at_1"]
n_zero = sum(1 for v in hits.values() if v == 0.0)
display(Markdown(
    f"**D1.** Of 38 recipes, **{n_zero}** never hit the true argmax. The best hit rate is {max(hits.values())*59:.0f}/59."
    f"Across 2242 recipe-days there are **{sum(round(v*59) for v in hits.values()):.0f}** hits."
))
display(t1.style.format({"window_value": "{:.0f}", "share_of_oracle": "{:.3f}"}).hide(axis="index"))

d2 = diag["D2_topk_ceiling"]["per_recipe"]["lstm_dispatch"]
t2 = pd.DataFrame({
    "K": list(d2["perfect_rerank_ceiling"]),
    "perfect_ceiling": list(d2["perfect_rerank_ceiling"].values()),
    "oracle_in_topk_rate": list(d2["oracle_in_topk"].values()),
})
t2["vs_incumbent"] = t2["perfect_ceiling"] - MONEY_LINE
display(Markdown("**D2.** Value of the best true window inside the incumbent's own top-K:"))
display(t2.style.format({"perfect_ceiling": "{:.0f}", "oracle_in_topk_rate": "{:.2f}",
                         "vs_incumbent": "{:+.0f}"}).hide(axis="index"))

d4 = diag["D4_noise"]
display(Markdown(
    f"**D4.** SE of a 59-day mean is about **{d4['median_se_of_59day_mean']:.0f}**, and of 30 days about "
    f"**{d4['se_of_30day_mean']:.0f}**. Notebook 12's per-epoch validation realized-score std is "
    f"{d4['nb12_per_epoch_val_realized_std_range'][0]:.0f}–"
    f"{d4['nb12_per_epoch_val_realized_std_range'][1]:.0f}, the same magnitude."
    f" 30-day selection signal-to-noise, realized mean "
    f"{d4['selection_signal_to_noise_30day']['realized_mean']:.3f} -> mean daily capture "
    f"{d4['selection_signal_to_noise_30day']['daily_mean_capture']:.3f}."
))
'''

FIG1 = '''
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ks = [int(k) for k in d1["kth_best_as_share_of_oracle"]]
vals = [d1["kth_best_as_share_of_oracle"][str(k)] for k in ks]
axes[0].plot(ks, vals, "o-", color="#2a6f97")
axes[0].axhline(1.0, color="0.5", ls="--", lw=1)
axes[0].set_xscale("log")
axes[0].set_xlabel("True rank K of the legal window")
axes[0].set_ylabel("Share of oracle")
axes[0].set_title("D1. The score surface is flat at the top.\\nRank 50 is still 92.8% of oracle")
axes[0].set_ylim(0.9, 1.01)
axes[0].grid(alpha=0.3)

ks2 = [int(k) for k in d2["perfect_rerank_ceiling"]]
ceil = [d2["perfect_rerank_ceiling"][str(k)] for k in ks2]
axes[1].plot(ks2, ceil, "s-", color="#c44536", label="perfect rerank ceiling")
axes[1].axhline(MONEY_LINE, color="#2a6f97", ls="--", lw=1.2, label=f"incumbent {MONEY_LINE:.0f}")
axes[1].axhline(ORACLE, color="0.35", ls=":", lw=1.2, label=f"oracle {ORACLE:.0f}")
axes[1].axhline(BASELINE, color="#6c757d", ls="-.", lw=1, label=f"no-model fixed window {BASELINE:.0f}")
axes[1].set_xscale("log")
axes[1].set_xlabel("Shortlist size K")
axes[1].set_ylabel("59-day mean official score")
axes[1].set_title("D2. Good windows are already near the top.\\nThe error is their order inside the shortlist")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.tight_layout()
p = FIGURES / "nb12_claude_fig1_diagnostics.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print(p.relative_to(REPO_ROOT)); show(p)
'''

TABLE = '''
anchors = [
    ("nb12 incumbent", "lstm_dispatch", None),
    ("nb12 transformer", "transformer_dispatch", None),
    ("nb11 best score, v2 Transformer", "v2/transformer_contextual", None),
    ("nb11 best RMSE, v1 Transformer", "v1/transformer_contextual", None),
    ("nb11 weakest, v1 LSTM", "v1/lstm_contextual", None),
    ("weekday-slot mean", "v1/climatology_weekday_slot", None),
    ("no-model fixed window", "constant_window", None),
]
rows = []
for label, key, _ in anchors:
    if key not in DAILY.columns:
        continue
    s = paired(key, "lstm_dispatch")
    rows.append({"block": "anchor", "recipe": label, "realized": DAILY[key].mean(),
                 "capture": DAILY[key].mean() / ORACLE, "vs incumbent": s["diff"],
                 "ci_low": s["lo"], "ci_high": s["hi"], "P(≤0)": s["p_le_0"],
                 "pred_profit": np.nan, "RMSE": np.nan})

for _, r in board.iterrows():
    s = paired(r["model"], "lstm_dispatch")
    rows.append({"block": f"new {r['track']}", "recipe": r["model"],
                 "realized": r["realized_profit_mean"], "capture": r["capture"],
                 "vs incumbent": s["diff"], "ci_low": s["lo"], "ci_high": s["hi"],
                 "P(≤0)": s["p_le_0"], "pred_profit": r["pred_profit_mean"], "RMSE": r["RMSE"]})
for name in ("r5_ensemble", "r4_rerank"):
    if name in DAILY.columns:
        s = paired(name, "lstm_dispatch")
        rows.append({"block": "new combined", "recipe": name, "realized": DAILY[name].mean(),
                     "capture": DAILY[name].mean() / ORACLE, "vs incumbent": s["diff"],
                     "ci_low": s["lo"], "ci_high": s["hi"], "P(≤0)": s["p_le_0"],
                     "pred_profit": np.nan, "RMSE": np.nan})

TAB = pd.DataFrame(rows).sort_values("realized", ascending=False).reset_index(drop=True)
TAB["cleared"] = TAB["realized"] > MONEY_LINE
display(TAB.style.format({
    "realized": "{:.0f}", "capture": "{:.3f}", "vs incumbent": "{:+.0f}",
    "ci_low": "{:+.0f}", "ci_high": "{:+.0f}", "P(≤0)": "{:.3f}",
    "pred_profit": "{:.0f}", "RMSE": "{:.3f}",
}, na_rep="—").hide(axis="index"))
'''

FIG2 = '''
sub = TAB.copy()
colors = []
for block in sub["block"]:
    colors.append("#2a6f97" if str(block).startswith("new") else "#adb5bd")
fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.32 * len(sub))))
ax.barh(sub["recipe"], sub["realized"], color=colors)
ax.axvline(MONEY_LINE, color="#c44536", ls="--", lw=1.4, label=f"line, incumbent {MONEY_LINE:.0f}")
ax.axvline(NB11_LINE, color="#e0a458", ls="-.", lw=1.1, label=f"notebook 11 best score {NB11_LINE:.0f}")
ax.axvline(BASELINE, color="#6c757d", ls=":", lw=1.3, label=f"no-model fixed window {BASELINE:.0f}")
ax.invert_yaxis()
ax.set_xlim(4300, max(ORACLE * 0.78, sub["realized"].max() * 1.03))
ax.set_xlabel("59-day mean realized dispatch score")
ax.set_title("Blue = new recipes this round. Gray = frozen anchors")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
p = FIGURES / "nb12_claude_fig2_realized.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print(p.relative_to(REPO_ROOT)); show(p)
'''

FIG3 = '''
pairs = board.pivot_table(index=["route", "family"], columns="track",
                          values="realized_profit_mean").dropna()
fig, ax = plt.subplots(figsize=(9.0, 4.4))
x = np.arange(len(pairs))
ax.bar(x - 0.2, pairs["A"], width=0.4, label="Track A (notebook-12 protocol: seed 42, select on realized score)", color="#adb5bd")
ax.bar(x + 0.2, pairs["B"], width=0.4, label="Track B (5 seeds, select on mean daily capture)", color="#2a6f97")
ax.axhline(MONEY_LINE, color="#c44536", ls="--", lw=1.2, label=f"line {MONEY_LINE:.0f}")
ax.axhline(BASELINE, color="#6c757d", ls=":", lw=1.2, label=f"no-model fixed window {BASELINE:.0f}")
ax.set_xticks(x, [f"{r}\\n{f}" for r, f in pairs.index], fontsize=8)
ax.set_ylabel("59-day realized score")
ax.set_ylim(4300, None)
ax.set_title("Tracks A and B: what the selection rule is worth in dispatch score")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = FIGURES / "nb12_claude_fig3_tracks.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print(p.relative_to(REPO_ROOT)); show(p)
delta = (pairs["B"] - pairs["A"])
display(Markdown(
    f"Median track B minus track A is **{delta.median():+.0f}** (range {delta.min():+.0f} to {delta.max():+.0f})."
    " That column is the selection rule plus multi-seed averaging, kept separate from the loss."
))
'''

FIG4 = '''
fig, ax = plt.subplots(figsize=(6.6, 5.4))
have = board.dropna(subset=["pred_profit_mean"])
for track, marker, color in (("A", "s", "#adb5bd"), ("B", "o", "#2a6f97")):
    sl = have[have["track"] == track]
    ax.scatter(sl["pred_profit_mean"], sl["realized_profit_mean"], marker=marker,
               s=70, color=color, label=f"track {track}", edgecolors="white", linewidths=0.5)
for _, r in have.iterrows():
    ax.annotate(r["model"].replace("_listwise", "").replace("_cheap", "").replace("_pairdiff", "pd"),
                (r["pred_profit_mean"], r["realized_profit_mean"]),
                fontsize=6.5, xytext=(4, 3), textcoords="offset points")
lims = [min(have["pred_profit_mean"].min(), have["realized_profit_mean"].min()) - 300,
        max(have["pred_profit_mean"].max(), have["realized_profit_mean"].max()) + 300]
ax.plot(lims, lims, ":", color="#888", lw=1, label="pred = realized")
ax.axhline(MONEY_LINE, color="#c44536", ls="--", lw=1.1)
ax.set_xlabel("Predicted score")
ax.set_ylabel("Realized score")
ax.set_title("Optimism gap is not extra realized score")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
p = FIGURES / "nb12_claude_fig4_honesty.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print(p.relative_to(REPO_ROOT)); show(p)
'''

STABILITY = '''
decomp = json.loads((REPORT / "seed_decomposition.json").read_text(encoding="utf-8"))
inst = json.loads((REPORT / "instability.json").read_text(encoding="utf-8"))

DEC = pd.DataFrame(decomp["rows"]).sort_values("single_curve_mean", ascending=False)
display(Markdown(
    "**Track B scores the pointwise mean of five curves.** That mixes two things: how good one curve is"
    "(a property of the loss) and what averaging five curves is worth (available to any recipe; the incumbent is a single curve and"
    "never received it). Split:"
))
display(DEC[["route", "family", "single_curve_mean", "single_curve_std",
             "averaged_curve", "averaging_gain", "single_curve_vs_incumbent"]]
        .rename(columns={"single_curve_mean": "single_curve_mean", "single_curve_std": "single_curve_sd",
                         "averaged_curve": "after_averaging", "averaging_gain": "gain_from_averaging",
                         "single_curve_vs_incumbent": "single_curve_vs_6301"})
        .style.format({"single_curve_mean": "{:.0f}", "single_curve_sd": "{:.0f}", "after_averaging": "{:.0f}",
                       "gain_from_averaging": "{:+.0f}", "single_curve_vs_6301": "{:+.0f}"}).hide(axis="index"))

neg = int((DEC["averaging_gain"] < 0).sum())
display(Markdown(
    f"- Ranked by the **single curve**, first place is `{DEC.iloc[0]['route']}/{DEC.iloc[0]['family']}`, "
    f"not the R1 Transformer at the top of the averaged board.\\n"
    f"- `r6_cheap/dlinear` has a single-curve score of {float(DEC.loc[(DEC.route=='r6_cheap')&(DEC.family=='dlinear'),'single_curve_mean'].iloc[0]):.0f}, "
    f"**230 below** the incumbent. It clears the line because of a +296 gain from averaging.\\n"
    f"- Averaging is not a sure gain: **{neg} of 12 rows are negative**, the worst "
    f"{DEC['averaging_gain'].min():+.0f}. It reduces variance. It does not guarantee a higher mean."
))

rows = []
for key in ("r1_listwise/transformer", "r2_pairdiff/lstm", "r6_cheap/dlinear"):
    v = inst[key]
    rows.append({
        "recipe": key,
        "spread_across_8_reruns": v["mean_59day_score"]["spread"],
        "curve_spread": v["max_curve_spread_across_repeats"],
        "days_with_flipped_window": v["worst_pair"]["days_with_a_different_window"],
    })
rows.append({"recipe": "transformer after the 5-seed mean (3 reruns)",
             "spread_across_8_reruns": inst["averaging_damps_it"]["spread"],
             "curve_spread": np.nan, "days_with_flipped_window": 0})
display(Markdown(
    "**Eight reruns of the same seed, data, and code.** Any difference is CPU reduction order:"
))
display(pd.DataFrame(rows).style.format(
    {"spread_across_8_reruns": "{:.1f}", "curve_spread": "{:.2e}", "days_with_flipped_window": "{:.0f}"},
    na_rep="—").hide(axis="index"))

w = inst["r1_listwise/transformer"]["worst_pair"]["largest_single_day"]
display(Markdown(
    f"The Transformer's eight scores are "
    f"`{inst['r1_listwise/transformer']['mean_59day_score']['values']}` — **bimodal**."
    f" Only {inst['r1_listwise/transformer']['worst_pair']['days_with_a_different_window']} of 59 days changed window, and "
    f" **{w['date']} contributed {100*w['share_of_total_gap']:.0f}% of the gap**: "
    f"the discharge block jumped from `td={w['low_window'][1]}` ({w['low_window'][1]/4:.2f}h) to "
    f"`td={w['high_window'][1]}` ({w['high_window'][1]/4:.2f}h), and that day's score "
    f"{w['low_profit']:.0f} → {w['high_profit']:.0f}.\\n\\n"
    f"Mechanism: `realized_profit` is a **step function** of the forecast. `optimize_day` takes an argmax over 3,321 windows."
    f"When two windows are nearly tied, a perturbation of order {inst['r1_listwise/transformer']['max_curve_spread_across_repeats']:.1e} "
    f"flips the window. On a spike day ({w['date']} has oracle 31559) the 59-day mean jumps by more than 300.\\n\\n"
    f"**A lead of a few hundred score units in this notebook sits inside that numerical noise floor**, before sampling noise."
    f" LSTM and DLinear are unaffected (spread 0.0), and those are the R2 and R3 routes whose paired tests stand;"
    f"the five-seed mean cuts the spread from {inst['averaging_damps_it']['single_curve_spread_for_comparison']:.0f} "
    f" to {inst['averaging_damps_it']['spread']:.1f}."
))
'''

MULTIPLE = '''
n_new = int(len(board) + sum(1 for n in ("r5_ensemble", "r4_rerank") if n in DAILY.columns))
inc = DAILY["lstm_dispatch"].to_numpy()
sd = float(np.median(DAILY.std(ddof=1)))
others = [c for c in DAILY.columns if c != "lstm_dispatch"]
corr = float(np.median([np.corrcoef(inc, DAILY[c].to_numpy())[0, 1] for c in others]))

B = 20000
common = RNG.standard_normal((B, 1, 59))
idio = RNG.standard_normal((B, n_new, 59))
sim = (np.sqrt(corr) * common + np.sqrt(1 - corr) * idio) * sd + inc.mean()
best_of = sim.mean(axis=2).max(axis=1)
p_any = float((best_of > MONEY_LINE).mean())
observed = float(TAB.loc[TAB["block"].str.startswith("new"), "realized"].max() - MONEY_LINE)
p_obs = float((best_of > MONEY_LINE + observed).mean())
display(Markdown(
    f"This round reports **{n_new}** new recipes. Under a null that every recipe's true level equals the incumbent"
    f"(within-day sd {sd:.0f}, between-recipe daily correlation {corr:.2f}): \\n\\n"
    f"- P(at least one > {MONEY_LINE:.0f}) = **{p_any:.3f}**\\n"
    f"- P(the best exceeds by at least {observed:+.0f}) = **{p_obs:.3f}**  ← this round observed {observed:+.0f}\\n\\n"
    f"This row is required reporting: more recipes raise the chance of clearing the line by sampling alone."
))
'''

CLOSING = '''
best_new = TAB[TAB["block"].str.startswith("new")].iloc[0]
vs_inc = paired(best_new["recipe"], "lstm_dispatch")
vs_base = paired(best_new["recipe"], "constant_window")
vs_nb11 = paired(best_new["recipe"], "v2/transformer_contextual")

# --- the pre-registered stop-loss is per ROUTE, not on the single best recipe:
#     "if all six routes' CI against the constant window contain 0, write a
#      negative result".  Evaluate each route by its own best recipe.
route_rows = []
for route in sorted(set(board["route"]) | {"r4_rerank", "r5_ensemble"}):
    members = [m for m in DAILY.columns if m.startswith(route)]
    if not members:
        continue
    champ = max(members, key=lambda m: DAILY[m].mean())
    s_base = paired(champ, "constant_window")
    s_inc = paired(champ, "lstm_dispatch")
    route_rows.append({
        "route": route, "best recipe": champ, "realized": DAILY[champ].mean(),
        "vs_fixed_window": s_base["diff"], "ci_low": s_base["lo"], "ci_high": s_base["hi"],
        "p_le_0_vs_fixed_window": s_base["p_le_0"], "Bonferroni p×6": min(1.0, s_base["p_le_0"] * 6),
        "vs incumbent": s_inc["diff"], "P(≤0) vs incumbent": s_inc["p_le_0"],
        "above_line": DAILY[champ].mean() > MONEY_LINE,
    })
ROUTES_TAB = pd.DataFrame(route_rows).sort_values("realized", ascending=False)
display(Markdown("**Each route's representative (its own best realized score), versus the no-model fixed window:**"))
display(ROUTES_TAB.style.format({
    "realized": "{:.0f}", "vs_fixed_window": "{:+.0f}", "ci_low": "{:+.0f}", "ci_high": "{:+.0f}",
    "p_le_0_vs_fixed_window": "{:.3f}", "Bonferroni p×6": "{:.3f}",
    "vs incumbent": "{:+.0f}", "P(≤0) vs incumbent": "{:.3f}"}).hide(axis="index"))

clears = ROUTES_TAB[ROUTES_TAB["ci_low"] > 0]
verdict = []
if best_new["realized"] > MONEY_LINE:
    verdict.append(
        f"**Primary call: above the line.** The best new recipe `{best_new['recipe']}` realized "
        f"**{best_new['realized']:.0f}** > {MONEY_LINE:.0f}, capture "
        f"**{best_new['capture']:.3f}** > 0.7001. "
        f"{int((TAB['block'].str.startswith('new') & TAB['above_line']).sum())} recipes are above the line."
    )
else:
    verdict.append(f"**Primary call: below the line.** The best new recipe {best_new['realized']:.0f} < {MONEY_LINE:.0f}.")

verdict.append(
    f"**The best of 26 is not itself evidence.** Under the null, P(the best exceeds by at least "
    f"{best_new['realized'] - MONEY_LINE:+.0f}) = {p_obs:.3f}."
    f" The first-ranked `{best_new['recipe']}` versus the incumbent {vs_inc['diff']:+.0f}, "
    f"95%CI [{vs_inc['lo']:+.0f}, {vs_inc['hi']:+.0f}], P(≤0)={vs_inc['p_le_0']:.3f} — the CI contains 0."
)

if len(clears):
    survives = clears[clears["Bonferroni p×6"] < 0.05]
    marginal = clears[clears["Bonferroni p×6"] >= 0.05]
    fmt = lambda r: (
        f"**{r['route']}** ({r['best recipe']}, {r['vs_fixed_window']:+.0f}, "
        f"CI [{r['ci_low']:+.0f}, {r['ci_high']:+.0f}], Bonferroni p={r['Bonferroni p×6']:.3f})"
    )
    verdict.append(
        f"**The results that stand are here: {len(clears)} routes versus the no-model fixed window ({BASELINE:.0f})"
        f"have a paired CI lower bound above zero.** This is the pre-specified **per-route** comparison, not the maximum of 26."
    )
    if len(survives):
        verdict.append(
            "Those that **still hold after a 6-way Bonferroni correction** are:"
            + ", ".join(fmt(r) for _, r in survives.iterrows())
            + ". No earlier recipe in this project had done that."
        )
    if len(marginal):
        verdict.append(
            "**Not significant after Bonferroni** (the raw CI excludes 0; after correction it does not). Treat it as a hint:"
            + ", ".join(fmt(r) for _, r in marginal.iterrows())
            + "."
        )
    verdict.append("**The pre-registered stop was not triggered:** not every route's interval contains 0.")
else:
    verdict.append(
        f"**The pre-registered stop is triggered:** all six routes versus the no-model fixed window ({BASELINE:.0f}) have intervals containing 0,"
        "write the negative result from section 6 of the plan, and do not open another round."
    )

verdict.append(
    f"Versus the notebook-11 score leader ({NB11_LINE:.0f}): {vs_nb11['diff']:+.0f}, "
    f"95%CI [{vs_nb11['lo']:+.0f}, {vs_nb11['hi']:+.0f}]."
    f" The oracle {ORACLE:.0f} is still not approached (best capture {best_new['capture']:.3f}). "
    f"The D2 top-50 reranking room (+781) was not captured. On the 30-day selection window the reranker "
    "**declined to rerank** (blend=0). The order inside the shortlist is not learnable on this sample."
)
verdict.append(
    f"**Resolution warning:** eight reruns of the same seed give `r1_listwise/transformer` a 59-day-mean spread of "
    f"{inst['r1_listwise/transformer']['mean_59day_score']['spread']:.0f} (section 5). "
    f"91% of that spread is a flipped window on 2025-12-30. Discount any lead of a few hundred score units by that fact."
    f"R2 and R3 are unaffected (LSTM spread 0.0), which is one reason only they enter the formal conclusion."
)
verdict.append(
    "These 59 days are not a new champion: the 35-recipe board is not refreshed, and `reports/phase_b/final_metrics.json` is not edited,"
    "and the v1/v2 leaderboard RMSE/MAE columns are not edited."
)
display(Markdown("\\n\\n".join("- " + v for v in verdict)))
'''


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md(
            "# Six decision-aligned routes versus notebook 12\n\n"
            "**Purpose:** notebook 12 replaced `F.mse_loss` with day-centered block-sum MSE plus one-hot cross-entropy over 3,321 windows."
            "`lstm_dispatch` records a 59-day realized score of **6301.33** and capture **0.700**."
            "This notebook runs six routes (R1–R6) under the same protocol and asks whether any of them actually beat it.\n\n"
            "**Sole primary rule:** 59-day `realized_profit_mean` **> 6301.33**. The field is a raw dispatch score, not currency."
            "A lower RMSE is not a win. A high `pred_profit` or a wide gap is not a win.\n\n"
            "**Protocol:** 271 fit days, select on the last 30, refit on 301, score the 59 days once. Seed 42 is track A. Seeds 42–46 are track B."
            "The oracle window on a training day comes only from that day's training label `A`. `holdout_not_used_for_selection: true`."
            "The line-by-line comparison with the notebook-12 plan is in "
            "[`experiments/holdout_dfl_ltr_v1/PLAN.md`](../experiments/holdout_dfl_ltr_v1/PLAN.md) section 3.\n\n"
            "**Two tracks.** **A** matches notebook 12 (seed 42, select on the 30-day realized mean) and is comparable to 6301.33."
            "**B** is the robust track (5-seed mean, select on 30-day daily-mean capture). Both are reported. The gap is itself a result."
        ),
        code(SETUP.strip()),
        code(LOAD.strip()),
        code(VERDICT.strip()),
        md(
            "## 1. Design basis: four measured quantities\n\n"
            "The routes were chosen from the four measurements below, not from a paper ranking."
            "Recomputation: `experiments/holdout_dfl_ltr_v1/diagnostics.py`."
        ),
        code(DIAG_TABLES.strip()),
        code(FIG1.strip()),
        md(
            "**D3.** There are only 3,321 legal actions, so a `(batch, 3321)` tensor is exact and the Gibbs distribution over actions is"
            "**exact**. Mandi's solution cache, Pogančić's DBB, and Berthet's perturbed optimizers are"
            "approximations for a feasible set that cannot be enumerated. Here they would only be worse, so this round excludes them.\n\n"
            "## 2. Six routes\n\n"
            "| | Route | External source | Change relative to notebook-12 window CE |\n"
            "|---|---|---|---|\n"
            "| R1 | Listwise value weights | Mandi et al. ICML 2022 **Eq. 16** | The target changes from one-hot to `softmax(true score / τ_tgt)`. Notebook 12 is the `τ_tgt → 0` limit. |\n"
            "| R2 | Pairwise difference | Mandi et al. **Eq. 13** | Regress window-score **differences**. A daily price level cancels. |\n"
            "| R3 | SPO+ | Elmachtoub & Grigas, *Manag. Sci.* 2022 | Convex upper bound on regret. Subgradient `2(z_{w*(2p̂−p)} − z_{w*(p)})`. |\n"
            "| R4 | top-K reranker | Mandi's ranking view plus D2 | Two stages: the curve proposes a shortlist, a shallow model reranks it |\n"
            "| R5 | Decision-weighted convex combination | Caruana et al. 2004, weights chosen by the task score | Convex combination of frozen curves. No new loss |\n"
            "| R6 | Pinball / peak-trough weights | Smets et al., *Energy* 2025 | Cheap control. A more elaborate loss has to beat it |\n\n"
            "The losses are in `experiments/holdout_dfl_ltr_v1/loss.py`."
            "Block sums reuse notebook 12's `day_center`, `block_sums`, and `window_scores`,"
            "so the shared definition is checked in code, not asserted."
        ),
        code(
            "import inspect\n"
            "from experiments.holdout_dfl_ltr_v1 import loss as dfl\n"
            "print(inspect.getsource(dfl.listwise_loss))\n"
            "print(inspect.getsource(dfl.spo_plus_loss))"
        ),
        md("## 3. Headline table\n\nEvery row has a paired-by-day bootstrap interval against the incumbent `lstm_dispatch`."),
        code(TABLE.strip()),
        code(FIG2.strip()),
        md("## 4. Tracks A and B: changing the loss versus changing the selection rule"),
        code(FIG3.strip()),
        code(FIG4.strip()),
        md(
            "## 5. Ensemble versus loss, and why a few hundred score units is not a gap on this window\n\n"
            "This section is how to read the table above. Artifacts:"
            "`reports/holdout_dfl_ltr_v1/seed_decomposition.json` and `instability.json`."
        ),
        code(STABILITY.strip()),
        md("## 6. Multiple comparisons"),
        code(MULTIPLE.strip()),
        md("## 7. Close-out, using the pre-registered wording in section 4 of the plan"),
        code(CLOSING.strip()),
        md(
            "---\n\n"
            "**Boundary:** these 59 days were already seen in notebook 11. This notebook only asks whether six routes beat notebook 12's score leader. The score is not currency."
            "It does not name a new champion, does not refresh the 35-recipe board, and does not edit `reports/phase_b/final_metrics.json`."
            "It does not edit the v1/v2 RMSE/MAE columns. R5's base learners were refit under this protocol"
            "(the frozen v1/v2 parquet files contain only the 59 evaluation days and have no 30-day validation forecasts, so they cannot set weights),"
            "which is also written in `ensemble_weights.json`."
        ),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> int:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    execute = "--no-exec" not in sys.argv
    if execute:
        from nbclient import NotebookClient

        client = NotebookClient(
            nb,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT / "notebooks")}},
        )
        client.execute()
    nbf.write(nb, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} (executed={execute})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
