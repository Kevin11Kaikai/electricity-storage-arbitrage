"""Generate the English numeric tables from exported records."""
from pathlib import Path
import argparse
import json
import pandas as pd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
args=parser.parse_args()
root=args.root
packaged = not (root/'shared').is_dir()
data=root/'data' if packaged else root/'shared'/'data'
M=pd.read_csv(data/'model_metrics.csv'); C=pd.read_csv(data/'correlations.csv')
B=pd.read_csv(data/'ensemble_decomposition.csv'); I=pd.read_csv(data/'paired_intervals.csv')
D=pd.read_csv(data/'rmse_provenance.csv'); CP=pd.read_csv(data/'controlled_comparisons_machine.csv',encoding='utf-8-sig')
S=json.loads((data/'summary.json').read_text(encoding='utf-8'))
def esc(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
             '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in str(s))
def f(x,n=2):
    # Typeset a real minus sign; a bare ASCII hyphen renders as a short dash in text mode.
    s=f'{x:.{n}f}'
    return r'$-$'+s[1:] if s.startswith('-') else s
def rng(lo,hi,n=2): return '['+f(lo,n)+', '+f(hi,n)+']'
def emit(out,name,caption,label,columns,header,rows,note=''):
    body='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    text=(r'\begin{table}[!htbp]'+'\n'+r'\centering\small'+'\n'+
          '\\caption{'+caption+'}\\label{'+label+'}\n'+
          '\\begin{tabularx}{\\linewidth}{'+columns+'}\n\\toprule\n'+
          ' & '.join(header)+r' \\'+'\n\\midrule\n'+body+'\n\\bottomrule\n\\end{tabularx}\n')
    if note: text+='\\par\\vspace{3pt}{\\footnotesize '+note+'}\n'
    text+='\\end{table}\n'
    assert not any(ord(ch)<32 and ch!='\n' for ch in text), (name, 'Control character in generated TeX')
    (out/(name+'.tex')).write_text(text,encoding='utf-8')
if packaged:
    language = json.loads((root/'BUILD_INFO.json').read_text())['language']
    if language != 'en':
        raise SystemExit('This repository builds the English report only.')
    out = root/'tables'
else:
    out = root/'en'/'tables'
out.mkdir(parents=True, exist_ok=True)
roles=[('holdout_rmse_v2','lgb_baseline','Matched LightGBM'),
       ('holdout_rmse_v2','transformer_contextual','MSE Transformer reference'),
       ('holdout_seed_bag_v1','listwise_transformer_mean','Frozen listwise ensemble')]
rows=[]
for src,model,label in roles:
    r=M[M.source.eq(src)&M.model.eq(model)].iloc[0]
    rows.append([label,f(r.predicted_mean),f(r.realized_mean),f(100*r.capture_aggregate)+r'\%',f(r.realized_mean-S['reference'])])
emit(out,'main_results','Main outcomes under the same 59-day evaluation and action contract.','tab:main','@{}Xrrrr@{}',
     ['Configuration','Predicted','Realized',r'$C_{\rm agg}$','vs. ref.'],rows,
     'Scores are daily means in the original scoring units. The shared oracle mean is 9000.89. Predicted values are not settlements.')
r=I[I.comparison.eq('headline_vs_reference')&I.block_length.eq(7)].iloc[0]
rows=[['Paired mean difference',f(S['paired_gain'])],
      ['Paired IID standard error (reference only)',f(S['paired_iid_se'])],
      ['7-day block 95\% conditional resampling interval',rng(r.ci_low,r.ci_high)],
      ['Wins / ties / losses (tolerance $10^{-6}$)',f"{S['wins']} / {S['ties']} / {S['losses']}"],
      ['Median daily difference',f(S['paired_median'])],
      ['Dec 30 share of total net gain',f(S['dec30_share_total_gain_pct'])+r'\%'],
      ['Mean difference excluding Dec 30',f(S['without_dec30_gain'])],
      ['Winning-day total / losing-day total',f(S['gross_winning_day_total'])+' / '+f(S['gross_losing_day_total'])],
      ['Largest single-day loss and date',f(S['worst_paired_difference'])+' ('+S['worst_paired_day']+')'],
      ['Worst realized day: headline / reference',f(S['worst_day_headline'])+' / '+f(S['worst_day_reference'])],
      ['Negative-score days: headline / reference',f"{S['negative_days_headline']} / {S['negative_days_reference']}"]]
emit(out,'paired','Paired evidence, loss tail, and uncertainty for the main comparison.','tab:paired','@{}Xr@{}',['Statistic','Value'],rows,
     'Date exclusion is a post hoc sensitivity analysis; the primary result retains all 59 days. Intervals do not correct for historical evaluation exposure. The worst-day and negative-day rows describe realized downside in this period; they are not a tail-risk model.')
rows=[]
for _,r in B.iterrows():
    label={'window_ce_lstm':'Window CE / LSTM','listwise_transformer':'Listwise / Transformer','spo_lstm':'SPO+ / LSTM'}[r.bag]
    rows.append([label,f(r.single_seed_mean),f(r.single_seed_sd),f(r.curve_mean_score),f(r.ensemble_increment)])
emit(out,'bags','Ensemble decomposition in fixed role order; no headline reselection.','tab:bags','@{}Xrrrr@{}',
     ['Recipe','Seed mean','Seed SD','Curve mean','Increment'],rows,
     'Every recipe uses seeds 42--46. Seed SD describes these five fits, not uncertainty over future periods.')
rows=[]
for family in ['Transformer','LSTM','DLinear']:
    def get(metric):
        selected=CP[CP.candidate.eq('codex_mse_'+family.lower()+'_'+metric)]
        assert len(selected)==1, (family,metric,'candidate must be unique')
        return float(selected.iloc[0].realized_profit_mean)
    a=get('rmse'); b=get('realized')
    rows.append([family,f(a),f(b),f(b-a)])
emit(out,'selection_control','Existing control: MSE training with a changed validation selection metric.','tab:selection','@{}Xrrr@{}',
     ['Family','Select by RMSE','Select by realized','Difference'],rows,
     r'These are the recorded control experiment, not refits of every historical candidate in Table~\ref{tab:main}.')
rows=[[r.source.replace('holdout_rmse_',''),r'\texttt{'+esc(r.model)+'}',f(r.original_rmse,6),f(r.same_prediction_rmse,6),f(r.realized_mean)] for _,r in D[D.source.isin(['holdout_rmse_v1','holdout_rmse_v2'])].iterrows()]
emit(out,'provenance','Historical RMSE versus the RMSE of the settled saved curve.','tab:provenance','@{}lXrrr@{}',
     ['Set','Model','Historical','Same curve','Realized'],rows,
     'No original RMSE/MAE field is overwritten. Other point candidates agree within $10^{-5}$.')
rows=[]
names={'point_mse':'RMSE','centered_mse':'Centered RMSE','listwise_proxy':'Listwise proxy','window_spearman':'Window rank $T$'}
for _,r in C[C.block_length.eq(7)].iterrows():
    rows.append([r.version,names[r.metric],f(r.pearson,3),rng(r.pearson_ci_low,r.pearson_ci_high,3),f(r.spearman,3),rng(r.spearman_ci_low,r.spearman_ci_high,3)])
emit(out,'correlations','Across-candidate associations and shared day-resampling intervals.','tab:corr','@{}lXrcr c@{}'.replace(' ',''),
     ['Set','Metric',r'$r$','95\% interval',r'$\rho$','95\% interval'],rows,
     r'$r$: Pearson; $\rho$: across-candidate Spearman. $T$ separately uses within-day Spearman. Percentile intervals: 7-day blocks, 10,000 replicates, dates resampled and candidates fixed. They exclude candidate-search and candidate-sampling uncertainty.')
rows=[]
for _,r in I.iterrows():
    label='Headline minus reference' if r.comparison=='headline_vs_reference' else 'Curve mean minus seed mean'
    rows.append([label,int(r.block_length),f(r.difference_mean),rng(r.ci_low,r.ci_high)])
emit(out,'bootstrap','Block-length sensitivity on the same frozen evidence.','tab:bootstrap','@{}Xrrc@{}',
     ['Comparison','Days/block','Mean diff.','95\% interval'],rows)
print('Generated English numeric tables')
