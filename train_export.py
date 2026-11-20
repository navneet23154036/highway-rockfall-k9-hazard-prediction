"""Train a deployable k=9 ensemble from the user's v4 notebook functions.
Run in Colab or locally: python train_export.py
"""
import numpy as np
import pandas as pd
import joblib
import notebook_source as n
from pathlib import Path

n.DATA_PATH = str(Path(__file__).with_name('2007 CDOTslope scores.xls'))
ctx = n.build_dataset()
X, total = ctx['X'], ctx['total']
idx = np.arange(len(X))
flags = n.infer_traffic_rule(ctx, idx)
known = ctx['base_known'] + flags * ctx['traffic_sum']
remainder = total - known
rank = {}
for mode, items in n.ITEMS.items():
    rows = idx[ctx['mode'] == mode]
    cols = [n.ALL_ITEMS.index(c) for c in items]
    rank[mode] = n.forward_rank(items, ctx['V'][rows][:, cols], remainder[rows])
obs = np.zeros_like(ctx['V'])
for mode in n.ITEMS:
    rows = np.where(ctx['mode'] == mode)[0]
    for item in rank[mode][:9]:
        obs[rows, n.ALL_ITEMS.index(item)] = ctx['V'][rows, n.ALL_ITEMS.index(item)]
osum = obs.sum(axis=1)
F = X[n.BASE_NUM + n.CAT].copy()
F['traffic_rule'] = flags
for j, item in enumerate(n.ALL_ITEMS):
    F[f'obs_{item}'] = obs[:, j]
num_cols = n.BASE_NUM + ['traffic_rule'] + [f'obs_{it}' for it in n.ALL_ITEMS]
target = total - known - osum
models = []
for seed in n.SEEDS:
    for name in ('gbr', 'ridge'):
        print('Training', name, seed, flush=True)
        models.append(n.make_pipeline(num_cols, name, seed).fit(F, target))
thresholds = {}
for key, group in pd.DataFrame({'key':ctx['key'],'total':total}).groupby('key'):
    if len(group) >= n.MIN_GROUP:
        thresholds[key] = np.quantile(group.total, [.3,.7]).tolist()
raters = {str(r): bool(flags[np.where(ctx['raters']==r)[0][0]]) for r in np.unique(ctx['raters'])}
bundle = dict(version='v4-k9', models=models, features=list(F.columns), num_cols=num_cols,
              rank=rank, item_names=ctx['item_names'], known_ids=n.KNOWN_IDS,
              all_items=n.ALL_ITEMS, thresholds=thresholds,
              global_thresholds=np.quantile(total,[.3,.7]).tolist(),
              rater_traffic_rule=raters, category_values={c:sorted(X[c].dropna().astype(str).unique().tolist()) for c in n.CAT}, min_group=n.MIN_GROUP,
              class_names=n.CLASS_NAMES,
              notes='Full-data fit; notebook k=9 evaluation is cross-validated. No calibrated k=9 probability supplied.')
joblib.dump(bundle, Path(__file__).with_name('rockfall_v4_k9.joblib'), compress=3)
print('Saved rockfall_v4_k9.joblib')
