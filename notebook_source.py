"""
Slope hazard classification, v4 = v3 (exact assembly + short geologic form) plus:
  1. validated triage: confidence cutoff chosen from inner out-of-fold margins on the
     training routes only, then applied to the held-out route;
  2. calibrated P(High): Platt scaling on (predicted total - group 70th percentile),
     fit on inner out-of-fold predictions; Brier, AUC, ECE, alert-threshold table;
  3. paired route-bootstrap tests between tiers;
  4. an audit of team-B totals that do not equal formula + traffic.

Everything that v3 printed is printed again, computed the same way.

usage: python slope_hazard_v4.py [data.xls] [n_seeds] [n_inner_seeds]
  n_inner_seeds defaults to n_seeds. The inner leave-one-route-out loop costs about
  8x the v3 run (roughly 25 min on one core with 5 seeds; joblib uses all cores).
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

args = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]
DATA_PATH = args[0] if len(args) > 0 else "2007 CDOTslope scores.xls"
N_SEEDS = int(args[1]) if len(args) > 1 else 5
N_INNER_SEEDS = int(args[2]) if len(args) > 2 else N_SEEDS
SHEET = "Slope Scores"
OUT_DIR = "artifacts_v4"
SEEDS = [42 + i for i in range(N_SEEDS)]
INNER_SEEDS = SEEDS[:N_INNER_SEEDS]
N_BOOT = 2000
N_JOBS = -1
MIN_GROUP = 5
K_GRID = [0, 1, 2, 3, 4, 5, 9]
K_REPORT = [0, 2, 3]
CLASS_NAMES = ["Low", "Moderate", "High"]

# --- settings for the new analyses, fixed before running them ----------------
K_INNER = [0, 2, 3]  # k values that need inner out-of-fold predictions
K_TRIAGE = [2, 3]  # stage-2 depth
TRIAGE_BUDGETS = [0.2, 0.3, 0.4, 0.5, 0.6]  # share of training slopes to assess
TARGET_ACC = 0.70  # target-rule: smallest share whose inner OOF accuracy reaches this
TARGET_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
K_CALIB = [0, 2, 3]
ALERT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ECE_BINS = 10
MARGIN_SCALE = 100.0  # score points; only for numerical conditioning of Platt
SCALE_VALUES = np.array([0, 3, 9, 27, 81, 243], float)

GEO_COLS_IDX = {
    "slope_height": 11,
    "slope_angle": 12,
    "slope_length": 13,
    "ditch_slope": 14,
    "ditch_width": 15,
    "ditch_depth": 16,
}
TRAFFIC_COLS_IDX = {"accidents": 10, "sight_dist": 17, "speed": 18, "adt": 19, "avr": 20}
GEO = list(GEO_COLS_IDX)
CAT = ["rock_type", "slope_type", "region"]
sc = lambda ids: [f"sc{i}" for i in ids]
ENV = sc([28, 32, 33, 34, 35])
DER = sc([27, 29, 31])
TRAF = sc([51, 52, 53])
KNOWN_IDS = [27, 28, 29, 31, 32, 33, 34, 35]
BASE_NUM = GEO + ENV + DER + TRAF

SECTIONS = {
    "slope_sum": range(27, 32),
    "clim_sum": range(32, 36),
    "sed_sum": range(36, 39),
    "cr_sum": range(39, 42),
    "disc_sum": range(42, 48),
    "blk_sum": range(48, 51),
}
ITEMS = {
    "cr": sc(range(39, 48)),
    "sed": sc(list(range(36, 39)) + list(range(42, 48))),
    "blk": sc(range(48, 51)),
}
ALL_ITEMS = sorted({c for v in ITEMS.values() for c in v}, key=lambda s: int(s[2:]))


# ----------------------------------------------------------------------------
# Data (unchanged from v3, plus raw cells kept for the audit)
# ----------------------------------------------------------------------------
def to_number(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    found = re.findall(r"\d+\.?\d*", str(x))
    return float(np.mean([float(v) for v in found])) if found else np.nan


def load_sheet():
    engine = "xlrd" if DATA_PATH.lower().endswith(".xls") else None
    raw = pd.read_excel(DATA_PATH, sheet_name=SHEET, header=None, engine=engine)
    group_row = raw.iloc[0].ffill().astype(str).tolist()
    label_row = [str(c).replace("\n", " ").strip() for c in raw.iloc[1]]
    names = [f"{g}|{c}" for g, c in zip(group_row, label_row)]
    d = raw.iloc[2:].copy()
    d.columns = names
    d = d[d[names[1]].notna()].reset_index(drop=True)
    return d, names


def build_dataset():
    d, names = load_sheet()
    col = lambda i: d[names[i]]
    X = pd.DataFrame(index=d.index)
    X["route"] = col(0).astype(str).str.strip()
    X["segment"] = col(1).astype(str).str.strip()
    X["region"] = col(2).astype(str).str.strip()
    X["rater"] = col(9).astype(str).str.strip()
    X["rock_type"] = col(22).astype(str).str.strip()
    X["slope_type"] = col(25).astype(str).str.strip()
    for name, i in {**GEO_COLS_IDX, **TRAFFIC_COLS_IDX}.items():
        X[name] = col(i).map(to_number)

    scores = d[names[27:54]].apply(pd.to_numeric, errors="coerce")
    scores.columns = sc(range(27, 54))
    for c in scores.columns:
        X[c] = scores[c]
    for name, rng in SECTIONS.items():
        X[name] = scores[sc(rng)].fillna(0).sum(axis=1)

    is_cr = X["rock_type"].str.startswith("Cryst")
    is_sed = X["rock_type"].str.startswith("Sed")
    X["mode"] = np.where(is_cr, "cr", np.where(is_sed, "sed", "blk"))
    X["rock_specific"] = np.where(
        is_cr,
        X["cr_sum"] + X["disc_sum"],
        np.where(is_sed, X["sed_sum"] + X["disc_sum"], X["blk_sum"]),
    )
    X["formula"] = X["slope_sum"] + X["clim_sum"] + X["rock_specific"]
    X["traffic_sum"] = X[TRAF].sum(axis=1)

    total = pd.to_numeric(col(54), errors="coerce").to_numpy()
    key = (X["rock_type"] + "|" + X["slope_type"]).to_numpy()
    ranks = pd.DataFrame({"t": total, "k": key}).groupby("k")["t"].rank(method="average", pct=True)
    y = np.where(ranks <= 0.3, 0, np.where(ranks <= 0.7, 1, 2))

    V = X[ALL_ITEMS].fillna(0).to_numpy(float)
    sig = X[sc(range(27, 54))].fillna(-1).astype(str).agg("|".join, axis=1) + "|" + pd.Series(total).astype(str)
    dup_w = 1.0 / sig.map(sig.value_counts()).to_numpy()
    cross_route = int((pd.DataFrame({"s": sig, "r": X["route"]}).groupby("s")["r"].nunique() > 1).sum())
    return {
        "raw": d,
        "names": names,
        "dup_w": dup_w,
        "profiles_across_routes": cross_route,
        "X": X,
        "total": total,
        "key": key,
        "y": y,
        "groups": X["route"].to_numpy(),
        "mode": X["mode"].to_numpy(),
        "V": V,
        "base_known": X[sc(KNOWN_IDS)].fillna(0).sum(axis=1).to_numpy(),
        "traffic_sum": X["traffic_sum"].to_numpy(),
        "raters": X["rater"].to_numpy(),
        "item_names": {c: names[int(c[2:])].split("|", 1)[1].replace(" Score", "") for c in ALL_ITEMS},
        "item_group": {c: names[int(c[2:])].split("|", 1)[0] for c in ALL_ITEMS},
    }


# ----------------------------------------------------------------------------
# Fold ingredients (unchanged from v3): all from training routes only
# ----------------------------------------------------------------------------
def infer_traffic_rule(ctx, tr):
    X, total = ctx["X"], ctx["total"]
    rule = {}
    for r in np.unique(ctx["raters"][tr]):
        rows = tr[ctx["raters"][tr] == r]
        plain = np.mean(np.abs(total[rows] - X["formula"].to_numpy()[rows]) < 1e-6)
        plus = np.mean(np.abs(total[rows] - X["formula"].to_numpy()[rows] - ctx["traffic_sum"][rows]) < 1e-6)
        rule[r] = bool(plus > plain)
    return np.array([float(rule.get(r, False)) for r in ctx["raters"]])


def group_thresholds(key_te, total_tr, key_tr):
    global_q = np.quantile(total_tr, [0.3, 0.7])
    lo, hi = np.empty(len(key_te)), np.empty(len(key_te))
    for k in np.unique(key_te):
        m = key_tr == k
        q = np.quantile(total_tr[m], [0.3, 0.7]) if m.sum() >= MIN_GROUP else global_q
        sel = key_te == k
        lo[sel], hi[sel] = q
    return lo, hi


def classify(p, lo, hi):
    return np.where(p <= lo, 0, np.where(p <= hi, 1, 2))


def margin_of(p, lo, hi):
    """Distance in score points to the nearest class cut."""
    return np.minimum(np.abs(p - lo), np.abs(p - hi))


def forward_rank(items, Xm, Y):
    chosen = []
    ss_tot = ((Y - Y.mean()) ** 2).sum() or 1.0
    while len(chosen) < len(items):
        best, best_j = -np.inf, None
        for j in range(len(items)):
            if j in chosen:
                continue
            A = np.c_[np.ones(len(Y)), Xm[:, chosen + [j]]]
            beta = np.linalg.lstsq(A, Y, rcond=None)[0]
            r2 = 1 - ((Y - A @ beta) ** 2).sum() / ss_tot
            if r2 > best:
                best, best_j = r2, j
        chosen.append(best_j)
    return [items[j] for j in chosen]


def fold_setup(ctx, tr, te):
    flag = infer_traffic_rule(ctx, tr)
    known = ctx["base_known"] + flag * ctx["traffic_sum"]
    remainder = ctx["total"] - known
    rank = {}
    for m, items in ITEMS.items():
        rows = tr[ctx["mode"][tr] == m]
        cols = [ALL_ITEMS.index(c) for c in items]
        rank[m] = forward_rank(items, ctx["V"][rows][:, cols], remainder[rows]) if len(rows) else list(items)
    lo, hi = group_thresholds(ctx["key"][te], ctx["total"][tr], ctx["key"][tr])
    return {"flag": flag, "known": known, "rank": rank, "lo": lo, "hi": hi}


# ----------------------------------------------------------------------------
# Models (unchanged from v3)
# ----------------------------------------------------------------------------
def make_pipeline(num_cols, name, seed):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if name == "ridge":
        steps.append(("sc", StandardScaler()))
        model = RidgeCV(alphas=np.logspace(-2, 3, 12))
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed
        )
    pre = ColumnTransformer(
        [
            ("num", Pipeline(steps), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def assemble_fold(ctx, tr, te, k, seed, fs):
    obs = np.zeros_like(ctx["V"])
    for m in ITEMS:
        rows = np.where(ctx["mode"] == m)[0]
        for it in fs["rank"][m][:k]:
            j = ALL_ITEMS.index(it)
            obs[rows, j] = ctx["V"][rows, j]
    osum = obs.sum(axis=1)
    target = ctx["total"] - fs["known"] - osum

    F = ctx["X"][BASE_NUM + CAT].copy()
    F["traffic_rule"] = fs["flag"]
    num_cols = BASE_NUM + ["traffic_rule"]
    if k > 0:
        for j, it in enumerate(ALL_ITEMS):
            F[f"obs_{it}"] = obs[:, j]
        num_cols = num_cols + [f"obs_{it}" for it in ALL_ITEMS]
    preds = [
        make_pipeline(num_cols, name, seed).fit(F.iloc[tr], target[tr]).predict(F.iloc[te])
        for name in ("gbr", "ridge")
    ]
    return fs["known"][te] + osum[te] + np.mean(preds, axis=0)


def run_k(ctx, k, seed, setups):
    pred = np.zeros(len(ctx["y"]))
    for (tr, te), fs in zip(ctx["splits"], setups):
        pred[te] = assemble_fold(ctx, tr, te, k, seed, fs)
    return pred


def run_legacy(ctx, seed):
    X, total = ctx["X"], ctx["total"]
    cols = GEO + ENV + DER
    pred = np.zeros(len(total))
    for tr, te in ctx["splits"]:
        pred[te] = make_pipeline(cols, "gbr", seed).fit(X.iloc[tr], total[tr]).predict(X.iloc[te])
    return pred


# ----------------------------------------------------------------------------
# NEW: inner leave-one-route-out inside one outer training set
# ----------------------------------------------------------------------------
def inner_oof(ctx, tr_outer):
    """Out-of-fold predictions for every outer-training slope, from models that never saw
    its route nor the outer held-out route. Flag, item ranking and cut points are refit
    inside each inner fold, exactly as in the outer loop."""
    n = len(ctx["y"])
    pred = {k: np.full(n, np.nan) for k in K_INNER}
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    for a, b in LeaveOneGroupOut().split(tr_outer, groups=ctx["groups"][tr_outer]):
        itr, ite = tr_outer[a], tr_outer[b]
        fs = fold_setup(ctx, itr, ite)
        lo[ite], hi[ite] = fs["lo"], fs["hi"]
        for k in K_INNER:
            pred[k][ite] = np.mean([assemble_fold(ctx, itr, ite, k, s, fs) for s in INNER_SEEDS], axis=0)
    return {"pred": {k: pred[k][tr_outer] for k in K_INNER}, "lo": lo[tr_outer], "hi": hi[tr_outer]}


def cutoff_for_share(m_in, f):
    if f <= 0:
        return -np.inf
    if f >= 1:
        return np.inf
    return float(np.quantile(m_in, f))


def choose_triage(inner, y_tr, kt):
    """Cutoffs from inner OOF margins: one per budget, plus the target rule."""
    m_in = margin_of(inner["pred"][0], inner["lo"], inner["hi"])
    c0 = classify(inner["pred"][0], inner["lo"], inner["hi"])
    ck = classify(inner["pred"][kt], inner["lo"], inner["hi"])
    out = {f"budget_{f:.1f}": cutoff_for_share(m_in, f) for f in TRIAGE_BUDGETS}
    chosen_f, reached, inner_acc = 1.0, False, np.nan
    for f in TARGET_GRID:
        c = cutoff_for_share(m_in, f)
        final = np.where(m_in < c, ck, c0)
        acc = float(np.mean(final == y_tr))
        inner_acc = acc  # if the target is never reached this ends as the all-assessed accuracy
        if acc >= TARGET_ACC:
            chosen_f, reached = float(f), True
            break
    out[f"target_{TARGET_ACC:.2f}"] = cutoff_for_share(m_in, chosen_f)
    return out, {"share": chosen_f, "reached": reached, "inner_acc": inner_acc}


def fit_platt(inner, y_tr, k):
    m = (inner["pred"][k] - inner["hi"]) / MARGIN_SCALE
    return LogisticRegression(C=1e6, max_iter=5000).fit(m.reshape(-1, 1), (y_tr == 2).astype(int))


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def macro_f1_fast(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0)))


def score(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "qwk": cohen_kappa_score(y, p, weights="quadratic"),
        "high_recall": recall_score(y == 2, p == 2, zero_division=0),
    }


def boot_indices(groups, seed):
    rng = np.random.default_rng(seed)
    routes = np.unique(groups)
    index = {r: np.where(groups == r)[0] for r in routes}
    for _ in range(N_BOOT):
        yield np.concatenate([index[r] for r in rng.choice(routes, size=len(routes), replace=True)])


def route_bootstrap(y, p, groups, seed=42):
    accs, f1s = [], []
    for sel in boot_indices(groups, seed):
        accs.append(float(np.mean(y[sel] == p[sel])))
        f1s.append(macro_f1_fast(y[sel], p[sel]))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])


def paired_bootstrap(y, pa, pb, groups, seed=7):
    """Accuracy and macro-F1 difference A - B, same route resample for both."""
    da, df1 = [], []
    for sel in boot_indices(groups, seed):
        da.append(np.mean(y[sel] == pa[sel]) - np.mean(y[sel] == pb[sel]))
        df1.append(macro_f1_fast(y[sel], pa[sel]) - macro_f1_fast(y[sel], pb[sel]))
    da, df1 = np.array(da), np.array(df1)
    p_two = lambda d: float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
    per_route = pd.DataFrame({"r": groups, "a": y == pa, "b": y == pb}).groupby("r")[["a", "b"]].mean()
    return {
        "acc_diff": float(np.mean(y == pa) - np.mean(y == pb)),
        "acc_95CI": f"{np.percentile(da, 2.5):+.3f} to {np.percentile(da, 97.5):+.3f}",
        "p_acc": round(p_two(da), 4),
        "f1_diff": macro_f1_fast(y, pa) - macro_f1_fast(y, pb),
        "f1_95CI": f"{np.percentile(df1, 2.5):+.3f} to {np.percentile(df1, 97.5):+.3f}",
        "p_f1": round(p_two(df1), 4),
        "routes_A_better/tie/worse": f"{(per_route.a > per_route.b).sum()}/"
        f"{(per_route.a == per_route.b).sum()}/{(per_route.a < per_route.b).sum()}",
    }


def ece(p, yb, bins=ECE_BINS):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - yb[m].mean())
    return float(tot)


def icc(values, groups):
    df = pd.DataFrame({"v": values, "g": groups})
    g = df.groupby("g")["v"]
    n, k, N = g.size(), g.ngroups, len(df)
    msb = (((g.mean() - df["v"].mean()) ** 2) * n).sum() / (k - 1)
    msw = ((df["v"] - g.transform("mean")) ** 2).sum() / (N - k)
    n0 = (N - (n**2).sum() / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def items_count(ctx, k):
    cnt = {m: min(k, len(v)) for m, v in ITEMS.items()}
    return np.array([cnt[m] for m in ctx["mode"]], float)


def items_per_slope(ctx, k):
    return float(items_count(ctx, k).mean())


# ----------------------------------------------------------------------------
# NEW: audit of totals that do not follow the rater's convention
# ----------------------------------------------------------------------------
def audit_totals(ctx):
    X, total, raw, names = ctx["X"], ctx["total"], ctx["raw"], ctx["names"]
    flag = infer_traffic_rule(ctx, np.arange(len(total)))
    formula = X["formula"].to_numpy()
    offset = total - formula - flag * ctx["traffic_sum"]

    print("\n=== Audit: recorded total minus the rater's convention ===")
    for r in np.unique(ctx["raters"]):
        m = ctx["raters"] == r
        conv = "formula + traffic" if flag[m][0] else "formula"
        vc = pd.Series(offset[m]).round(1).value_counts().sort_index()
        print(f"{r} ({m.sum()} slopes, convention: {conv}) offsets: " + ", ".join(f"{k:g}: {v}" for k, v in vc.items()))

    bad = np.where(np.abs(offset) > 1e-6)[0]
    if len(bad) == 0:
        print("No offsets.")
        return

    # 1. non-numeric or off-scale cells in the score block
    raw_scores = raw[names[27:55]]
    text_cells, off_scale = [], []
    for i in bad:
        for j, c in enumerate(names[27:55]):
            v = raw_scores.iloc[i, j]
            if pd.isna(v):
                continue
            num = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            if pd.isna(num):
                text_cells.append((X["segment"].iat[i], c, repr(v)))
            elif j < 27 and not np.isclose(SCALE_VALUES, num).any():
                off_scale.append((X["segment"].iat[i], c, num))
    print(f"\nRows with an offset: {len(bad)}. Non-numeric score cells in them: {len(text_cells)}; off-scale values: {len(off_scale)}")
    for t in text_cells[:30]:
        print("  text cell:", t)
    for t in off_scale[:30]:
        print("  off-scale:", t)

    # 2. alternative conventions that would make the row exact
    S = {s: X[s].to_numpy() for s in SECTIONS}
    alts = {
        "all rock sections counted": S["slope_sum"] + S["clim_sum"] + S["sed_sum"] + S["cr_sum"] + S["disc_sum"] + S["blk_sum"],
        "discontinuities left out": formula - np.where(ctx["mode"] != "blk", S["disc_sum"], 0),
        "climate left out": formula - S["clim_sum"],
    }
    for t in TRAF:
        alts[f"traffic without {ctx['names'][int(t[2:])].split('|', 1)[1]}"] = formula + ctx["traffic_sum"] - X[t].fillna(0).to_numpy()
    print("\nAlternative conventions: rows (among the offset rows) they make exact")
    for name, v in alts.items():
        print(f"  {name}: {int((np.abs(total[bad] - v[bad]) < 1e-6).sum())}")

    # 3. one item rescored on the 3/9/27/81 scale
    counted = {
        m: sc(range(27, 36)) + ITEMS[m] + TRAF for m in ITEMS
    }
    single = {}
    rows_out = []
    for i in bad:
        cands = []
        for c in counted[ctx["mode"][i]]:
            v = X[c].iat[i]
            if pd.isna(v):
                continue
            for alt in SCALE_VALUES[1:]:
                if alt != v and np.isclose(alt - v, offset[i]):
                    cands.append(f"{ctx['names'][int(c[2:])].split('|', 1)[1]}: {v:g}->{alt:g}")
                    single[c] = single.get(c, 0) + 1
        pairs = 0
        if not cands:  # two items rescored (e.g. -12 = two items 9->3)
            vals = [X[c].iat[i] for c in counted[ctx["mode"][i]] if not pd.isna(X[c].iat[i])]
            steps = [[a - v for a in SCALE_VALUES[1:] if a != v] for v in vals]
            for u in range(len(vals)):
                for w in range(u + 1, len(vals)):
                    pairs += sum(np.isclose(a + b, offset[i]) for a in steps[u] for b in steps[w])
        rows_out.append(
            {
                "segment": X["segment"].iat[i],
                "route": X["route"].iat[i],
                "rater": X["rater"].iat[i],
                "rock_mode": ctx["mode"][i],
                "total": total[i],
                "convention_sum": total[i] - offset[i],
                "offset": offset[i],
                "single_item_explanations": len(cands),
                "two_item_explanations": int(pairs),
                "candidates": "; ".join(cands),
            }
        )
    out = pd.DataFrame(rows_out)
    print("\nOffset rows by route:", out["route"].value_counts().to_dict())
    print("Offset size by rock mode:")
    print(pd.crosstab(out["rock_mode"], out["offset"]).to_string())
    print(
        f"\nRows explained by exactly one single-item rescore: {(out.single_item_explanations == 1).sum()}; "
        f"by several (ambiguous): {(out.single_item_explanations > 1).sum()}; by none: {(out.single_item_explanations == 0).sum()}"
    )
    print(
        "Of those with no single-item explanation, rows a two-item rescore could explain: "
        f"{((out.single_item_explanations == 0) & (out.two_item_explanations > 0)).sum()}; "
        f"rows nothing on the scale explains: {((out.single_item_explanations == 0) & (out.two_item_explanations == 0)).sum()}"
    )
    print("A rescore that fits is only a candidate. Many items sit at 9, so -6 fits many cells; only the sheet can confirm.")
    print("Items whose rescoring would explain an offset (count of rows):")
    for c, n in sorted(single.items(), key=lambda t: -t[1])[:10]:
        print(f"  {ctx['names'][int(c[2:])].split('|', 1)[1]}: {n}")
    unexplained = out[out.single_item_explanations == 0]
    print("\nRows no single rescore explains (check these cells in Excel):")
    print(unexplained.drop(columns="candidates").to_string(index=False))

    raw_dump = raw.iloc[bad][names[:1] + names[1:2] + names[9:10] + names[22:23] + names[27:55]].copy()
    raw_dump.insert(0, "offset", offset[bad])
    out.to_csv(os.path.join(OUT_DIR, "audit_offsets.csv"), index=False)
    raw_dump.to_csv(os.path.join(OUT_DIR, "audit_offset_rows_raw.csv"), index=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = build_dataset()
    y, total, groups, X = ctx["y"], ctx["total"], ctx["groups"], ctx["X"]
    routes = np.unique(groups)
    ctx["splits"] = list(LeaveOneGroupOut().split(X, y, groups))
    setups = [fold_setup(ctx, tr, te) for tr, te in ctx["splits"]]
    lo, hi = np.zeros(len(y)), np.zeros(len(y))
    for (tr, te), fs in zip(ctx["splits"], setups):
        lo[te], hi[te] = fs["lo"], fs["hi"]

    print(f"Slopes: {len(y)}  Routes: {len(routes)}  Seeds: {SEEDS}  Inner seeds: {INNER_SEEDS}")
    print("Class counts:", dict(zip(CLASS_NAMES, np.bincount(y, minlength=3).tolist())))
    base = score(y, np.full(len(y), int(np.bincount(y).argmax())))
    print("Majority baseline:", {k: round(v, 3) for k, v in base.items()})
    print(
        f"Distinct score profiles: {int(round(ctx['dup_w'].sum()))} of {len(y)} slopes; "
        f"profiles shared across routes: {ctx['profiles_across_routes']}"
    )

    print("\n=== Label convention by rater (exact matches) ===")
    rows = []
    for r in np.unique(ctx["raters"]):
        m = ctx["raters" ] == r
        rows.append(
            {
                "rater": r,
                "slopes": int(m.sum()),
                "total == hazard formula": int((np.abs(total[m] - X["formula"].to_numpy()[m]) < 1e-6).sum()),
                "total == formula + traffic": int(
                    (np.abs(total[m] - X["formula"].to_numpy()[m] - ctx["traffic_sum"][m]) < 1e-6).sum()
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    flag_all = infer_traffic_rule(ctx, np.arange(len(y)))
    known_all = ctx["base_known"] + flag_all * ctx["traffic_sum"]
    unknown = total - known_all
    print("\n=== Where the variance is ===")
    print(
        f"Var(total) {total.var():.0f} | known exactly {known_all.var():.0f} "
        f"({known_all.var() / total.var():.0%}) | unknown {unknown.var():.0f} ({unknown.var() / total.var():.0%})"
    )
    print(f"ICC of unknown part: route {icc(unknown, groups):.2f}   region {icc(unknown, X['region'].to_numpy()):.2f}")

    # --- outer runs (as v3) -------------------------------------------------
    jobs = [(k, s) for k in K_GRID for s in SEEDS]
    results = Parallel(n_jobs=N_JOBS)(delayed(run_k)(ctx, k, s, setups) for k, s in jobs)
    legacy = Parallel(n_jobs=N_JOBS)(delayed(run_legacy)(ctx, s) for s in SEEDS)
    pred_seed = {(k, s): p for (k, s), p in zip(jobs, results)}
    ens = {k: np.mean([pred_seed[(k, s)] for s in SEEDS], axis=0) for k in K_GRID}
    cls = {k: classify(ens[k], lo, hi) for k in K_GRID}
    cls_legacy = classify(np.mean(legacy, axis=0), lo, hi)

    print("\n=== Previous best tier (direct GBR on geometry + expanded + derived) ===")
    leg = pd.DataFrame([score(y, classify(p, lo, hi)) for p in legacy])
    print(leg.agg(["mean", "std"]).round(3).to_string())

    print("\n=== Accuracy vs. geologic items assessed per slope (leave-one-route-out) ===")
    rows = []
    for k in K_GRID:
        per_seed = pd.DataFrame([score(y, classify(pred_seed[(k, s)], lo, hi)) for s in SEEDS])
        acc_ci, f1_ci = route_bootstrap(y, cls[k], groups)
        row = {"k_items_per_mode": k, "items_per_slope": round(items_per_slope(ctx, k), 2)}
        for c in per_seed.columns:
            row[c] = f"{per_seed[c].mean():.3f} +/- {per_seed[c].std(ddof=1) if len(SEEDS) > 1 else 0:.3f}"
        row["acc_dedup"] = round(float(np.average(cls[k] == y, weights=ctx["dup_w"])), 3)
        row["acc_95CI"] = f"{acc_ci[0]:.3f}-{acc_ci[1]:.3f}"
        row["f1_95CI"] = f"{f1_ci[0]:.3f}-{f1_ci[1]:.3f}"
        rows.append(row)
    voi = pd.DataFrame(rows)
    print(voi.to_string(index=False))
    voi.to_csv(os.path.join(OUT_DIR, "voi_summary.csv"), index=False)

    print("\n=== Items chosen by forward selection, folds in which each ranks in the top 3 ===")
    for m, items in ITEMS.items():
        cnt = pd.Series([it for fs in setups for it in fs["rank"][m][:3]]).value_counts()
        mode_name = {"cr": "Crystalline", "sed": "Sedimentary", "blk": "Block in matrix"}[m]
        print(f"{mode_name}: " + ", ".join(f"{ctx['item_names'][c]} ({n}/{len(setups)})" for c, n in cnt.items()))

    route_table = pd.DataFrame({"n_slopes": pd.Series(groups).value_counts()})
    for k in K_REPORT:
        cm = np.bincount(y * 3 + cls[k], minlength=9).reshape(3, 3)
        print(f"\n=== Detail, k = {k} === QWK {cohen_kappa_score(y, cls[k], weights='quadratic'):.3f}")
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))
        pr = pd.DataFrame({"route": groups, "ok": (y == cls[k]).astype(float)}).groupby("route")["ok"].mean()
        route_table[f"k={k}"] = pr.round(3)
        print(f"Per-route: mean {pr.mean():.3f} sd {pr.std():.3f} min {pr.min():.3f}  routes >= 0.70: {(pr >= 0.70).sum()}/{len(pr)}")
    route_table.to_csv(os.path.join(OUT_DIR, "per_route.csv"))

    # --- NEW: inner out-of-fold predictions per outer fold -----------------
    print(f"\nRunning inner leave-one-route-out for k in {K_INNER} ({len(ctx['splits'])} outer folds)...", flush=True)
    inners = Parallel(n_jobs=N_JOBS)(delayed(inner_oof)(ctx, tr) for tr, te in ctx["splits"])

    # =========================================================================
    # TASK 1: validated triage
    # =========================================================================
    print("\n=== Task 1. Validated triage (cutoff chosen on inner out-of-fold margins, training routes only) ===")
    print("Stage 1 = k=0 model. A held-out slope goes to stage 2 if its margin is below the training cutoff.")
    print("budget_f: cutoff = f-quantile of inner margins. target: smallest inner share reaching "
          f"{TARGET_ACC:.2f} inner OOF accuracy (grid step 0.05).")
    tri_rows, target_log, tri_final = [], [], {}
    for kt in K_TRIAGE:
        cut = {}  # rule -> per-slope cutoff (held-out slope gets its own fold's cutoff)
        for (tr, te), inner in zip(ctx["splits"], inners):
            rules, info = choose_triage(inner, y[tr], kt)
            for r, c in rules.items():
                cut.setdefault(r, np.zeros(len(y)))[te] = c
            target_log.append({"stage2_k": kt, "held_out_route": groups[te][0], **info})
        n_items = items_count(ctx, kt)
        for rule, c in cut.items():
            per_seed = []
            for s in SEEDS:
                m_s = margin_of(pred_seed[(0, s)], lo, hi)
                f_s = np.where(m_s < c, classify(pred_seed[(kt, s)], lo, hi), classify(pred_seed[(0, s)], lo, hi))
                per_seed.append(score(y, f_s))
            per_seed = pd.DataFrame(per_seed)
            assess = margin_of(ens[0], lo, hi) < c
            final = np.where(assess, cls[kt], cls[0])
            tri_final[(kt, rule)] = final
            acc_ci, _ = route_bootstrap(y, final, groups)
            tri_rows.append(
                {
                    "stage2_k": kt,
                    "rule": rule,
                    "share_assessed": round(assess.mean(), 3),
                    "items_per_slope": round(float((assess * n_items).mean()), 2),
                    "accuracy": f"{per_seed.accuracy.mean():.3f} +/- {per_seed.accuracy.std(ddof=1) if len(SEEDS) > 1 else 0:.3f}",
                    "macro_f1": round(per_seed.macro_f1.mean(), 3),
                    "high_recall": round(per_seed.high_recall.mean(), 3),
                    "acc_ens": round(accuracy_score(y, final), 3),
                    "acc_dedup": round(float(np.average(final == y, weights=ctx["dup_w"])), 3),
                    "acc_95CI": f"{acc_ci[0]:.3f}-{acc_ci[1]:.3f}",
                    "routes>=0.70": int((pd.Series(final == y).groupby(groups).mean() >= 0.70).sum()),
                }
            )
    tri = pd.DataFrame(tri_rows)
    print(tri.to_string(index=False))
    tri.to_csv(os.path.join(OUT_DIR, "triage_validated.csv"), index=False)
    tl = pd.DataFrame(target_log)
    print("\nTarget rule, share chosen on each fold's training routes:")
    print(tl.groupby("stage2_k").agg(share_mean=("share", "mean"), share_min=("share", "min"),
                                     share_max=("share", "max"), folds_reached=("reached", "sum")).round(3).to_string())
    tl.to_csv(os.path.join(OUT_DIR, "triage_target_by_fold.csv"), index=False)

    # =========================================================================
    # TASK 2: calibrated P(High)
    # =========================================================================
    print("\n=== Task 2. Calibrated P(High): Platt on (predicted total - group 70th pct), fit on inner OOF ===")
    cal_rows, alert_rows, rel_rows = [], [], []
    for k in K_CALIB:
        p = np.zeros(len(y))
        p_clim = np.zeros(len(y))
        for (tr, te), inner in zip(ctx["splits"], inners):
            lr = fit_platt(inner, y[tr], k)
            p[te] = lr.predict_proba(((ens[k][te] - hi[te]) / MARGIN_SCALE).reshape(-1, 1))[:, 1]
            p_clim[te] = np.mean(y[tr] == 2)
        yb = (y == 2).astype(int)
        b, bc = brier_score_loss(yb, p), brier_score_loss(yb, p_clim)
        boots = [(brier_score_loss(yb[s], p[s]), roc_auc_score(yb[s], p[s]) if 0 < yb[s].sum() < len(s) else np.nan)
                 for s in boot_indices(groups, 11)]
        boots = np.array(boots)
        cal_rows.append(
            {
                "k": k,
                "brier": round(b, 4),
                "brier_95CI": f"{np.nanpercentile(boots[:, 0], 2.5):.3f}-{np.nanpercentile(boots[:, 0], 97.5):.3f}",
                "brier_baseline(prevalence)": round(bc, 4),
                "brier_skill": round(1 - b / bc, 3),
                "auc": round(roc_auc_score(yb, p), 3),
                "auc_95CI": f"{np.nanpercentile(boots[:, 1], 2.5):.3f}-{np.nanpercentile(boots[:, 1], 97.5):.3f}",
                "ece_10bin": round(ece(p, yb), 4),
                "mean_p": round(p.mean(), 3),
                "observed_rate": round(yb.mean(), 3),
            }
        )
        for t in ALERT_THRESHOLDS:
            a = p >= t
            alert_rows.append(
                {
                    "k": k,
                    "p_threshold": t,
                    "alerts": int(a.sum()),
                    "alert_rate": round(a.mean(), 3),
                    "precision": round(yb[a].mean(), 3) if a.any() else np.nan,
                    "recall": round(a[yb == 1].mean(), 3),
                }
            )
        edges = np.linspace(0, 1, ECE_BINS + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, ECE_BINS - 1)
        for bi in range(ECE_BINS):
            m = idx == bi
            if m.any():
                rel_rows.append({"k": k, "bin": f"{edges[bi]:.1f}-{edges[bi + 1]:.1f}", "n": int(m.sum()),
                                 "mean_p": round(p[m].mean(), 3), "observed": round(yb[m].mean(), 3)})
    print(pd.DataFrame(cal_rows).to_string(index=False))
    print("\nAlert table (thresholds fixed in advance, not tuned):")
    print(pd.DataFrame(alert_rows).to_string(index=False))
    print("\nReliability bins:")
    print(pd.DataFrame(rel_rows).to_string(index=False))
    pd.DataFrame(cal_rows).to_csv(os.path.join(OUT_DIR, "calibration_summary.csv"), index=False)
    pd.DataFrame(alert_rows).to_csv(os.path.join(OUT_DIR, "alert_table.csv"), index=False)
    pd.DataFrame(rel_rows).to_csv(os.path.join(OUT_DIR, "reliability_bins.csv"), index=False)

    # =========================================================================
    # TASK 3: paired route-bootstrap tests
    # =========================================================================
    print("\n=== Task 3. Paired route bootstrap (A - B, seed-ensembled classes, 2000 route resamples) ===")
    tgt = f"target_{TARGET_ACC:.2f}"
    pairs = [
        ("k=2", cls[2], "k=0", cls[0]),
        ("k=2", cls[2], "previous tier", cls_legacy),
        ("k=2", cls[2], "k=3", cls[3]),
        ("triage k=2 target", tri_final[(2, tgt)], "k=0", cls[0]),
        ("triage k=2 target", tri_final[(2, tgt)], "k=2 all slopes", cls[2]),
    ]
    pt = pd.DataFrame([{"A": a, "B": b, **paired_bootstrap(y, pa, pb, groups)} for a, pa, b, pb in pairs])
    print(pt.round(3).to_string(index=False))
    pt.to_csv(os.path.join(OUT_DIR, "paired_tests.csv"), index=False)

    # =========================================================================
    # TASK 4: audit
    # =========================================================================
    audit_totals(ctx)

    pd.DataFrame(
        {"route": groups, "y_true": y, "lo": lo, "hi": hi,
         **{f"class_k{k}": cls[k] for k in K_GRID}, **{f"pred_k{k}": ens[k] for k in K_GRID}}
    ).to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)
    print(f"\nWrote {OUT_DIR}/")


if __name__ == "__main__":
    main()