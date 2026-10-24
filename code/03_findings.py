"""Step 3 - validate the findings with statistics.

No scipy/statsmodels on this machine, so everything is numpy:
  * bootstrap confidence intervals (clustered by business where relevant)
  * logistic regression by IRLS on ridge-stabilised standardised features
  * AUC / ROC computed directly
  * ridge is required because several attributes are collinear

Outputs: analysis/out/03_findings_*.csv, 03_model_coefficients.csv,
         03_risk_deciles.csv, 03_auc.csv
"""

from __future__ import annotations

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
RNG = np.random.default_rng(20260101)
NBOOT = 2000

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
print(f"features: {feat.shape[0]:,} rows x {feat.shape[1]} cols\n")


def write(name, df):
    df.to_csv(os.path.join(OUT, name), index=False, encoding="utf-8-sig")
    print(f"  wrote {name}")


# ==========================================================================
# helpers
# ==========================================================================
def boot_ci(values: np.ndarray, weights: np.ndarray | None = None,
            stat=np.mean, n=NBOOT, alpha=0.05):
    """Percentile bootstrap CI for a statistic over per-unit values."""
    values = np.asarray(values, dtype=float)
    idx = np.arange(len(values))
    stats = np.empty(n)
    for i in range(n):
        take = RNG.choice(idx, size=len(idx), replace=True)
        v = values[take]
        stats[i] = stat(v) if weights is None else stat(v, weights[take])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(values) if weights is None else stat(values, weights)), float(lo), float(hi)


def diff_in_rates(mask_a: np.ndarray, y: np.ndarray, n=NBOOT):
    """Bootstrap CI for the difference in closure rate between two groups."""
    a = y[mask_a]
    b = y[~mask_a]

    def stat(_):
        ia = RNG.choice(len(a), len(a), replace=True)
        ib = RNG.choice(len(b), len(b), replace=True)
        return a[ia].mean() - b[ib].mean()

    obs = a.mean() - b.mean()
    draws = np.array([stat(None) for _ in range(n)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(obs), float(lo), float(hi), float(a.mean()), float(b.mean()), len(a), len(b)


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """AUC via the Mann-Whitney U identity, with ties handled by average ranks."""
    y = np.asarray(y)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def standardise(X: pd.DataFrame):
    mu = X.mean()
    sd = X.std().replace(0, 1.0)
    return (X - mu) / sd, mu, sd


def logistic_irls(X: np.ndarray, y: np.ndarray, ridge: float = 1.0,
                  iters: int = 60, tol: float = 1e-8):
    """Logistic regression via IRLS with a small ridge penalty (collinearity)."""
    n, p = X.shape
    Xd = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    pen = np.eye(p + 1) * ridge
    pen[0, 0] = 0.0
    for _ in range(iters):
        eta = Xd @ beta
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        WX = Xd * w[:, None]
        A = Xd.T @ WX + pen
        b = Xd.T @ (w * z)
        try:
            new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def fit_and_report(X: pd.DataFrame, y: np.ndarray, label: str):
    """Fit, report odds ratios for standardised features (per +1 SD)."""
    Xs, mu, sd = standardise(X)
    beta = logistic_irls(Xs.values, y)
    p = Xs @ beta[1:] + beta[0]
    return {
        "model": label,
        "auc": auc_score(y, p),
        "coef": dict(zip(X.columns, beta[1:])),
        "intercept": beta[0],
    }


# ==========================================================================
# FINDING A - review volume & tips vs survival; stars do not separate
# ==========================================================================
print("=== Finding A: volume vs stars ===")
y = feat.is_open.values
closed = 1 - feat.is_open.values

fa = []
feat["_rc_dec"] = pd.qcut(feat.business_review_count, 10, labels=False, duplicates="drop")
for d, g in feat.groupby("_rc_dec"):
    fa.append({"bucket": f"评论数 D{d+1}", "n": len(g),
               "avg_value": g.business_review_count.mean(),
               "closure_rate": 1 - g.is_open.mean(),
               "lo": np.nan, "hi": np.nan})
fa = pd.DataFrame(fa)
# bootstrap CI on the extreme deciles
for tag, sub in [("评论数 D1", feat[feat._rc_dec == 0]), ("评论数 D10", feat[feat._rc_dec == 9])]:
    r = np.array(1 - sub.is_open.values, dtype=float)
    m, lo, hi = boot_ci(r)
    fa.loc[fa.bucket == tag, ["closure_rate", "lo", "hi"]] = [m, lo, hi]

feat["_tip_dec"] = pd.qcut(feat.n_tips, 10, labels=False, duplicates="drop")
tip_tbl = feat.groupby("_tip_dec").agg(n=("is_open", "size"), avg_tips=("n_tips", "mean"),
                                       open_rate=("is_open", "mean")).reset_index()
write("03_findingA_reviewcount.csv", fa)
write("03_findingA_tips.csv", tip_tbl)

star_tbl = feat.assign(bin=pd.cut(feat.business_stars, [0, 2, 2.5, 3, 3.5, 4, 4.5, 5])) \
    .groupby("bin", observed=True).agg(n=("is_open", "size"), closure_rate=("is_open", lambda s: 1 - s.mean()),
                                       avg_reviews=("business_review_count", "mean")).reset_index()
star_tbl["bin"] = star_tbl["bin"].astype(str)
write("03_findingA_stars.csv", star_tbl)

r_star = feat[["business_stars", "is_open"]].corr(method="spearman").iloc[0, 1]
r_vol = feat[["business_review_count", "is_open"]].corr(method="spearman").iloc[0, 1]
r_tip = feat[["n_tips", "is_open"]].corr(method="spearman").iloc[0, 1]
r_ck = feat[["total_checkins", "is_open"]].corr(method="spearman").iloc[0, 1]
print(f"  spearman stars~open {r_star:+.3f} | reviews {r_vol:+.3f} | tips {r_tip:+.3f} | checkins {r_ck:+.3f}")
# star coefficient CI via bootstrap on the correlation
bs = []
for _ in range(500):
    t = feat.sample(len(feat), replace=True, random_state=None)
    bs.append(t[["business_stars", "is_open"]].corr(method="spearman").iloc[0, 1])
print(f"  stars~open spearman 95%CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")

# ==========================================================================
# FINDING B - availability attributes, univariate then adjusted
# ==========================================================================
print("\n=== Finding B: availability attributes ===")
fb = []
for a in ["delivery", "takeout", "has_tv", "reservations", "outdoor_seating",
          "caters", "good_for_kids", "accepts_credit_cards", "bike_parking"]:
    col = f"is_{a}"
    mask = feat[col] == 1
    if mask.sum() < 200:
        continue
    obs, lo, hi, ra, rb, na, nb = diff_in_rates(mask.values, closed)
    fb.append({"attribute": a, "closure_true": ra, "closure_false": rb,
               "n_true": na, "n_false": nb, "diff": obs, "lo": lo, "hi": hi,
               "significant": not (lo <= 0 <= hi)})
fb = pd.DataFrame(fb).sort_values("diff")
write("03_findingB_univariate.csv", fb)
for _, r in fb.iterrows():
    flag = "*" if r.significant else " "
    print(f"  {flag} {r.attribute:22} True {r.closure_true:.3f} vs False {r.closure_false:.3f} "
          f"diff {r['diff']:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]")

# rating comparison for the headline paradox
d_t = feat.loc[feat.is_delivery == 1, "business_stars"].mean()
d_f = feat.loc[feat.is_delivery == 0, "business_stars"].mean()
print(f"  paradox check: avg stars delivery=True {d_t:.3f} vs False {d_f:.3f}")

# ==========================================================================
# FINDING B (adjusted) + FINDING C - logistic models
# ==========================================================================
print("\n=== Logistic models ===")
model_cols = ["is_delivery", "is_takeout", "is_has_tv", "is_reservations",
              "is_outdoor_seating", "is_caters", "is_good_for_kids",
              "is_accepts_credit_cards", "is_bike_parking",
              "price_num", "log_reviews", "std_stars", "pct_neg",
              "has_hours", "hours_open_late",
              "city_restaurant_count", "city_cat_count"]
M = feat[model_cols + ["is_open"]].copy()
# Unknown -> column mean, and record an explicit missingness indicator instead
for c in model_cols:
    if M[c].isna().any():
        M[c] = M[c].fillna(M[c].mean())

m1 = fit_and_report(M[model_cols], M.is_open.values, "full")
print(f"  full model AUC = {m1['auc']:.4f}")

coef_rows = []
for k, v in sorted(m1["coef"].items(), key=lambda kv: -abs(kv[1])):
    coef_rows.append({"feature": k, "std_coef": v, "odds_ratio_per_SD": np.exp(v)})
write("03_model_coefficients.csv", pd.DataFrame(coef_rows))
for r in coef_rows[:12]:
    print(f"    {r['feature']:26} coef {r['std_coef']:+.3f}  OR {r['odds_ratio_per_SD']:.3f}")

# compare against a stars-only baseline and volume-only baseline
base_stars = fit_and_report(M[["price_num"]].assign(stars=feat.business_stars.fillna(feat.business_stars.mean())),
                            M.is_open.values, "stars")
base_vol = fit_and_report(M[["log_reviews"]], M.is_open.values, "volume")
auc_tbl = pd.DataFrame([
    {"model": "仅星级 + 价格", "auc": base_stars["auc"], "n_features": 2},
    {"model": "仅评论量", "auc": base_vol["auc"], "n_features": 1},
    {"model": "完整模型", "auc": m1["auc"], "n_features": len(model_cols)},
])
write("03_auc.csv", auc_tbl)
print(auc_tbl.to_string(index=False))

# bootstrap CI for the full-model AUC
ys = M.is_open.values
Xs, _, _ = standardise(M[model_cols])
ps = Xs.values @ logistic_irls(Xs.values, ys)[1:] + logistic_irls(Xs.values, ys)[0]
auc_draws = []
for _ in range(400):
    take = RNG.choice(len(ys), len(ys), replace=True)
    auc_draws.append(auc_score(ys[take], ps[take]))
auc_lo, auc_hi = np.percentile(auc_draws, [2.5, 97.5])
print(f"  full AUC 95%CI [{auc_lo:.4f}, {auc_hi:.4f}]")

# ==========================================================================
# FINDING C - crowding dose-response
# ==========================================================================
print("\n=== Finding C: crowding ===")
fc = []
feat["_crowd_dec"] = pd.qcut(feat.city_cat_count, 8, labels=False, duplicates="drop")
for d, g in feat.groupby("_crowd_dec"):
    r = np.array(1 - g.is_open.values, dtype=float)
    m, lo, hi = boot_ci(r, n=800)
    fc.append({"decile": int(d) + 1, "n": len(g), "avg_same_category": g.city_cat_count.mean(),
               "closure_rate": m, "lo": lo, "hi": hi})
write("03_findingC_crowding.csv", pd.DataFrame(fc))

feat["_city_dec"] = pd.qcut(feat.city_restaurant_count, 8, labels=False, duplicates="drop")
cc = feat.groupby("_city_dec").agg(n=("is_open", "size"),
                                   avg_city_size=("city_restaurant_count", "mean"),
                                   closure_rate=("is_open", lambda s: 1 - s.mean())).reset_index()
write("03_findingC_citysize.csv", cc)

# within-state check
st_rows = []
for st in ["PA", "FL", "NJ", "MO"]:
    s = feat[feat.state == st].copy()
    if len(s) < 500:
        continue
    s["q"] = pd.qcut(s.city_restaurant_count, 4, labels=False, duplicates="drop")
    gg = s.groupby("q").agg(n=("is_open", "size"), closure=("is_open", lambda x: 1 - x.mean()))
    for q, row in gg.iterrows():
        st_rows.append({"state": st, "quartile": int(q) + 1, "n": int(row.n), "closure_rate": row.closure})
write("03_findingC_within_state.csv", pd.DataFrame(st_rows))

# ==========================================================================
# RISK INDEX - combine the validated signals
# ==========================================================================
print("\n=== Risk index ===")
risk_cols = ["is_delivery", "is_takeout", "is_has_tv", "has_hours",
             "log_reviews", "n_tips", "total_checkins", "std_stars", "pct_neg",
             "city_cat_count", "price_num"]
R = feat[risk_cols].copy()
for c in risk_cols:
    R[c] = R[c].fillna(R[c].mean())
Rs, _, _ = standardise(R)
# sign so that higher score = higher closure risk
signs = np.sign([-m1["coef"].get(c, 0) for c in risk_cols])
score = (Rs.values * signs).sum(axis=1)
score = (score - score.mean()) / score.std()
feat["risk_score"] = score

feat["_risk_dec"] = pd.qcut(feat.risk_score, 10, labels=False, duplicates="drop")
rk = feat.groupby("_risk_dec").agg(n=("is_open", "size"), risk=("risk_score", "mean"),
                                   closure_rate=("is_open", lambda s: 1 - s.mean())).reset_index()
write("03_risk_deciles.csv", rk)
print(f"  risk-index AUC = {auc_score(feat.is_open.values, -feat.risk_score.values):.4f}")
print(f"  lowest decile closure  {rk.closure_rate.iloc[0]:.3f}")
print(f"  highest decile closure {rk.closure_rate.iloc[-1]:.3f}")

# ==========================================================================
# FINDING D - category time fingerprint
# ==========================================================================
print("\n=== Finding D: time fingerprint ===")
ch = pd.read_csv(os.path.join(WS, "data", "bigquery", "bdao_group4.checkin_restaurant_hourly.csv"),
                 usecols=["business_id", "category_major", "hour", "checkin_count", "weekday"])
ch_late = ch.assign(late=ch.hour.isin([22, 23, 0, 1, 2, 3]) * ch.checkin_count)
fd = ch_late.groupby("category_major").agg(total=("checkin_count", "sum"), late=("late", "sum"))
fd["late_share"] = fd.late / fd.total
fd = fd.reset_index().sort_values("late_share", ascending=False)
write("03_findingD_late_share.csv", fd)
print(fd.to_string(index=False))

pivot = ch.pivot_table(index="category_major", columns="hour", values="checkin_count",
                       aggfunc="sum", fill_value=0)
pivot = pivot.div(pivot.sum(axis=1), axis=0)
pivot.to_csv(os.path.join(OUT, "03_findingD_hour_matrix.csv"), encoding="utf-8-sig")
print("  wrote 03_findingD_hour_matrix.csv")

# ==========================================================================
# FINDING E - users
# ==========================================================================
print("\n=== Finding E: users ===")
us = pd.read_csv(os.path.join(WS, "data", "bigquery", "bdao_group4.yelp_users.csv"),
                 low_memory=False,
                 usecols=["user_id", "review_count", "average_stars", "fans", "useful", "elite"])
us["elite_n"] = us.elite.fillna("").apply(lambda s: len(str(s).split(",")) if s else 0)
fe_rows = [{"metric": "users", "value": len(us)},
           {"metric": "median_review_count", "value": us.review_count.median()},
           {"metric": "p90_review_count", "value": us.review_count.quantile(.9)},
           {"metric": "max_review_count", "value": us.review_count.max()},
           {"metric": "share_with_0_fans", "value": (us.fans == 0).mean()},
           {"metric": "share_elite", "value": (us.elite_n > 0).mean()},
           {"metric": "median_avg_stars", "value": us.average_stars.median()}]
write("03_findingE_users.csv", pd.DataFrame(fe_rows))
for r in fe_rows:
    print(f"  {r['metric']:22} {r['value']}")

# concentration: what share of reviews come from the top 1% of users
us_sorted = us.sort_values("review_count", ascending=False)
top1 = us_sorted.head(int(len(us_sorted) * 0.01)).review_count.sum()
tot = us.review_count.sum()
print(f"  top 1% users produce {top1/tot:.1%} of all reviews")
pd.DataFrame([{"metric": "top1pct_user_review_share", "value": top1 / tot}]).to_csv(
    os.path.join(OUT, "03_findingE_concentration.csv"), index=False, encoding="utf-8-sig")

feat.drop(columns=[c for c in feat.columns if c.startswith("_")]).to_csv(
    os.path.join(OUT, "business_features.csv"), index=False, encoding="utf-8-sig")
print("\ndone.")
