"""Step 3c - refit the risk model with TRUE spatial density replacing the crude
city-count proxy, and quantify the improvement.

Reports odds ratios per +1 SD and the AUC for each specification so the gain from
real geospatial density is explicit.
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

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)


def standardise(X):
    return (X - X.mean()) / X.std().replace(0, 1.0)


def irls(X, y, ridge=1.0, iters=60):
    n, p = X.shape
    Xd = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    pen = np.eye(p + 1) * ridge
    pen[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(Xd @ beta, -30, 30)
        m = 1 / (1 + np.exp(-eta))
        w = np.clip(m * (1 - m), 1e-9, None)
        z = eta + (y - m) / w
        beta = np.linalg.solve(Xd.T @ (Xd * w[:, None]) + pen, Xd.T @ (w * z))
    return beta


def auc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); sr = s[order]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


BASE = ["is_delivery", "is_takeout", "is_has_tv", "is_reservations",
        "is_outdoor_seating", "is_caters", "is_good_for_kids",
        "is_accepts_credit_cards", "is_bike_parking",
        "price_num", "log_reviews", "std_stars", "pct_neg",
        "has_hours", "hours_open_late"]

SPECS = {
    "A 旧代理：城市店数": BASE + ["city_restaurant_count"],
    "B 旧代理：+ 同品类": BASE + ["city_restaurant_count", "city_cat_count"],
    "C1 真实密度：0.5km": BASE + ["n_within_0_5km"],
    "C2 真实密度：1km": BASE + ["n_within_1_0km"],
    "C3 真实密度：2km": BASE + ["n_within_2_0km"],
    "D 真实密度 + 同品类 1km": BASE + ["n_within_1_0km", "n_same_cat_1km"],
    # three nested scales together are ~0.99 correlated, which makes their
    # individual coefficients unstable (sign flips); kept only to show that effect
    "X 三尺度同时入模（共线，仅作对照）": BASE + ["n_within_0_5km", "n_within_1_0km", "n_within_2_0km"],
}

y = feat.is_open.values
rows = []
details = {}
for name, cols in SPECS.items():
    M = feat[cols].copy()
    for c in cols:
        M[c] = M[c].fillna(M[c].mean())
    Xs = standardise(M)
    beta = irls(Xs.values, y)
    p = 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(y), 1)), Xs.values]) @ beta, -30, 30)))
    a = auc(y, p)
    rows.append({"spec": name, "n_features": len(cols), "auc": a})
    details[name] = dict(zip(cols, beta[1:]))
    print(f"{name:34} AUC {a:.4f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "03_model_density_auc.csv"), index=False, encoding="utf-8-sig")

# bootstrap CI for the best spec vs the proxy spec
def boot_auc(cols, n=300):
    M = feat[cols].copy()
    for c in cols:
        M[c] = M[c].fillna(M[c].mean())
    Xs = standardise(M)
    beta = irls(Xs.values, y)
    p = 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(y), 1)), Xs.values]) @ beta, -30, 30)))
    out = []
    for _ in range(n):
        t = RNG.choice(len(y), len(y), replace=True)
        out.append(auc(y[t], p[t]))
    return np.percentile(out, [2.5, 97.5])


print()
for nm in ["A 旧代理：城市店数", "C2 真实密度：1km"]:
    lo, hi = boot_auc(SPECS[nm], n=200)
    print(f"  {nm:34} AUC 95%CI [{lo:.4f}, {hi:.4f}]")

# coefficient table for the recommended single-scale spec
best = "C2 真实密度：1km"
d = details[best]
co = pd.DataFrame([{"feature": k, "std_coef": v, "odds_ratio_per_SD": float(np.exp(v))}
                   for k, v in sorted(d.items(), key=lambda kv: -abs(kv[1]))])
co.to_csv(os.path.join(OUT, "03_model_density_coefficients.csv"), index=False, encoding="utf-8-sig")
print(f"\n=== {best} coefficients (predicting is_open; OR>1 = higher survival odds) ===")
for _, r in co.iterrows():
    print(f"  {r.feature:26} coef {r.std_coef:+.3f}  OR {r.odds_ratio_per_SD:.3f}")

# pairwise correlation of the three scales, to document the collinearity
cc = feat[["n_within_0_5km", "n_within_1_0km", "n_within_2_0km"]].corr(method="spearman")
cc.to_csv(os.path.join(OUT, "03_density_scale_collinearity.csv"), encoding="utf-8-sig")
print("\n=== 三个尺度的 Spearman 相关（说明为何不能同时入模）===")
print(cc.round(4).to_string())
print(f"\nwrote 03_model_density_auc.csv / 03_model_density_coefficients.csv")
