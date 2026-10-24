"""Discrete-time hazard model of the review series stopping ("going quiet").

DESIGN
    Pooled logistic regression on the person-period data IS the discrete-time hazard
    model: with one row per store-quarter and y = 1 only in the quarter the series
    stops, the fitted probability is the conditional hazard h(t | x). The baseline
    hazard enters as period dummies, so no parametric shape is imposed.

THE QUESTION THIS ANSWERS
    The survival analysis could only say "which stores are open". This asks "when do
    stores go quiet, and does a sliding rating bring it forward?" - the factor the
    earlier report listed as the natural next step but could not test.

DISCIPLINE AGAINST LEAKAGE
    * predictors use only quarters strictly before the current one
    * `is_open` is never an input (it is mechanically tied to the event)
    * full-history features (`business_stars`, `n_reviews`) are excluded: they are
      computed over the whole review history, including the future
    * static attributes (density, price tier, delivery) are snapshot attributes and
      are kept only with that caveat stated

Outputs (analysis/out/):
    08_hazard_coefficients.csv   hazard ratios with bootstrap CI
    08_hazard_fit.md             fit, calibration, validation against is_open
    08_hazard_curves.csv         predicted survival curves by rating-trend tercile
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
RNG = np.random.default_rng(20261024)
NBOOT = int(os.environ.get("NBOOT", "120"))

print("loading person-period data ...")
pp = pd.read_csv(os.path.join(OUT, "08_hazard_person_period.csv.gz"))
print("  rows %s, events %s" % (format(len(pp), ","), format(int(pp.y.sum()), ",")))

# ---------------------------------------------------------- static covariates
feat = pd.read_csv(os.path.join(OUT, "business_features.csv"),
                   usecols=["business_id", "n_within_1_0km", "price_num",
                            "is_delivery"],
                   low_memory=False)
pp = pp.merge(feat, on="business_id", how="left")
pp["log_density"] = np.log1p(pp.n_within_1_0km)
pp["delivery"] = pp.is_delivery.fillna(pp.is_delivery.mean()).astype(float)
pp["price_num"] = pp.price_num.fillna(pp.price_num.mean())

# ------------------------------------------------------------- design matrix
print("building design matrix ...")
BANDS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 32), (33, 48), (49, 10_000)]
band_cols = []
for lo, hi in BANDS[:-1]:                     # last band is the reference level
    c = "band_%d_%d" % (lo, hi)
    pp[c] = ((pp.quarters_elapsed >= lo) & (pp.quarters_elapsed <= hi)).astype(float)
    band_cols.append(c)

pp["has_trend"] = pp.stars_trend.notna().astype(float)
for c in ["rev_lag4", "stars_lag4", "rev_trend", "stars_trend", "log_density",
          "price_num"]:
    pp[c] = pp[c].fillna(pp[c].mean())
pp["log_rev_lag4"] = np.log1p(pp["rev_lag4"])
pp["log_rev_trend"] = np.log(pp.rev_trend.clip(lower=0.25, upper=4.0))

FEATS = band_cols + ["log_rev_lag4", "stars_lag4", "log_rev_trend",
                     "stars_trend", "has_trend", "log_density", "price_num",
                     "delivery"]

X = pp[FEATS].to_numpy(np.float32)
y = pp.y.to_numpy(np.float32)
# standardise non-band features so the horse-race is on a common scale
scale_idx = [FEATS.index(f) for f in FEATS if not f.startswith("band_")]
mu = X[:, scale_idx].mean(axis=0)
sd = X[:, scale_idx].std(axis=0)
sd[sd == 0] = 1.0
Xs = X.copy()
Xs[:, scale_idx] = (Xs[:, scale_idx] - mu) / sd
A = np.hstack([np.ones((len(y), 1), np.float32), Xs]).astype(np.float32)


def irls(Am, yv, ridge=2.0, iters=14):
    beta = np.zeros(Am.shape[1], dtype=np.float64)
    pen = np.eye(Am.shape[1]) * ridge
    pen[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(Am @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        z = eta + (yv - p) / w
        beta = np.linalg.solve((Am.T @ (Am * w[:, None].astype(np.float32))).astype(np.float64) + pen, (Am.T @ (w * z).astype(np.float32)).astype(np.float64))
    return beta


print("fitting discrete-time hazard model ...")
beta = irls(A, y, iters=30)
p = 1.0 / (1.0 + np.exp(-np.clip(A @ beta, -30, 30)))
eps = 1e-12
ll = float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
ll0 = float(np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())))
pseudo_r2 = 1 - ll / ll0
auc_num = np.sum(p[y == 0][None, :] < p[y == 1][:, None]) if False else None
# AUC via Mann-Whitney ranks (fast, no full matrix)
r = pd.Series(p).rank().to_numpy()
n1, n0 = int(y.sum()), int(len(y) - y.sum())
auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print("  log-likelihood %.1f, pseudo-R2 %.4f, AUC %.4f" % (ll, pseudo_r2, auc))

# ------------------------------------------------------------- bootstrap CI
print("bootstrapping %d replicates (cluster: store) ..." % NBOOT)
stores = pp.business_id.to_numpy()
uniq = np.unique(stores)
idx_by_store = {s: np.flatnonzero(stores == s) for s in uniq}
boot = np.empty((NBOOT, A.shape[1]))
for b in range(NBOOT):
    pick = RNG.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([idx_by_store[s] for s in pick])
    boot[b] = irls(A[idx], y[idx])
    if (b + 1) % 25 == 0:
        print("    %d/%d" % (b + 1, NBOOT), flush=True)
lo = np.percentile(boot, 2.5, axis=0)
hi = np.percentile(boot, 97.5, axis=0)

LABEL = {
    "log_rev_lag4": "recent review volume (log, +1 SD)",
    "stars_lag4": "recent mean rating (+1 SD)",
    "log_rev_trend": "review-volume trend (+1 SD)",
    "stars_trend": "rating trend, change (+1 SD)",
    "has_trend": "has enough history for a trend",
    "log_density": "log 1 km competitor count (+1 SD)",
    "price_num": "price tier (+1 SD)",
    "delivery": "offers delivery",
}
rows = []
for i, f in enumerate(FEATS):
    if f.startswith("band_"):
        lab = "baseline h(t): " + f.replace("band_", "").replace("_", "-") + " quarters"
    else:
        lab = LABEL.get(f, f)
    rows.append({
        "feature": f,
        "label": lab,
        "coef_std": beta[i + 1],
        "hazard_ratio": np.exp(beta[i + 1]),
        "hr_lo": np.exp(lo[i + 1]),
        "hr_hi": np.exp(hi[i + 1]),
        "significant": not (lo[i + 1] < 0 < hi[i + 1]),
    })
coefs = pd.DataFrame(rows)
coefs.to_csv(os.path.join(OUT, "08_hazard_coefficients.csv"), index=False,
             encoding="utf-8-sig")

print()
print("=== hazard ratios (HR>1 = goes quiet sooner) ===")
show = coefs[~coefs.feature.str.startswith("band_")].sort_values("hazard_ratio")
for _, r_ in show.iterrows():
    print("  %-42s HR %5.3f  [%5.3f, %5.3f]%s"
          % (r_.label[:42], r_.hazard_ratio, r_.hr_lo, r_.hr_hi,
             "  *" if r_.significant else ""))

# --------------------------------------------------- survival curves by tercile
print()
print("computing predicted survival curves ...")
base = {f: 0.0 for f in FEATS}
for f in ["has_trend"]:
    base[f] = pp[f].mean()
# a synthetic store at the median of everything else
base_x = np.array([base[f] for f in FEATS], float)[None, :]
base_xs = base_x.copy()
base_xs[0, scale_idx] = (base_x[0, scale_idx] - mu) / sd

c = "stars_trend"
col = FEATS.index(c)
q1, q2 = pp[c].quantile([1 / 3, 2 / 3])


def hazard_path(trend_z, quarters=24):
    """Hazard sequence for a store at a given standardised rating trend."""
    hs = []
    for t in range(1, quarters + 1):
        x = base_xs.copy()
        x[0, col] = trend_z
        for f in band_cols:
            x[0, FEATS.index(f)] = 1.0 if eval_band(f, t) else 0.0
        xa = np.concatenate([[1.0], x[0]])
        hs.append(1.0 / (1.0 + np.exp(-float(xa @ beta))))
    return np.array(hs)


def eval_band(name, t):
    lo, hi = name.replace("band_", "").split("_")
    return int(lo) <= t <= int(hi)


def z_of(value):
    return (value - mu[scale_idx.index(col)]) / sd[scale_idx.index(col)]


curve_rows = []
for label, val in (("rating falling (P33)", q1), ("flat (median)", pp[c].median()),
                   ("rating rising (P67)", q2)):
    z = z_of(val)
    hs = hazard_path(z, 24)
    surv = np.cumprod(1 - hs)
    for t, (h, s) in enumerate(zip(hs, surv), start=1):
        curve_rows.append({"group": label, "quarter": t, "hazard": h, "survival": s})
    curve_rows.append({"group": label, "quarter": 0, "hazard": np.nan, "survival": 1.0})
curves = pd.DataFrame(curve_rows).sort_values(["group", "quarter"])
curves.to_csv(os.path.join(OUT, "08_hazard_curves.csv"), index=False,
              encoding="utf-8-sig")

print("  survival after 8 quarters:")
for label in ["rating falling (P33)", "flat (median)", "rating rising (P67)"]:
    s8 = curves[(curves.group == label) & (curves.quarter == 8)].survival.iloc[0]
    print("    %-22s %.3f" % (label, s8))

# --------------------------------------------------------------- validation
print()
print("validation against is_open (which the model never saw):")
store_pred = pd.DataFrame({"business_id": pp.business_id, "p": p, "seq": pp.seq,
                           "n": pp.n_periods})
# 4-quarter predicted risk at each store's last observed period
lastp = store_pred[store_pred.seq == store_pred.n].groupby("business_id").p.mean()
op = pp.groupby("business_id").is_open.first()
j = pd.DataFrame({"pred": lastp, "is_open": op}).dropna()
jj = j[j.is_open == 0].index.union(j[j.is_open == 1].index)
r2 = j["pred"].rank().to_numpy()
n1v, n0v = int((j.is_open == 0).sum()), int((j.is_open == 1).sum())
auc_v = (r2[j.is_open == 0].sum() - n1v * (n1v + 1) / 2) / (n1v * n0v)
print("  AUC of last-period hazard vs is_open: %.4f" % auc_v)

with open(os.path.join(OUT, "08_hazard_fit.md"), "w", encoding="utf-8") as fh:
    fh.write("# Discrete-time hazard model - fit summary\n\n")
    fh.write("Person-period rows: %s | stores: %s | events: %s\n\n"
             % (format(len(pp), ","), format(pp.business_id.nunique(), ","),
                format(int(y.sum()), ",")))
    fh.write("| metric | value |\n|---|---:|\n")
    fh.write("| log-likelihood | %.1f |\n" % ll)
    fh.write("| McFadden pseudo-R2 | %.4f |\n" % pseudo_r2)
    fh.write("| in-sample AUC | %.4f |\n" % auc)
    fh.write("| AUC vs is_open (external) | %.4f |\n" % auc_v)
    fh.write("| bootstrap replicates | %d |\n" % NBOOT)
print("\nwrote 08_hazard_coefficients.csv, 08_hazard_curves.csv, 08_hazard_fit.md")
