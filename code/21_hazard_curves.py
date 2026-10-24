"""Recompute hazard-based survival curves for several dimensions.

Purpose: the hazard model found the rating LEVEL predicts cessation (HR 0.809) but the
rating TREND does not (HR 0.997, CI [0.980, 1.017]). Two figures need curves:
  * rating trend  - three curves lying exactly on top of each other; the overlap IS
                    the result (the hypothesis we hoped for does not hold)
  * delivery      - the largest single effect (HR 0.606), visibly separated
so both are emitted here, plus rating level and density for completeness.

Refits only the point model (no bootstrap), so this runs in seconds.

Output: analysis/out/08_hazard_curves.csv
        columns: dimension, dimension_cn, group, quarter, hazard, survival
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

pp = pd.read_csv(os.path.join(OUT, "08_hazard_person_period.csv.gz"))
feat = pd.read_csv(os.path.join(OUT, "business_features.csv"),
                   usecols=["business_id", "n_within_1_0km", "price_num", "is_delivery"],
                   low_memory=False)
pp = pp.merge(feat, on="business_id", how="left")
pp["log_density"] = np.log1p(pp.n_within_1_0km)
pp["delivery"] = pp.is_delivery.fillna(pp.is_delivery.mean()).astype(float)
pp["price_num"] = pp.price_num.fillna(pp.price_num.mean())

BANDS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 32), (33, 48)]
band_cols = []
for lo, hi in BANDS:
    c = "band_%d_%d" % (lo, hi)
    pp[c] = ((pp.quarters_elapsed >= lo) & (pp.quarters_elapsed <= hi)).astype(float)
    band_cols.append(c)
pp["has_trend"] = pp.stars_trend.notna().astype(float)
for c in ["rev_lag4", "stars_lag4", "rev_trend", "stars_trend", "log_density",
          "price_num"]:
    pp[c] = pp[c].fillna(pp[c].mean())
pp["log_rev_lag4"] = np.log1p(pp.rev_lag4)
pp["log_rev_trend"] = np.log(pp.rev_trend.clip(lower=0.25, upper=4.0))

FEATS = band_cols + ["log_rev_lag4", "stars_lag4", "log_rev_trend", "stars_trend",
                     "has_trend", "log_density", "price_num", "delivery"]
X = pp[FEATS].to_numpy(np.float32)
y = pp.y.to_numpy(np.float32)
scale_idx = [FEATS.index(f) for f in FEATS if not f.startswith("band_")]
mu = X[:, scale_idx].mean(axis=0)
sd = X[:, scale_idx].std(axis=0)
sd[sd == 0] = 1.0
Xs = X.copy()
Xs[:, scale_idx] = (Xs[:, scale_idx] - mu) / sd
A = np.hstack([np.ones((len(y), 1), np.float32), Xs]).astype(np.float32)


def irls(Am, yv, ridge=2.0, iters=30):
    beta = np.zeros(Am.shape[1], dtype=np.float64)
    pen = np.eye(Am.shape[1]) * ridge
    pen[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(Am @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        z = eta + (yv - p) / w
        beta = np.linalg.solve(
            (Am.T @ (Am * w[:, None].astype(np.float32))).astype(np.float64) + pen,
            (Am.T @ (w * z).astype(np.float32)).astype(np.float64))
    return beta


beta = irls(A, y)
np.savez(os.path.join(OUT, "08_hazard_params.npz"), beta=beta, mu=mu, sd=sd,
         feats=np.array(FEATS, dtype=object))
print("refit done (point estimate only)")

# Baseline store = the mean of every feature, which in standardised space is zero.
base_xs = np.zeros(len(FEATS))


def z_of(feature: str, raw_value: float) -> float:
    col = FEATS.index(feature)
    return (raw_value - mu[scale_idx.index(col)]) / sd[scale_idx.index(col)]


def curve(feature: str, z: float, quarters: int = 24):
    surv, hz, cum = [], [], 1.0
    col = FEATS.index(feature)
    for t in range(1, quarters + 1):
        x = base_xs.copy()
        x[col] = z
        for i, (lo, hi) in enumerate(BANDS):
            x[FEATS.index(band_cols[i])] = 1.0 if (lo <= t <= hi) else 0.0
        xa = np.concatenate([[1.0], x])
        h = 1.0 / (1.0 + np.exp(-float(xa @ beta)))
        cum *= (1 - h)
        hz.append(h)
        surv.append(cum)
    return np.array(hz), np.array(surv)


# (feature, chinese name, [(label, kind, spec)]) -- kind: 'raw' or 'quantile'
SPECS = [
    ("delivery", "外送 availability",
     [("提供外送", "raw", 1.0), ("不提供外送", "raw", 0.0)]),
    ("stars_lag4", "评分水平",
     [("评分较高（P67）", "q", 2 / 3), ("评分中位", "q", 0.5),
      ("评分较低（P33）", "q", 1 / 3)]),
    ("stars_trend", "评分趋势",
     [("评分上升（P67）", "q", 2 / 3), ("评分持平（中位）", "q", 0.5),
      ("评分下滑（P33）", "q", 1 / 3)]),
    ("log_density", "竞争密度",
     [("密集商圈（P67）", "q", 2 / 3), ("中位", "q", 0.5),
      ("稀疏区域（P33）", "q", 1 / 3)]),
]

rows = []
for feature, dim_cn, groups in SPECS:
    for label, kind, spec in groups:
        raw = spec if kind == "raw" else float(pp[feature].quantile(spec))
        z = z_of(feature, raw)
        hs, sv = curve(feature, z)
        rows.append({"dimension": feature, "dimension_cn": dim_cn, "group": label,
                     "value": raw, "quarter": 0, "hazard": np.nan, "survival": 1.0})
        for t in range(1, 25):
            rows.append({"dimension": feature, "dimension_cn": dim_cn, "group": label,
                         "value": raw, "quarter": t,
                         "hazard": hs[t - 1], "survival": sv[t - 1]})
curves = pd.DataFrame(rows)
curves.to_csv(os.path.join(OUT, "08_hazard_curves.csv"), index=False,
              encoding="utf-8-sig")

print()
for feature, dim_cn, groups in SPECS:
    print("=== %s (%s) ===" % (feature, dim_cn))
    for _, g in curves[curves.dimension == feature].groupby("group", sort=False):
        s8 = g[g.quarter == 8].survival.iloc[0]
        s24 = g[g.quarter == 24].survival.iloc[0]
        print("  %-20s 第8季 %.3f   第24季 %.3f" % (g.group.iloc[0], s8, s24))
print("\nwrote 08_hazard_curves.csv (%d rows)" % len(curves))
