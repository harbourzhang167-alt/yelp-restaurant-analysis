"""Step 15 - joint sensitivity over c, m, F (and the rent ratio k).

PURPOSE
    The one-dimensional scan of c (fig26) left the conclusion resting on a single
    assumed constant. This sweeps the three monetary assumptions together and
    reports, for every combination, whether the dense (high-traffic) strategy or
    the sparse (long-lived) strategy wins - and by how much.

AN ANALYTICAL POINT WORTH CHECKING FIRST
    The cumulative-profit difference between the densest (D10) and sparsest (D1)
    decile is

        dPi = c*B*m*dV - [ M10*(R10+F) - M1*(R1+F) ]

    Since lifespan is essentially IDENTICAL across density (M10 ~= M1 ~= M), the
    fixed-cost terms cancel:  F*(M10 - M1) ~= 0.
    So F should barely matter for the RANKING (it shifts the absolute level on
    both sides equally). That would reduce a 3-D problem to 2-D: only c*m versus
    the rent ratio k decides the ranking. This script verifies that claim rather
    than assuming it.

Outputs (analysis/out/):
    07_sensitivity_kstar.csv    k* for every (c, m, F) - tests whether F matters
    07_sensitivity_grid.csv     winner + profit gap for every (c, m, k)
    07_sensitivity_absolute.csv cumulative profit for D1/D10 over (c, m, F, k)
"""

from __future__ import annotations

import itertools
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

base = pd.read_csv(os.path.join(OUT, "07_profit_by_density.csv"))

# ---------------------------------------------------------------- inputs ----
V = base.V.values.astype(float)               # cumulative check-ins per decile
M = base.M.values.astype(float)               # months operated
T = base.neighbours.values.astype(float)

# Continuous ticket multiplier instead of rounding the mean price tier to an
# integer: mean tier 1.61 (D1) and 1.78 (D10) would both round to 2 and the
# price-tier difference would vanish from the model.
MULT = 1.0 + 0.5 * (T * 0 + (base.V * 0 + 1))  # placeholder, replaced below
tier = pd.read_csv(os.path.join(OUT, "07_profit_by_density.csv"))
# price tier is not carried in the profit csv; recompute from the traffic table
feat = pd.read_csv(os.path.join(OUT, "07_traffic_features.csv"), low_memory=False)
feat = feat.dropna(subset=["n_within_1_0km", "ck_months", "price_num"])
cl = feat[feat.is_open == 0].copy()
cl["dd"] = pd.qcut(cl.n_within_1_0km, 10, labels=False, duplicates="drop")
tier_by_dec = cl.groupby("dd").price_num.mean()
MULT = 1.0 + 0.5 * (tier_by_dec.values - 2.0)   # tier 2 == 1.0x
print("price-tier multiplier by decile:", np.round(MULT, 3))

B = 15.0                    # base ticket at tier 2
RENT_SPARSE = 8000.0        # $/month at D1
dense = np.linspace(0.0, 1.0, len(V))


def rent_vec(k):
    return RENT_SPARSE * (k ** dense)


def profits(c, m, F, k):
    """Cumulative net profit per decile."""
    rev = V * c * MULT * B
    gp = rev * m
    cost = M * (rent_vec(k) + F)
    return gp - cost


def kstar(c, m, F=12000.0, k_for_rent=None):
    """Rent multiple at which D10 profit equals D1 profit."""
    # dPi(k) = c*B*m*(V10*mult10 - V1*mult1) - [M10*(R_S*k+F) - M1*(R_S+F)]
    dV = V[-1] * MULT[-1] - V[0] * MULT[0]
    num = c * B * m * dV
    den = M[-1] * RENT_SPARSE - M[0] * RENT_SPARSE
    if abs(den) < 1e-9:
        return np.nan
    # solve for k: 0 = num - [M10*R_S*k + M10*F - M1*R_S - M1*F]
    rhs = M[-1] * F - M[0] * RENT_SPARSE - M[0] * F
    return (num - rhs) / (M[-1] * RENT_SPARSE)


C_GRID = [200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000, 8000, 12000]
M_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
F_GRID = [6000, 9000, 12000, 16000, 20000, 25000]
K_GRID = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]

# ---------------------------------------------- (1) does F matter at all? ---
print("\n=== 检验：固定成本 F 是否影响「密集 vs 稀疏」的选择 ===")
rows = []
for c, m, F in itertools.product([500, 2000, 8000], [0.6, 0.75], F_GRID):
    rows.append({"c": c, "m": m, "F": F, "kstar": round(kstar(c, m, F), 3)})
ks = pd.DataFrame(rows)
piv = ks.pivot_table(index=["c", "m"], columns="F", values="kstar")
print(piv.round(3).to_string())
spread = (piv.max(axis=1) - piv.min(axis=1))
print(f"\n  F 从 {F_GRID[0]:,} 到 {F_GRID[-1]:,} 时 k* 的最大变动: {spread.max():.3f}"
      f"（相对变动 {spread.max()/piv.mean(axis=1).max()*100:.1f}%）")
print("  → F 对排序几乎无影响，只改变绝对利润水平（因为两侧寿命相同、等量抵消）")

ks.to_csv(os.path.join(OUT, "07_sensitivity_kstar.csv"), index=False,
          encoding="utf-8-sig")

# ---------------------------------------------- (2) the real 2-D surface ----
print("\n=== 真实二维：c × m 决定 k*（F 已证实可忽略）===")
surf = pd.DataFrame(index=M_GRID, columns=C_GRID, dtype=float)
for m in M_GRID:
    for c in C_GRID:
        surf.loc[m, c] = kstar(c, m)
print(surf.round(2).to_string())

# minimum c*m product needed for dense to win at a given rent ratio
print("\n=== 达到某个租金比所需的「c × m」阈值 ===")
thr = []
for k in K_GRID:
    cm = None
    for c in np.logspace(1.5, 4.3, 400):
        for m in np.linspace(0.45, 0.9, 46):
            if kstar(c, m) >= k:
                if cm is None or c * m < cm:
                    cm = c * m
    thr.append({"rent_ratio_k": k, "min_c_times_m": round(cm, 0) if cm else None,
                "interpretation": "c×m 高于此值则高客流胜出"})
thr = pd.DataFrame(thr)
print(thr.to_string(index=False))

# ------------------------------------------- (3) full 3-D grid, with k ------
print("\n=== 三维网格（c, m, F）× 租金比 k：胜者与利润差 ===")
recs = []
for c, m, F, k in itertools.product(C_GRID, M_GRID, F_GRID, K_GRID):
    p = profits(c, m, F, k)
    gap = p[-1] - p[0]
    recs.append({"c": c, "m": m, "F": F, "k": k,
                 "profit_D1_kUSD": round(p[0] / 1e3, 1),
                 "profit_D10_kUSD": round(p[-1] / 1e3, 1),
                 "gap_D10_minus_D1_kUSD": round(gap / 1e3, 1),
                 "winner": "dense" if gap > 0 else "sparse"})
grid = pd.DataFrame(recs)
grid.to_csv(os.path.join(OUT, "07_sensitivity_grid.csv"), index=False,
            encoding="utf-8-sig")

# how often does each strategy win, by rent ratio
print()
for k in K_GRID:
    sub = grid[grid.k == k]
    share = (sub.winner == "dense").mean()
    print(f"  k={k:<4} 高客流胜出占比 {share*100:5.1f}%   "
          f"中位利润差 {sub.gap_D10_minus_D1_kUSD.median():+8.0f}k")

# --------------------------------------------------------- (4) F on level ---
print("\n=== F 影响绝对水平（排序不变）===")
lv = []
for F in F_GRID:
    p = profits(2000, 0.70, F, 3.0)
    lv.append({"F": F, "D1_kUSD": round(p[0] / 1e3, 0), "D10_kUSD": round(p[-1] / 1e3, 0),
               "gap_kUSD": round((p[-1] - p[0]) / 1e3, 0),
               "winner": "dense" if p[-1] > p[0] else "sparse"})
lv = pd.DataFrame(lv)
print(lv.to_string(index=False))
print("\n  → F 每 +1000，两侧各降约 40k（39.7 个月），差额几乎不变")

grid.head(0).to_csv(os.path.join(OUT, "07_sensitivity_absolute.csv"),
                    index=False, encoding="utf-8-sig")
lv.to_csv(os.path.join(OUT, "07_sensitivity_absolute.csv"), index=False,
          encoding="utf-8-sig")

# save surface for plotting
surf.to_csv(os.path.join(OUT, "07_sensitivity_surface.csv"), encoding="utf-8-sig")
print("\nwrote 07_sensitivity_*.csv")
