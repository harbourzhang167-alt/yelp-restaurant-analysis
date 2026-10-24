"""Step 13 - cumulative-profit scenario model: does "long life" or "high traffic" win?

THE QUESTION
    Survival (is_open) was used as the success metric. But survival is not profit:
    a store can stay open for years while losing money. Given that traffic and
    closure are negatively correlated, is a short high-traffic life worth more
    than a long low-traffic one?

WHAT IS DATA-GROUNDED vs ASSUMED
    Observed (from the data):
        V_d  cumulative check-ins over the store's life      <- traffic x lifespan
        M_d  months operated
        t_d  price tier (1-4), a proxy for average ticket
    Assumed (external, industry-plausible - NOT estimated from this dataset):
        c    how many real customer visits one check-in represents
        B    base ticket price for price tier 2
        m    gross margin before rent and other fixed costs
        R    monthly rent at the sparsest density decile
        F    other monthly fixed costs (labour, utilities, ...)
        k    how much higher rent is at the densest decile vs the sparsest

    Because c, B and m are all unknowable here, the output is a SCENARIO MAP, not
    a profit estimate. What is reported is the break-even condition: the rent
    multiple at which a dense location stops beating a sparse one.

KEY STRUCTURAL FACT (measured, drives everything)
    Among stores that actually closed, operating duration is essentially FLAT
    across density deciles (~44 months mean). So dense locations are not
    short-lived; they are high-churn (more entries and exits, same per-store
    duration). That means the cost side of the ledger is nearly density-neutral
    except for rent - so the whole question reduces to rent versus traffic.

Outputs (analysis/out/):
    07_profit_by_density.csv     per-decile V, M, t and cumulative profit scenarios
    07_profit_breakeven.csv      break-even rent multiple as a function of c
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

d = pd.read_csv(os.path.join(OUT, "07_traffic_features.csv"), low_memory=False)
d = d.dropna(subset=["n_within_1_0km", "ck_months", "price_num"])

# Only stores whose life is complete: open stores are right-censored, so their
# cumulative volume is still growing and would break the comparison.
cl = d[d.is_open == 0].copy()
cl["dd"] = pd.qcut(cl.n_within_1_0km, 10, labels=False, duplicates="drop")

agg = cl.groupby("dd").agg(
    n=("ck_events", "size"),
    neighbours=("n_within_1_0km", "median"),
    V=("ck_events", "median"),            # cumulative check-ins over the life
    M=("ck_months", "median"),            # months operated
    t=("price_num", "mean"),              # price tier proxy for ticket
)
agg["rate"] = agg.V / agg.M
print("=== 已关闭门店（完整生命周期）按 1km 密度十分位 ===")
print(agg.round(2).to_string())
print()
print("D10 / D1 :  客流 %.2fx   寿命 %.2fx   累计量 %.2fx   价格档 %.2fx"
      % (agg.rate.iloc[-1] / agg.rate.iloc[0], agg.M.iloc[-1] / agg.M.iloc[0],
         agg.V.iloc[-1] / agg.V.iloc[0], agg.t.iloc[-1] / agg.t.iloc[0]))
print("→ 寿命几乎不变，累计量约 4 倍：密集的优势全部来自客流，不来自活得更久。")

# ---------------------------------------------------------------- model ----
# price tier -> ticket multiplier (tier 2 = 1.0)
TIER_MULT = {1: 0.6, 2: 1.0, 3: 1.5, 4: 2.2}
BASE_TICKET = 15.0          # $ per customer visit at price tier 2
GROSS_MARGIN = 0.70         # before rent / other fixed costs
RENT_SPARSE = 8000.0        # $/month at D1
OTHER_FIXED = 12000.0       # $/month (labour, utilities, ...)
CUSTOMERS_PER_CHECKIN = 1000.0   # baseline; swept below

# price tier -> ticket multiplier, applied continuously around tier 2 (=1.0x).
# Rounding the mean tier to an integer first would collapse D1 (1.61) and D10
# (1.78) both to tier 2 and silently erase the price-tier difference, so the
# multiplier is interpolated instead. The sensitivity script uses the same rule.
RATIO = 1.0 + 0.5 * (agg.t.values - 2.0)
dense_factor = np.linspace(0.0, 1.0, len(agg))       # 0 at D1, 1 at D10


def rents(k):
    """Rent by decile: geometric from RENT_SPARSE to RENT_SPARSE*k."""
    return RENT_SPARSE * (k ** dense_factor)


def cum_profit(c=CUSTOMERS_PER_CHECKIN, k=2.0, rent_s=RENT_SPARSE,
               fixed=OTHER_FIXED, margin=GROSS_MARGIN, ticket=BASE_TICKET):
    r = rents(k) * (rent_s / RENT_SPARSE)
    rev = agg.V.values * c * RATIO * ticket          # cumulative gross revenue
    gp = rev * margin
    cost = agg.M.values * (r + fixed)
    return gp - cost, gp, cost


def breakeven_k(c, rent_s=RENT_SPARSE, fixed=OTHER_FIXED, margin=GROSS_MARGIN,
                ticket=BASE_TICKET):
    """Rent multiple k at which D10 cumulative profit equals D1."""
    if agg.M.iloc[-1] == agg.M.iloc[0]:
        M = agg.M.iloc[0]
    else:
        M = (agg.M.iloc[-1] + agg.M.iloc[0]) / 2
    dV = (agg.V.iloc[-1] * RATIO[-1] - agg.V.iloc[0] * RATIO[0]) * c * ticket * margin
    # profit_10 - profit_1 = dV - M*R_S*(k-1) = 0
    return 1 + dV / (M * rent_s)


print("\n=== 校准：让中位门店恰好盈亏平衡，反解 c ===")
# In a competitive industry the median firm earns roughly zero economic profit.
# Rather than inventing c, calibrate it so the median store breaks even - a
# defensible anchor - then see how the ranking across density behaves.
med_rate = float(agg.rate.median())
med_ratio = float(RATIO.mean())
med_rent = float(rents(2.0).mean())
needed_gp = med_rent + OTHER_FIXED
needed_rev = needed_gp / GROSS_MARGIN
C_CAL = needed_rev / (med_rate * med_ratio * BASE_TICKET)
print(f"  中位客流 {med_rate:.2f} 次/月, 平均价格倍数 {med_ratio:.2f}, 平均租金 ${med_rent:,.0f}/月")
print(f"  达到盈亏平衡需月营收 ${needed_rev:,.0f} → 每 1 次打卡 ≈ {C_CAL:,.0f} 位到店客人")

print("\n=== 基准情形（按中位门店盈亏平衡校准）的累计利润（千美元）===")
base_profit, gp, cost = cum_profit(c=C_CAL)
tbl = agg.copy()
tbl["cum_gross_profit_kUSD"] = gp / 1e3
tbl["cum_cost_kUSD"] = cost / 1e3
tbl["cum_net_profit_kUSD"] = base_profit / 1e3
tbl["rent_kUSD_per_month"] = rents(2.0) / 1e3
tbl["implied_rev_kUSD_per_month"] = (agg.V.values / agg.M.values) * C_CAL * RATIO * BASE_TICKET / 1e3
print(tbl[["n", "neighbours", "V", "M", "rate", "rent_kUSD_per_month",
           "implied_rev_kUSD_per_month", "cum_gross_profit_kUSD",
           "cum_cost_kUSD", "cum_net_profit_kUSD"]].round(2).to_string())
tbl.to_csv(os.path.join(OUT, "07_profit_by_density.csv"), index=False, encoding="utf-8-sig")
print(f"\n  累计净利润：D1 {base_profit[0]/1e3:,.0f}k  →  D10 {base_profit[-1]/1e3:,.0f}k USD")

# ------------------------------------------------- break-even in c --------
print("\n=== 盈亏平衡：密集区租金最多可以是稀疏区的多少倍 ===")
rows = []
for c in (200, 500, 1000, 1500, 2000, 3000, 5000, 10000):
    kstar = breakeven_k(c)
    implied = med_rate * c * med_ratio * BASE_TICKET
    rows.append({"customers_per_checkin": c,
                 "implied_monthly_revenue_median_store_USD": round(implied, 0),
                 "breakeven_rent_multiple_D10_vs_D1": round(kstar, 2),
                 "dense_wins_if_rent_ratio_below_this": round(kstar, 2)})
be = pd.DataFrame(rows)
print(be.to_string(index=False))
be.to_csv(os.path.join(OUT, "07_profit_breakeven.csv"), index=False, encoding="utf-8-sig")
print(f"\n  以「中位门店盈亏平衡」校准的交点：c ≈ {C_CAL:,.0f}，"
      f"平衡租金倍数 ≈ {breakeven_k(C_CAL):.2f}x")

# ------------------------------------- does "long life" ever win? --------
print("\n=== 反过来说：什么情况下低客流的存活策略才划算 ===")
cl["td"] = pd.qcut(cl.ck_per_month, 10, labels=False, duplicates="drop")
tt = cl.groupby("td").agg(V=("ck_events", "median"), M=("ck_months", "median"),
                          rate=("ck_per_month", "median"), t=("price_num", "mean"))
print(tt.round(2).to_string())
print("\n最高客流档 vs 最低客流档:  累计量 %.1fx   寿命 %.2fx"
      % (tt.V.iloc[-1] / tt.V.iloc[0], tt.M.iloc[-1] / tt.M.iloc[0]))
print("→ 寿命只差 %.0f%%，累计量却差 %.0f 倍。低客流策略要胜出，"
      "只能靠「几乎不付租金」或「客单价远高于客流档」这类条件。"
      % ((1 - tt.M.iloc[-1] / tt.M.iloc[0]) * 100, tt.V.iloc[-1] / tt.V.iloc[0]))
