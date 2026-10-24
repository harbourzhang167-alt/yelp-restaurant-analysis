"""Step 14 - charts for the cumulative-profit scenario model.

fig26  break-even rent multiple as a function of the check-in->customer factor
fig27  cumulative net profit by density, and the survival/profit contradiction
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import ACCENT, ACCENT2, ACCENT3, GRID, INK, MUTED, WARN, Chart, font  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
IMG = os.path.join(OUT, "charts")

be = pd.read_csv(os.path.join(OUT, "07_profit_breakeven.csv"))
tbl = pd.read_csv(os.path.join(OUT, "07_profit_by_density.csv"))
feat = pd.read_csv(os.path.join(OUT, "07_traffic_features.csv"), low_memory=False)

C_CAL = 2716.0
K_CAL = 6.34


# ---------------------------------------------------------------- fig26
def fig26():
    c = Chart(1180, 760,
              title="图 26｜密集区租金能高到什么程度，高客流才不再划算？",
              subtitle="横轴 = 每 1 次打卡代表的真实到店人次（唯一的未知换算）；纵轴 = 盈亏平衡租金倍数（D10 ÷ D1）",
              xlabel="每 1 次打卡 ≈ 多少位到店客人（对数刻度）", ylabel="盈亏平衡租金倍数",
              pad_b=150)
    xs = np.log10(be.customers_per_checkin.values)
    ys = be.breakeven_rent_multiple_D10_vs_D1.values
    c.set_xlim(xs.min() - 0.08, xs.max() + 0.08)
    c.set_ylim(0, 24)
    ticks = [200, 500, 1000, 2000, 5000, 10000]
    c.draw_frame(x_ticks=[np.log10(t) for t in ticks], x_fmt="", y_fmt="{:.0f}×")
    for t in ticks:
        tw = c.d.textbbox((0, 0), f"{t:,}", font=font(14))[2]
        c.d.text((c.px(np.log10(t)) - tw / 2, c.H - c.pad_b + 26), f"{t:,}",
                 font=font(14), fill=MUTED)

    # realistic dense-vs-sparse rent band (2x - 4x)
    c.d.rectangle([c.plot[0], c.py(4.0), c.plot[2], c.py(2.0)],
                  fill="#eaf2e8")
    c.d.text((c.plot[0] + 16, c.py(4.0) + 8),
             "现实租金比区间：密集区约为稀疏区的 2–4 倍", font=font(14, True), fill=ACCENT3)

    c.line(list(zip(xs, ys)), color=ACCENT2, width=4)
    for x, y in zip(xs, ys):
        c.marker(x, y, r=7, color=ACCENT2)
    # calibration point
    c.vline(np.log10(C_CAL), color=WARN)
    c.marker(np.log10(C_CAL), K_CAL, r=10, color=WARN)
    c.d.text((c.px(np.log10(C_CAL)) + 18, c.py(K_CAL) - 52),
             f"中位门店盈亏平衡校准\nc ≈ {C_CAL:,.0f}　平衡倍数 ≈ {K_CAL:.1f}×",
             font=font(14, True), fill=WARN)
    # The y value is the ACTUAL rent ratio (D10/D1); the curve is the break-even
    # ratio. Dense wins when the actual ratio sits BELOW the curve.
    c.d.text((c.px(xs.min()) + 24, c.py(22)),
             "曲线上方＝实际租金比高于临界值 → 低客流长寿命更划算", font=font(14, True), fill=MUTED)
    c.d.text((c.px(xs.min()) + 24, c.py(1.2)),
             "曲线下方＝租金还没吃掉客流优势 → 高客流更划算", font=font(14, True), fill=ACCENT2)
    c.legend([("盈亏平衡线", ACCENT2, "line"), ("现实租金比 2–4×", ACCENT3)],
             x=c.px(xs.max()) - 300, y=c.plot[1] + 12)
    c.footnote("纵轴＝「密集区租金÷稀疏区租金」实际倍数，曲线＝临界倍数。实际低于曲线则高客流赢。"
               "校准后临界 6.3×，高于现实 2–4×，故高客流胜出。")
    c.save(os.path.join(IMG, "fig26_profit_breakeven.png"))


# ---------------------------------------------------------------- fig27
def fig27():
    W, H = 1180, 800
    img = Image.new("RGB", (W, H), "#ffffff")
    d2 = ImageDraw.Draw(img)
    d2.text((44, 22), "图 27｜存活与盈利可能指向相反方向", font=font(25, True), fill=INK)
    d2.text((44, 60), "上：累计净利润（按中位门店盈亏平衡校准）；下：该密度档的关闭率。"
                      "关闭率最低的档，累计利润为负", font=font(15), fill=MUTED)

    x0, x1 = 120, W - 80
    n = len(tbl)
    def pxp(i):
        return x0 + (i + 0.5) / n * (x1 - x0)

    # --- top: cumulative net profit
    top, bot = 130, 470
    vals = tbl.cum_net_profit_kUSD.values
    lo, hi = min(vals.min(), 0) * 1.15, max(vals.max(), 0) * 1.2
    def pyv(v):
        return bot - (v - lo) / (hi - lo) * (bot - top)
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        yy = pyv(v)
        d2.line([(x0, yy), (x1, yy)], fill="#eef1f4")
        d2.text((x0 - 14 - 54, yy - 8), f"{v:+,.0f}k".replace("+", ""), font=font(12), fill=MUTED)
    d2.line([(x0, pyv(0)), (x1, pyv(0))], fill=INK, width=2)
    d2.line([(x0, top), (x0, bot)], fill=INK, width=2)
    d2.text((x0, top - 28), "累计净利润（千美元，完整生命周期）", font=font(15, True), fill=INK)
    for i, v in enumerate(vals):
        col = ACCENT3 if v > 0 else ACCENT2
        xa, xb = pxp(i) - 26, pxp(i) + 26
        d2.rectangle([xa, min(pyv(0), pyv(v)), xb, max(pyv(0), pyv(v))], fill=col)
        lab = f"{v:,.0f}k"
        d2.text((pxp(i) - 22, pyv(v) - 20 if v > 0 else pyv(v) + 4), lab,
                font=font(11, True), fill=col)

    # --- bottom: closure rate
    top2, bot2 = 540, 700
    cr = 1 - tbl.get("closure_rate", pd.Series(np.nan, index=tbl.index))
    dens = tbl.neighbours.values
    # closure rate by density comes from the traffic table
    feat2 = feat.dropna(subset=["n_within_1_0km"]).copy()
    feat2["dd"] = pd.qcut(feat2.n_within_1_0km, 10, labels=False, duplicates="drop")
    _cls = feat2.groupby("dd").is_open.mean()
    clos = (1 - _cls).values
    cmax = clos.max() * 1.35
    def pyc(v):
        return bot2 - v / cmax * (bot2 - top2)
    for k in range(4):
        v = cmax * k / 3
        yy = pyc(v)
        d2.line([(x0, yy), (x1, yy)], fill="#eef1f4")
        d2.text((x0 - 14 - 40, yy - 8), f"{v:.0%}", font=font(12), fill=MUTED)
    d2.line([(x0, bot2), (x1, bot2)], fill=INK, width=2)
    d2.line([(x0, top2), (x0, bot2)], fill=INK, width=2)
    d2.text((x0, top2 - 28), "关闭率", font=font(15, True), fill=ACCENT2)
    for i, v in enumerate(clos):
        xa, xb = pxp(i) - 26, pxp(i) + 26
        d2.rectangle([xa, pyc(v), xb, pyc(0)], fill=ACCENT2)
        d2.text((pxp(i) - 18, pyc(v) - 20), f"{v:.0%}", font=font(11, True), fill=ACCENT2)

    for i, r in tbl.iterrows():
        d2.text((pxp(i) - 10, 718), f"D{i+1}", font=font(12, True), fill=INK)
        tw = d2.textbbox((0, 0), f"{r.neighbours:.0f}", font=font(11))[2]
        d2.text((pxp(i) - tw / 2, 738), f"{r.neighbours:.0f}", font=font(11), fill=MUTED)
    d2.text((x0, 764), "1 公里内竞争对手数（每档中位）", font=font(13, True), fill=INK)
    img.save(os.path.join(IMG, "fig27_survival_vs_profit.png"))
    print("  rendered fig27_survival_vs_profit.png")


fig26()
fig27()
print("profit charts rendered")
