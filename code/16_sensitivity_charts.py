"""Step 16 - charts for the joint sensitivity.

fig28  k* surface over (c, m): the 2-D face that actually decides the ranking,
       with iso-lines where the break-even sits exactly at 2x / 3x / 4x rent
fig29  why F drops out: k* is flat in F, while the absolute profit level falls
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

surf = pd.read_csv(os.path.join(OUT, "07_sensitivity_surface.csv"), index_col=0)
surf.columns = [float(c) for c in surf.columns]
ks = pd.read_csv(os.path.join(OUT, "07_sensitivity_kstar.csv"))
absol = pd.read_csv(os.path.join(OUT, "07_sensitivity_absolute.csv"))

C_CAL, M_CAL = 2716.0, 0.70


# ---------------------------------------------------------------- fig28
def fig28():
    W, H = 1320, 880
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((48, 24), "图 28｜联合敏感性：决定胜负的其实只有一个乘积", font=font(25, True), fill=INK)
    d.text((48, 62), "横轴 = 每 1 次打卡代表的到店人次 c；纵轴 = 毛利率 m；格内数字 = 盈亏平衡租金倍数 k*。"
                     "固定成本 F 已验证几乎不影响（见图 29）", font=font(14), fill=MUTED)

    cs = list(surf.columns)
    ms = list(surf.index)
    x0, x1 = 150, W - 230
    y0, y1 = 150, H - 170
    nc, nm = len(cs), len(ms)
    # The c axis is log-scaled, so columns must be placed at their log positions
    # (equal-width cells would misalign the analytic iso-lines by up to ~half a
    # column, since the c grid is only approximately log-spaced).
    lc = np.log10(np.array(cs, dtype=float))
    def x_at(cval):
        return x0 + (np.log10(cval) - lc[0]) / (lc[-1] - lc[0]) * (x1 - x0)
    edges = [x0] + [(x_at(cs[j]) + x_at(cs[j + 1])) / 2 for j in range(nc - 1)] + [x1]
    ch = (y1 - y0) / nm

    vals = surf.values.astype(float)
    vmax = 12.0

    def colour(v):
        """light -> dark as k* rises: low k* = sparse-favouring, high = dense."""
        t = min(max(v / vmax, 0.0), 1.0)
        if t < 0.5:
            u = t / 0.5
            r, g, b = 236 - 20 * u, 210, 198 + 10 * u
        else:
            u = (t - 0.5) / 0.5
            r, g, b = 216 - 100 * u, 210 + 24 * u, 208 - 60 * u
        return (int(r), int(g), int(b))

    for i, m in enumerate(ms):
        for j, c in enumerate(cs):
            v = surf.loc[m, c]
            xx = edges[j]
            wcell = edges[j + 1] - edges[j]
            yy = y0 + (nm - 1 - i) * ch
            d.rectangle([xx, yy, xx + wcell, yy + ch],
                        fill=colour(v), outline="#ffffff", width=2)
            lab = f"{v:.1f}"
            tw = d.textbbox((0, 0), lab, font=font(13, True))[2]
            d.text((xx + wcell / 2 - tw / 2, yy + ch / 2 - 9), lab,
                   font=font(13, True), fill="#1f2933")

    # Iso-lines solved ANALYTICALLY rather than interpolated from this coarse
    # grid: k* is linear in (c*m), so for a fixed target k the curve is
    # m = A_k / c  (a hyperbola). Interpolating the 11x8 grid missed the nearly
    # vertical k*=3 line entirely (it crossed only one grid column).
    V10, V1 = 87.0, 22.0
    dV = 87.0 * (1.0 + 0.5 * (1.78 - 2.0)) - 22.0 * (1.0 + 0.5 * (1.61 - 2.0))
    M10, M1 = 40.31, 39.10
    R_S, F = 8000.0, 12000.0
    B = 15.0

    lines_drawn = 0
    cgrid = np.logspace(np.log10(cs[0]), np.log10(cs[-1]), 240)
    for target, col, wd in ((2.0, ACCENT2, 3), (3.0, "#7a4a8c", 4), (4.0, ACCENT, 3)):
        A = target * M10 * R_S + M10 * F - M1 * R_S - M1 * F
        mcurve = A / (cgrid * B * dV)
        pts = []
        for cc, mm in zip(cgrid, mcurve):
            if not (ms[0] - 0.02 <= mm <= ms[-1] + 0.02):
                continue
            jj = (np.log10(cc) - lc[0]) / (lc[-1] - lc[0]) * (x1 - x0)
            # m -> y: rows are ms[0] at the bottom
            frac_m = (mm - ms[0]) / (ms[-1] - ms[0]) * (nm - 1)
            yy = y0 + (nm - 1 - frac_m) * ch + ch / 2
            pts.append((x0 + jj, yy))
        if len(pts) > 1:
            d.line(pts, fill=col, width=wd)
            lines_drawn += 1
            # label at the TOP end of the curve, on a white plate so it does not
            # sit on top of a cell's k* number
            mx, my = pts[-1]
            lab = f"k*={target:.0f}×"
            tw = d.textbbox((0, 0), lab, font=font(14, True))[2]
            d.rectangle([mx - tw / 2 - 4, my - 26, mx + tw / 2 + 4, my - 4],
                        fill="#ffffff", outline=col)
            d.text((mx - tw / 2, my - 25), lab, font=font(14, True), fill=col)
    print(f"    iso-lines drawn: {lines_drawn}")

    # legend: what the fill colour means
    lx, ly = 48, 100
    d.rectangle([lx, ly, lx + 20, ly + 14], fill=colour(1.3), outline="#ffffff")
    d.text((lx + 28, ly - 2), "浅色 = 低 k*，低客流长寿命更划算", font=font(12), fill=MUTED)
    d.rectangle([lx, ly + 22, lx + 20, ly + 36], fill=colour(11.0), outline="#ffffff")
    d.text((lx + 28, ly + 20), "绿色 = 高 k*，高客流更划算", font=font(12), fill=MUTED)

    # calibrated point
    cj = min(range(nc), key=lambda j: abs(cs[j] - C_CAL))
    mi = min(range(nm), key=lambda i: abs(ms[i] - M_CAL))
    xx = x_at(cs[cj])
    yy = y0 + (nm - 1 - mi) * ch + ch / 2
    d.ellipse([xx - 12, yy - 12, xx + 12, yy + 12], outline="#ffffff", width=4)
    d.ellipse([xx - 9, yy - 9, xx + 9, yy + 9], fill=WARN, outline="#ffffff", width=2)

    # axes
    d.line([(x0, y0), (x0, y1)], fill=INK, width=2)
    d.line([(x0, y1), (x1, y1)], fill=INK, width=2)
    for j, c in enumerate(cs):
        lab = f"{c:,.0f}"
        tw = d.textbbox((0, 0), lab, font=font(12))[2]
        d.text((x_at(c) - tw / 2, y1 + 8), lab, font=font(12), fill=MUTED)
    for i, m in enumerate(ms):
        lab = f"{m:.2f}"
        tw = d.textbbox((0, 0), lab, font=font(12))[2]
        d.text((x0 - tw - 12, y0 + (nm - 1 - i) * ch + ch / 2 - 8), lab,
               font=font(12), fill=MUTED)
    d.text((x0 + (x1 - x0) / 2 - 150, H - 120), "每 1 次打卡 ≈ 到店人次 c（对数刻度）",
           font=font(16, True), fill=INK)

    # explanatory text
    box_y = H - 104
    d.text((48, box_y),
           "读法：格内 k* 是该组合下密集区能承受的最高租金倍数。k* 高于现实租金比（约 2–4×）则高客流胜出。",
           font=font(14), fill=INK)
    d.text((48, box_y + 26),
           "校准点 c≈2,716、m=0.70 → k*≈6.3×，高于现实 2–4×，故按「中位门店盈亏平衡」校准，高客流胜出。",
           font=font(14, True), fill=WARN)
    d.text((48, box_y + 52),
           "但若 c 只有 500（打卡是很稀疏的高信号），k*≈2.0×，正落在现实租金比区间内——结论变成不确定。",
           font=font(14), fill=ACCENT2)
    img.save(os.path.join(IMG, "fig28_sensitivity_surface.png"))
    print("  rendered fig28_sensitivity_surface.png")


def nice_ticks(lo, hi, target=5):
    """Round tick values covering [lo, hi] with a 1/2/5 x 10^k step."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(target - 1, 1)
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    start = np.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.001:
        ticks.append(round(v, 10))
        v += step
    return ticks


# ---------------------------------------------------------------- fig29
def fig29():
    W, H = 1220, 620
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((48, 22), "图 29｜为什么固定成本 F 可以先不管：它只平移利润，不改变胜负", font=font(24, True), fill=INK)
    d.text((48, 58), "左：F 变化时 k* 几乎不动（三条线）；右：F 变化时绝对利润同幅下降，但两档差额保持不变",
           font=font(14), fill=MUTED)

    # ---- left panel: k* vs F
    lx0, lx1 = 100, 590
    ly0, ly1 = 120, 480
    sub = ks[(ks.m == 0.75)]
    cs3 = sorted(sub.c.unique())[:3]
    if len(cs3) < 3:
        cs3 = sorted(ks.c.unique())[:3]
    fmin, fmax = ks.F.min(), ks.F.max()
    def px_l(F):
        return lx0 + (F - fmin) / (fmax - fmin) * (lx1 - lx0)
    allv = ks[ks.c.isin(cs3)].kstar
    ymin, ymax = allv.min() * 0.85, allv.max() * 1.08
    def py_l(v):
        return ly1 - (v - ymin) / (ymax - ymin) * (ly1 - ly0)
    for v in nice_ticks(ymin, ymax, 5):
        yy = py_l(v)
        d.line([(lx0, yy), (lx1, yy)], fill="#eef1f4")
        d.text((lx0 - 56, yy - 8), f"{v:.1f}×", font=font(12), fill=MUTED)
    d.line([(lx0, ly0), (lx0, ly1)], fill=INK, width=2)
    d.line([(lx0, ly1), (lx1, ly1)], fill=INK, width=2)
    d.text((lx0, ly0 - 30), "盈亏平衡租金倍数 k*", font=font(14, True), fill=INK)
    cols = [ACCENT, ACCENT2, ACCENT3]
    for n, c in enumerate(cs3):
        s = ks[(ks.m == 0.75) & (ks.c == c)].sort_values("F")
        if s.empty:
            s = ks[ks.c == c].sort_values("F")
        pts = [(px_l(F), py_l(v)) for F, v in zip(s.F, s.kstar)]
        d.line(pts, fill=cols[n], width=4)
        for x, y in pts:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=cols[n])
        d.text((pts[-1][0] + 8, pts[-1][1] - 24), f"c={c:,.0f}", font=font(13, True), fill=cols[n])
    d.text((lx0, ly1 + 10), f"F = ${fmin:,} → ${fmax:,}/月", font=font(13), fill=MUTED)
    d.text((lx0, ly1 + 34), "k* 变动 < 0.1（<0.5%）", font=font(13, True), fill=ACCENT3)

    # ---- right panel: absolute profit vs F
    rx0, rx1 = 720, 1170
    ry0, ry1 = 120, 480
    def px_r(F):
        return rx0 + (F - absol.F.min()) / (absol.F.max() - absol.F.min()) * (rx1 - rx0)
    lo = min(absol.D1_kUSD.min(), absol.D10_kUSD.min()) * 1.1
    hi = max(absol.D1_kUSD.max(), absol.D10_kUSD.max()) * 1.12
    def py_r(v):
        return ry1 - (v - lo) / (hi - lo) * (ry1 - ry0)
    for v in nice_ticks(lo, hi, 5):
        yy = py_r(v)
        d.line([(rx0, yy), (rx1, yy)], fill="#eef1f4")
        d.text((rx0 - 70, yy - 8), f"{v:,.0f}k", font=font(12), fill=MUTED)
    d.line([(rx0, py_r(0)), (rx1, py_r(0))], fill=INK, width=2)
    d.line([(rx0, ry0), (rx0, ry1)], fill=INK, width=2)
    d.line([(rx0, ry1), (rx1, ry1)], fill=INK, width=2)
    d.text((rx0, ry0 - 30), "累计净利润（千美元）", font=font(14, True), fill=INK)
    for series, col, lab in ((absol.D10_kUSD, ACCENT3, "D10 最密集"),
                             (absol.D1_kUSD, ACCENT2, "D1 最稀疏")):
        pts = [(px_r(F), py_r(v)) for F, v in zip(absol.F, series)]
        d.line(pts, fill=col, width=4)
        for x, y in pts:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
    d.text((px_r(absol.F.iloc[-1]) - 130, py_r(absol.D10_kUSD.iloc[-1]) - 26),
           "D10 最密集", font=font(13, True), fill=ACCENT3)
    d.text((px_r(absol.F.iloc[-1]) - 132, py_r(absol.D1_kUSD.iloc[-1]) - 24),
           "D1 最稀疏", font=font(13, True), fill=ACCENT2)
    d.text((rx0, ry1 + 10), "两线近似平行 → 差额恒定 ≈ 570–590k", font=font(13, True), fill=INK)
    d.text((rx0, ry1 + 34), "（c=2000, m=0.70, k=3.0）", font=font(12), fill=MUTED)

    img.save(os.path.join(IMG, "fig29_fixed_cost_irrelevance.png"))
    print("  rendered fig29_fixed_cost_irrelevance.png")


fig28()
fig29()
print("sensitivity charts rendered")
