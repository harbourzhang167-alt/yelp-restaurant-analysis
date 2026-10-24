"""Figures for the discrete-time hazard model.

fig30  hazard ratios for the time-varying and static predictors
fig31  predicted survival curves by rating trend (the factor the earlier report
       could not test)
fig32  baseline hazard by quarters since the store's first observed quarter
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

co = pd.read_csv(os.path.join(OUT, "08_hazard_coefficients.csv"))
cv = pd.read_csv(os.path.join(OUT, "08_hazard_curves.csv"))


# ------------------------------------------------------------------ fig30
def fig30():
    d = co[~co.feature.str.startswith("band_")].copy()
    d = d.reindex(d.hazard_ratio.sort_values().index)
    n = len(d)
    H = 330 + n * 46
    ch = Chart(1180, H,
               title="图 30｜离散时间 hazard：什么让门店「安静下来」",
               subtitle="每一季一个观测，事件＝评论序列在该季停止；HR>1 表示更早停止。"
                        "仅用事件发生前的滞后信息，is_open 不入模",
               xlabel="风险比（Hazard Ratio）", ylabel="",
               pad_l=340, pad_r=110, pad_t=150, pad_b=170)
    ch.set_xlim(0.55, 1.34)
    ch.set_ylim(-0.8, n - 0.2)
    ch.draw_frame(x_ticks=[0.6, 0.8, 1.0, 1.2], x_fmt="{:.1f}",
                  y_ticks=[], grid=True)
    for i, (_, r) in enumerate(d.iterrows()):
        yy = ch.py(i)
        col = ACCENT2 if r.hazard_ratio > 1 else ACCENT3
        ch.d.line([(ch.px(1.0), yy), (ch.px(r.hazard_ratio), yy)], fill=col, width=5)
        ch.d.line([(ch.px(r.hr_lo), yy), (ch.px(r.hr_hi), yy)], fill=INK, width=2)
        for v in (r.hr_lo, r.hr_hi):
            ch.d.line([(ch.px(v), yy - 6), (ch.px(v), yy + 6)], fill=INK, width=2)
        ch.marker(r.hazard_ratio, i, r=7, color=col)
        lab = r.label
        tw = ch.d.textbbox((0, 0), lab, font=font(14))[2]
        ch.d.text((ch.px(ch.xlim[0]) - tw - 14, yy - 9), lab, font=font(14), fill=INK)
        # value label sits right of the CI when the bar runs left, and right of the
        # bar when it runs right - otherwise it lands on top of the bar
        s = "%.3f" % r.hazard_ratio
        anchor = max(float(r.hr_hi), 1.0) + 0.018
        ch.d.text((ch.px(anchor), yy - 9), s, font=font(13, True), fill=col)
    ch.vline(1.0, color=MUTED)
    # "HR = 1" goes at the BOTTOM of the reference line, and the legend to the empty
    # top-left, so neither collides with the top row's bar or its value label
    ch.d.text((ch.px(1.0) + 6, ch.plot[3] - 20), "HR = 1（无影响）", font=font(12), fill=MUTED)
    ch.legend([("更早安静（风险↑）", ACCENT2), ("更晚安静（风险↓）", ACCENT3)],
              x=ch.px(0.575), y=ch.plot[1] + 12)
    ch.footnote("误差棒为 120 次按门店聚类的 bootstrap 95% CI；标准化系数转换为风险比")
    ch.save(os.path.join(IMG, "fig30_hazard_ratios.png"))


# ------------------------------------------------------------------ fig31
def fig31():
    """Two panels: the factor we hoped would predict (flat) vs the one that does.

    The rating-trend coefficient is statistically indistinguishable from zero, so its
    three curves coincide exactly. Showing that next to delivery - the largest effect -
    makes the negative result visible instead of hiding it behind a coefficient.
    """
    W, H = 1260, 760
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((48, 22), "图 31｜我们以为有用的信号，和真正有用的信号", font=font(25, True), fill=INK)
    d.text((48, 60), "同一「典型门店」，只改一个变量；纵轴为仍未停止评论的比例。"
                     "左：评分趋势（三条线完全重合）；右：是否提供外送", font=font(14), fill=MUTED)

    panels = [
        (60, 620, 130, 590, "stars_trend",
         [("评分上升（P67）", ACCENT3), ("评分持平（中位）", ACCENT), ("评分下滑（P33）", ACCENT2)],
         "评分趋势：三条曲线重合 → 无预警价值"),
        (670, 1210, 130, 590, "delivery",
         [("提供外送", ACCENT3), ("不提供外送", ACCENT2)],
         "是否提供外送：差距在 24 季拉大到 30pp"),
    ]

    for (x0, x1, y0, y1, dim, series, panel_title) in panels:
        sub = cv[cv.dimension == dim]
        # ---- axes
        for k in range(5):
            v = k / 4
            yy = y1 - v * (y1 - y0)
            d.line([(x0, yy), (x1, yy)], fill="#eef1f4")
            d.text((x0 - 44, yy - 8), "%.0f%%" % (v * 100), font=font(12), fill=MUTED)
        for q in (0, 4, 8, 12, 16, 20, 24):
            xx = x0 + q / 24 * (x1 - x0)
            d.line([(xx, y1), (xx, y1 + 5)], fill=INK)
            d.text((xx - 8, y1 + 10), "%d" % q, font=font(12), fill=MUTED)
        d.line([(x0, y0), (x0, y1)], fill=INK, width=2)
        d.line([(x0, y1), (x1, y1)], fill=INK, width=2)
        d.text((x0, y0 - 28), panel_title, font=font(15, True), fill=INK)
        # axis titles
        lab = "首次观测后的季数"
        tw = d.textbbox((0, 0), lab, font=font(13))[2]
        d.text(((x0 + x1) / 2 - tw / 2, y1 + 34), lab, font=font(13), fill=MUTED)

        # ---- curves
        for label, col in series:
            g = sub[sub.group == label].sort_values("quarter")
            if g.empty:
                continue
            pts = [(x0 + q / 24 * (x1 - x0), y1 - s * (y1 - y0))
                   for q, s in zip(g.quarter, g.survival)]
            d.line(pts, fill=col, width=4)
        # ---- end labels, nudged apart only when the curves actually differ
        ends = []
        for label, col in series:
            g = sub[sub.group == label].sort_values("quarter")
            if g.empty:
                continue
            s24 = float(g.survival.iloc[-1])
            ends.append((s24, label, col))
        ends.sort()
        for i, (s24, label, col) in enumerate(ends):
            yy = y1 - s24 * (y1 - y0)
            d.text((x1 - 150, yy - 10 + i * 17), "%s %.0f%%" % (label.split("（")[0], s24 * 100),
                   font=font(12, True), fill=col)

    d.text((48, H - 92), "第 8 季与第 24 季仍未停止的比例：", font=font(14, True), fill=INK)
    # y-axis title for the left panel (rotated text is avoided; a label above works)
    d.text((60, 100), "仍未停止评论的比例", font=font(13), fill=MUTED)
    s8_del = {g: cv[(cv.dimension == "delivery") & (cv.group == g) & (cv.quarter == 8)].survival.iloc[0]
              for g in ("提供外送", "不提供外送")}
    s8_tr = cv[(cv.dimension == "stars_trend") & (cv.quarter == 8)].survival.max()
    d.text((48, H - 64),
           "外送  第8季 %.1f%% vs %.1f%%（差 %.1fpp）　｜　评分趋势 第8季 三组均为 %.1f%%（差 0.0pp）"
           % (s8_del["提供外送"] * 100, s8_del["不提供外送"] * 100,
              100 * (s8_del["提供外送"] - s8_del["不提供外送"]), s8_tr * 100),
           font=font(14), fill=ACCENT2)
    d.text((48, H - 36),
           "注：评分趋势的系数 HR 0.997，95% CI [0.980, 1.017] 跨过 1 —— 这是一个被证伪的假设，不是绘图失败",
           font=font(12), fill=MUTED)
    img.save(os.path.join(IMG, "fig31_hazard_survival_curves.png"))
    print("  rendered fig31_hazard_survival_curves.png")


# ------------------------------------------------------------------ fig32
def fig32():
    d = co[co.feature.str.startswith("band_")].copy()
    d["lo"] = d.feature.str.replace("band_", "").str.split("_").str[0].astype(int)
    d["hi"] = d.feature.str.replace("band_", "").str.split("_").str[1].astype(int)
    d = d.sort_values("lo")
    # the last band is the reference, implied HR = 1
    med = list((d.lo + d.hi) / 2) + [56]
    hr = list(d.hazard_ratio) + [1.0]
    ch = Chart(1180, 660,
               title="图 32｜基线风险：门店「安静下来」的概率随经营时长如何变化",
               subtitle="以第 49 季及以后为参照（HR=1）；其余变量固定，纵轴为各时段的相对风险",
               xlabel="首次观测后的季数（分段）", ylabel="风险比（相对 49 季以上）",
               pad_b=150)
    ch.set_xlim(0, max(med) + 6)
    ch.set_ylim(0, 3.0)
    ch.draw_frame(x_ticks=[2, 8, 14, 20, 28, 40, 56], x_fmt="{:.0f}",
                  y_ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], y_fmt="{:.1f}")
    for x, y in zip(med, hr):
        ch.bars([x], [y], width_frac=0.55, color=ACCENT if y < 1 else ACCENT2, y_base=0)
        ch.d.text((ch.px(x) - 14, ch.py(y) - 22), "%.2f" % y, font=font(12, True), fill=INK)
    ch.hline(1.0, color=MUTED)
    # reference label on the right, where no bar reaches
    ch.d.text((ch.px(43), ch.py(1.0) - 26), "参照水平 HR = 1", font=font(12), fill=MUTED)
    ch.footnote("风险在 5–8 季后达到峰值（2.73×），此后单调下降；首 4 季偏低，"
                "更可能是「首次被观测到」的左截断效应，而非真正的低风险")
    ch.save(os.path.join(IMG, "fig32_baseline_hazard.png"))


for fn in (fig30, fig31, fig32):
    print("---", fn.__name__)
    fn()
print("\nhazard charts rendered")
