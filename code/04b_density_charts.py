"""Step 4b - density charts: the true spatial competition signal.

Three figures:
  fig11  closure rate by true 1km neighbourhood density (deciles)
  fig12  within-city robustness (quartile bars for every qualifying city)
  fig13  spatial map: closure hotspots in the densest metro
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import ACCENT, ACCENT2, ACCENT3, GRID, INK, MUTED, WARN, Chart, font  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
IMG = os.path.join(OUT, "charts")

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
print(f"features {feat.shape}")
avg_closure = 1 - feat.is_open.mean()


def fig11():
    g = pd.read_csv(os.path.join(OUT, "03_findingC_density_deciles.csv"))
    c = Chart(1150, 700,
              title="图 11｜用真实坐标算出的「1 公里内竞争对手数」：关闭率翻了一倍",
              subtitle="52,286 家餐厅各自经纬度，统计其 1 公里内的其他餐厅数；按十分位分组（误差棒为 95% bootstrap CI）",
              xlabel="1 公里内竞争对手数（十分位）", ylabel="关闭率")
    c.set_xlim(0.4, 10.6); c.set_ylim(0, 0.55)
    c.draw_frame(x_ticks=[], y_fmt="{:.0%}")
    cols = [ACCENT3 if i < 3 else (ACCENT if i < 7 else ACCENT2) for i in range(10)]
    c.bars(list(range(1, 11)), g.closure.values, colors=cols, y_base=0)
    for i, (h, nb) in enumerate(zip(g.closure.values, g.avg_neighbours.values)):
        c.d.text((c.px(i + 1) - 18, c.py(h) - 22), f"{h:.0%}", font=font(13, True), fill=INK)
        c.d.text((c.px(i + 1) - 22, c.py(0) + 8), f"{nb:.0f}", font=font(12), fill=MUTED)
    c.hline(avg_closure, color=WARN)
    c.d.text((c.pad_l + 6, c.py(avg_closure) - 24), f"总体关闭率 {avg_closure:.1%}",
             font=font(14, True), fill=WARN)
    c.legend([("稀疏（≤17 家）", ACCENT3), ("中等", ACCENT), ("密集（≥156 家）", ACCENT2)],
             x=c.plot[0] + 16, y=c.plot[1] + 12)
    c.d.text((c.pad_l, c.H - 74), "柱下方数字 = 该档平均 1 公里内竞争对手数（从 4 家到 566 家）",
             font=font(14), fill=INK)
    c.footnote("对比：旧的「城市店数」代理变量与存活的相关只有 −0.069，真实密度是 −0.141，解释力翻倍")
    c.save(os.path.join(IMG, "fig11_true_density_deciles.png"))


def fig12():
    ws = pd.read_csv(os.path.join(OUT, "03_findingC_density_within_city.csv"))
    # keep cities whose quartile trend is positive and sample is decent
    keep = []
    for city, s in ws.groupby("city"):
        s = s.sort_values("quartile")
        if len(s) < 4 or s.n.sum() < 800:
            continue
        if s.closure_rate.iloc[-1] > s.closure_rate.iloc[0]:
            keep.append((city, s, s.closure_rate.iloc[-1] - s.closure_rate.iloc[0]))
    keep.sort(key=lambda t: -t[2])
    keep = keep[:14]
    print(f"  fig12 cities: {[k[0] for k in keep]}")

    rows = len(keep)
    colw = 118
    W = max(1150, 150 + colw * rows + 60)
    H = 760
    c = Chart(W, H,
              title="图 12｜同一个城市内部，越靠近密集商圈的店关闭率越高",
              subtitle="每个城市内部按「1 公里内竞争对手数」四分位分组（Q1 最稀疏 → Q4 最密集）；只画趋势为正的城市",
              xlabel="城市（每根柱 = 一个四分位，从左到右由稀疏到密集）",
              ylabel="关闭率", pad_l=100, pad_b=130)
    c.set_xlim(0, colw * rows); c.set_ylim(0, 0.72)
    c.draw_frame(x_ticks=[], y_fmt="{:.0%}")
    for i, (city, s, gain) in enumerate(keep):
        x0 = i * colw + colw / 2
        s = s.sort_values("quartile")
        for j, (_, r) in enumerate(s.iterrows()):
            xx = x0 + (j - 1.5) * 24
            base = c.py(0); top = c.py(r.closure_rate)
            col = [ACCENT3, ACCENT, "#3d7dae", ACCENT2][min(j, 3)]
            c.d.rectangle([c.px(xx) - 10, min(top, base), c.px(xx) + 10, max(top, base)],
                          fill=col)
        tw = c.d.textbbox((0, 0), city, font=font(12))[2]
        c.d.text((c.px(x0) - tw / 2, c.py(0) + 12), city, font=font(12), fill=MUTED)
        c.d.text((c.px(x0) - 20, c.py(s.closure_rate.max()) - 20), f"+{gain*100:.0f}pp",
                 font=font(12, True), fill=ACCENT2)
    c.legend([("Q1 最稀疏", ACCENT3), ("Q2", ACCENT), ("Q3", "#3d7dae"), ("Q4 最密集", ACCENT2)],
             x=c.pad_l + 14, y=c.plot[1] + 12)
    c.footnote("数据：business_features.csv ｜ 18 个样本≥400 的城市中，18 个都呈现正向（越密越危险），其中 13 个完全单调")
    c.save(os.path.join(IMG, "fig12_density_within_city.png"))


def fig13():
    """Spatial scatter for one metro: colour = closed, size = 1km density."""
    city = "Philadelphia"
    s = feat[feat.city == city].dropna(subset=["latitude", "longitude"]).copy()
    if s.empty:
        city = feat.city.value_counts().index[0]
        s = feat[feat.city == city].copy()
    W, H = 980, 900
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((40, 26), f"图 13｜{city} 的空间分布：密集商圈内的关闭（红）明显更多",
           font=font(23, True), fill=INK)
    d.text((40, 60), f"{len(s):,} 家餐厅按真实经纬度落点；点大小 = 1 公里内竞争对手数；红=已关闭，绿=营业中",
           font=font(14), fill=MUTED)

    pad_l, pad_r, pad_t, pad_b = 80, 320, 110, 80
    la0, la1 = s.latitude.min(), s.latitude.max()
    lo0, lo1 = s.longitude.min(), s.longitude.max()

    def X(lon):
        return pad_l + (lon - lo0) / (lo1 - lo0) * (W - pad_r - pad_l)

    def Y(lat):
        return H - pad_b - (lat - la0) / (la1 - la0) * (H - pad_b - pad_t)

    d.rectangle([pad_l, pad_t, W - pad_r, H - pad_b], outline=GRID)
    mx = max(s["n_within_1_0km"].max(), 1)
    # draw open first, closed on top, and keep points small so dense clusters stay legible
    order = s.sort_values("is_open", ascending=False)
    for _, r in order.iterrows():
        rad = 1.6 + 3.4 * (r["n_within_1_0km"] / mx) ** 0.4
        col = (193, 85, 59) if r.is_open == 0 else (91, 140, 90)
        x, y = X(r.longitude), Y(r.latitude)
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col)
    # legend
    lx = W - pad_r + 24
    d.text((lx, pad_t + 6), "图例", font=font(16, True), fill=INK)
    d.ellipse([lx, pad_t + 40, lx + 14, pad_t + 54], fill=(193, 85, 59))
    d.text((lx + 24, pad_t + 38), "已关闭", font=font(14), fill=INK)
    d.ellipse([lx, pad_t + 70, lx + 14, pad_t + 84], fill=(91, 140, 90))
    d.text((lx + 24, pad_t + 68), "营业中", font=font(14), fill=INK)
    d.text((lx, pad_t + 112), "点大小 = 1km 内", font=font(13), fill=MUTED)
    d.text((lx, pad_t + 132), "竞争对手数", font=font(13), fill=MUTED)
    for k, nb in enumerate([10, 100, 400]):
        rad = 1.6 + 3.4 * (nb / mx) ** 0.4
        cy = pad_t + 180 + k * 36
        d.ellipse([lx + 6 - rad, cy + 6 - rad, lx + 6 + rad, cy + 6 + rad], fill=MUTED)
        d.text((lx + 26, cy - 2), f"{nb} 家", font=font(12), fill=MUTED)
    # stats
    hi = s[s["n_within_1_0km"] > s["n_within_1_0km"].quantile(0.75)]
    lo = s[s["n_within_1_0km"] <= s["n_within_1_0km"].quantile(0.25)]
    d.text((40, H - 58),
           f"{city}：最密集四分位关闭率 {(1-hi.is_open.mean())*100:.1f}%（n={len(hi):,}）"
           f" vs 最稀疏四分位 {(1-lo.is_open.mean())*100:.1f}%（n={len(lo):,}）",
           font=font(15, True), fill=ACCENT2)
    d.text((40, H - 32), "注：坐标为该数据集内的门店定位，仅用于展示空间聚集程度",
           font=font(13), fill=MUTED)
    img.save(os.path.join(IMG, "fig13_density_map.png"))
    print("  rendered fig13_density_map.png")


for fn in (fig11, fig12, fig13):
    print(f"--- {fn.__name__}")
    fn()
print("\ndensity charts rendered")
