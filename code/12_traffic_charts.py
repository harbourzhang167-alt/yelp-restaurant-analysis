"""Step 12 - charts for the traffic-as-outcome analysis.

fig21  traffic distribution (why a log model) + check-in platform decline caveat
fig22  what explains traffic - percent effect per +1 SD, bootstrap CI
fig23  the density tension: traffic AND closure both rise with competition
fig24  category scatter: traffic vs closure rate
fig25  traffic deciles vs closure rate (inverted U)
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import (ACCENT, ACCENT2, ACCENT3, GRID, INK, MUTED, WARN,  # noqa: E402
                      Chart, font)
from PIL import Image, ImageDraw  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
D = os.path.join(WS, "data", "bigquery")
IMG = os.path.join(OUT, "charts")
os.makedirs(IMG, exist_ok=True)

d = pd.read_csv(os.path.join(OUT, "07_traffic_features.csv"), low_memory=False)
coefs = pd.read_csv(os.path.join(OUT, "07_traffic_coefs.csv"))
dens = pd.read_csv(os.path.join(OUT, "07_traffic_density.csv"))
cats = pd.read_csv(os.path.join(OUT, "07_traffic_category.csv"))

LABEL = {
    "n_within_1_0km": "1 km 内竞争对手数",
    "pct_neg": "差评占比（1–2 星）",
    "alcohol_full_bar": "设有全酒吧",
    "alcohol_beer_wine": "供应啤酒/葡萄酒",
    "wifi_free": "提供免费 WiFi",
    "wifi_none": "标注“无 WiFi”",
    "wifi_paid": "提供收费 WiFi",
    "hours_days_listed": "营业时间登记完整度",
    "has_hours": "有营业时间信息",
    "hours_open_late": "营业至深夜",
    "is_outdoor_seating": "有户外座位",
    "is_delivery": "提供外送",
    "is_takeout": "提供外带",
    "is_has_tv": "店内有电视",
    "is_reservations": "接受预订",
    "is_caters": "承办宴席",
    "is_good_for_kids": "适合儿童",
    "is_accepts_credit_cards": "可刷信用卡",
    "is_bike_parking": "有自行车停车",
    "price_num": "价格档",
    "business_stars": "店铺星级",
    "std_stars": "星级波动",
    "city_restaurant_count": "城市餐厅总数",
    "mid_year": "活跃期中点年份",
}


def lab(name):
    if name in LABEL:
        return LABEL[name]
    if name.startswith("cat_"):
        return "品类：" + name[4:].replace("_", " ")
    return name


# ------------------------------------------------------------------ fig21
def fig21():
    c = Chart(1120, 700,
              title="图 21｜客流强度是长尾分布——所以模型用对数尺度",
              subtitle="51,318 家门店。客流强度 = 打卡总量 ÷ 活跃月数（中位 1.0 次/月，最高 278 次/月）",
              xlabel="客流强度（次/月，对数刻度）", ylabel="门店数", pad_b=135)
    v = d.ck_per_month.clip(upper=d.ck_per_month.quantile(0.999))
    logs = np.log10(v)
    hist, edges = np.histogram(logs, bins=44)
    c.set_xlim(logs.min(), logs.max())
    c.set_ylim(0, 1.05)
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    c.draw_frame(x_ticks=[np.log10(t) for t in ticks if logs.min() <= np.log10(t) <= logs.max()],
                 x_fmt="", y_ticks=[], grid=True)
    for t in ticks:
        if logs.min() <= np.log10(t) <= logs.max():
            tw = c.d.textbbox((0, 0), f"{t:g}", font=font(14))[2]
            c.d.text((c.px(np.log10(t)) - tw / 2, c.H - c.pad_b + 26), f"{t:g}",
                     font=font(14), fill=MUTED)
    mx = hist.max()
    cols = []
    for i in range(len(hist)):
        mid = (edges[i] + edges[i + 1]) / 2
        cols.append(ACCENT2 if mid > np.log10(10) else ACCENT)
    for i in range(len(hist)):
        c.bars([(edges[i] + edges[i + 1]) / 2], [hist[i] / mx], width_frac=0.92,
               color=cols[i], y_base=0)
    med = np.log10(d.ck_per_month.median())
    c.vline(med, color=WARN)
    # place the callout to the right of the line and below the peak, so it does not
    # sit on top of the tallest bar
    c.d.text((c.px(med) + 14, c.py(0.55)), "中位 1.0 次/月", font=font(14, True), fill=WARN)
    c.legend([("一般门店", ACCENT), ("高强度（>10 次/月）", ACCENT2)],
             x=c.plot[0] + 16, y=c.plot[1] + 14)
    c.footnote("注：打卡总量 2015 年达峰、2021 年下降约 70%，那是 Yelp 打卡功能使用率下降，不是门店客流下降——"
               "故不用“近期打卡”衡量客流趋势")
    c.save(os.path.join(IMG, "fig21_traffic_distribution.png"))


# ------------------------------------------------------------------ fig22
def fig22():
    c = coefs.copy()
    c = c.reindex(c.coef_std.abs().sort_values(ascending=True).index)
    keep = c.tail(16)
    H = 320 + len(keep) * 36
    ch = Chart(1200, H,
               title="图 22｜什么在解释客流？——结构特征的影响（每 +1 个标准差）",
               subtitle="OLS（因变量 = log 客流强度），控制其余变量；条形为百分比变化，误差棒为 600 次 bootstrap 的 95% CI。R² = 0.410",
               xlabel="客流强度的百分比变化（%）", ylabel="",
               pad_l=310, pad_r=90, pad_t=150, pad_b=160)
    lo = min(keep.ci_lo_pct.min(), -30) - 4
    hi = max(keep.ci_hi_pct.max(), 30) + 4
    ch.set_xlim(lo, hi)
    # margins so the top and bottom rows sit clear of the subtitle/axis
    ch.set_ylim(-0.75, len(keep) - 0.25)
    step = 10
    ticks = list(range(int(lo // step * step), int(hi // step * step) + step, step))
    ch.draw_frame(x_ticks=ticks, x_fmt="{:+.0f}%", y_ticks=[], grid=True)
    for i, (_, r) in enumerate(keep.iterrows()):
        col = ACCENT3 if r.pct_effect_per_SD > 0 else ACCENT2
        yy = ch.py(i)
        # chartkit.errorbar() draws a VERTICAL bar (y-direction); a forest plot
        # needs a HORIZONTAL CI, so it is drawn explicitly here.
        ch.d.line([(ch.px(r.ci_lo_pct), yy), (ch.px(r.ci_hi_pct), yy)], fill=INK, width=2)
        for v in (r.ci_lo_pct, r.ci_hi_pct):
            ch.d.line([(ch.px(v), yy - 6), (ch.px(v), yy + 6)], fill=INK, width=2)
        ch.d.line([(ch.px(0), yy), (ch.px(r.pct_effect_per_SD), yy)], fill=col, width=4)
        ch.marker(r.pct_effect_per_SD, i, r=7, color=col)
        l = lab(r.feature)
        tw = ch.d.textbbox((0, 0), l, font=font(15))[2]
        ch.d.text((ch.px(ch.xlim[0]) - tw - 14, yy - 10), l, font=font(15), fill=INK)
        s = f"{r.pct_effect_per_SD:+.1f}%"
        sw = ch.d.textbbox((0, 0), s, font=font(14, True))[2]
        # put the value label clear of the CI end cap, not on top of it
        if r.pct_effect_per_SD > 0:
            tx = ch.px(r.ci_hi_pct) + 12
        else:
            tx = ch.px(r.ci_lo_pct) - sw - 12
        ch.d.text((tx, yy - 10), s, font=font(14, True), fill=col)
    ch.vline(0, color=MUTED)
    # legend in the empty lower-left corner, away from the top rows' labels
    ch.legend([("与更高客流相关", ACCENT3), ("与更低客流相关", ACCENT2)],
              x=ch.plot[0] + 24, y=ch.plot[3] - 96)
    ch.footnote("注：属性来自门店的挂牌信息，反映的是“什么样的店”而非“加了这项会增加多少客流”，不可当因果杠杆")
    ch.save(os.path.join(IMG, "fig22_traffic_drivers.png"))


# ------------------------------------------------------------------ fig23
def fig23():
    """Two stacked panels sharing the x categories.

    A dual axis let the two lines cross, which reads as "closure falls below
    traffic" - meaningless, since the scales differ. Stacking removes that
    ambiguity: both panels rise, left to right.
    """
    e = dens.copy()
    W, H = 1180, 820
    img = Image.new("RGB", (W, H), "#ffffff")
    d2 = ImageDraw.Draw(img)
    d2.text((44, 22), "图 23｜密度悖论：越密集的地方客流越高，关闭率也越高",
            font=font(25, True), fill=INK)
    d2.text((44, 60), "按门店 1 公里内竞争对手数分十分位（D1 最稀疏 → D10 最密集）；上下两幅共用横轴",
            font=font(15), fill=MUTED)

    x0, x1 = 110, W - 70
    n = len(e)
    def pxp(i): return x0 + (i + 0.5) / n * (x1 - x0)

    panels = [
        # (top, bottom, values, label, colour, fmt)
        (130, 430, e.median_traffic_per_month, "客流强度（次/月）", ACCENT, "{:.1f}"),
        (480, 740, e.closure_rate, "关闭率", ACCENT2, "{:.0%}"),
    ]
    for top, bot, vals, ylab, col, fmt in panels:
        vmax = vals.max() * 1.28
        def pyv(v, top=top, bot=bot, vmax=vmax):
            return bot - v / vmax * (bot - top)
        # grid + y ticks
        for k in range(5):
            v = vmax * k / 4
            yy = pyv(v)
            d2.line([(x0, yy), (x1, yy)], fill="#eef1f4")
            d2.text((x0 - 14 - 40, yy - 8), fmt.format(v), font=font(12), fill=MUTED)
        d2.line([(x0, top), (x0, bot)], fill=INK, width=2)
        d2.line([(x0, bot), (x1, bot)], fill=INK, width=2)
        d2.text((x0, top - 28), ylab, font=font(15, True), fill=col)
        pts = [(pxp(i), pyv(v)) for i, v in enumerate(vals)]
        d2.line(pts, fill=col, width=5)
        for (xx, yy), v in zip(pts, vals):
            d2.ellipse([xx - 6, yy - 6, xx + 6, yy + 6], fill=col, outline="#ffffff", width=2)
        # end-value callouts
        d2.text((pxp(0) + 10, pyv(vals.iloc[0]) - 22), fmt.format(vals.iloc[0]),
                font=font(13, True), fill=col)
        d2.text((pxp(n - 1) - 52, pyv(vals.iloc[-1]) - 22), fmt.format(vals.iloc[-1]),
                font=font(13, True), fill=col)

    # x labels (shared)
    for i, r in e.iterrows():
        xx = pxp(i)
        tw = d2.textbbox((0, 0), f"{r.avg_neighbours_1km:.0f}", font=font(12))[2]
        d2.text((xx - tw / 2, 748), f"{r.avg_neighbours_1km:.0f}", font=font(12), fill=MUTED)
        d2.text((xx - 9, 768), f"D{i+1}", font=font(12, True), fill=INK)
    d2.text((x0, 792), "1 公里内竞争对手数（每档平均）", font=font(14, True), fill=INK)
    img.save(os.path.join(IMG, "fig23_density_tension.png"))
    print("  rendered fig23_density_tension.png")


# ------------------------------------------------------------------ fig24
def fig24():
    c = Chart(1300, 840,
              title="图 24｜品类坐标：客流高的品类，关闭率也偏高（并非低客流＝高风险）",
              subtitle="20 个主要品类；横轴 = 相对其他品类的客流强度，纵轴 = 关闭率；点大小 = 门店数。"
                       "品类层面 Spearman ρ = +0.66",
              xlabel="相对其他品类的客流强度（%）", ylabel="关闭率", pad_b=150, pad_r=70)
    lo, hi = cats.vs_others_pct.min() - 14, cats.vs_others_pct.max() + 20
    c.set_xlim(lo, hi)
    c.set_ylim(0.08, 0.52)
    c.draw_frame(x_ticks=[-50, 0, 50, 100, 150], x_fmt="{:+.0f}%", y_fmt="{:.0%}")
    mx = cats.n.max()
    placed = []
    for _, r in cats.iterrows():
        rad = 4 + 11 * (r.n / mx) ** 0.5
        col = ACCENT2 if r.closure_rate > 0.33 else (ACCENT3 if r.closure_rate < 0.28 else ACCENT)
        cx, cy = c.px(r.vs_others_pct), c.py(r.closure_rate)
        c.d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                    fill=col, outline="#ffffff", width=2)
    # labels last, with greedy vertical offset so they do not collide
    for _, r in cats.sort_values("n", ascending=False).iterrows():
        rad = 4 + 11 * (r.n / mx) ** 0.5
        cx, cy = c.px(r.vs_others_pct), c.py(r.closure_rate)
        name = r.category.replace("_", " ")
        f = font(12, True)
        tw = c.d.textbbox((0, 0), name, f)[2]
        for dy in (-(rad + 16), rad + 4, -(rad + 30), rad + 18, -(rad + 44)):
            bb = (cx - tw / 2 - 2, cy + dy - 1, cx + tw / 2 + 2, cy + dy + 15)
            if any(not (bb[2] < u[0] or bb[0] > u[2] or bb[3] < u[1] or bb[1] > u[3])
                   for u in placed):
                continue
            placed.append(bb)
            c.d.text((cx - tw / 2, cy + dy), name, font=f, fill=INK)
            break
    c.hline(0.327, color=WARN)
    c.d.text((c.px(c.xlim[0]) + 10, c.py(0.327) + 6), "整体关闭率 32.7%",
             font=font(13, True), fill=WARN)
    c.legend([("关闭率 >33%", ACCENT2), ("28–33%", ACCENT), ("<28%", ACCENT3)],
             x=c.px(c.xlim[1]) - 190, y=c.plot[3] - 110)
    c.footnote("读法：Fast Food 客流最低（−56%）关闭率也最低（14.7%）；Nightlife 客流最高（+143%）关闭率 33.5%；"
               "Indian 是反例（客流仅 +20% 但关闭率 45.4%）")
    c.save(os.path.join(IMG, "fig24_category_traffic_closure.png"))


# ------------------------------------------------------------------ fig25
def fig25():
    t = d.copy()
    t["dec"] = pd.qcut(t.ck_per_month, 10, labels=False, duplicates="drop")
    g = t.groupby("dec").agg(n=("is_open", "size"), med=("ck_per_month", "median"),
                             closure=("is_open", lambda s: 1 - s.mean())).reset_index()
    c = Chart(1150, 720,
              title="图 25｜客流与存活不是线性关系：中间档最危险",
              subtitle=f"按客流强度十分位分组（各档中位客流 {g.med.min():.2f} → {g.med.max():.2f} 次/月）；"
                       "最低档关闭率仅 15%，中高档升至 40%，最高档略回落",
              xlabel="客流强度十分位（1 = 最低）", ylabel="关闭率", pad_b=185)
    c.set_xlim(0.4, 10.6)
    c.set_ylim(0, 0.50)
    c.draw_frame(x_ticks=list(range(1, 11)), x_fmt="{:.0f}", y_fmt="{:.0%}")
    cols = [ACCENT3] * 2 + [ACCENT] * 4 + [ACCENT2] * 4
    c.bars(list(range(1, 11)), g.closure.values, colors=cols, y_base=0)
    for i, h in enumerate(g.closure.values):
        # value above the bar only; the median-traffic figures live in the subtitle
        # (drawn under the bars they collided with the x tick labels)
        c.d.text((c.px(i + 1) - 18, c.py(h) - 22), f"{h:.0%}", font=font(13, True), fill=INK)
    avg = 1 - d.is_open.mean()
    c.hline(avg, color=WARN)
    c.d.text((c.pad_l + 6, c.py(avg) - 24), f"整体 {avg:.1%}", font=font(14, True), fill=WARN)
    c.legend([("低客流（稳定小店型）", ACCENT3), ("中等", ACCENT), ("高客流", ACCENT2)],
             x=c.plot[0] + 16, y=c.plot[1] + 12)
    c.footnote("与星级发现同构：两端的店更稳，中间地带（竞争最激烈、类型最同质）最危险")
    c.save(os.path.join(IMG, "fig25_traffic_deciles_closure.png"))


for fn in (fig21, fig22, fig23, fig24, fig25):
    print(f"--- {fn.__name__}")
    fn()
print("\ntraffic charts rendered")
