"""Step 4 - render the evidence charts with Pillow.

Each chart is written to analysis/out/ and then visually inspected.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import (ACCENT, ACCENT2, ACCENT3, GRID, INK, MUTED, WARN, Chart, font)  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
IMG = os.path.join(OUT, "charts")
os.makedirs(IMG, exist_ok=True)

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
print(f"features {feat.shape}")


# ==========================================================================
def chart01_overview():
    """Distribution: review count (long tail) and closure rate by decile."""
    c = Chart(1150, 700,
              title="图 1｜关闭率随评论量单调下降：从 47% 降到 15%",
              subtitle="52,286 家餐厅，按评论数十分位分组（D1 = 评论最少，D10 = 最多）；D10 平均 455 条评论",
              xlabel="评论数十分位", ylabel="关闭率")
    d = feat.copy()
    d["dec"] = pd.qcut(d.business_review_count, 10, labels=False, duplicates="drop")
    g = d.groupby("dec").agg(avg=("business_review_count", "mean"),
                             closure=("is_open", lambda s: 1 - s.mean())).reset_index()
    c.set_xlim(0.4, 10.6)
    c.set_ylim(0, 0.55)
    c.draw_frame(x_ticks=list(range(1, 11)), y_fmt="{:.0%}", x_fmt="{:.0f}")
    heights = g.closure.values
    cols = [ACCENT2 if i == 0 or i == 9 else ACCENT for i in range(10)]
    c.bars(list(range(1, 11)), heights, colors=cols, y_base=0)
    # value labels + the two extremes
    for i, (cen, h, av) in enumerate(zip(range(1, 11), heights, g.avg.values)):
        c.d.text((c.px(cen) - 18, c.py(h) - 22), f"{h:.0%}", font=font(13, True), fill=INK)
    c.d.text((c.px(1) - 30, c.py(heights[0]) - 48), f"{g.avg.values[0]:.0f} 条", font=font(12), fill=MUTED)
    c.d.text((c.px(10) - 40, c.py(heights[9]) - 48), f"{g.avg.values[9]:.0f} 条", font=font(12), fill=MUTED)
    c.legend([("关闭率", ACCENT), ("最低/最高档（重点）", ACCENT2)], x=c.plot[0] + 14, y=c.plot[1] + 12)
    c.footnote("数据：cleaned_restaurants + final_performance_master ｜ 关闭率 = 1 − is_open 均值")
    c.save(os.path.join(IMG, "fig01_reviewcount_closure.png"))


def chart02_corr():
    """Pearson vs Spearman side by side - why the choice of metric matters."""
    cols = ["business_stars", "business_review_count", "total_checkins", "n_tips",
            "price_num", "city_restaurant_count", "is_open"]
    label_map = {"business_stars": "星级", "business_review_count": "评论数",
                 "total_checkins": "打卡量", "n_tips": "tips数", "price_num": "价格档",
                 "city_restaurant_count": "城市店数", "is_open": "存活"}
    sub = feat[cols].apply(pd.to_numeric, errors="coerce").dropna()
    pear = sub.corr(method="pearson")
    spear = sub.corr(method="spearman")

    W, H = 1120, 2240
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((64, 28), "图 2｜同一批数据，Pearson 与 Spearman 差很多——必须两个都看",
           font=font(25, True), fill=INK)
    d.text((64, 66), "店铺级 52,286 家。看「存活」这一行：星级几乎无色（≈0），评论数/打卡量明显有色",
           font=font(15), fill=MUTED)

    cell = 128
    n = len(cols)
    # stacked vertically: two 7x7 grids cannot share a row at a readable size
    for gi, (title, mat, oy) in enumerate([("Pearson", pear, 170),
                                           ("Spearman", spear, 170 + n * cell + 220)]):
        ox = 150
        d.text((ox, oy - 40), title, font=font(20, True), fill=INK)
        for i in range(n):
            for j in range(n):
                v = mat.iloc[i, j]
                mag = min(abs(v), 1.0)
                base = (47, 111, 159) if v >= 0 else (193, 85, 59)
                col = tuple(int(255 - (255 - cc) * mag) for cc in base)
                x0, y0 = ox + j * cell, oy + i * cell
                d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=col, outline="#ffffff")
                if i != j:
                    tc = "#ffffff" if mag > 0.45 else INK
                    s = f"{v:+.2f}".replace("+", "").replace("-0", "−0")
                    tw = d.textbbox((0, 0), s, font=font(17))[2]
                    d.text((x0 + cell / 2 - tw / 2, y0 + cell / 2 - 11), s,
                           font=font(17), fill=tc)
        for i in range(n):
            lab = label_map[cols[i]]
            tw = d.textbbox((0, 0), lab, font=font(16))[2]
            d.text((ox - tw - 14, oy + i * cell + cell / 2 - 11), lab, font=font(16), fill=INK)
        for j in range(n):
            lab = label_map[cols[j]]
            tw = d.textbbox((0, 0), lab, font=font(16))[2]
            d.text((ox + j * cell + cell / 2 - tw / 2, oy + n * cell + 10), lab,
                   font=font(16), fill=INK)

    d.text((64, H - 56), "数据：business_features.csv ｜ 正相关偏蓝、负相关偏红，颜色深浅 = 强度",
           font=font(15), fill=MUTED)
    d.text((64, H - 32), "注：Pearson 低估了非单调与重尾变量的关联，因此报告中两者并列",
           font=font(15), fill=MUTED)
    img.save(os.path.join(IMG, "fig02_corr_pearson_vs_spearman.png"))
    print("  rendered fig02_corr_pearson_vs_spearman.png")


def chart03_stars_nonmonotonic():
    """Stars vs closure is U-shaped, not monotonic - the honest picture."""
    d = feat.copy()
    d["bin"] = pd.cut(d.business_stars, [0, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    g = d.groupby("bin", observed=True).agg(n=("is_open", "size"),
                                            closure=("is_open", lambda s: 1 - s.mean())).reset_index()
    g["lab"] = ["≤2.0", "2.0–2.5", "2.5–3.0", "3.0–3.5", "3.5–4.0", "4.0–4.5", "4.5–5.0"]
    c = Chart(1100, 680,
              title="图 3｜星级与关闭率不是单调关系，而是中间高、两头低",
              subtitle="52,286 家餐厅。中间档（2.5–3.5 星）最危险，最低分档反而更安全",
              xlabel="店铺星级区间", ylabel="关闭率")
    c.set_xlim(0.5, 7.5)
    c.set_ylim(0, 0.45)
    c.draw_frame(x_ticks=[], y_fmt="{:.0%}")
    # descriptive band labels only (no numeric ticks)
    for i, lab in enumerate(g.lab):
        tw = c.d.textbbox((0, 0), lab, font=font(14))[2]
        c.d.text((c.px(i + 1) - tw / 2, c.H - c.pad_b + 12), lab, font=font(14), fill=MUTED)
    c.bars(list(range(1, 8)), g.closure.values, color=ACCENT, y_base=0)
    for i, (h, n) in enumerate(zip(g.closure.values, g.n.values)):
        c.d.text((c.px(i + 1) - 18, c.py(h) - 22), f"{h:.0%}", font=font(13, True), fill=INK)
        c.d.text((c.px(i + 1) - 22, c.py(h) + 6), f"n={n:,}", font=font(11), fill=MUTED)
    avg = 1 - feat.is_open.mean()
    c.hline(avg, color=WARN)
    c.d.text((c.plot[2] - 190, c.py(avg) - 22), f"总体关闭率 {avg:.1%}", font=font(14), fill=WARN)
    c.footnote("数据：business_features.csv ｜ 「≤2.0 星」档存活率高，多为有固定客源的老店（选择效应）")
    c.save(os.path.join(IMG, "fig03_stars_nonmonotonic.png"))


def chart04_delivery_paradox():
    """The headline: delivery availability dominates; better-rated venues close more."""
    fb = pd.read_csv(os.path.join(OUT, "03_findingB_univariate.csv"))
    fb = fb.sort_values("diff")
    c = Chart(1150, 720,
              title="图 4｜「可达性」压倒「口碑」：能配送的店关闭率低 28.6 个百分点",
              subtitle="横轴为 关闭率(有该特征) − 关闭率(无该特征)，负值=该特征与更低关闭率相关；误差棒为 95% bootstrap CI",
              xlabel="关闭率之差（百分点）", ylabel="", pad_b=125)
    lo, hi = fb.lo.min(), fb.hi.max()
    c.set_xlim(lo - 0.03, hi + 0.03)
    c.set_ylim(-0.6, len(fb) - 0.4)
    n = len(fb)
    c.pad_l = 250
    c.draw_frame(x_ticks=[x / 100 for x in range(-30, 16, 5)], y_ticks=[], grid=True,
                 x_fmt="{:+.0%}")
    name_map = {"delivery": "外送 delivery", "takeout": "外带 takeout", "has_tv": "有电视 has_tv",
                "reservations": "接受预订", "outdoor_seating": "户外座位", "caters": "承办宴席 caters",
                "good_for_kids": "适合儿童", "accepts_credit_cards": "可刷卡",
                "bike_parking": "自行车停车"}
    for i, (_, r) in enumerate(fb.iterrows()):
        yy = i
        # semantics: a negative diff means LOWER closure = safer = green;
        # a positive diff means HIGHER closure = riskier = red.
        col = ACCENT3 if r["diff"] < 0 else ACCENT2
        c.d.line([(c.px(0), c.py(yy)), (c.px(r["diff"]), c.py(yy))], fill=col, width=3)
        c.errorbar(r["diff"], yy, r.lo, r.hi, color=INK, width=2, cap=6)
        c.marker(r["diff"], yy, r=7, color=col)
        lab = name_map.get(r.attribute, r.attribute)
        tw = c.d.textbbox((0, 0), lab, font=font(15))[2]
        c.d.text((c.px(c.xlim[0]) - tw - 12, c.py(yy) - 10), lab, font=font(15), fill=INK)
        s = f"{r['diff']*100:+.1f}pp"
        c.d.text((c.px(r["diff"]) + (12 if r["diff"] > 0 else -70), c.py(yy) - 10), s,
                 font=font(14, True), fill=col)
    c.vline(0, color=MUTED)
    # Legend goes to the PLOT'S TOP-LEFT: every positive-diff bar sits on the
    # right half, so the top-left is empty. It used to be pinned at a hardcoded
    # pixel position that landed across the bike_parking row and hid its bar.
    c.legend([("该特征与更低关闭率相关（更安全）", ACCENT3),
              ("该特征与更高关闭率相关（更危险）", ACCENT2)],
             x=c.plot[0] + 16, y=c.plot[1] + 10)
    c.footnote("关键对照：delivery=True 组平均星级 3.41，delivery=False 组 3.61 —— 口碑更好的一组反而更容易倒")
    c.save(os.path.join(IMG, "fig04_delivery_paradox.png"))


def chart05_forest():
    """Adjusted odds ratios per +1 SD from the logistic model."""
    co = pd.read_csv(os.path.join(OUT, "03_model_coefficients.csv"))
    co = co.sort_values("std_coef")
    label = {"is_delivery": "外送 delivery", "log_reviews": "评论量 log", "is_has_tv": "有电视",
             "is_reservations": "接受预订", "hours_open_late": "营业到深夜", "city_restaurant_count": "城市店数",
             "price_num": "价格档", "has_hours": "有营业时间", "is_caters": "承办宴席",
             "is_bike_parking": "自行车停车", "is_takeout": "外带", "is_accepts_credit_cards": "可刷卡",
             "is_outdoor_seating": "户外座位", "is_good_for_kids": "适合儿童",
             "std_stars": "星级波动", "pct_neg": "差评占比", "city_cat_count": "同城同品类店数"}
    co["lab"] = co.feature.map(lambda f: label.get(f, f))
    # odds ratio axis, log scale via manual mapping
    c = Chart(1150, 760,
              title="图 5｜多变量 logistic：控制彼此与地理后，哪些因素独立影响存活",
              subtitle="纵轴为每个特征 +1 个标准差的优势比（OR>1 表示存活几率更高）；模型 AUC 0.763",
              xlabel="优势比 Odds Ratio（对数刻度，1.0 = 无关联）", ylabel="", pad_b=125)
    c.pad_l = 250
    ors = co.odds_ratio_per_SD.values
    lx = np.log(ors)
    lo, hi = min(lx.min(), -0.45), max(lx.max(), 0.45)
    c.set_xlim(lo, hi)
    c.set_ylim(-0.6, len(co) - 0.4)
    ticks = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    c.draw_frame(x_ticks=[np.log(t) for t in ticks], y_ticks=[], grid=True,
                 x_fmt="")
    for t in ticks:
        c.d.text((c.px(np.log(t)) - 12, c.H - c.pad_b + 26), f"{t:g}", font=font(14), fill=MUTED)
    for i, (_, r) in enumerate(co.iterrows()):
        col = ACCENT3 if r.std_coef > 0 else ACCENT2
        c.d.line([(c.px(0), c.py(i)), (c.px(r.std_coef), c.py(i))], fill=col, width=3)
        c.marker(r.std_coef, i, r=7, color=col)
        tw = c.d.textbbox((0, 0), r.lab, font=font(15))[2]
        c.d.text((c.px(c.xlim[0]) - tw - 12, c.py(i) - 10), r.lab, font=font(15), fill=INK)
        c.d.text((c.px(r.std_coef) + (10 if r.std_coef > 0 else -52), c.py(i) - 10),
                 f"{r.odds_ratio_per_SD:.2f}", font=font(14, True), fill=col)
    c.vline(0, color=MUTED)
    c.legend([("OR>1：存活几率更高", ACCENT3), ("OR<1：存活几率更低", ACCENT2)],
             x=760, y=430)
    c.footnote("注：系数方向按「预测存活」输出；OR 为标准化系数取指数，非因果效应")
    c.save(os.path.join(IMG, "fig05_model_forest.png"))


def chart06_auc():
    """ROC comparison: full model vs stars-only baseline."""
    model_cols = ["is_delivery", "is_takeout", "is_has_tv", "is_reservations",
                  "is_outdoor_seating", "is_caters", "is_good_for_kids",
                  "is_accepts_credit_cards", "is_bike_parking",
                  "price_num", "log_reviews", "std_stars", "pct_neg",
                  "has_hours", "hours_open_late",
                  "city_restaurant_count", "city_cat_count"]
    M = feat[model_cols + ["is_open"]].copy()
    for c_ in model_cols:
        M[c_] = M[c_].fillna(M[c_].mean())
    y = M.is_open.values
    mu, sd = M[model_cols].mean(), M[model_cols].std().replace(0, 1)
    Xs = (M[model_cols] - mu) / sd

    def irls(X, y, ridge=1.0):
        n, p = X.shape
        Xd = np.hstack([np.ones((n, 1)), X])
        beta = np.zeros(p + 1)
        pen = np.eye(p + 1) * ridge
        pen[0, 0] = 0
        for _ in range(60):
            eta = np.clip(Xd @ beta, -30, 30)
            m = 1 / (1 + np.exp(-eta))
            w = np.clip(m * (1 - m), 1e-9, None)
            z = eta + (y - m) / w
            A = Xd.T @ (Xd * w[:, None]) + pen
            b = Xd.T @ (w * z)
            beta = np.linalg.solve(A, b)
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
        return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    def roc(y, s):
        pts = []
        for thr in np.unique(np.quantile(s, np.linspace(0, 1, 300))):
            pred = s >= thr
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            tn = ((pred == 0) & (y == 0)).sum()
            pts.append((fp / (fp + tn), tp / (tp + fn)))
        pts.append((0, 0)); pts.append((1, 1))
        return sorted(pts)

    full = irls(Xs.values, y)
    p_full = 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(y), 1)), Xs.values]) @ full, -30, 30)))
    vol_mu, vol_sd = M[["log_reviews"]].mean().iloc[0], M[["log_reviews"]].std().iloc[0]
    Xv = ((M[["log_reviews"]] - vol_mu) / vol_sd).values
    pv = 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(y), 1)), Xv]) @ irls(Xv, y), -30, 30)))
    st_mu, st_sd = feat.business_stars.mean(), feat.business_stars.std()
    Xst = ((feat[["business_stars"]].fillna(st_mu) - st_mu) / st_sd).values
    ps = 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(y), 1)), Xst]) @ irls(Xst, y), -30, 30)))

    c = Chart(900, 800, title="图 6｜完整模型 vs 只用星级的预警能力",
              subtitle="ROC 曲线，越靠左上越好；AUC 0.5 = 与随机猜测无异",
              xlabel="假阳性率（误报为存活）", ylabel="真阳性率（正确识别存活）",
              pad_l=110, pad_b=110)
    c.set_xlim(0, 1); c.set_ylim(0, 1)
    c.draw_frame(n_xticks=6, n_yticks=6, x_fmt="{:.0%}", y_fmt="{:.0%}")
    c.d.line([(c.px(0), c.py(0)), (c.px(1), c.py(1))], fill=MUTED, width=2)
    c.d.text((c.px(0.62), c.py(0.56)), "随机猜测", font=font(14), fill=MUTED)
    for pts, col, lab, w in [
        (roc(y, ps), ACCENT2, f"仅星级 AUC {auc(y, ps):.3f}", 3),
        (roc(y, pv), ACCENT3, f"仅评论量 AUC {auc(y, pv):.3f}", 3),
        (roc(y, p_full), ACCENT, f"完整模型 AUC {auc(y, p_full):.3f}", 4),
    ]:
        c.line(pts, color=col, width=w)
    c.legend([(f"仅星级  {auc(y, ps):.3f}", ACCENT2, "line"),
              (f"仅评论量 {auc(y, pv):.3f}", ACCENT3, "line"),
              (f"完整模型 {auc(y, p_full):.3f}", ACCENT, "line")],
             x=c.px(0.40), y=c.py(0.30))
    c.footnote("结论：仅用星级的预警能力接近随机（AUC 0.52），加入可达性、活跃度与拥挤度后升至 0.76")
    c.save(os.path.join(IMG, "fig06_roc.png"))


def chart07_crowding():
    """Crowding dose-response plus the within-state robustness check."""
    fc = pd.read_csv(os.path.join(OUT, "03_findingC_crowding.csv"))
    ws = pd.read_csv(os.path.join(OUT, "03_findingC_within_state.csv"))
    c = Chart(1200, 700,
              title="图 7｜同城同品类的竞争越密集，关闭率越高（州内控制后仍成立）",
              subtitle="左：按同城同品类店铺数八分位的关闭率（95% CI）；右：四个大州内部按城市规模四分位",
              xlabel="", ylabel="关闭率", pad_b=120)
    c.set_xlim(0.4, 8.6); c.set_ylim(0, 0.85)
    c.draw_frame(x_ticks=[], y_fmt="{:.0%}")
    for i, (_, r) in enumerate(fc.iterrows()):
        c.bars([i + 1], [r.closure_rate], width_frac=0.6, color=ACCENT, y_base=0)
        c.errorbar(i + 1, r.closure_rate, r.lo, r.hi, color=INK, width=2, cap=6)
        c.d.text((c.px(i + 1) - 18, c.py(r.closure_rate) - 22), f"{r.closure_rate:.0%}",
                 font=font(13, True), fill=INK)
        c.d.text((c.px(i + 1) - 8, c.py(0) + 8), f"{r.avg_same_category:.0f}",
                 font=font(12), fill=MUTED)
    c.d.text((c.pad_l, c.H - 100), "八分位（柱下方数字 = 该档平均同品类店数，从 1 家到 988 家）",
             font=font(14), fill=INK)
    c.d.text((c.pad_l, c.H - 68), f"最低档关闭率 {fc.closure_rate.iloc[0]:.1%} → 最高档 {fc.closure_rate.iloc[-1]:.1%}"
             f"（+{(fc.closure_rate.iloc[-1]-fc.closure_rate.iloc[0])*100:.1f}pp）",
             font=font(14, True), fill=ACCENT2)
    # right panel: within-state quartiles
    ox, oy, size = 700, 130, 150
    states = ["PA", "FL", "NJ", "MO"]
    for si, st in enumerate(states):
        sub = ws[ws.state == st].sort_values("quartile")
        if sub.empty:
            continue
        d = c.d
        d.text((ox + si * 118, oy - 24), st, font=font(15, True), fill=INK)
        for _, r in sub.iterrows():
            h = r.closure_rate
            x0 = ox + si * 118 + (r.quartile - 1) * 24
            y0 = oy + size - h * size / 0.85
            d.rectangle([x0, y0, x0 + 18, oy + size], fill=ACCENT, outline=None)
            d.text((x0 + 1, y0 - 18), f"{h:.0%}", font=font(11), fill=INK)
    c.d.text((ox - 10, oy + size + 14), "城市规模 Q1→Q4（州内比较）", font=font(13), fill=MUTED)
    c.footnote("数据：business_features.csv ｜ 缺失人口数据，拥挤度以「同城店数」近似，属局限")
    c.save(os.path.join(IMG, "fig07_crowding.png"))


def chart08_heatmap_hours():
    """Category x hour check-in fingerprint."""
    hm = pd.read_csv(os.path.join(OUT, "03_findingD_hour_matrix.csv"), index_col=0)
    order = ["Breakfast & Brunch", "Coffee & Tea", "Fast Food", "Other", "Vegan",
             "Mexican", "Chinese", "Japanese", "Seafood", "Bars", "Pizza"]
    hm = hm.reindex([o for o in order if o in hm.index])
    c = Chart(1250, 700, title="", subtitle="",
              pad_l=210, pad_b=95, pad_t=115)
    d = c.d
    d.text((210, 30), "图 8｜不同品类的客流时间指纹明显不同", font=font(25, True), fill=INK)
    d.text((210, 68), "每行 = 该品类打卡量在 24 小时上的占比（行内归一化），颜色越深客流占比越高",
           font=font(15), fill=MUTED)
    x0, y0 = 220, 165
    cw = (1250 - 260) / 24
    chh = 32
    mx = hm.values.max()
    for i, (name, row) in enumerate(hm.iterrows()):
        for h in range(24):
            v = row.iloc[h]
            frac = v / mx
            col = tuple(int(255 - (255 - cc) * frac) for cc in (47, 111, 159))
            d.rectangle([x0 + h * cw, y0 + i * chh, x0 + (h + 1) * cw, y0 + (i + 1) * chh],
                        fill=col, outline="#ffffff")
        tw = d.textbbox((0, 0), name, font=font(14))[2]
        d.text((x0 - 14 - tw, y0 + i * chh + 8), name, font=font(14), fill=INK)
    for h in range(0, 24, 2):
        tw = d.textbbox((0, 0), str(h), font=font(13))[2]
        d.text((x0 + h * cw + cw / 2 - tw / 2, y0 + len(hm) * chh + 10), str(h),
               font=font(13), fill=MUTED)
    d.text((x0, y0 + len(hm) * chh + 34), "小时（0–23）", font=font(15, True), fill=INK)
    # highlight late-night band
    d.rectangle([x0 + 22 * cw, y0 - 6, x0 + 24 * cw, y0 + len(hm) * chh + 4],
                outline=WARN, width=3)
    d.text((x0 + 20 * cw, y0 - 30), "深夜时段", font=font(14, True), fill=WARN)
    c.footnote("注意：打卡时间戳存在统一时区偏移（各时区咖啡店 6–10 点均极低），只宜做品类间相对比较")
    c.save(os.path.join(IMG, "fig08_category_hour_heatmap.png"))


def chart09_users():
    """User activity concentration (the Customer direction)."""
    us = pd.read_csv(os.path.join(WS, "data", "bigquery", "bdao_group4.yelp_users.csv"),
                     low_memory=False, usecols=["review_count"])
    rc = us.review_count.clip(upper=us.review_count.quantile(0.999))
    c = Chart(1100, 660,
              title="图 9｜评论贡献极度集中：1% 的用户写了近四分之一的评论",
              subtitle="1,446,025 名用户的评论数分布（横轴对数刻度）——中位数仅 7 条，右尾极长",
              xlabel="用户评论数（对数刻度）", ylabel="用户数")
    c.set_xlim(0, np.log10(rc.max() + 1)); c.set_ylim(0, 1)
    hist, edges = np.histogram(np.log10(us.review_count + 1), bins=40)
    c.draw_frame(x_ticks=[np.log10(v) for v in [1, 5, 20, 100, 500, 2000]],
                 x_fmt="", y_ticks=[], n_yticks=0)
    for t in [1, 5, 20, 100, 500, 2000]:
        c.d.text((c.px(np.log10(t)) - 12, c.H - c.pad_b + 26), str(t), font=font(14), fill=MUTED)
    maxh = hist.max()
    for i in range(len(hist)):
        cx = (edges[i] + edges[i + 1]) / 2
        h = hist[i] / maxh
        c.bars([cx], [h], width_frac=0.9, color=ACCENT3, y_base=0)
    c.d.text((c.px(np.log10(2)) + 10, c.py(0.92)), "中位数 7 条", font=font(14), fill=INK)
    c.d.text((c.px(np.log10(300)), c.py(0.72)), "长尾：少数超级用户", font=font(14), fill=ACCENT2)
    c.footnote("说明：top 1% 用户贡献 24.2% 的评论；76.5% 用户没有粉丝，仅 5.8% 曾是 elite")
    c.save(os.path.join(IMG, "fig09_user_concentration.png"))


def chart10_risk_deciles():
    """The deliverable: a risk score with a 4.8x spread in closure."""
    rk = pd.read_csv(os.path.join(OUT, "03_risk_deciles.csv"))
    c = Chart(1150, 700,
              title="图 10｜餐厅风险评分：最低分组关闭率 13.7%，最高分组 66.1%（相差 4.8 倍）",
              subtitle="RiskIndex 由已验证的信号合成（外送、评论量、tips、打卡、星级波动、差评占比、拥挤度、价格）",
              xlabel="风险评分十分位（1 = 风险最低）", ylabel="关闭率")
    c.set_xlim(0.4, 10.6); c.set_ylim(0, 0.8)
    c.draw_frame(x_ticks=list(range(1, 11)), y_fmt="{:.0%}", x_fmt="{:.0f}")
    cols = [ACCENT3] * 3 + [ACCENT] * 4 + [ACCENT2] * 3
    c.bars(list(range(1, 11)), rk.closure_rate.values, colors=cols, y_base=0)
    for i, (h, n) in enumerate(zip(rk.closure_rate.values, rk.n.values)):
        c.d.text((c.px(i + 1) - 18, c.py(h) - 22), f"{h:.0%}", font=font(13, True), fill=INK)
    avg = 1 - feat.is_open.mean()
    c.hline(avg, color=WARN)
    c.d.text((c.pad_l + 6, c.py(avg) - 24), f"总体关闭率 {avg:.1%}", font=font(14, True), fill=WARN)
    c.legend([("低风险", ACCENT3), ("中风险", ACCENT), ("高风险", ACCENT2)],
             x=c.plot[0] + 16, y=c.plot[1] + 12)
    c.footnote("用途：NROA 可把该评分做成会员自检工具，高风险店优先获得运营辅导")
    c.save(os.path.join(IMG, "fig10_risk_deciles.png"))


for fn in [chart01_overview, chart02_corr, chart03_stars_nonmonotonic, chart04_delivery_paradox,
           chart05_forest, chart06_auc, chart07_crowding, chart08_heatmap_hours,
           chart09_users, chart10_risk_deciles]:
    print(f"--- {fn.__name__}")
    fn()

print("\nall charts rendered to analysis/out/charts/")
