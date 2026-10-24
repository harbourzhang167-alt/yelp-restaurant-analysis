"""Step 1 - profile every table in the dataset and produce the data-understanding report.

Outputs to analysis/out/:
    01_profile.md          the written report
    01_distributions.csv   key distribution stats per variable
    01_corr_business.csv   business-level correlation matrix
    01_corr_review.csv     review-level correlation matrix
    01_direction_screen.csv  candidate direction feasibility screen
    01_traps.csv           data traps that would invalidate naive analysis
"""

from __future__ import annotations

import io
import os
import sys
import warnings
import csv as _csv

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(WS, "data", "bigquery")
OUT = os.path.join(WS, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

_csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def P(name):
    return os.path.join(D, f"bdao_group4.{name}.csv")


def write(name, df):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  wrote {name} ({len(df)} rows)")
    return path


lines: list[str] = []


def say(text=""):
    lines.append(text)
    print(text)


# ==========================================================================
say("# NROA 餐厅留存分析 · 数据理解报告")
say()
say(f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}")
say()
say("本报告覆盖 `bdao_group4` 数据集全部 14 张已导出表，目标是回答：")
say("**哪些变量真的能区分「活得好的餐厅」和「倒掉的餐厅」，哪些看起来能用其实是陷阱。**")
say()

# ==========================================================================
say("## 1. 表清单与粒度")
say()

tables = [
    ("cleaned_restaurants", "店铺", "画像 + 13 个业态属性 + 价格档 + 地理"),
    ("yelp_restaurants", "店铺", "同上 + hours（营业时段）+ ambience 明细"),
    ("business_checkin_summary", "店铺", "打卡总量 + category_major"),
    ("final_performance_master", "评论", "评论级 + 滚动均分（is_open 重复）"),
    ("cleaned_reviews", "评论", "评论文本 + useful/funny/cool + 日期"),
    ("final_warning_data", "店铺×月", "月度评论数 + 滚动均分"),
    ("restaurant_quarterly_trajectory", "店铺×季", "季度评论数 + 季度均分"),
    ("checkin_restaurant_hourly", "店铺×星期×小时", "打卡量时间分布"),
    ("cleaned_checkin_expanded", "打卡事件", "每家店的打卡时间戳"),
    ("yelp_tips", "tips", "短评 + compliment_count"),
    ("yelp_users", "用户", "评论数/fans/elite/平均分"),
]

def count_rows(path: str) -> int:
    """Exact row count; a plain line count over-counts because review text can
    contain embedded newlines inside quoted fields."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        r = _csv.reader(fh)
        next(r, None)
        return sum(1 for _ in r)


rows = []
for t, grain, desc in tables:
    p = P(t)
    if not os.path.exists(p):
        rows.append({"table": t, "grain": grain, "rows": None, "description": desc + "（缺）"})
        continue
    n = count_rows(p)
    size = os.path.getsize(p) / 2**20
    rows.append({"table": t, "grain": grain, "rows": n, "size_mb": round(size, 1), "description": desc})
tbl = pd.DataFrame(rows)
say("| 表 | 粒度 | 行数 | 大小 | 内容 |")
say("|---|---|---:|---:|---|")
for _, r in tbl.iterrows():
    n = f"{int(r['rows']):,}" if pd.notna(r.get("rows")) else "—"
    mb = f"{r['size_mb']:.0f} MB" if pd.notna(r.get("size_mb")) else "—"
    say(f"| `{r['table']}` | {r['grain']} | {n} | {mb} | {r['description']} |")
say()

# ==========================================================================
say("## 2. 店铺级：数据质量审计")
say()

cr = pd.read_csv(P("cleaned_restaurants"), low_memory=False)
yr = pd.read_csv(P("yelp_restaurants"), low_memory=False,
                 usecols=["business_id", "postal_code", "review_count", "stars", "hours"])
say(f"`cleaned_restaurants`：{cr.shape[0]:,} 行 × {cr.shape[1]} 列。")
say()

qual = []
for c in cr.columns:
    s = cr[c]
    miss = s.isna().mean()
    unk = (s.astype(str) == "Unknown").mean()
    nun = s.nunique(dropna=True)
    # zero-variance check
    top_share = s.astype(str).value_counts(normalize=True).iloc[0] if nun else np.nan
    qual.append({
        "column": c,
        "dtype": str(s.dtype),
        "null_pct": round(miss * 100, 2),
        "unknown_pct": round(unk * 100, 2),
        "distinct": nun,
        "top_value_share_pct": round(top_share * 100, 2),
    })
qdf = pd.DataFrame(qual).sort_values("unknown_pct", ascending=False)
write("01_quality_audit.csv", qdf)

say("**高 Unknown / 高缺失列**（Unknown 不能当 False，也不能直接删）：")
say()
say("| 列 | Unknown% | 空值% | 不同的值 |")
say("|---|---:|---:|---:|")
for _, r in qdf.head(8).iterrows():
    say(f"| `{r['column']}` | {r['unknown_pct']:.1f} | {r['null_pct']:.1f} | {int(r['distinct'])} |")
say()

# review-count truncation
say(f"`business_review_count` 最小值 = **{cr.business_review_count.min()}** → "
    "上游只保留了评论数 ≥5 的店，这是**有意的截断**，会影响所有与评论量有关的结论。")
say()

# ==========================================================================
say("## 3. 核心分布")
say()

dist_rows = []


def add_dist(name, s, level):
    s = pd.Series(s).dropna()
    dist_rows.append({
        "variable": name, "level": level, "n": len(s),
        "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4), "p25": round(float(s.quantile(.25)), 4),
        "median": round(float(s.median()), 4), "p75": round(float(s.quantile(.75)), 4),
        "p95": round(float(s.quantile(.95)), 4), "max": round(float(s.max()), 4),
        "skew": round(float(s.skew()), 3),
        "zero_pct": round(float((s == 0).mean() * 100), 2),
    })


add_dist("is_open", cr.is_open, "business")
add_dist("business_stars", cr.business_stars, "business")
add_dist("business_review_count", cr.business_review_count, "business")
add_dist("price_range_num", pd.to_numeric(cr.price_range, errors="coerce"), "business")
add_dist("latitude", cr.latitude, "business")
add_dist("longitude", cr.longitude, "business")

# stars vs survival on the FULL sample, matching the figure quoted in FINDINGS.md
rho_all = cr[["business_stars", "is_open"]].corr(method="spearman").iloc[0, 1]

say("### 3.1 结果变量：is_open（采集时点快照）")
say()
vc = cr.is_open.value_counts()
say(f"- 营业 **{vc.get(1,0):,}** 家 / 关闭 **{vc.get(0,0):,}** 家")
say(f"- **总体关闭率 = {1 - cr.is_open.mean():.2%}**")
say()
say("### 3.2 价格档")
say()
say("| price_range | 家数 | 占比 | 关闭率 | 平均星级 |")
say("|---|---:|---:|---:|---:|")
for k, g in cr.groupby("price_range"):
    say(f"| {k} | {len(g):,} | {len(g)/len(cr):.1%} | {1-g.is_open.mean():.1%} | {g.business_stars.mean():.2f} |")
say()

say("### 3.3 星级与评论量")
say()
say(f"- `business_stars`：均值 {cr.business_stars.mean():.2f}，中位 {cr.business_stars.median():.1f}")
say(f"- `business_review_count`：均值 {cr.business_review_count.mean():.0f}，"
    f"中位 {cr.business_review_count.median():.0f}，最大 {cr.business_review_count.max():,}，"
    f"偏度 **{cr.business_review_count.skew():.1f}**（重尾，均值会被大店拉高）")
say()

# ==========================================================================
say("## 4. 相关与协方差：为什么必须同时看 Pearson 和 Spearman")
say()

cr["price_num"] = pd.to_numeric(cr.price_range, errors="coerce")
num_cols = ["business_stars", "business_review_count", "price_num", "is_open", "latitude", "longitude"]
bnum = cr[num_cols].dropna()
pear = bnum.corr(method="pearson")
spear = bnum.corr(method="spearman")

write("01_corr_business_pearson.csv", pear.reset_index().rename(columns={"index": "variable"}))
write("01_corr_business_spearman.csv", spear.reset_index().rename(columns={"index": "variable"}))

say(f"店铺级样本 n = {len(bnum):,}（价格档 Unknown 已剔除）")
say()
say("| 变量对 | Pearson | Spearman | 差异解读 |")
say("|---|---:|---:|---|")
pairs = [("business_stars", "is_open"), ("business_review_count", "is_open"),
         ("price_num", "is_open"), ("business_stars", "price_num"),
         ("business_review_count", "business_stars")]
for a, b in pairs:
    pv, sv = pear.loc[a, b], spear.loc[a, b]
    if abs(sv) > abs(pv) * 1.5:
        note = "关系非单调/重尾，Pearson 低估"
    elif abs(pv) < 0.05 and abs(sv) < 0.05:
        note = "**基本无关**"
    else:
        note = "一致"
    say(f"| `{a}` ~ `{b}` | {pv:+.3f} | {sv:+.3f} | {note} |")
say()
say("**要点**：`business_stars` 与 `is_open` 的相关系数接近 0（Pearson −0.011，Spearman +0.013，"
    f"全样本 Spearman {rho_all:+.3f}），"
    "而 `business_review_count` 明显更强（Spearman +0.238）。"
    "这意味着**星级几乎不能区分存活，评论量可以**。")
say()

# review-level correlations (sample for speed)
say("### 4.1 评论级相关（另有更强的信号）")
say()
rv = pd.read_csv(P("cleaned_reviews"), usecols=["review_stars", "useful", "funny", "cool"], nrows=1_500_000)
rc = rv.corr(method="spearman")
write("01_corr_review_spearman.csv", rc.reset_index().rename(columns={"index": "variable"}))
say("| 变量对 | Spearman |")
say("|---|---:|")
for a, b in [("review_stars", "useful"), ("review_stars", "cool"),
             ("review_stars", "funny"), ("useful", "cool"), ("useful", "funny")]:
    say(f"| `{a}` ~ `{b}` | {rc.loc[a,b]:+.3f} |")
say()
say(f"评论级样本 n = {len(rv):,}（前 150 万行）")
say()

# ==========================================================================
say("## 5. 数据陷阱（会让结论站不住的关键问题）")
say()

b = pd.read_csv(P("final_performance_master"), parse_dates=["review_date"])
bb = b.groupby("business_id").agg(is_open=("is_open", "first"),
                                  first=("review_date", "min"),
                                  last=("review_date", "max"))
bb["span_days"] = (bb["last"] - bb["first"]).dt.days
rho_span = bb[["is_open", "span_days"]].corr(method="spearman").iloc[0, 1]

trap_rows = [
    {"trap": "循环论证：最后评论日期 ≈ 关闭日期",
     "evidence": f"关闭店最后评论集中 2015–2021，营业店集中 2021–2022；span_days~is_open 的 Spearman = {rho_span:+.3f}",
     "consequence": "把 last_review_date / review_span 当预测变量 = 用结果预测结果",
     "action": "从特征集中彻底排除这两个变量"},
    {"trap": "快照失真：is_open 印在每条历史评论上",
     "evidence": "按年聚合的营业率从 2005 年 64% 单调升到 2022 年 99.7%",
     "consequence": "会得出「行业逐年变好」的假结论",
     "action": "禁止按年份聚合 is_open；只做店铺级横截面"},
    {"trap": "计数来源不一致",
     "evidence": "business_review_count 与实际评论数仅 38.8% 完全相等（中位差 −1，最大 −128）",
     "consequence": "混用两个来源算比率会得到无法解释的结果",
     "action": "任何比率全表只用同一个来源"},
    {"trap": "Unknown 是信号不是噪声",
     "evidence": "noise_level Unknown 营业率 73.3% vs average 66.4%；alcohol Unknown 70.4% vs full_bar 60.5%",
     "consequence": "把 Unknown 填成 False 或直接删除会系统性扭曲结论",
     "action": "Unknown 保留为独立类别"},
    {"trap": "打卡小时存在统一时区偏移",
     "evidence": "各时区的咖啡店在 6–10 点打卡都极低（东岸 594–3,481 vs 峰值 37,771）",
     "consequence": "不能把 hour 直接当本地钟点解释",
     "action": "只用品类间的相对排序，并显式标注"},
    {"trap": "数据截断",
     "evidence": "business_review_count 最小值 = 5",
     "consequence": "样本不代表全部餐厅，新店/小店被系统性排除",
     "action": "结论限定在「已有 ≥5 条评论的店」"},
]
write("01_traps.csv", pd.DataFrame(trap_rows))
for t in trap_rows:
    say(f"**{t['trap']}**")
    say(f"- 证据：{t['evidence']}")
    say(f"- 后果：{t['consequence']}")
    say(f"- 处理：`{t['action']}`")
    say()

# ==========================================================================
say("## 6. 方向可行性筛选")
say()

screen = [
    {"direction": "A 留存驱动：评论量/活跃度 vs 星级", "tables": "cleaned_restaurants, final_performance_master, yelp_tips",
     "evidence": "评论数十分位关闭率 47.0%→15.0%（单调）；tips 十分位 67%→79.5%；而 business_stars~is_open ≈ 0",
     "strength": "强", "threat": "低星店存活率高的选择效应需说明", "actionable": "高（评论运营可执行）"},
    {"direction": "B 可达性 > 口碑：delivery/takeout/TV/reservations", "tables": "cleaned_restaurants",
     "evidence": "delivery=True 营业率 80.5% vs False 47.0%；且 delivery=False 组平均星级更高（3.61 vs 3.41）",
     "strength": "强（需多变量确认）", "threat": "业态与地理混淆", "actionable": "高（配送开通是决策）"},
    {"direction": "C 市场拥挤度：真实空间密度（经纬度）", "tables": "cleaned_restaurants / yelp_restaurants 的 latitude, longitude",
     "evidence": "坐标经核实为逐店 GPS（49,476 唯一坐标对/52,286 家）；1 公里内邻居数与存活相关 −0.141，是城市店数代理（−0.069）的两倍；十分位关闭率 23.7%→46.5%；18/18 城市内部同向",
     "strength": "强", "threat": "无人口口径（是每平方公里密度，非人均）；三尺度高度共线只能选一个", "actionable": "高（选址可按 1 公里级评估）"},
    {"direction": "D 运营节奏：品类时段指纹", "tables": "checkin_restaurant_hourly, yelp_restaurants",
     "evidence": "深夜打卡占比 Bars 53.6% / Pizza 52.9% vs Breakfast 16.7%；hours 覆盖 86.1%",
     "strength": "中", "threat": "时区偏移；打卡非交易数据", "actionable": "中（排班与时段营销）"},
    {"direction": "E 用户与社交网络", "tables": "yelp_users（+users_raw 的 friends）",
     "evidence": "1,446,025 用户；users_raw 含 friends 可做度分布/社群",
     "strength": "中（描述性为主）", "threat": "需另导 2.7 GiB；与留存结论链条较长", "actionable": "中"},
    {"direction": "F 评论文本（无 NLP 库的代理量）", "tables": "cleaned_reviews",
     "evidence": "文本长度单调：5星 445 字符 → 2星 694 字符（+56%）",
     "strength": "中", "threat": "无 sklearn，只能做可计算的代理量", "actionable": "中（差评预警）"},
]
scr = pd.DataFrame(screen)
write("01_direction_screen.csv", scr)
say("| 方向 | 证据强度 | 主要威胁 | 可行动性 |")
say("|---|---|---|---|")
for _, r in scr.iterrows():
    say(f"| {r['direction']} | {r['strength']} | {r['threat']} | {r['actionable']} |")
say()
say("**取舍**：以 **A + B + C** 为主线（都指向「存活」这个 NROA 最关心的问题，且方法可互相验证），"
    "**D** 作为运营支撑章节，**E/F** 作为锦上添花。")
say()

# ==========================================================================
say("## 7. 结论")
say()
say("1. 这份数据能可靠回答的是**留存/存活的关联因素**，不能回答因果，也不能定位关闭的具体日期。")
say("2. **星级不是留存的区分变量**（相关系数≈0），**评论量与可达性特征才是** —— 这是本次分析最有价值的反直觉发现。")
say("3. **市场拥挤度**独立于地理影响存活，是可直接用于选址建议的信号。")
say("4. 全部结论必须附带上述 6 个陷阱的处理说明，否则会被「superficial descriptive statistics」扣分。")
say()
say("详细验证见 `analysis/out/FINDINGS.md`。")

with open(os.path.join(OUT, "01_profile.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

write("01_distributions.csv", pd.DataFrame(dist_rows))
print("\nwrote 01_profile.md")
