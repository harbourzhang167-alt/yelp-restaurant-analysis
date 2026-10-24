"""Render the full methodology chain diagram.

Shows, end to end: how the question was framed, which angle was chosen and why,
how the data was interrogated, how findings were validated, and what was finally
produced - including the paths that were tried and abandoned, because those carry
most of the reasoning.

Output: analysis/out/charts/fig20_method_chain.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flowkit import (INK, MUTED, arrow, badge, font, rounded_box, text_w, wrap)  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "out", "charts")
os.makedirs(OUT, exist_ok=True)

W = 2040
MARGIN = 56
LANE_GAP = 26

LANES = [
    {
        "no": "①", "title": "提问题：先定义“为谁、算什么”",
        "goal": "不先确定服务对象，任何指标都可以被解释成相反结论",
        "colour": "#2f6f9f", "tint": "#eef4f9",
        "steps": [
            ("业务背景", "美国餐饮规模约 1.5 万亿美元，但每年大量新店开张，竞争激烈，业主缺数据化策略"),
            ("客户与交付物", "NROA（4 万+ 独立餐厅协会）要出首届 Restaurant Success Index，我们交付其中一章的“分析基础”"),
            ("服务对象判定", "被分析单元 = 餐厅（1 行 1 家）；结果变量 = 是否关闭 → 结论服务【商家】，平台仅为数据源"),
            ("问题收敛", "thriving vs struggling：什么在区分活着与倒闭的餐厅？并限定为“可观测、可行动”的特征"),
        ],
        "why": "为什么这样提：简报要求 findings 必须能落到“对餐厅老板的具体建议”，所以问题必须挂在商家可决策的特征上，而不是平台指标上。",
    },
    {
        "no": "②", "title": "选角度：在四个方向里做取舍",
        "goal": "数据能支撑 ≠ 值得做；要选“能验证 + 可行动”的组合",
        "colour": "#5b8c5a", "tint": "#f0f6ef",
        "steps": [
            ("四个候选", "Market（地理/竞争）｜Customer（评价/用户）｜Operation（运营/时段）｜Performance（绩效/留存）"),
            ("筛选准则", "① 结论能否被统计验证 ② 是否对业主可行动 ③ 主要威胁能否排除（循环论证/缺失/截断）"),
            ("锁定主轴", "以 Performance 为核心：存活与否是最硬的商业结果，且能容纳 Market/Operation 作为支撑章节"),
            ("放弃的方向", "未做用户增长/平台变现/情感对流量影响——那会服务平台而非商家"),
        ],
        "why": "为什么这样选：存活是唯一“非黑即白、可观测、覆盖全部样本”的商业结果；其他方向只能做描述性切片。",
    },
    {
        "no": "③", "title": "审数据：先找“哪些变量不能用”",
        "goal": "这一层决定了后面所有结论是否站得住",
        "colour": "#b7791f", "tint": "#fbf4e8",
        "steps": [
            ("审全部 17 张表", "识别粒度：餐厅级 52,286 / 评论级 4,724,683 / 打卡 8.5M 事件 / 月度·季度轨迹"),
            ("发现循环论证", "关闭店“最后评论日期”≈关闭日期（span~is_open 的 Spearman 0.352）→ 该变量必须排除"),
            ("发现快照失真", "is_open 是采集时点快照却印在每条历史评论上 → 禁止按年份聚合存活率"),
            ("发现 Unknown 是信号", "alcohol 62% Unknown、by_appointment_only 94%；且 Unknown 组存活率更高 → 不能填 False"),
            ("发现覆盖不均", "PA 占 25%、CA 仅 2%，且 CA/IL 内部只覆盖一个角落 → 全国平均不可说"),
        ],
        "why": "为什么先审数据：评分表里最容易被扣分的是“方法不严谨”。把陷阱写进报告，比多做两张图更能体现分析质量。",
    },
    {
        "no": "④", "title": "做验证：把“看起来有关”变成“确实有关、幅度多大”",
        "goal": "每个结论都要有：点估计 + 置信区间 + 样本量 + 效应量",
        "colour": "#c1553b", "tint": "#fbeeeb",
        "steps": [
            ("相关怎么算", "Pearson 与 Spearman 并列——关系非单调且重尾，Pearson 会低估（星级 0.038 vs 评论数 0.180）"),
            ("模型怎么选", "logistic 预测 is_open（33% 关闭率，正例充足）；样本不足无法做严谨时序生存模型"),
            ("工具受限怎么办", "本机无 scipy/statsmodels → numpy 手写 IRLS + 岭惩罚；bootstrap 自置信区间"),
            ("如何防过拟合", "按 business_id 聚类重抽；AUC 而非准确率；效应量而非只看 p 值"),
            ("变量怎么迭代", "城市店数代理（−0.069）→ 真实 1 km 邻居数（−0.141）；同品类密度经检验无增量解释力"),
        ],
        "why": "为什么这样验证：数据量极大，几乎所有变量都会“统计显著”，所以必须改用效应量与区间判断重要性。",
    },
    {
        "no": "⑤", "title": "出结论：可辩护的四条 + 边界与修正",
        "goal": "结论必须自带反面证据与适用边界，否则只是营销话术",
        "colour": "#5a4f9b", "tint": "#f1eff8",
        "steps": [
            ("结论 1", "星级几乎不区分存活（≈0）；评论量十分位关闭率 47% → 15%"),
            ("结论 2", "可达性压倒口碑：有外送关闭率 19.5% vs 无外送 48.1%（差 28.6pp）；而无外送组评分反而更高"),
            ("结论 3", "真实空间密度：1 km 内邻居数让关闭率翻倍（23.7% → 46.5%），18/18 城市内部同向"),
            ("结论 4", "风险评分把关闭率拉开 4.8 倍（13.7% vs 66.1%）；AUC 0.763，仅用星级仅 0.523"),
            ("诚实修正", "加入“城市店数”后同品类密度 OR≈0.99 → 撤回“换品类可脱困”的建议"),
            ("边界声明", "非因果 · 坐标含加拿大 · 覆盖不均 · 无人口口径 → 结论限定“本样本内”"),
        ],
        "why": "为什么保留修正：一条被自己推翻的假设，比五条顺理成章的结论更能证明分析是“验证”而非“编故事”。",
    },
]

REJECTED = [
    ("连通分量聚城市", "按距离合并城市标签，结果因传递性把整个新泽西并成一个“城市”", "改用密度峰值做核心，无法串联"),
    ("Overpass 矢量街道", "大 bbox 反复 504，单城 25 块瓦片几乎抓不完", "改用 OSM 栅格底图，每图 9–20 个请求"),
    ("星级作为筛选指标", "AUC 0.523 ≈ 随机猜测", "改用评论量+可达性+密度，AUC 0.763"),
    ("同品类密度单独入模", "三尺度共线（0.72–0.89）导致系数符号翻转", "只取单一尺度 1 km，并公开共线性"),
]

OUTPUTS = [
    ("数据层", "17 张表 / 42,377,029 行 / 12 个已导出 CSV（逐表整表多重集核对通过）"),
    ("分析层", "business_features.csv（52,286 × 125）+ 25 份结论汇总 CSV"),
    ("证据层", "84 张图：24 州级 · 24 矢量街道 · 36 真实 OSM 底图"),
    ("交付层", "FINDINGS.md（4 条结论 + 1 条诚实修正 + 12 条局限）· 01_profile.md · Tableau 实施方案 · 小尺度地图说明"),
]


def main():
    probe = Image.new("RGB", (10, 10))
    pd = ImageDraw.Draw(probe)

    f_h1 = font(34, True)
    f_h2 = font(17)
    f_lane = font(23, True)
    f_goal = font(16)
    f_box_t = font(17, True)
    f_box_b = font(15)
    f_why = font(16)
    f_badge = font(14, True)

    # ---------------- measure lanes
    # Column count must be the MAX steps across lanes, otherwise a 5-step lane
    # overflows the canvas (it did at 4 columns).
    n = max(len(l["steps"]) for l in LANES)
    inner = W - MARGIN * 2 - 124 - 16
    box_gap = 16
    box_w = (inner - box_gap * (n - 1)) / n

    lane_heights = []
    step_layouts = []
    for lane in LANES:
        rows = []
        for title, body in lane["steps"]:
            tl = wrap(pd, body, f_box_b, box_w - 32)
            rows.append((title, tl))
        maxh = max(40 + 22 + len(tl) * 21 + 18 for _, tl in rows)
        why_h = 24 + 46
        lane_heights.append(int(30 + maxh + 14 + why_h + 16))
        step_layouts.append(rows)

    head_h = 150
    strip_h = 96
    total = head_h + strip_h + sum(lane_heights) + LANE_GAP * len(LANES) \
        + 40 + 240 + 40 + 260 + 70

    img = Image.new("RGB", (W, int(total)), "#ffffff")
    d = ImageDraw.Draw(img)

    # ---------------- header
    d.text((MARGIN, 30), "从提问到产出：这套分析是怎么一路推导出来的",
           font=f_h1, fill=INK)
    d.text((MARGIN, 78),
           "服务对象：NROA 会员（独立餐厅经营者）｜分析单元：52,286 家餐厅｜结果变量：is_open（总体关闭率 33.1%）",
           font=f_h2, fill=MUTED)
    d.text((MARGIN, 104),
           "下面每一层都标注了「这一步解决什么」与「为什么这样选」，并在最后列出被放弃的路径——那些才是推导的核心。",
           font=f_h2, fill=MUTED)

    # ---------------- input strip
    y = head_h
    rounded_box(img, d, MARGIN, y, W - MARGIN * 2, strip_h - 12, "#f4f6f8", "#d3dbe2")
    d.text((MARGIN + 18, y + 12), "输入", font=font(17, True), fill=INK)
    d.text((MARGIN + 76, y + 12),
           "Yelp 学术数据集（2022 采集）：business 52,286 · review 4.72M · tip 649k · check-in 8.5M 事件 · user 1.45M"
           "　　渠道：BigQuery（bdao-group.bdao_group4）→ 本地 CSV",
           font=font(15), fill=MUTED)
    d.text((MARGIN + 76, y + 44),
           "外部数据：PublicaMundi 美国州界 GeoJSON · OpenStreetMap 栅格瓦片与矢量街道（Overpass）",
           font=font(15), fill=MUTED)

    # ---------------- lanes
    y = head_h + strip_h
    lane_color_prev = None
    for li, lane in enumerate(LANES):
        h = lane_heights[li]
        # container
        rounded_box(img, d, MARGIN, y, W - MARGIN * 2, h, lane["tint"], "#d3dbe2",
                    radius=12, width=1)
        # left label strip
        d.rounded_rectangle([MARGIN, y, MARGIN + 104, y + h], radius=12,
                            fill=lane["colour"])
        d.text((MARGIN + 30, y + 20), lane["no"], font=font(30, True), fill="#ffffff")
        # stacked lane title on the strip
        fv = font(17, True)
        ty = y + 74
        for ch in lane["title"].split("：")[0]:
            d.text((MARGIN + 40, ty), ch, font=fv, fill="#ffffff")
            ty += 21

        # title + goal
        d.text((MARGIN + 124, y + 16), lane["title"], font=f_lane, fill=INK)
        d.text((MARGIN + 124, y + 48), "目标：" + lane["goal"], font=f_goal, fill=MUTED)

        # step boxes
        bx0 = MARGIN + 124
        by = y + 84
        rows = step_layouts[li]
        box_h = max(40 + 22 + len(tl) * 21 + 18 for _, tl in rows)
        for si, (title, tl) in enumerate(rows):
            x = bx0 + si * (box_w + box_gap)
            rounded_box(img, d, x, by, box_w, box_h, "#ffffff", lane["colour"],
                        radius=9, width=2)
            d.text((x + 16, by + 12), f"{si+1}. {title}", font=f_box_t, fill=lane["colour"])
            yy = by + 40
            for line in tl:
                d.text((x + 16, yy), line, font=f_box_b, fill=INK)
                yy += 21
            if si < len(rows) - 1:
                ax = x + box_w + 3
                arrow(d, [(ax, by + box_h / 2), (ax + box_gap - 6, by + box_h / 2)],
                      lane["colour"], width=3, head=8)

        # why strip
        wy = by + box_h + 14
        d.line([(bx0, wy), (W - MARGIN - 16, wy)], fill=lane["colour"], width=1)
        why_lines = wrap(pd, lane["why"], f_why, W - MARGIN - 16 - bx0 - 8)
        wy += 8
        for line in why_lines:
            d.text((bx0, wy), line, font=f_why, fill=lane["colour"])
            wy += 22

        y += h + LANE_GAP

        # connector between lanes: the chain is a derivation, so make the hand-off
        # explicit and visible (a bare 4px line in a 26px gap was invisible).
        if li < len(LANES) - 1:
            cx = MARGIN + 52
            gy0 = y - LANE_GAP + 1
            gy1 = y - 1
            d.line([(cx, gy0), (cx, gy1)], fill=lane["colour"], width=5)
            arrow(d, [(cx, gy1 - 12), (cx, gy1)], lane["colour"], width=5, head=13)
            label = f"得出下一步"
            fcon = font(13, True)
            d.text((cx + 14, gy0 + 3), label, font=fcon, fill=lane["colour"])

    # ---------------- rejected paths
    y += 10
    d.text((MARGIN, y), "被放弃的路径（推导的真正证据）", font=font(25, True), fill=INK)
    y += 40
    rounded_box(img, d, MARGIN, y, W - MARGIN * 2, 232, "#fdf8f6", "#e3c9c2")
    ry = y + 16
    for name, why, fix in REJECTED:
        # ASCII-safe mark: the ✓/✕ glyphs are missing from the CJK fonts used here
        d.text((MARGIN + 20, ry), "×", font=font(19, True), fill="#c1553b")
        d.text((MARGIN + 48, ry), name, font=font(16, True), fill="#8c3a26")
        d.text((MARGIN + 250, ry), why, font=font(15), fill=INK)
        d.text((W - MARGIN - 520, ry), "→ " + fix, font=font(15), fill="#5b8c5a")
        ry += 52
    y += 250

    # ---------------- outputs
    d.text((MARGIN, y), "最终产出", font=font(25, True), fill=INK)
    y += 40
    colw = (W - MARGIN * 2) / 4
    for i, (label, body) in enumerate(OUTPUTS):
        x = MARGIN + i * colw
        rounded_box(img, d, x + 6, y, colw - 12, 150, "#f4f7fa", "#cdd8e2")
        badge(d, x + 20, y + 14, label, "#2f6f9f", size=15)
        lines = wrap(pd, body, font(14), colw - 44)
        yy = y + 54
        for line in lines[:5]:
            d.text((x + 20, yy), line, font=font(14), fill=INK)
            yy += 20

    # footer
    d.text((MARGIN, total - 44),
           "复现：analysis/01_profile.py → 02_features.py → 03_findings.py → 03b/03c(density) → "
           "06_metros.py → 07/08/09(maps) → 10_method_chain.py。全部统计量用 numpy 实现（本机无 scipy/statsmodels）。",
           font=font(13), fill=MUTED)

    path = os.path.join(OUT, "fig20_method_chain.png")
    img.save(path)
    print(f"wrote {path}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
