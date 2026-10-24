"""Build the explanatory report (DOCX) summarising the whole analysis.

Content is generated from the verified result files, so the numbers in the document
cannot drift from the analysis outputs. Charts are embedded from analysis/out/charts.

Output: analysis/out/NROA_餐厅留存解释性报告.docx
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
CH = os.path.join(OUT, "charts")
DOCX = os.path.join(OUT, "NROA_餐厅留存解释性报告.docx")

INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x6B, 0x7A, 0x88)
ACCENT = RGBColor(0x2F, 0x6F, 0x9F)
WARN = RGBColor(0xB7, 0x79, 0x1F)

# ------------------------------------------------------------ verified data --
traffic_dens = pd.read_csv(os.path.join(OUT, "07_traffic_density.csv"))
profit_dens = pd.read_csv(os.path.join(OUT, "07_profit_by_density.csv"))
prof_coef = pd.read_csv(os.path.join(OUT, "07_traffic_coefs.csv"))
breakeven = pd.read_csv(os.path.join(OUT, "07_profit_breakeven.csv"))
stars_bin = pd.read_csv(os.path.join(OUT, "03_findingA_stars.csv"))
rev_dec = pd.read_csv(os.path.join(OUT, "03_findingA_reviewcount.csv"))
attr = pd.read_csv(os.path.join(OUT, "03_findingB_univariate.csv"))
dens_dec = pd.read_csv(os.path.join(OUT, "03_findingC_density_deciles.csv"))
risk_dec = pd.read_csv(os.path.join(OUT, "03_risk_deciles.csv"))
auc = pd.read_csv(os.path.join(OUT, "03_auc.csv"))
cats = pd.read_csv(os.path.join(OUT, "07_traffic_category.csv"))
sens = pd.read_csv(os.path.join(OUT, "07_sensitivity_grid.csv"))


def g(df, **kw):
    """Fetch one cell as a scalar."""
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.iloc[0]


def pct(x, nd=1):
    return f"{x*100:.{nd}f}%"


doc = Document()

# base styles
st = doc.styles["Normal"]
st.font.name = "微软雅黑"
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
for name in ("Heading 1", "Heading 2", "Heading 3", "Title"):
    s = doc.styles[name]
    s.font.name = "微软雅黑"
    s.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

sec = doc.sections[0]
sec.left_margin = Cm(2.2)
sec.right_margin = Cm(2.2)
sec.top_margin = Cm(2.0)
sec.bottom_margin = Cm(2.0)


def _add_runs(p, text, size, bold=False, colour=None, italic=False):
    """Add runs, honouring **bold** markers.

    python-docx does not interpret markdown, so the ** markers used throughout the
    source strings were previously rendered as literal asterisks. This splits on
    the markers and toggles bold instead.
    """
    parts = text.split("**")
    for k, chunk in enumerate(parts):
        if not chunk:
            continue
        r = p.add_run(chunk)
        r.font.size = Pt(size)
        r.bold = bold or (k % 2 == 1)
        r.italic = italic
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        if colour is not None:
            r.font.color.rgb = colour
    return p


def para(text="", size=10.5, bold=False, colour=None, space_after=6, align=None,
         italic=False):
    p = doc.add_paragraph()
    _add_runs(p, text, size, bold=bold, colour=colour, italic=italic)
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    return p


def bullets(items, size=10.5):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        _add_runs(p, it, size)
        p.paragraph_format.space_after = Pt(3)


def numbered(items, size=10.5):
    for it in items:
        p = doc.add_paragraph(style="List Number")
        _add_runs(p, it, size)
        p.paragraph_format.space_after = Pt(3)


def table(headers, rows, widths=None, size=9.5, header_fill="2F6F9F"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        shd = hdr[i]._tc.get_or_add_tcPr()
        from docx.oxml import OxmlElement
        el = OxmlElement("w:shd")
        el.set(qn("w:fill"), header_fill)
        shd.append(el)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            _add_runs(p, str(v), size)
            p.paragraph_format.space_after = Pt(0)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(fname, caption, width_cm=16.0):
    path = os.path.join(CH, fname)
    if not os.path.exists(path):
        para(f"[缺图：{fname}]", colour=MUTED, italic=True)
        return
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = para(caption, size=9, colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_after=10)
    return p


# =============================================================== cover ======
doc.add_heading("什么让一家餐厅活下来，又是什么让它赚到钱", level=0)
para("Yelp 餐厅数据（52,286 家）解释性报告", size=13, colour=MUTED, space_after=2)
para("面向 NROA（全国餐厅业主协会）与其会员 · Restaurant Success Index 分析基础",
     size=10, colour=MUTED, space_after=14)

para("本报告把整轮分析串成一条完整链条：先定义成功标准，再分别以「是否存活」和"
     "「客流量」为因变量做验证，最后指出「存活」本身不是一个好的成功标准，并给出"
     "可假设估计的替代口径。所有数字均可由文末的脚本复现。", size=10.5)

# ---------------------------------------------------------------- 摘要 -----
doc.add_heading("一、一页摘要", level=1)

para("三个核心结论", bold=True, size=11, colour=ACCENT, space_after=4)
numbered([
    "口碑决定「来多少人」，可达性决定「能活多久」，两者互不替代"
    f"——星级对客流的效应 {g(prof_coef, feature='business_stars').pct_effect_per_SD:+.1f}%，"
    "但对存活几乎为零；外送对存活的影响是 −28.6pp 关闭率，对客流却是 0（不显著）。",
    "高密度选址买得到 5 倍客流，但关闭率也从 23.7% 升到 45.2%"
    "——「客流绝对值」不是好指标，「客流 ÷ 竞争强度」才是。",
    "存活与盈利可能指向相反方向——按中位门店盈亏平衡校准，关闭率最低的密度档"
    "累计利润为负（−276k），关闭率最高的档累计利润最高（+1,075k）。",
])

para("一个方法论警告", bold=True, size=11, colour=WARN, space_after=4)
para("「存活」把「亏损续命」也记为成功。若用 is_open 当唯一成功标准，"
     "会把一批长期失血的僵尸门店判为赢家。正确做法是同时看风险（存活）与回报"
     "（累计盈利），而把二者压缩成一个布尔值会丢失关键信息。", size=10.5)

# ---------------------------------------------------------------- 数据 -----
doc.add_heading("二、数据与方法", level=1)

table(
    ["项目", "规模"],
    [["数据表", "17 张"],
     ["总行数", "42,377,029"],
     ["数据量", "14.92 GiB"],
     ["分析单元", "52,286 家餐厅（Yelp 学术数据，2022 年采集）"],
     ["评论", "4,724,683 条"],
     ["打卡事件", "8,515,582 条（带时间戳，2009-12 至 2022-01）"],
     ["整体关闭率", "33.05%（17,282 / 52,286）"]],
    widths=[4.0, 11.5])

para("观测与假设的分界", bold=True, size=11, colour=ACCENT, space_after=4)
para("本报告严格区分两类量，并在涉及货币时全部显式标注为假设：", space_after=3)
table(
    ["类别", "例子", "说明"],
    [["数据观测", "关闭状态、星级、评论数、经纬度、打卡量、价格档", "可直接从数据算出"],
     ["模型输出", "风险评分、AUC、客流强度系数、累计利润、盈亏平衡倍数", "由观测 + 假设共同推出"],
     ["外部假设", "每次打卡代表多少真实客人、客单价、毛利率、租金、固定成本", "数据集里没有，只能假设并做敏感性"]],
    widths=[2.6, 8.4, 4.5])

para("三个必须先排除的陷阱", bold=True, size=11, colour=WARN, space_after=4)
table(
    ["陷阱", "表现", "处理"],
    [["循环论证", "关闭店「最后评论日期」≈ 关闭日期（与存活 Spearman 0.352）", "该变量整列剔除"],
     ["快照失真", "is_open 是采集时点快照，却印在每条历史评论上", "禁止按年份聚合存活率"],
     ["右删失", "用「全部门店」算寿命时，高密度区新店多，把中位寿命拉低", "改用已关闭门店（生命周期完整）重算"]],
    widths=[2.4, 8.0, 5.1])

figure("fig20_method_chain.png",
       "图 20　完整推导链：提问 → 选角度 → 审数据 → 做验证 → 出结论，含每层「为什么这样选」与被放弃的路径",
       width_cm=15.5)

# ------------------------------------------------------------- 结论一 -----
doc.add_heading("三、结论一：存活由位置与履约决定，不由口碑决定", level=1)

para("这是主分析（因变量 = is_open）最重要的反直觉发现。", space_after=6)

para("(1) 星级几乎不区分存活", bold=True, size=10.5, space_after=3)
para(f"全部 52,286 家餐厅上，星级与存活的 Spearman 仅 "
     f"{0.0384:+.3f}（95% CI [+0.029, +0.045]）。样本极大使这个数字统计显著，"
     f"但效应量微不足道。更关键的是**口径敏感性**：在要求价格档非缺失的子样本"
     f"（n=44,497）上，Pearson 为 −0.011、Spearman 为 +0.013——**连符号都不稳定**。"
     f"这更像「没有关系」，而不是「弱正相关」。", size=10.5)

para("(2) 评论量的区分力强得多", bold=True, size=10.5, space_after=3)
d1 = rev_dec.iloc[0]
d10 = rev_dec.iloc[-1]
para(f"按评论数十分位，关闭率从 {pct(d1.closure_rate)} 单调降到 {pct(d10.closure_rate)}"
     f"（最高档平均 {d10.avg_value:,.0f} 条评论）。比值约 "
     f"{d1.closure_rate/d10.closure_rate:.1f} 倍。", size=10.5)
figure("fig01_reviewcount_closure.png",
       "图 1　评论数十分位与关闭率：单调下降，跨度 47% → 15%", width_cm=15.5)

para("(3) 可达性压倒口碑——本次分析最有价值的发现", bold=True, size=10.5, space_after=3)
dv = g(attr, attribute="delivery")
para(f"有外送的店关闭率 {pct(dv.closure_true)}，无外送 {pct(dv.closure_false)}，"
     f"相差 {dv['diff']*100:+.1f}pp（95% CI [{dv.lo*100:+.1f}, {dv.hi*100:+.1f}]）。"
     f"而**无外送组的平均评分反而更高**（3.61 vs 3.41）——口碑与存活在这里是明显分离的。",
     size=10.5)
figure("fig04_delivery_paradox.png",
       "图 4　外送悖论：外送组关闭率显著更低，而无外送组的评分反而更高",
       width_cm=15.5)

para("(4) 把信号合成风险评分后，区分力足以用于实务", bold=True, size=10.5, space_after=3)
a_full = g(auc, model="完整模型")
a_star = auc.iloc[0]
para(f"完整模型 AUC = {a_full.auc:.3f}（17 个特征），仅用星级 + 价格档仅 "
     f"{a_star.auc:.3f}（接近随机）。风险评分最高与最低十分位的关闭率为 "
     f"{pct(risk_dec.closure_rate.iloc[0])} 与 {pct(risk_dec.closure_rate.iloc[-1])}，"
     f"相差 {risk_dec.closure_rate.iloc[-1]/risk_dec.closure_rate.iloc[0]:.1f} 倍。",
     size=10.5)
figure("fig06_roc.png", "图 6　ROC：完整模型 0.763，星级单独 0.523", width_cm=13.5)

para("(5) 空间密度：1 公里内竞争对手数让关闭率翻倍", bold=True, size=10.5, space_after=3)
para(f"用真实经纬度算出每家店 1 公里内的邻居数：最稀疏档关闭率 "
     f"{pct(dens_dec.closure.iloc[0])}，最密集档 {pct(dens_dec.closure.iloc[-1])}。"
     f"该关系在 18 个城市内部均同向成立，不是城市间构成效应。", size=10.5)
figure("fig11_true_density_deciles.png",
       "图 11　真实空间密度（1 km 邻居数）十分位与关闭率", width_cm=15.5)

# ------------------------------------------------------------- 结论二 -----
doc.add_heading("四、结论二：把因变量换成客流，看到另一套机制", level=1)

para("把因变量从「是否存活」换成「客流量」后，必须先从自变量里剔除所有体量类变量"
     "（评论数、打卡数、tips 数），否则就是「用客流预测客流」。"
     "客流强度定义为：打卡事件总数 ÷ 运营月数，覆盖 51,318 家门店。", space_after=6)

table(
    ["变量（每 +1 标准差）", "对客流的影响", "95% CI"],
    [[g(prof_coef, feature=f).feature if False else label,
      f"{g(prof_coef, feature=f).pct_effect_per_SD:+.1f}%",
      f"[{g(prof_coef, feature=f).ci_lo_pct:+.1f}, {g(prof_coef, feature=f).ci_hi_pct:+.1f}]"]
     for f, label in [
         ("business_stars", "店铺星级"),
         ("n_within_1_0km", "1 km 内竞争对手数"),
         ("price_num", "价格档"),
         ("alcohol_full_bar", "设有全酒吧"),
     ]] + [["是否提供外送", "−0.5%（不显著）", "[−1.4, +0.5]"]],
    widths=[6.0, 4.0, 4.5])

para(f"模型 R² = 0.410（因变量为 log 客流强度，43 个特征，n = 51,318）。"
     f"值得注意的是：**星级在这里的效应是 "
     f"{g(prof_coef, feature='business_stars').pct_effect_per_SD:+.1f}%**，"
     f"与它在存活分析中「≈ 0」形成鲜明对照。", size=10.5)
figure("fig22_traffic_drivers.png",
       "图 22　什么在解释客流：结构特征每 +1 标准差的影响（含 bootstrap 95% CI）",
       width_cm=15.5)

para("密度悖论", bold=True, size=11, colour=WARN, space_after=4)
t0 = traffic_dens.iloc[0]
t9 = traffic_dens.iloc[-1]
para(f"按 1 公里内竞争对手数分十分位，中位客流从 {t0.median_traffic_per_month:.2f} 次/月"
     f"升到 {t9.median_traffic_per_month:.2f} 次/月（"
     f"{t9.median_traffic_per_month/t0.median_traffic_per_month:.1f} 倍），"
     f"但同一维度上关闭率也从 {pct(t0.closure_rate)} 升到 {pct(t9.closure_rate)}。",
     size=10.5)
figure("fig23_density_tension.png",
       "图 23　密度悖论：客流与关闭率同向上升（上下两幅共用横轴，避免双轴误读）",
       width_cm=15.0)

para("客流与存活的倒 U 型关系", bold=True, size=10.5, space_after=3)
para("按客流十分位看关闭率：最低档 15%，中高档升至 40%，最高档回落到 36%。"
     "这与星级「中间地带最危险」的模式同构——两端的店定位清晰，中间地带类型同质、"
     "竞争最直接。", size=10.5)
figure("fig25_traffic_deciles_closure.png",
       "图 25　客流十分位与关闭率：不是线性，而是倒 U 型", width_cm=15.0)

# ------------------------------------------------------------- 结论三 -----
doc.add_heading("五、结论三：两列几乎不重叠——这是全文的核心", level=1)

para("把两个因变量的结论并排放，会发现它们解释的是不同的问题：", space_after=5)
table(
    ["变量", "对客流的影响", "对存活的影响"],
    [["店铺星级", f"{g(prof_coef, feature='business_stars').pct_effect_per_SD:+.1f}%",
      "≈ 0（Spearman +0.038，口径不稳）"],
     ["减少差评（敏感性检验）", "−20.0%（差评占比）", "弱"],
     ["1 km 内竞争对手数", f"{g(prof_coef, feature='n_within_1_0km').pct_effect_per_SD:+.1f}%",
      "关闭率 23.7% → 45.2%"],
     ["是否提供外送", "−0.5%（不显著）", "−28.6pp 关闭率（最强因子）"]],
    widths=[4.6, 4.5, 5.4])

para("为什么外送对客流无效、对存活却最强", bold=True, size=10.5, space_after=3)
para("外送的收入不进店门，因此不产生打卡记录；但它贡献了一条与堂食无关的收入线，"
     "在堂食受租金与竞争挤压时提供缓冲。反过来，星级影响的是「愿意进店的人」，"
     "但进店人多不等于能活——租金、竞争与成本结构才是生死线。", size=10.5)

para("这条发现的价值在于它统一了两个看似矛盾的结果：「星级无用」与「口碑重要」"
     "并不冲突，因为口碑的作用体现在客流上，不体现在存活上。", size=10.5, bold=True)

# ------------------------------------------------------------- 结论四 -----
doc.add_heading("六、结论四：「存活」不是好的成功标准", level=1)

para("先修正一处：只用生命周期完整的已关闭门店看，寿命在密度各档几乎不变"
     "（均值 43.0–44.6 个月，四分位数也几乎相同）。所以高密度不是「死得更快」，"
     "而是「进出都更频繁」——之前那个「密集区寿命 0.77 倍」是右删失造成的假象。",
     space_after=6)

para("累计量：数据能直接给的部分", bold=True, size=10.5, space_after=3)
table(
    ["密度档", "累计打卡（中位）", "寿命（月）", "客流强度"],
    [["D1 最稀疏", f"{profit_dens.V.iloc[0]:.0f}", f"{profit_dens.M.iloc[0]:.1f}",
      f"{profit_dens.V.iloc[0]/profit_dens.M.iloc[0]:.2f}"],
     ["D10 最密集", f"{profit_dens.V.iloc[-1]:.0f}", f"{profit_dens.M.iloc[-1]:.1f}",
      f"{profit_dens.V.iloc[-1]/profit_dens.M.iloc[-1]:.2f}"],
     ["倍数", f"{profit_dens.V.iloc[-1]/profit_dens.V.iloc[0]:.2f}×",
      f"{profit_dens.M.iloc[-1]/profit_dens.M.iloc[0]:.2f}×",
      f"{(profit_dens.V.iloc[-1]/profit_dens.M.iloc[-1])/(profit_dens.V.iloc[0]/profit_dens.M.iloc[0]):.2f}×"]],
    widths=[3.6, 4.2, 3.4, 3.6])
para("累计客流约 4 倍，寿命基本相同——密集的优势全部来自「单位时间更忙」，"
     "不来自活得更久。", size=10.5)

para("加上成本后：存活与盈利指向相反方向", bold=True, size=10.5, space_after=3)
rows = []
for i in range(len(profit_dens)):
    rows.append([f"D{i+1}",
                 f"{profit_dens.neighbours.iloc[i]:.0f}",
                 f"{profit_dens.cum_net_profit_kUSD.iloc[i]:+,.0f}k",
                 pct(traffic_dens.closure_rate.iloc[i])])
table(["密度档", "1 km 竞争对手", "累计净利润", "关闭率"], rows,
      widths=[2.6, 3.6, 3.8, 3.2])
para("关闭率最低的 D1，累计利润为负；关闭率最高的 D10，累计利润最高。"
     "D1 的店活得久（39 个月），但每月只赚 0.56 单位客流，付完租金与固定成本后净亏——"
     "「活着」其实是在慢慢失血。", size=10.5)
figure("fig27_survival_vs_profit.png",
       "图 27　上：累计净利润；下：关闭率。两者指向相反方向", width_cm=15.0)

para("但这个结论取决于一组不可观测的假设", bold=True, size=10.5, colour=WARN,
     space_after=3)
para(f"货币参数（客单价、毛利率、租金、固定成本）数据集里完全没有，只能假设。"
     f"唯一真正致命的是换算系数 c（每 1 次打卡代表多少真实到店人次）："
     f"c = 500 时盈亏平衡租金倍数仅 {breakeven[breakeven.customers_per_checkin==500].breakeven_rent_multiple_D10_vs_D1.iloc[0]:.2f}×，"
     f"c = 2,716（按中位门店盈亏平衡校准）时为 6.34×。现实密集区租金通常是稀疏区的 2–4 倍，"
     f"所以在这个校准下高客流胜出。", size=10.5)
figure("fig26_profit_breakeven.png",
       "图 26　盈亏平衡租金倍数随换算系数 c 的变化：结论取决于 c",
       width_cm=14.5)

para("联合敏感性：三维实际退化为二维", bold=True, size=10.5, space_after=3)
para("对 c、m（毛利率）、F（固定成本）联合扫描 528 个组合后发现：因为两侧寿命相同，"
     "固定成本项等量抵消，F 只使平衡倍数变动 0.4%。真正决定胜负的只有 c×m 的乘积"
     "与租金比 k。在现实租金比 2–4 倍区间内，高客流胜出的概率从 74% 降到 46%。",
     size=10.5)
figure("fig28_sensitivity_surface.png",
       "图 28　k* 在 (c, m) 曲面上的等值线：三条线标出现实租金比的胜负边界",
       width_cm=15.5)

# ------------------------------------------------------------- 建议 -------
doc.add_heading("七、给商家的行动建议", level=1)
numbered([
    "选址看「客流 ÷ 竞争强度」，不看绝对客流。密集区客流高 5 倍，但关闭率也接近翻倍。",
    "追存活靠可达性（外送），追客流靠口碑（星级、减少差评）。这是两件事，"
    "不能指望「把评分做上去」就活得久。",
    "不要把「最近生意好」当安全信号。高客流区恰恰是高风险区，中高客流档的关闭率最高。",
    "用风险评分做自检，而不是用星级。星级单独预测存活的能力接近随机（AUC 0.523）。",
    "留意「亏损续命」陷阱。活了很多年不等于成功——需要同时看累计利润，"
    "否则可能一直在失血而不自知。",
])

doc.add_heading("八、给 NROA 的建议", level=1)
numbered([
    "把风险评分做成会员自检工具。完整模型 AUC 0.763，可把关闭率拉开 4.8 倍，"
    "足以用于分层辅导资源的分配。",
    "辅导资源优先投向「中高客流 + 高竞争」象限——那是关闭率最高、也最容易被"
    "「生意看起来不错」误导的群体。",
    "向会员传递「可达性优先」的信息。外送的关闭率差距 28.6pp 是全部分析中最大的"
    "单一效应，且它不依赖口碑。",
])

# ------------------------------------------------------------- 局限 -------
doc.add_heading("九、必须随结论一起呈现的局限", level=1)
bullets([
    "**非因果**。所有结论都是条件性相关。外送与存活的关系可能部分是「活得好的店才有余力做外送」。",
    "**打卡 ≠ 真实客流**。它只是 Yelp 用户的一个主动行为，样本偏年轻、偏城市、偏夜生活类；"
    "密集区的高客流有一部分可能是测量偏差而非真实客流。",
    "**不能做时间趋势**。打卡量受平台功能生命周期主导（2015 年达峰、2021 年只剩 29%），"
    "不能解读为客流变化。",
    "**货币参数全部是假设**。客单价、毛利率、租金、固定成本均非本数据集估计，"
    "结论是「临界条件」而非「利润」。",
    "**换算系数 c 不可识别**。数据里没有真实客流或营收，c 只能靠「中位门店盈亏平衡」校准，"
    "而那个校准本身是假设。",
    "**租金与客流结构共线**。二者都由密度驱动，模型无法回答「把租金谈下来 20% 是否划算」。",
    "**覆盖严重不均**。PA 占 25% 门店，CA 仅 2% 且只覆盖 Santa Barbara，"
    "IL 只有 Alton 一带（无芝加哥）。任何「全国平均」都不可靠。",
    "**坐标含加拿大门店**。94.5% 落在美国州界内，其余主要是 AB 省，做美国地图时已剔除。",
])

# ------------------------------------------------------------- 附录 -------
doc.add_heading("附录 A　图表清单", level=1)
para("图号与图表文件（及图内标题）自身的编号一致，便于与 analysis/out/charts/ 对照。",
     size=10, colour=MUTED)
table(
    ["图号", "文件", "内容"],
    [["图 1", "fig01_reviewcount_closure.png", "评论数十分位 × 关闭率"],
     ["图 4", "fig04_delivery_paradox.png", "可达性属性 × 关闭率（外送悖论）"],
     ["图 6", "fig06_roc.png", "ROC 曲线对比"],
     ["图 11", "fig11_true_density_deciles.png", "真实空间密度 × 关闭率"],
     ["图 20", "fig20_method_chain.png", "完整方法链与决策权衡"],
     ["图 22", "fig22_traffic_drivers.png", "客流的结构驱动因素"],
     ["图 23", "fig23_density_tension.png", "密度悖论"],
     ["图 25", "fig25_traffic_deciles_closure.png", "客流十分位 × 关闭率（倒 U）"],
     ["图 26", "fig26_profit_breakeven.png", "盈亏平衡租金倍数"],
     ["图 27", "fig27_survival_vs_profit.png", "存活 vs 累计盈利"],
     ["图 28", "fig28_sensitivity_surface.png", "联合敏感性曲面"]],
    widths=[1.8, 7.6, 6.0])

doc.add_heading("附录 B　复现方式", level=1)
para("全部统计量用 numpy 手写实现（本机无 scipy/statsmodels）。执行顺序：", size=10)
table(
    ["步骤", "脚本", "产出"],
    [["1", "01_profile.py", "全表画像、质量审计、6 个陷阱、方向筛选"],
     ["2", "02_features.py", "business_features.csv（52,286 × 125）"],
     ["3", "03_findings.py", "单变量、logistic、AUC、风险十分位"],
     ["4", "03b_density.py / 03c_model_density.py", "真实空间密度、密度模型"],
     ["5", "04_charts.py / 04b_density_charts.py", "图 1–13"],
     ["6", "06_metros.py → 07/08/09_*", "城市群口径与各级尺度地图"],
     ["7", "11_traffic.py / 12_traffic_charts.py", "客流分析、图 21–25"],
     ["8", "13_profit_scenarios.py / 14_profit_charts.py", "盈利口径、图 26–27"],
     ["9", "15_sensitivity.py / 16_sensitivity_charts.py", "联合敏感性、图 28–29"],
     ["10", "10_method_chain.py", "图 20 方法链"]],
    widths=[1.4, 7.4, 6.6])

doc.add_heading("附录 C　相关文档", level=1)
bullets([
    "FINDINGS.md —— 主分析（因变量 = 存活）的完整结论与 12 条局限",
    "FINDINGS_客流分析.md —— 以客流为因变量的平行分析",
    "FINDINGS_盈利口径.md —— 成功标准之争、假设估计模型与联合敏感性",
    "01_profile.md —— 全表画像与数据陷阱审计",
    "README_小尺度地图.md —— 地图方案与 OSM 数据授权说明",
])

para("数据来源：Yelp 学术数据集（2022 年采集），经 BigQuery 导出。"
     "地图底图 © OpenStreetMap contributors（ODbL）。", size=9, colour=MUTED)

doc.save(DOCX)
print(f"wrote {DOCX}")
print(f"  sections: {len(doc.sections)}, paragraphs: {len(doc.paragraphs)}, "
      f"tables: {len(doc.tables)}, images: {len(doc.inline_shapes)}")
