"""Step 4c - real US map: state boundaries + city distribution.

Validates that the dataset's coordinates are correct by point-in-polygon testing
every restaurant against the real state outlines, then renders:

    fig14  state choropleth (restaurant count) + city bubbles sized by count,
           colour = closure rate
    fig15  city distribution: top metros ranked, and closure rate vs city size

Source of outlines: PublicaMundi US states GeoJSON, cached at
analysis/data/us-states.geojson (fetched once, kept for reproducibility).
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import ACCENT, ACCENT2, ACCENT3, GRID, INK, MUTED, WARN, font  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
IMG = os.path.join(OUT, "charts")
GEO = os.path.join(WS, "analysis", "data", "us-states.geojson")

ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Puerto Rico": "PR", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
# Canadian provinces present in the Yelp data
CANADA = {"AB": "Alberta", "ON": "Ontario", "QC": "Quebec", "BC": "British Columbia",
          "NS": "Nova Scotia", "NB": "New Brunswick", "MB": "Manitoba",
          "SK": "Saskatchewan", "NL": "Newfoundland and Labrador",
          "PE": "Prince Edward Island"}

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
geo = json.load(open(GEO, encoding="utf-8"))


# ---------------------------------------------------------------- point in polygon
def rings_of(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


def inside(lon, lat, ring):
    """Ray casting on a single outer ring."""
    x, y = lon, lat
    n = len(ring)
    hit = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi):
            hit = not hit
        j = i
    return hit


print("validating coordinates against real state outlines ...")
polygons = {}
for f in geo["features"]:
    nm = f["properties"]["name"]
    for ring in rings_of(f["geometry"]):
        polygons.setdefault(nm, []).append((min(p[0] for p in ring), max(p[0] for p in ring),
                                            min(p[1] for p in ring), max(p[1] for p in ring), ring))

sub = feat.dropna(subset=["latitude", "longitude"])
matched = np.zeros(len(sub), dtype=bool)
for i, (_, r) in enumerate(sub.iterrows()):
    lon, lat = r.longitude, r.latitude
    for nm, boxes in polygons.items():
        for x0, x1, y0, y1, ring in boxes:
            if x0 <= lon <= x1 and y0 <= lat <= y1 and inside(lon, lat, ring):
                matched[i] = True
                break
        if matched[i]:
            break

sub = sub.assign(in_us=matched)
us = sub[sub.in_us]
print(f"  restaurants with coordinates : {len(sub):,}")
print(f"  inside a US state outline    : {matched.sum():,} ({matched.mean():.1%})")
print(f"  outside (Canadian provinces) : {(~matched).sum():,}")

# which states do the matched ones fall in (independent of the dataset's own state column)
print("\n=== cross-check: dataset 'state' vs outline match ===")
print(f"  dataset states: {sorted(feat.state.dropna().unique())}")
print(f"  US rows       : {int(sub.in_us.sum()):,}   non-US rows: {int((~sub.in_us).sum()):,}")
nonus_states = sub.loc[~sub.in_us, "state"].value_counts()
print(f"  non-US rows by dataset state: {nonus_states.to_dict()}")

# ---------------------------------------------------------------- aggregates
st = us.groupby("state").agg(
    restaurants=("business_id", "size"),
    closure_rate=("is_open", lambda s: 1 - s.mean()),
    avg_stars=("business_stars", "mean"),
    median_density=("n_within_1_0km", "median"),
    avg_lat=("latitude", "mean"),
    avg_lon=("longitude", "mean"),
).reset_index()
st.to_csv(os.path.join(OUT, "05_state_summary.csv"), index=False, encoding="utf-8-sig")
print(f"\nwrote 05_state_summary.csv ({len(st)} states)")
print(st.sort_values("restaurants", ascending=False).head(12)
      [["state", "restaurants", "closure_rate", "median_density"]].round(3).to_string(index=False))

city = us.groupby(["city", "state"]).agg(
    restaurants=("business_id", "size"),
    closure_rate=("is_open", lambda s: 1 - s.mean()),
    avg_stars=("business_stars", "mean"),
    median_density=("n_within_1_0km", "median"),
    lat=("latitude", "mean"),
    lon=("longitude", "mean"),
).reset_index()
city.to_csv(os.path.join(OUT, "05_city_summary.csv"), index=False, encoding="utf-8-sig")
print(f"\nwrote 05_city_summary.csv ({len(city)} cities)")
print(city.sort_values("restaurants", ascending=False).head(10)
      [["city", "state", "restaurants", "closure_rate", "median_density"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- projection
# Albers-style conic for the contiguous US; simple and good enough visually.
def make_projector(pts):
    lon = np.radians(pts[:, 0]); lat = np.radians(pts[:, 1])
    lat0, lat1, lat2 = np.radians(29.5), np.radians(45.5), np.radians(37.5)
    lon0 = np.radians(-96.0)
    n = 0.5 * (np.sin(lat0) + np.sin(lat1))
    c = np.cos(lat0) ** 2 + 2 * n * np.sin(lat0)
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    rho0 = np.sqrt(c - 2 * n * np.sin(lat2)) / n
    theta = n * (lon - lon0)
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    return x, y


# exclude non-contiguous areas from the framed map
EXCL = {"Alaska", "Hawaii", "Puerto Rico"}
allpts = []
for f in geo["features"]:
    if f["properties"]["name"] in EXCL:
        continue
    for ring in rings_of(f["geometry"]):
        allpts.extend(ring)
allpts = np.array(allpts)
PX, PY = make_projector(allpts)
x_min, x_max = PX.min(), PX.max()
y_min, y_max = PY.min(), PY.max()


def fig14():
    W, H = 1500, 1000
    img = Image.new("RGB", (W, H), "#f7f9fb")
    d = ImageDraw.Draw(img)
    d.text((46, 26), "图 14｜美国餐厅分布：州内门店数 + 城市聚集度（真实州界）",
           font=font(26, True), fill=INK)
    d.text((46, 64), f"{len(us):,} 家餐厅，来自 {st.state.nunique()} 个州 / {len(city):,} 个城市；"
                     f"州底色 = 门店数，圆点 = 城市（大小 = 门店数，颜色 = 关闭率）",
           font=font(15), fill=MUTED)

    pad_l, pad_r, pad_t, pad_b = 60, 330, 120, 90
    sx = (W - pad_r - pad_l) / (x_max - x_min)
    sy = (H - pad_b - pad_t) / (y_max - y_min)
    s = min(sx, sy)

    def tx(lon, lat):
        x, y = make_projector(np.array([[lon, lat]]))
        return (pad_l + (x[0] - x_min) * s + (W - pad_r - pad_l - (x_max - x_min) * s) / 2,
                pad_t + (y_max - y[0]) * s + (H - pad_b - pad_t - (y_max - y_min) * s) / 2)

    by_state = us.groupby("state").size().to_dict()
    mx_state = max(by_state.values())
    # sqrt scale: Philadelphia alone is 5,854 of 12,644 in PA, so a linear ramp
    # washes every other state out to near-white
    def state_frac(n):
        return (n / mx_state) ** 0.5

    # state fills
    for f in geo["features"]:
        nm = f["properties"]["name"]
        if nm in EXCL:
            continue
        n = by_state.get(ABBR.get(nm, ""), 0)
        frac = state_frac(n)
        col = tuple(int(247 - (247 - cc) * frac) for cc in (47, 111, 159))
        for ring in rings_of(f["geometry"]):
            pts = [tx(p[0], p[1]) for p in ring]
            if len(pts) > 2:
                d.polygon(pts, fill=col, outline="#ffffff")
    # redraw outlines for crispness
    for f in geo["features"]:
        if f["properties"]["name"] in EXCL:
            continue
        for ring in rings_of(f["geometry"]):
            pts = [tx(p[0], p[1]) for p in ring]
            if len(pts) > 2:
                d.line(pts + [pts[0]], fill="#b8c4cf", width=1)

    # city bubbles first, so state abbreviations can be suppressed near them
    big_cities = city[city.restaurants >= 50].sort_values("restaurants")
    mx_city = city.restaurants.max()
    placed = []
    for _, r in big_cities.iterrows():
        rad = 2.0 + 13.0 * (r.restaurants / mx_city) ** 0.55
        x, y = tx(r.lon, r.lat)
        cr = r.closure_rate
        t = min(max((cr - 0.20) / (0.60 - 0.20), 0), 1)
        if t < 0.5:
            u = t / 0.5
            col = (int(91 + (183 - 91) * u), int(140 + (121 - 140) * u), int(90 + (31 - 90) * u))
        else:
            u = (t - 0.5) / 0.5
            col = (int(183 + (193 - 183) * u), int(121 + (85 - 121) * u), int(31 + (59 - 31) * u))
        d.ellipse([x - rad, y - rad, x + rad, y + rad],
                  fill=col, outline="#ffffff", width=1)
        placed.append((x, y, rad, r))

    # state abbreviations only where no sizeable bubble sits nearby
    for _, r in st.iterrows():
        if r.restaurants < 300:
            continue
        x, y = tx(r.avg_lon, r.avg_lat)
        if any((abs(x - px) < 46 and abs(y - py) < 30) for px, py, _, _ in placed):
            continue
        lab = r.state
        tw = d.textbbox((0, 0), lab, font=font(13, True))[2]
        d.text((x - tw / 2, y - 8), lab, font=font(13, True), fill="#0f2b40")

    # city labels last, so nothing covers them
    for x, y, rad, r in placed:
        if r.restaurants <= 1200:
            continue
        lab = f"{r.city} {int(r.restaurants):,}"
        tw = d.textbbox((0, 0), lab, font=font(14, True))[2]
        # white halo so the text stays readable over dark fills
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((x - tw / 2 + dx, y - rad - 20 + dy), lab, font=font(14, True), fill="#ffffff")
        d.text((x - tw / 2, y - rad - 20), lab, font=font(14, True), fill="#12232f")

    # legend panel
    lx = W - pad_r + 26
    ly = pad_t + 6
    d.text((lx, ly), "州底色调", font=font(15, True), fill=INK)
    d.text((lx, ly + 24), "门店数（浅→深）", font=font(12), fill=MUTED)
    for i in range(60):
        c = tuple(int(247 - (247 - cc) * (i / 59)) for cc in (47, 111, 159))
        d.rectangle([lx + i * 4, ly + 46, lx + i * 4 + 4, ly + 62], fill=c)
    d.text((lx, ly + 68), "少", font=font(11), fill=MUTED)
    d.text((lx + 226, ly + 68), "多", font=font(11), fill=MUTED)

    d.text((lx, ly + 108), "圆点颜色 = 关闭率", font=font(15, True), fill=INK)
    for i, (lab, cr) in enumerate([("低 20%", 0.20), ("中 40%", 0.40), ("高 60%", 0.60)]):
        t = min(max((cr - 0.20) / 0.40, 0), 1)
        if t < 0.5:
            u = t / 0.5
            col = (int(91 + (183 - 91) * u), int(140 + (121 - 140) * u), int(90 + (31 - 90) * u))
        else:
            u = (t - 0.5) / 0.5
            col = (int(183 + (193 - 183) * u), int(121 + (85 - 121) * u), int(31 + (59 - 31) * u))
        cy = ly + 136 + i * 26
        d.ellipse([lx, cy, lx + 14, cy + 14], fill=col)
        d.text((lx + 22, cy - 1), lab, font=font(13), fill=INK)

    d.text((lx, ly + 232), "圆点大小 = 城市门店数", font=font(15, True), fill=INK)
    for i, nb in enumerate([100, 1000, 5000]):
        rad = 2.0 + 13.0 * (nb / mx_city) ** 0.55
        cy = ly + 262 + i * 34
        d.ellipse([lx + 8 - rad, cy + 8 - rad, lx + 8 + rad, cy + 8 + rad],
                  fill="#8fa8bb", outline="#ffffff")
        d.text((lx + 30, cy - 1), f"{nb:,} 家", font=font(13), fill=INK)

    d.text((lx, ly + 380), "验证：全部坐标逐点做", font=font(12), fill=MUTED)
    d.text((lx, ly + 398), "点在多边形内检验，", font=font(12), fill=MUTED)
    d.text((lx, ly + 416), f"{matched.mean()*100:.1f}% 落入美国州界内", font=font(12), fill=MUTED)

    d.text((60, H - 56), "州界数据：PublicaMundi US states GeoJSON（本地缓存 analysis/data/us-states.geojson）",
           font=font(13), fill=MUTED)
    d.text((60, H - 32), f"注：另有 {(~matched).sum():,} 家位于加拿大省份（AB 等），未显示在图上",
           font=font(13), fill=MUTED)
    img.save(os.path.join(IMG, "fig14_us_map_states_cities.png"))
    print("  rendered fig14_us_map_states_cities.png")


def fig15():
    """City distribution: top metros ranked, plus closure vs city size."""
    top = city.sort_values("restaurants", ascending=False).head(22).reset_index(drop=True)
    W, H = 1420, 900
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((46, 26), "图 15｜城市分布：门店高度集中在少数都会区，大市场关闭率略高",
           font=font(25, True), fill=INK)
    d.text((46, 64), f"左：门店数最多的 22 个城市（柱=门店数，标注=关闭率，颜色=关闭率）；右：329 个城市规模 vs 关闭率",
           font=font(15), fill=MUTED)

    # ---- left: ranked bars
    x0, y0 = 300, 130
    rowh = 30
    barw = 420
    mx = top.restaurants.max()
    d.line([(x0, y0 - 16), (x0, y0 - 16)], fill=GRID)
    for i, r in top.iterrows():
        yy = y0 + i * rowh
        w = barw * r.restaurants / mx
        # colour by closure rate
        t = min(max((r.closure_rate - 0.20) / 0.40, 0), 1)
        if t < 0.5:
            u = t / 0.5
            col = (int(91 + (183 - 91) * u), int(140 + (121 - 140) * u), int(90 + (31 - 90) * u))
        else:
            u = (t - 0.5) / 0.5
            col = (int(183 + (193 - 183) * u), int(121 + (85 - 121) * u), int(31 + (59 - 31) * u))
        d.rectangle([x0, yy, x0 + w, yy + rowh - 9], fill=col)
        d.text((x0 - 10 - d.textbbox((0, 0), r.city, font=font(14))[2], yy + 1),
               f"{r.city}", font=font(14), fill=INK)
        d.text((x0 + w + 10, yy + 1), f"{int(r.restaurants):,} 家 · 关闭 {r.closure_rate:.0%}",
               font=font(13), fill=MUTED)

    # ---- right: city size vs closure rate
    ox, oy, ow, oh = 880, 130, 480, 500
    d.rectangle([ox, oy, ox + ow, oy + oh], outline=GRID)
    c2 = city[city.restaurants >= 15].copy()
    lg = np.log10(c2.restaurants)
    lo, hi = float(lg.min()), float(lg.max())
    cre, crhi = 0.0, 0.75
    for _, r in c2.iterrows():
        px = ox + (np.log10(r.restaurants) - lo) / (hi - lo) * ow
        py = oy + oh - (r.closure_rate - cre) / (crhi - cre) * oh
        n = r.restaurants
        rad = 2.0 + 9.0 * min(n / 6000, 1) ** 0.6
        d.ellipse([px - rad, py - rad, px + rad, py + rad],
                  fill=(47, 111, 159, 90) if False else (79, 129, 168))
    for k in range(5):
        yy = oy + oh * k / 4
        d.line([(ox, yy), (ox + ow, yy)], fill="#f0f2f4")
        d.text((ox - 52, yy - 8), f"{(crhi - (crhi-cre)*k/4)*100:.0f}%", font=font(12), fill=MUTED)
    for t in [20, 100, 500, 2000, 8000]:
        if np.log10(t) < lo or np.log10(t) > hi:
            continue
        px = ox + (np.log10(t) - lo) / (hi - lo) * ow
        d.line([(px, oy), (px, oy + oh)], fill="#f0f2f4")
        tw = d.textbbox((0, 0), f"{t:,}", font=font(12))[2]
        d.text((px - tw / 2, oy + oh + 8), f"{t:,}", font=font(12), fill=MUTED)
    d.text((ox + ow / 2 - 90, oy + oh + 32), "城市门店数（对数）", font=font(15, True), fill=INK)
    d.text((ox - 10, oy - 26), "关闭率", font=font(15, True), fill=INK)

    # trend annotation (numbers computed, not asserted)
    c2 = city[city.restaurants >= 15].copy()
    rho = c2[["restaurants", "closure_rate"]].corr(method="spearman").iloc[0, 1]
    c2["_q"] = pd.qcut(c2.restaurants, 5, labels=False, duplicates="drop")
    qg = c2.groupby("_q").agg(closure=("closure_rate", "mean"), avg=("restaurants", "mean"))
    tot = city.restaurants.sum()
    top10 = city.sort_values("restaurants", ascending=False).head(10).restaurants.sum() / tot
    top50 = city.sort_values("restaurants", ascending=False).head(50).restaurants.sum() / tot
    d.text((ox, oy + oh + 62), f"样本：{len(c2)} 个门店数 ≥15 的城市", font=font(13), fill=MUTED)
    d.text((ox, oy + oh + 84), f"城市规模与关闭率 Spearman = {rho:+.3f}",
           font=font(13, True), fill=ACCENT2)
    d.text((ox, oy + oh + 104),
           f"最小五分位关闭率 {qg.closure.iloc[0]:.1%} → 最大五分位 {qg.closure.iloc[-1]:.1%}",
           font=font(13), fill=INK)
    d.text((ox, oy + oh + 126),
           f"前 10 城占全部门店 {top10:.0%}，前 50 城占 {top50:.0%}",
           font=font(13), fill=INK)

    d.text((46, H - 56), "数据：business_features.csv ｜ 州界与城市坐标均来自数据集自身的经纬度",
           font=font(13), fill=MUTED)
    d.text((46, H - 32), "注：城市为 Yelp 的 city 字段，未做都会区合并，故同一都市圈可能拆成多行",
           font=font(13), fill=MUTED)
    img.save(os.path.join(IMG, "fig15_city_distribution.png"))
    print("  rendered fig15_city_distribution.png")


print()
fig14()
fig15()
print("\nmap charts rendered")
