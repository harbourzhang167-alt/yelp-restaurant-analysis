"""Step 7 - per-state spatial density maps (fig13 style, six states per image).

Each panel plots every restaurant in that state at its true coordinate, over the
real state outline, with neighbouring outlines for context.

Two variants:
    fig16a..c  colour = closure status (red closed / green open), size = density
    fig17a..c  colour = competition density (light -> dark)

Rendering notes: each panel is drawn onto its own layer the size of the panel so
it clips cleanly, then pasted; header and stats bands are painted on top of the
layer so map content can never cover the text.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import GRID, INK, MUTED, font  # noqa: E402
from mapkit import (ABBR, GREEN, RED, Projector, density_colour, load_states,  # noqa: E402
                    rings_of)
from PIL import Image, ImageDraw  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
IMG = os.path.join(OUT, "charts")
os.makedirs(IMG, exist_ok=True)

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
states_geo = load_states()

DS2FULL = {"AZ": "Arizona", "CA": "California", "CO": "Colorado", "DE": "Delaware",
           "FL": "Florida", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
           "LA": "Louisiana", "MO": "Missouri", "MT": "Montana",
           "NC": "North Carolina", "NJ": "New Jersey", "NV": "Nevada",
           "PA": "Pennsylvania", "TN": "Tennessee"}

us = feat[feat.state.isin(DS2FULL)].dropna(subset=["latitude", "longitude"])
order = us.groupby("state").size().sort_values(ascending=False)
PER_IMG = 6
panels = [list(order.index[i:i + PER_IMG]) for i in range(0, len(order), PER_IMG)]
print(f"states: {list(order.index)} -> {len(panels)} images")


def panel_image(st, w, h, colour_by, global_mx, half_lon, half_lat):
    """Render one state panel on its own layer, so content clips to the box."""
    layer = Image.new("RGB", (w, h), "#f8fafc")
    d = ImageDraw.Draw(layer)
    s = us[us.state == st]
    full = DS2FULL[st]

    # project the FOCAL state to fill the panel, then expand the data bounds a
    # little so neighbouring context is visible without the state going off-frame
    # Fit the projection to the STATE POLYGON, not to the data extent. Fitting to
    # the data makes a handful of far-flung outlets blow up the frame (Nevada's few
    # non-Reno rows otherwise shrink its core to a dot). With a polygon fit, a
    # state whose restaurants cluster in one corner (Idaho -> Boise, California ->
    # Santa Barbara) simply shows its outline with the cluster in that corner.
    arr_all = np.vstack([np.asarray(r, float) for r in rings_of(states_geo[full])])
    lo0, lo1 = arr_all[:, 0].min(), arr_all[:, 0].max()
    la0, la1 = arr_all[:, 1].min(), arr_all[:, 1].max()
    m_lon = (lo1 - lo0) * 0.10
    m_lat = (la1 - la0) * 0.10
    view_lon = (lo0 - m_lon, lo1 + m_lon)
    view_lat = (la0 - m_lat, la1 + m_lat)
    box = (6, 6, w - 6, h - 6)
    proj = Projector(box, list(view_lon), list(view_lat))

    # Only draw states that actually intersect the viewport. Drawing all 48 turns
    # a large-bounds state (Nevada) into a picture of the whole country.
    def intersects(geom):
        for ring in rings_of(geom):
            arr = np.asarray(ring, float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            lo0, lo1 = arr[:, 0].min(), arr[:, 0].max()
            la0, la1 = arr[:, 1].min(), arr[:, 1].max()
            if not (lo1 < view_lon[0] or lo0 > view_lon[1]
                    or la1 < view_lat[0] or la0 > view_lat[1]):
                return True
        return False

    for name, geom in states_geo.items():
        if name == full or name in ("Alaska", "Hawaii", "Puerto Rico"):
            continue
        if not intersects(geom):
            continue
        for ring in rings_of(geom):
            arr = np.asarray(ring, float)
            px, py = proj(arr[:, 0], arr[:, 1])
            pts = list(zip(px.tolist(), py.tolist()))
            if len(pts) > 2:
                d.polygon(pts, fill="#e3eaf0", outline="#cfdae3")

    for ring in rings_of(states_geo[full]):
        arr = np.asarray(ring, float)
        px, py = proj(arr[:, 0], arr[:, 1])
        pts = list(zip(px.tolist(), py.tolist()))
        if len(pts) > 2:
            d.polygon(pts, fill="#eef3f7", outline="#6f8496")
            d.line(pts + [pts[0]], fill="#6f8496", width=2)

    # points: closed last so red is not buried under green
    dens = s["n_within_1_0km"].to_numpy(float)
    px, py = proj(s.longitude.to_numpy(float), s.latitude.to_numpy(float))
    isopen = s.is_open.to_numpy()
    for i in np.argsort(isopen):
        r = 1.5 + 2.5 * (dens[i] / max(global_mx, 1)) ** 0.35
        col = (RED if isopen[i] == 0 else GREEN) if colour_by == "closure" \
            else density_colour(dens[i], global_mx)
        d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=col)

    # city labels: keep only top 3, and skip any that would overlap one already drawn
    top = s.groupby("city").size().sort_values(ascending=False).head(3)
    placed_boxes = []
    for city, n in top.items():
        cs = s[s.city == city]
        cx, cy = proj(cs.longitude.mean(), cs.latitude.mean())
        if not (20 < cx < w - 20 and 56 < cy < h - 50):
            continue
        txt = f"{city} {n}"
        f = font(12, True)
        tw = d.textbbox((0, 0), txt, f)[2]
        bx0, by0, bx1, by1 = cx - tw / 2 - 3, cy - 3, cx + tw / 2 + 3, cy + 16
        if any(not (bx1 < ax0 or bx0 > ax1 or by1 < ay0 or by0 > ay1)
               for ax0, ay0, ax1, ay1 in placed_boxes):
            continue
        placed_boxes.append((bx0, by0, bx1, by1))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((cx - tw / 2 + dx, cy + dy), txt, font=f, fill="#ffffff")
        d.text((cx - tw / 2, cy), txt, font=f, fill="#12232f")

    # opaque header and footer bands
    d.rectangle([0, 0, w, 34], fill="#ffffff")
    d.rectangle([0, h - 46, w, h], fill="#ffffff")
    d.text((10, 6), f"{full} ({st})", font=font(17, True), fill=INK)
    d.text((10, h - 42), f"{len(s):,} 家 · 关闭 {1 - s.is_open.mean():.1%} · "
                         f"1km 内中位 {s['n_within_1_0km'].median():.0f} 家 · "
                         f"平均星级 {s.business_stars.mean():.2f}",
           font=font(13), fill=MUTED)
    return layer


for colour_by, tag in (("closure", "16"), ("density", "17")):
    global_mx = float(us["n_within_1_0km"].quantile(0.995))
    for pi, group in enumerate(panels):
        W, H = 1560, 1120
        img = Image.new("RGB", (W, H), "#ffffff")
        d = ImageDraw.Draw(img)
        suffix = chr(ord('a') + pi)
        if colour_by == "closure":
            title = f"图 16{suffix}｜各州餐厅空间分布：红=已关闭，绿=营业中（点大小 = 1km 内竞争对手数）"
        else:
            title = f"图 17{suffix}｜各州竞争密度：颜色越深 = 1km 内竞争对手越多"
        d.text((36, 24), title, font=font(24, True), fill=INK)
        d.text((36, 60), f"每点 = 一家餐厅的真实坐标；本图 6 个州，"
                         f"全数据集 {len(us):,} 家（含加拿大门店已剔除）；"
                         f"坐标经点在多边形内检验，94.5% 落入美国州界",
               font=font(14), fill=MUTED)

        pw = (W - 4 * 22) / 3
        ph = (H - 210) / 2
        for k, st in enumerate(group):
            cx = 22 + (k % 3) * (pw + 22)
            cy = 100 + (k // 3) * (ph + 14)
            layer = panel_image(st, int(pw), int(ph), colour_by, global_mx, 0.18, 0.18)
            img.paste(layer, (int(cx), int(cy)))
            d.rectangle([cx, cy, cx + pw, cy + ph], outline=GRID)

        # legend
        lx, ly = W - 430, H - 96
        if colour_by == "closure":
            d.ellipse([lx, ly, lx + 14, ly + 14], fill=RED)
            d.text((lx + 22, ly - 1), "已关闭", font=font(14), fill=INK)
            d.ellipse([lx + 120, ly, lx + 134, ly + 14], fill=GREEN)
            d.text((lx + 142, ly - 1), "营业中", font=font(14), fill=INK)
            d.text((lx, ly + 26), "点大小 = 1km 内竞争对手数（越大越密集）",
                   font=font(13), fill=MUTED)
        else:
            for i, nb in enumerate([0, 40, 150, 400, 900]):
                col = density_colour(nb, global_mx)
                d.rectangle([lx + i * 52, ly, lx + i * 52 + 40, ly + 16], fill=col,
                            outline="#ffffff")
            d.text((lx, ly + 26), "浅 → 深：1km 内竞争对手数（0 / 40 / 150 / 400 / 900）",
                   font=font(13), fill=MUTED)

        d.text((36, H - 46), "数据：business_features.csv ｜ 州界：PublicaMundi US states "
                             "GeoJSON（本地缓存 analysis/data/us-states.geojson）",
               font=font(12), fill=MUTED)
        d.text((36, H - 26), "注：门店密集的都会区在整州尺度上会有叠压，局部细节见对应州的放大图",
               font=font(12), fill=MUTED)

        name = f"fig{tag}{suffix}_state_maps_{'closure' if colour_by == 'closure' else 'density'}.png"
        img.save(os.path.join(IMG, name))
        print("  rendered", name)

print("done.")
