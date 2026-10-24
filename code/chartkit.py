"""Minimal charting toolkit on Pillow - axes, gridlines, ticks, labels, legend.

No matplotlib on this machine, so this provides just enough to render clear,
well-labelled charts. All text uses a CJK-capable TrueType font.
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

# palette (colour-blind friendly-ish, consistent across all charts)
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
ACCENT = "#2f6f9f"      # primary series
ACCENT2 = "#c1553b"     # contrast series
ACCENT3 = "#5b8c5a"     # third series
WARN = "#b7791f"
BG = "#ffffff"

FONT_REG = "msyh.ttc"
FONT_BOLD = "msyhbd.ttc"


def font(size: int, bold: bool = False):
    for name in ([FONT_BOLD] if bold else []) + [FONT_REG, "simhei.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default(size=size)


def text_size(draw, s, f):
    box = draw.textbbox((0, 0), s, font=f)
    return box[2] - box[0], box[3] - box[1]


class Chart:
    """A figure with a single plot area, drawn in data coordinates."""

    def __init__(self, w=1100, h=700, title="", subtitle="", xlabel="", ylabel="",
                 pad_l=95, pad_r=40, pad_t=95, pad_b=95):
        self.W, self.H = w, h
        self.pad_l, self.pad_r, self.pad_t, self.pad_b = pad_l, pad_r, pad_t, pad_b
        self.img = Image.new("RGB", (w, h), BG)
        self.d = ImageDraw.Draw(self.img)
        self.title = title
        self.subtitle = subtitle
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = (0, 1)
        self.ylim = (0, 1)
        self.f_title = font(26, True)
        self.f_sub = font(15)
        self.f_tick = font(15)
        self.f_axis = font(17, True)
        self.f_leg = font(15)

    # -- coordinate mapping ------------------------------------------------
    @property
    def plot(self):
        return (self.pad_l, self.pad_t, self.W - self.pad_r, self.H - self.pad_b)

    def set_xlim(self, lo, hi):
        self.xlim = (lo, hi if hi != lo else lo + 1)

    def set_ylim(self, lo, hi):
        self.ylim = (lo, hi if hi != lo else lo + 1)

    def px(self, x):
        x0, _, x1, _ = self.plot
        lo, hi = self.xlim
        return x0 + (x - lo) / (hi - lo) * (x1 - x0)

    def py(self, y):
        _, y0, _, y1 = self.plot
        lo, hi = self.ylim
        return y1 - (y - lo) / (hi - lo) * (y1 - y0)

    # -- chrome ------------------------------------------------------------
    def draw_frame(self, n_yticks=6, n_xticks=6, y_fmt="{:.2f}", x_fmt="{:.2f}",
                   x_ticks=None, y_ticks=None, grid=True):
        x0, y0, x1, y1 = self.plot
        # gridlines + y ticks
        lo, hi = self.ylim
        ys = y_ticks if y_ticks is not None else [
            lo + (hi - lo) * i / (n_yticks - 1) for i in range(n_yticks)
        ]
        for v in ys:
            yy = self.py(v)
            if grid:
                self.d.line([(x0, yy), (x1, yy)], fill=GRID, width=1)
            lab = y_fmt.format(v)
            tw, th = text_size(self.d, lab, self.f_tick)
            self.d.text((x0 - 10 - tw, yy - th / 2 - 2), lab, font=self.f_tick, fill=MUTED)
        # x ticks
        lo, hi = self.xlim
        xs = x_ticks if x_ticks is not None else [
            lo + (hi - lo) * i / (n_xticks - 1) for i in range(n_xticks)
        ]
        for v in xs:
            xx = self.px(v)
            if grid:
                self.d.line([(xx, y0), (xx, y1)], fill="#f0f2f4", width=1)
            lab = x_fmt.format(v)
            tw, th = text_size(self.d, lab, self.f_tick)
            self.d.text((xx - tw / 2, y1 + 8), lab, font=self.f_tick, fill=MUTED)
        # axes lines
        self.d.line([(x0, y0), (x0, y1)], fill=INK, width=1)
        self.d.line([(x0, y1), (x1, y1)], fill=INK, width=1)
        # titles
        if self.title:
            self.d.text((self.pad_l, 26), self.title, font=self.f_title, fill=INK)
        if self.subtitle:
            self.d.text((self.pad_l, 62), self.subtitle, font=self.f_sub, fill=MUTED)
        if self.xlabel:
            tw, _ = text_size(self.d, self.xlabel, self.f_axis)
            self.d.text(((x0 + x1) / 2 - tw / 2, self.H - 42), self.xlabel,
                        font=self.f_axis, fill=INK)
        if self.ylabel:
            tmp = Image.new("RGBA", (10, 10))
            td = ImageDraw.Draw(tmp)
            tw, th = text_size(td, self.ylabel, self.f_axis)
            strip = Image.new("RGBA", (tw + 6, th + 6), (0, 0, 0, 0))
            sd = ImageDraw.Draw(strip)
            sd.text((3, 3), self.ylabel, font=self.f_axis, fill=INK)
            strip = strip.rotate(90, expand=True)
            self.img.paste(strip, (14, int((y0 + y1) / 2 - strip.height / 2)), strip)

    # -- primitives --------------------------------------------------------
    def line(self, pts, color=ACCENT, width=3):
        self.d.line([(self.px(x), self.py(y)) for x, y in pts], fill=color, width=width)

    def bars(self, centers, heights, width_frac=0.72, color=ACCENT,
             y_base=None, outline=None, colors=None):
        if len(centers) > 1:
            step = abs(self.px(centers[1]) - self.px(centers[0]))
        else:
            step = 30
        w = step * width_frac
        base = self.py(y_base if y_base is not None else self.ylim[0])
        for i, (c, h) in enumerate(zip(centers, heights)):
            x = self.px(c)
            col = colors[i] if colors else color
            top = self.py(h)
            self.d.rectangle([x - w / 2, min(top, base), x + w / 2, max(top, base)],
                             fill=col, outline=outline)

    def errorbar(self, x, y, lo, hi, color=INK, width=2, cap=5):
        xx = self.px(x)
        self.d.line([(xx, self.py(lo)), (xx, self.py(hi))], fill=color, width=width)
        for v in (lo, hi):
            self.d.line([(xx - cap, self.py(v)), (xx + cap, self.py(v))], fill=color, width=width)

    def marker(self, x, y, r=5, color=ACCENT, outline=BG):
        xx, yy = self.px(x), self.py(y)
        self.d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=color, outline=outline, width=2)

    def point(self, x, y, r=6, color=ACCENT, outline=BG, label=None, label_dy=-22):
        self.marker(x, y, r, color, outline)
        if label:
            tw, th = text_size(self.d, label, self.f_leg)
            self.d.text((self.px(x) - tw / 2, self.py(y) + label_dy), label,
                        font=self.f_leg, fill=INK)

    def hline(self, y, color=WARN, width=2, dash=8):
        x0, _, x1, _ = self.plot
        yy = self.py(y)
        x = x0
        while x < x1:
            self.d.line([(x, yy), (min(x + dash, x1), yy)], fill=color, width=width)
            x += dash * 2

    def vline(self, x, color=WARN, width=2, dash=8):
        _, y0, _, y1 = self.plot
        xx = self.px(x)
        y = y0
        while y < y1:
            self.d.line([(xx, y), (xx, min(y + dash, y1))], fill=color, width=width)
            y += dash * 2

    def legend(self, entries, x=None, y=None, boxed=True):
        """entries: list of (label, colour) or (label, colour, 'line'|'box')."""
        pad = 10
        f = self.f_leg
        widths = []
        for e in entries:
            tw, _ = text_size(self.d, e[0], f)
            widths.append(tw + 30)
        bw = max(widths) + pad * 2
        bh = len(entries) * 24 + pad * 2
        x0, y0, x1, y1 = self.plot
        lx = x if x is not None else x0 + 14
        ly = y if y is not None else y0 + 12
        if boxed:
            self.d.rectangle([lx, ly, lx + bw, ly + bh], fill=BG, outline=GRID)
        for i, e in enumerate(entries):
            yy = ly + pad + i * 24
            col = e[1]
            kind = e[2] if len(e) > 2 else "box"
            if kind == "line":
                self.d.line([(lx + pad, yy + 8), (lx + pad + 16, yy + 8)], fill=col, width=3)
            else:
                self.d.rectangle([lx + pad, yy + 2, lx + pad + 14, yy + 14], fill=col, outline=col)
            self.d.text((lx + pad + 22, yy), e[0], font=f, fill=INK)
        return lx + bw, ly + bh

    def footnote(self, text):
        self.d.text((self.pad_l, self.H - 24), text, font=self.f_sub, fill=MUTED)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.img.save(path, quality=95)
        print("  rendered", os.path.basename(path))
