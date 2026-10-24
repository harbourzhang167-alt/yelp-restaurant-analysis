"""Flow-diagram toolkit on Pillow: wrapped text, rounded boxes, arrows, lanes.

Used for the methodology chain diagram. No graphviz/matplotlib available, so this
provides just enough primitives for a clean, readable process map.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

INK = "#1f2933"
MUTED = "#6b7a88"
LINE = "#c6d0d8"
BG = "#ffffff"

FONT_REG = "msyh.ttc"
FONT_BOLD = "msyhbd.ttc"


def font(size: int, bold: bool = False):
    for name in ([FONT_BOLD] if bold else []) + [FONT_REG, "simhei.ttf"]:
        try:
            from PIL import ImageFont
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def text_w(draw, s, f):
    return draw.textbbox((0, 0), s, f)[2]


def wrap(draw, text, f, max_w):
    """Wrap by characters (CJK has no spaces) while keeping latin words whole."""
    lines = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        i = 0
        while i < len(para):
            ch = para[i]
            # keep a latin/number token together
            if ch.isascii() and (ch.isalnum() or ch in "_-./%"):
                j = i
                while j < len(para) and para[j].isascii() and (para[j].isalnum() or para[j] in "_-./%"):
                    j += 1
                token = para[i:j]
                if text_w(draw, cur + token, f) <= max_w:
                    cur += token
                    i = j
                    continue
                if cur:
                    lines.append(cur)
                    cur = ""
                    continue
                lines.append(token)
                i = j
                continue
            if text_w(draw, cur + ch, f) <= max_w:
                cur += ch
                i += 1
            else:
                lines.append(cur)
                cur = ch
                i += 1
        lines.append(cur)
    return lines


def rounded_box(img, draw, x, y, w, h, fill, outline, radius=10, width=2):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius,
                           fill=fill, outline=outline, width=width)


def arrow(draw, pts, colour, width=3, head=9):
    """Polyline arrow with a filled head on the last segment."""
    for a, b in zip(pts, pts[1:]):
        draw.line([a, b], fill=colour, width=width)
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    for s in (+1, -1):
        a2 = ang + s * math.radians(26)
        draw.line([(x2, y2),
                   (x2 - head * math.cos(a2), y2 - head * math.sin(a2))],
                  fill=colour, width=width)


def badge(draw, x, y, text, fill, fg="#ffffff", pad=8, size=15, radius=8):
    f = font(size, True)
    w = text_w(draw, text, f) + pad * 2
    h = size + pad * 1.4
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)
    draw.text((x + pad, y + pad * 0.35), text, font=f, fill=fg)
    return w, h
