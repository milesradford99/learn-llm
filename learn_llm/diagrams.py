"""Diagrams for lesson pages, drawn as SVG with no plotting library."""

from html import escape

from learn_llm.plot import AXIS, DOT, LINE, TEXT

# Mia's one-day recipe: p = w * t, g = p + b, e = g - s, L = e * e.
_BOXES = {"w": (40, 40), "t": (40, 120), "p": (200, 80), "b": (200, 175),
          "g": (360, 125), "s": (360, 215), "e": (520, 170), "L": (660, 170)}
_EDGES = [("w", "p", "×"), ("t", "p", "×"), ("p", "g", "+"), ("b", "g", "+"),
          ("g", "e", "−"), ("s", "e", "−"), ("e", "L", "²")]
_BW, _BH = 76, 44


def one_day(values: dict | None = None, blames: dict | None = None, highlight: str = ""):
    """The recipe for one day's squared error as boxes and arrows.

    `values` fills each box's value (forward); `blames` writes each box's blame under it
    (backward). `highlight` names one box to outline.
    """
    import marimo as mo

    values, blames = values or {}, blames or {}
    parts = ['<svg viewBox="0 0 760 270" width="100%" style="max-width:760px" role="img" '
             'aria-label="w times t gives p; p plus b gives g; g minus s gives e; e squared gives L" '
             'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, monospace" text-anchor="middle">',
             f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{AXIS}"/></marker></defs>']
    for a, b, op in _EDGES:
        (xa, ya), (xb, yb) = _BOXES[a], _BOXES[b]
        x1, y1, x2, y2 = xa + _BW / 2, ya, xb - _BW / 2 - 4, yb
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{AXIS}" marker-end="url(#arrow)"/>')
        parts.append(f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 6}" font-size="15" fill="{TEXT}">{op}</text>')
    for name, (x, y) in _BOXES.items():
        stroke = DOT if name == highlight else AXIS
        width = 3 if name == highlight else 1.2
        parts.append(f'<rect x="{x - _BW / 2}" y="{y - _BH / 2}" width="{_BW}" height="{_BH}" rx="6" '
                     f'fill="none" stroke="{stroke}" stroke-width="{width}"/>')
        value = values.get(name)
        label = f"{name} = {value:g}" if isinstance(value, (int, float)) else name
        parts.append(f'<text x="{x}" y="{y + 5}" font-size="14" fill="{TEXT}">{escape(label)}</text>')
        if name in blames:
            parts.append(f'<text x="{x}" y="{y + _BH / 2 + 17}" font-size="13" fill="{LINE}">'
                         f'{escape(str(blames[name]))}</text>')
    parts.append("</svg>")
    return mo.Html("".join(parts))


def looking(read: list[str], shares: list[float], asker: str, label: str):
    """One step of attention: the `asker` box below looks at each `read` box above.

    A thicker line means a bigger share; each line carries its share. `label` describes the
    picture for screen readers, in the lesson's language.
    """
    import marimo as mo

    n, gap, top, bottom = len(read), 150, 34, 170
    width = max(gap * n, 300)
    xs = [width / 2 + gap * (i - (n - 1) / 2) for i in range(n)]
    parts = [f'<svg viewBox="0 0 {width} 200" width="100%" style="max-width:{width}px" role="img" '
             f'aria-label="{escape(label)}" xmlns="http://www.w3.org/2000/svg" '
             'font-family="ui-monospace, monospace" text-anchor="middle">']
    for x, share in zip(xs, shares):
        parts.append(f'<line x1="{width / 2}" y1="{bottom - _BH / 2}" x2="{x}" y2="{top + _BH / 2}" stroke="{DOT}" '
                     f'stroke-width="{1 + 10 * share:.1f}" stroke-linecap="round" opacity="0.75"/>')
        mx, my = (width / 2 + 2 * x) / 3, (bottom - _BH / 2 + 2 * (top + _BH / 2)) / 3
        parts.append(f'<text x="{mx + 24:.0f}" y="{my + 4:.0f}" font-size="13" fill="{TEXT}">{share:.2f}</text>')
    for x, y, text in [(x, top, word) for x, word in zip(xs, read)] + [(width / 2, bottom, asker)]:
        parts.append(f'<rect x="{x - 55}" y="{y - _BH / 2}" width="110" height="{_BH}" rx="6" '
                     f'fill="none" stroke="{AXIS}" stroke-width="1.2"/>')
        parts.append(f'<text x="{x}" y="{y + 5}" font-size="14" fill="{TEXT}">{escape(text)}</text>')
    parts.append("</svg>")
    return mo.Html("".join(parts))


def block_road(x: str, norm: str, attention: str, think: str, label: str):
    """A transformer block: two parts, each on an `x + f(x)` road that passes it by.

    The four names are the box labels, in the lesson's language; `label` describes the
    picture for screen readers.
    """
    import marimo as mo

    y, up = 120, 44
    boxes = [(60, x), (190, norm), (330, attention), (530, norm), (670, think)]
    plus = [430, 770]
    parts = [f'<svg viewBox="0 0 840 170" width="100%" style="max-width:840px" role="img" aria-label="{escape(label)}" '
             'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, monospace" text-anchor="middle">',
             f'<defs><marker id="road" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{AXIS}"/></marker></defs>']

    def arrow(x1, x2):
        parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{AXIS}" marker-end="url(#road)"/>')

    half = 50
    arrow(60 + half, 190 - half - 4), arrow(190 + half, 330 - half - 4), arrow(330 + half, plus[0] - 18)
    arrow(plus[0] + 14, 530 - half - 4), arrow(530 + half, 670 - half - 4), arrow(670 + half, plus[1] - 18)
    arrow(plus[1] + 14, 832)
    for start, end in [(125, plus[0]), (plus[0] + 30, plus[1])]:  # the two roads that pass a part by
        parts.append(f'<polyline points="{start},{y} {start},{up} {end},{up} {end},{y - 18}" fill="none" '
                     f'stroke="{DOT}" stroke-width="2.5" marker-end="url(#road)"/>')
    for cx, text in boxes:
        parts.append(f'<rect x="{cx - half}" y="{y - _BH / 2}" width="{2 * half}" height="{_BH}" rx="6" fill="none" '
                     f'stroke="{AXIS}" stroke-width="1.2"/>')
        parts.append(f'<text x="{cx}" y="{y + 5}" font-size="14" fill="{TEXT}">{escape(text)}</text>')
    for cx in plus:
        parts.append(f'<circle cx="{cx}" cy="{y}" r="14" fill="none" stroke="{DOT}" stroke-width="2"/>')
        parts.append(f'<text x="{cx}" y="{y + 6}" font-size="18" fill="{TEXT}">+</text>')
    parts.append("</svg>")
    return mo.Html("".join(parts))


def pipeline(boxes: list[tuple[str, str]], arrows: list[str], label: str):
    """Values in a row, each made from the one before: a box per value, a named arrow per step.

    Each box is (name, shape); `arrows` names the step between neighbours. `label` describes
    the picture for screen readers, in the lesson's language.
    """
    import marimo as mo

    half, gap, y = 62, 250, 60
    width = gap * (len(boxes) - 1) + 2 * half + 20
    xs = [10 + half + gap * i for i in range(len(boxes))]
    parts = [f'<svg viewBox="0 0 {width} 120" width="100%" style="max-width:{width}px" role="img" aria-label="{escape(label)}" '
             'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, monospace" text-anchor="middle">',
             f'<defs><marker id="step" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{AXIS}"/></marker></defs>']
    for x1, x2, text in zip(xs, xs[1:], arrows):
        parts.append(f'<line x1="{x1 + half}" y1="{y}" x2="{x2 - half - 4}" y2="{y}" stroke="{AXIS}" marker-end="url(#step)"/>')
        parts.append(f'<text x="{(x1 + x2) / 2}" y="{y - 10}" font-size="13" fill="{TEXT}">{escape(text)}</text>')
    for x, (name, shape) in zip(xs, boxes):
        parts.append(f'<rect x="{x - half}" y="{y - 30}" width="{2 * half}" height="60" rx="6" fill="none" '
                     f'stroke="{AXIS}" stroke-width="1.2"/>')
        parts.append(f'<text x="{x}" y="{y - 4}" font-size="15" fill="{TEXT}">{escape(name)}</text>')
        parts.append(f'<text x="{x}" y="{y + 17}" font-size="12" fill="{AXIS}">{escape(shape)}</text>')
    parts.append("</svg>")
    return mo.Html("".join(parts))


def circle(angle: float, label: str):
    """The circle of radius 1 with the point at `angle` (radians), and its cos and sin as lengths.

    `label` describes the picture for screen readers, in the lesson's language.
    """
    import math

    import marimo as mo

    cx, cy, r = 170, 160, 110
    px, py = cx + r * math.cos(angle), cy - r * math.sin(angle)
    ax, ay = cx + 34 * math.cos(angle / 2), cy - 34 * math.sin(angle / 2)
    arc_x, arc_y = cx + 22 * math.cos(angle), cy - 22 * math.sin(angle)
    parts = [
        f'<svg viewBox="0 0 340 320" width="100%" style="max-width:340px" role="img" aria-label="{escape(label)}" '
        'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, monospace" text-anchor="middle">',
        f'<line x1="{cx - r - 26}" y1="{cy}" x2="{cx + r + 26}" y2="{cy}" stroke="{AXIS}"/>',
        f'<line x1="{cx}" y1="{cy - r - 26}" x2="{cx}" y2="{cy + r + 26}" stroke="{AXIS}"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{AXIS}" stroke-width="1.2"/>',
        f'<path d="M{cx + 22},{cy} A22,22 0 0 0 {arc_x:.1f},{arc_y:.1f}" fill="none" stroke="{TEXT}"/>',
        f'<text x="{ax:.0f}" y="{ay + 4:.0f}" font-size="13" fill="{TEXT}">θ</text>',
        f'<line x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}" stroke="{TEXT}" stroke-width="1.2"/>',
        f'<line x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{cy}" stroke="{LINE}" stroke-width="3"/>',
        f'<line x1="{px:.1f}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}" stroke="{DOT}" stroke-width="3"/>',
        f'<text x="{(cx + px) / 2:.0f}" y="{cy + 18}" font-size="13" fill="{TEXT}">cos θ</text>',
        f'<text x="{px + 30:.0f}" y="{(cy + py) / 2 + 4:.0f}" font-size="13" fill="{TEXT}">sin θ</text>',
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{DOT}"/>',
        f'<text x="{px + 40:.0f}" y="{py - 10:.0f}" font-size="13" fill="{TEXT}">(cos θ, sin θ)</text>',
        f'<circle cx="{cx + r}" cy="{cy}" r="4" fill="{TEXT}"/>',
        f'<text x="{cx + r + 26}" y="{cy + 18}" font-size="12" fill="{AXIS}">(1, 0)</text>',
        f'<text x="{cx + 24}" y="{cy - r - 8}" font-size="12" fill="{AXIS}">(0, 1)</text>',
        "</svg>",
    ]
    return mo.Html("".join(parts))


def ring(names: list[str], label: str):
    """Devices in a circle, each with an arrow to the neighbour it passes to.

    `label` describes the picture for screen readers, in the lesson's language.
    """
    import math

    import marimo as mo

    n, cx, cy, r, half = len(names), 190, 150, 100, 46
    spots = [(cx + r * math.sin(2 * math.pi * i / n), cy - r * math.cos(2 * math.pi * i / n)) for i in range(n)]
    parts = [f'<svg viewBox="0 0 380 300" width="100%" style="max-width:380px" role="img" aria-label="{escape(label)}" '
             'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, monospace" text-anchor="middle">',
             f'<defs><marker id="pass" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{DOT}"/></marker></defs>']
    for (x1, y1), (x2, y2) in zip(spots, spots[1:] + spots[:1]):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        gap = min(half / max(abs(ux), 1e-9), _BH / 2 / max(abs(uy), 1e-9)) + 8  # to the edge of the box
        parts.append(f'<line x1="{x1 + ux * gap:.1f}" y1="{y1 + uy * gap:.1f}" x2="{x2 - ux * gap:.1f}" y2="{y2 - uy * gap:.1f}" '
                     f'stroke="{DOT}" stroke-width="2" marker-end="url(#pass)"/>')
    for (x, y), name in zip(spots, names):
        parts.append(f'<rect x="{x - half:.1f}" y="{y - _BH / 2:.1f}" width="{2 * half}" height="{_BH}" rx="6" '
                     f'fill="none" stroke="{AXIS}" stroke-width="1.2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" font-size="14" fill="{TEXT}">{escape(name)}</text>')
    parts.append("</svg>")
    return mo.Html("".join(parts))
