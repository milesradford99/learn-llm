"""Diagrams for lesson pages, drawn as SVG with no plotting library."""

from html import escape

from learn_llm.plot import AXIS, DOT, GRID, LINE, TEXT

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


def _width(text: str, size: float) -> float:
    """About how wide `text` is in pixels: CJK characters are about twice a Latin one."""
    return sum(1.0 if ord(ch) > 0x2E80 else 0.62 for ch in str(text)) * size


def _entry(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def grids(parts: list, label: str):
    """Small matrices side by side, with their entries, to show which entries meet.

    A string in `parts` is drawn between the grids ("@", "=", "→"). A grid is a dict with
    `name` (written above it), `values` (its rows, numbers or text), and optionally `rows`
    and `cols` (a label for each row and column) and `marks`, mapping (row, column) to
    "blue", "red" or "box" (outlined), or several of them with spaces. `label` describes the
    picture for screen readers, in the lesson's language.
    """
    import marimo as mo

    fills = {"blue": DOT, "red": LINE}
    ch, top, gap, pad = 30, 52, 18, 10
    height = top + ch * max(len(p["values"]) for p in parts if isinstance(p, dict)) + 12
    x, body = pad, []
    for part in parts:
        if isinstance(part, str):
            w = _width(part, 18) + 2 * gap
            body.append(f'<text x="{x + w / 2:.1f}" y="{top + 20}" font-size="18" fill="{TEXT}">{escape(part)}</text>')
            x += w
            continue
        values, rows, cols = part["values"], part.get("rows") or [], part.get("cols") or []
        n_rows, n_cols = len(values), len(values[0])
        cw = max([52.0] + [_width(_entry(v), 14) + 16 for r in values for v in r]
                 + [_width(c, 11) + 16 for c in cols])
        cw = max(cw, (_width(part.get("name", ""), 14) + 12) / n_cols)  # room for the name above
        lw = max([0.0] + [_width(r, 11) + 8 for r in rows])
        gx = x + lw
        body.append(f'<text x="{gx + cw * n_cols / 2:.1f}" y="16" font-size="14" font-weight="600" '
                    f'fill="{TEXT}">{escape(part.get("name", ""))}</text>')
        for c, text in enumerate(cols):
            body.append(f'<text x="{gx + cw * (c + 0.5):.1f}" y="{top - 8}" font-size="11" '
                        f'fill="{AXIS}">{escape(text)}</text>')
        for r, text in enumerate(rows):
            body.append(f'<text x="{gx - 6:.1f}" y="{top + ch * r + 19}" font-size="11" text-anchor="end" '
                        f'fill="{AXIS}">{escape(text)}</text>')
        marks = part.get("marks") or {}
        for r in range(n_rows):
            for c in range(n_cols):
                kinds = marks.get((r, c), "").split()
                cx, cy = gx + cw * c, top + ch * r
                for kind in kinds:
                    if kind in fills:
                        body.append(f'<rect x="{cx:.1f}" y="{cy}" width="{cw:.1f}" height="{ch}" '
                                    f'fill="{fills[kind]}" fill-opacity="0.2"/>')
                body.append(f'<text x="{cx + cw / 2:.1f}" y="{cy + 20}" font-size="14" '
                            f'fill="{TEXT}">{escape(_entry(values[r][c]))}</text>')
        for c in range(1, n_cols):
            body.append(f'<line x1="{gx + cw * c:.1f}" y1="{top}" x2="{gx + cw * c:.1f}" y2="{top + ch * n_rows}" '
                        f'stroke="{GRID}"/>')
        for r in range(1, n_rows):
            body.append(f'<line x1="{gx:.1f}" y1="{top + ch * r}" x2="{gx + cw * n_cols:.1f}" y2="{top + ch * r}" '
                        f'stroke="{GRID}"/>')
        body.append(f'<rect x="{gx:.1f}" y="{top}" width="{cw * n_cols:.1f}" height="{ch * n_rows}" rx="4" '
                    f'fill="none" stroke="{AXIS}" stroke-width="1.2"/>')
        for r in range(n_rows):
            for c in range(n_cols):
                if "box" in marks.get((r, c), "").split():
                    body.append(f'<rect x="{gx + cw * c + 1.5:.1f}" y="{top + ch * r + 1.5}" width="{cw - 3:.1f}" '
                                f'height="{ch - 3}" rx="3" fill="none" stroke="{TEXT}" stroke-width="2.5"/>')
        x = gx + cw * n_cols
    width = x + pad
    head = (f'<svg viewBox="0 0 {width:.0f} {height}" width="100%" style="max-width:{width:.0f}px" role="img" '
            f'aria-label="{escape(label)}" xmlns="http://www.w3.org/2000/svg" '
            'font-family="ui-monospace, monospace" text-anchor="middle">')
    return mo.Html(head + "".join(body) + "</svg>")


def cards(entries: list[dict], label: str, arrows: str = ""):
    """Record cards in a row: one per box, or one per entry of the engine's book.

    Each entry is a dict with `title` (written on the card's top band) and `lines` (one slot
    per line, such as "data = 140"). `mark` set to "now" outlines the card in blue (the one
    being read); "done" greys it out. `arrows` is drawn between neighbouring cards (for
    example "→"), or nothing. `label` describes the picture for screen readers, in the
    lesson's language.
    """
    import marimo as mo

    size, line_h, band, pad, gap = 13, 20, 28, 10, 34 if arrows else 16
    widths = [max([_width(e["title"], 14)] + [_width(s, size) for s in e["lines"]]) + 2 * pad + 8
              for e in entries]
    height = band + line_h * max(len(e["lines"]) for e in entries) + 2 * pad + 8
    x, body = 4.0, []
    for i, (entry, w) in enumerate(zip(entries, widths)):
        mark = entry.get("mark", "")
        stroke, sw = (DOT, 3) if mark == "now" else (AXIS, 1.2)
        opacity = ' opacity="0.45"' if mark == "done" else ""
        body.append(f'<g{opacity}>')
        body.append(f'<rect x="{x:.1f}" y="4" width="{w:.1f}" height="{height - 8}" rx="8" fill="none" '
                    f'stroke="{stroke}" stroke-width="{sw}"/>')
        body.append(f'<rect x="{x:.1f}" y="4" width="{w:.1f}" height="{band}" rx="8" fill="{DOT}" fill-opacity="0.12"/>')
        body.append(f'<text x="{x + w / 2:.1f}" y="{4 + band - 9}" font-size="14" font-weight="600" '
                    f'text-anchor="middle" fill="{TEXT}">{escape(entry["title"])}</text>')
        for j, text in enumerate(entry["lines"]):
            body.append(f'<text x="{x + pad:.1f}" y="{4 + band + pad + line_h * j + 12}" font-size="{size}" '
                        f'text-anchor="start" fill="{TEXT}">{escape(text)}</text>')
        body.append('</g>')
        x += w
        if i < len(entries) - 1:
            if arrows:
                body.append(f'<text x="{x + gap / 2:.1f}" y="{height / 2 + 6:.1f}" font-size="18" '
                            f'text-anchor="middle" fill="{AXIS}">{escape(arrows)}</text>')
            x += gap
    width = x + 4
    head = (f'<svg viewBox="0 0 {width:.0f} {height}" width="100%" style="max-width:{width:.0f}px" role="img" '
            f'aria-label="{escape(label)}" xmlns="http://www.w3.org/2000/svg" '
            'font-family="ui-monospace, monospace">')
    return mo.Html(head + "".join(body) + "</svg>")


def bars(items: list[tuple[str, float]], label: str, top: float | None = None, digits: int = 2):
    """Horizontal bars, one per (name, value): blue to the right for positive, red to the left
    for negative, each with its value written beside it. `top` is the value of a full bar
    (by default the biggest size). `label` describes the picture for screen readers, in the
    lesson's language.
    """
    import marimo as mo

    top = top or max(abs(v) for _, v in items) or 1.0
    negative = any(v < 0 for _, v in items)
    name_w = max(_width(n, 14) for n, _ in items) + 16
    half = 200 if negative else 0
    span = 200 if negative else 360
    zero = name_w + (half + 50 if negative else 0)  # room for a negative bar's value
    row_h, width = 30, zero + span + 70
    parts = [f'<svg viewBox="0 0 {width:.0f} {row_h * len(items) + 8}" width="100%" style="max-width:{width:.0f}px" '
             f'role="img" aria-label="{escape(label)}" xmlns="http://www.w3.org/2000/svg" '
             'font-family="ui-monospace, monospace">']
    for i, (name, value) in enumerate(items):
        y = 4 + row_h * i
        size = min(abs(value) / top, 1.0) * span
        x = zero if value >= 0 else zero - size
        color = DOT if value >= 0 else LINE
        parts.append(f'<text x="{name_w - 10:.1f}" y="{y + 19}" font-size="14" text-anchor="end" fill="{TEXT}">'
                     f'{escape(name)}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y + 4}" width="{max(size, 1):.1f}" height="{row_h - 10}" rx="3" '
                     f'fill="{color}" fill-opacity="0.75"/>')
        tx, anchor = (zero + size + 6, "start") if value >= 0 else (zero - size - 6, "end")
        parts.append(f'<text x="{tx:.1f}" y="{y + 19}" font-size="13" text-anchor="{anchor}" fill="{TEXT}">'
                     f'{value:.{digits}f}</text>')
    parts.append(f'<line x1="{zero:.1f}" y1="2" x2="{zero:.1f}" y2="{row_h * len(items) + 6}" stroke="{AXIS}"/>')
    parts.append("</svg>")
    return mo.Html("".join(parts))


def network(layers: list[list], edges: list[tuple], label: str, heads: list[str] = ()):
    """A small network drawn as columns of nodes joined by arrows.

    `layers` is a list of columns, top to bottom; a node is a string or a list of lines (for
    example a value and, under it, a blame). `edges` are (column, from, to, text): an arrow from
    node `from` of that column to node `to` of the next, with `text` written on it near its
    start. `heads` names each column above it. `label` describes the picture for screen
    readers, in the lesson's language.
    """
    import marimo as mo

    layers = [[[n] if isinstance(n, str) else list(n) for n in col] for col in layers]
    node_w = max(_width(line, 14) for col in layers for n in col for line in n) + 22
    node_h = 18 * max(len(n) for col in layers for n in col) + 14
    gap_x, gap_y, top = node_w + 120, node_h + 46, 34 if heads else 8
    rows = max(len(col) for col in layers)
    height = top + gap_y * (rows - 1) + node_h + 10
    width = gap_x * (len(layers) - 1) + node_w + 20

    def spot(c, r):
        offset = (rows - len(layers[c])) * gap_y / 2
        return 10 + c * gap_x, top + offset + r * gap_y

    parts = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" style="max-width:{width:.0f}px" role="img" '
             f'aria-label="{escape(label)}" xmlns="http://www.w3.org/2000/svg" '
             'font-family="ui-monospace, monospace" text-anchor="middle">',
             f'<defs><marker id="net" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{AXIS}"/></marker></defs>']
    for c, text in enumerate(heads):
        x, _ = spot(c, 0)
        parts.append(f'<text x="{x + node_w / 2:.1f}" y="18" font-size="13" font-weight="600" fill="{AXIS}">'
                     f'{escape(text)}</text>')
    for c, a, b, text in edges:
        (xa, ya), (xb, yb) = spot(c, a), spot(c + 1, b)
        x1, y1, x2, y2 = xa + node_w, ya + node_h / 2, xb - 4, yb + node_h / 2
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{AXIS}" '
                     f'marker-end="url(#net)"/>')
        if text:
            tx, ty = x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.3
            w = _width(text, 13) + 8
            parts.append(f'<rect x="{tx - w / 2:.1f}" y="{ty - 10:.1f}" width="{w:.1f}" height="18" rx="4" '
                         f'fill="var(--background, white)" stroke="{GRID}"/>')
            parts.append(f'<text x="{tx:.1f}" y="{ty + 4:.1f}" font-size="13" fill="{DOT}">{escape(text)}</text>')
    for c, col in enumerate(layers):
        for r, lines in enumerate(col):
            x, y = spot(c, r)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_w:.1f}" height="{node_h}" rx="8" fill="none" '
                         f'stroke="{AXIS}" stroke-width="1.2"/>')
            for i, line in enumerate(lines):
                color = TEXT if i == 0 else LINE
                ty = y + node_h / 2 + 5 + 18 * (i - (len(lines) - 1) / 2)
                parts.append(f'<text x="{x + node_w / 2:.1f}" y="{ty:.1f}" font-size="14" '
                             f'fill="{color}">{escape(line)}</text>')
    parts.append("</svg>")
    return mo.Html("".join(parts))
