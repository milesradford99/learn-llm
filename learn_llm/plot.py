"""Small SVG charts for lesson pages: dots, lines and curves, with no plotting library.

Colors come from CSS variables with fallbacks, so charts follow the page's light/dark theme.
"""

import math
from html import escape

W, H = 520, 320
PAD_L, PAD_R, PAD_T, PAD_B = 56, 16, 16, 44

DOT = "var(--blue-9, #2563eb)"
LINE = "var(--crimson-9, #dc2626)"
AXIS = "var(--slate-9, #64748b)"
TEXT = "var(--slate-12, #1e293b)"
GRID = "var(--slate-4, #e2e8f0)"
SERIES = [LINE, DOT, "var(--grass-9, #16a34a)", "var(--amber-9, #d97706)"]


def _ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    """About `n` round-numbered ticks (steps of 1, 2 or 5 times a power of ten)."""
    raw = (hi - lo) / n
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    first = math.ceil(lo / step)
    return [round(i * step, 10) for i in range(first, math.floor(hi / step) + 1)]


def chart(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_label: str,
    y_label: str,
    dots: list[tuple[float, float]] = (),
    lines: list[tuple[float, float]] = (),  # (w, b): y = w * x + b
    curve: list[tuple[float, float]] = (),  # points joined in order
    marker: tuple[float, float] | None = None,  # one highlighted point on the curve
    segments: list[tuple[float, float, float, float]] = (),  # (x1, y1, x2, y2), dashed
    series: list[tuple[str, list[tuple[float, float]]]] = (),  # (name, points): several curves, with a key
):
    """Return an SVG chart as a marimo Html element."""
    import marimo as mo

    (x0, x1), (y0, y1) = x_range, y_range

    def sx(x):
        return PAD_L + (x - x0) / (x1 - x0) * (W - PAD_L - PAD_R)

    def sy(y):
        return H - PAD_B - (y - y0) / (y1 - y0) * (H - PAD_T - PAD_B)

    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" '
        f'aria-label="{escape(y_label)} against {escape(x_label)}" xmlns="http://www.w3.org/2000/svg">',
        f'<defs><clipPath id="plot"><rect x="{PAD_L}" y="{PAD_T}" width="{W - PAD_L - PAD_R}" '
        f'height="{H - PAD_T - PAD_B}"/></clipPath></defs>',
    ]
    for t in _ticks(x0, x1):
        parts.append(f'<line x1="{sx(t):.1f}" y1="{PAD_T}" x2="{sx(t):.1f}" y2="{H - PAD_B}" stroke="{GRID}"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{H - PAD_B + 16}" font-size="11" fill="{AXIS}" text-anchor="middle">{t:g}</text>')
    for t in _ticks(y0, y1):
        parts.append(f'<line x1="{PAD_L}" y1="{sy(t):.1f}" x2="{W - PAD_R}" y2="{sy(t):.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{PAD_L - 6}" y="{sy(t) + 4:.1f}" font-size="11" fill="{AXIS}" text-anchor="end">{t:g}</text>')
    parts.append(f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}" stroke="{AXIS}"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}" stroke="{AXIS}"/>')
    parts.append(f'<text x="{(PAD_L + W - PAD_R) / 2}" y="{H - 8}" font-size="12" fill="{TEXT}" text-anchor="middle">{escape(x_label)}</text>')
    parts.append(
        f'<text x="14" y="{(PAD_T + H - PAD_B) / 2}" font-size="12" fill="{TEXT}" text-anchor="middle" '
        f'transform="rotate(-90 14 {(PAD_T + H - PAD_B) / 2})">{escape(y_label)}</text>'
    )
    parts.append('<g clip-path="url(#plot)">')
    for xa, ya, xb, yb in segments:
        parts.append(
            f'<line x1="{sx(xa):.1f}" y1="{sy(ya):.1f}" x2="{sx(xb):.1f}" y2="{sy(yb):.1f}" '
            f'stroke="{LINE}" stroke-dasharray="3 3" opacity="0.7"/>'
        )
    for w, b in lines:
        parts.append(
            f'<line x1="{sx(x0):.1f}" y1="{sy(w * x0 + b):.1f}" x2="{sx(x1):.1f}" y2="{sy(w * x1 + b):.1f}" '
            f'stroke="{LINE}" stroke-width="2.5"/>'
        )
    if curve:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(curve))
        parts.append(f'<path d="{path}" fill="none" stroke="{LINE}" stroke-width="2.5"/>')
    for i, (_, points) in enumerate(series):
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(points))
        dash = ' stroke-dasharray="6 4"' if i % 2 else ""
        parts.append(f'<path d="{path}" fill="none" stroke="{SERIES[i % len(SERIES)]}" stroke-width="2.5"{dash}/>')
    for x, y in dots:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{DOT}"/>')
    if marker:
        parts.append(f'<circle cx="{sx(marker[0]):.1f}" cy="{sy(marker[1]):.1f}" r="7" fill="{DOT}" stroke="white" stroke-width="2"/>')
    parts.append("</g>")
    for i, (name, _) in enumerate(series):
        y = PAD_T + 14 + 18 * i
        dash = ' stroke-dasharray="6 4"' if i % 2 else ""
        parts.append(f'<line x1="{W - PAD_R - 150}" y1="{y - 4}" x2="{W - PAD_R - 126}" y2="{y - 4}" '
                     f'stroke="{SERIES[i % len(SERIES)]}" stroke-width="2.5"{dash}/>')
        parts.append(f'<text x="{W - PAD_R - 120}" y="{y}" font-size="12" fill="{TEXT}">{escape(name)}</text>')
    parts.append("</svg>")
    return mo.Html("".join(parts))
