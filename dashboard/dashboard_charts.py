"""
SVG chart helpers and Qt table factory for the Analytics Dashboard.
No matplotlib — pure SVG generated as strings, embedded in QTextBrowser.
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QTableWidget, QTableWidgetItem

_SVG_W = 560
_SVG_H = 280
_PAD = 50


def _svg_open(w: int = _SVG_W, h: int = _SVG_H) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'style="background:#FAFAFA;border-radius:6px;">'
    )


def _svg_close() -> str:
    return "</svg>"


def _svg_axes(
    w: int,
    h: int,
    pad: int,
    x_labels: List[str],
    y_min: float,
    y_max: float,
    n_y: int = 5,
) -> str:
    parts = []
    parts.append(
        f'<rect x="{pad}" y="{pad // 2}" width="{w - 2 * pad}" height="{h - pad - pad // 2}" '
        f'fill="none" stroke="#CCCCCC" stroke-width="1"/>'
    )
    plot_h = h - pad - pad // 2
    for i in range(n_y + 1):
        y = pad // 2 + plot_h * (1 - i / n_y)
        val = y_min + (y_max - y_min) * i / n_y
        parts.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{w - pad}" y2="{y:.1f}" '
            f'stroke="#EEEEEE" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        lbl = f"{val:.1f}" if abs(val) >= 0.1 else f"{val:.3f}"
        parts.append(
            f'<text x="{pad - 4}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="9" fill="#888">{lbl}</text>'
        )
    n_x = len(x_labels)
    plot_w = w - 2 * pad
    for i, lbl in enumerate(x_labels):
        x = pad + plot_w * (i + 0.5) / max(n_x, 1)
        parts.append(
            f'<text x="{x:.1f}" y="{h - 4}" text-anchor="middle" '
            f'font-size="9" fill="#888">{lbl[:12]}</text>'
        )
    return "".join(parts)


def _data_to_xy(
    values: List[float],
    y_min: float,
    y_max: float,
    w: int = _SVG_W,
    h: int = _SVG_H,
    pad: int = _PAD,
) -> List[tuple]:
    n = len(values)
    plot_w = w - 2 * pad
    plot_h = h - pad - pad // 2
    y_range = y_max - y_min or 1.0
    pts = []
    for i, v in enumerate(values):
        x = pad + plot_w * (i + 0.5) / n
        y_frac = (v - y_min) / y_range
        y = pad // 2 + plot_h * (1 - y_frac)
        pts.append((x, y))
    return pts


def _svg_bar_chart(
    labels: List[str],
    values: List[float],
    title: str = "",
    color: str = "#4A90D9",
    neg_color: str = "#E74C3C",
    w: int = _SVG_W,
    h: int = _SVG_H,
) -> str:
    if not values:
        return f"{_svg_open(w, h)}<text x='50%' y='50%' text-anchor='middle'>No data</text>{_svg_close()}"

    y_min = min(0.0, min(values)) * 1.1
    y_max = max(0.0, max(values)) * 1.1 or 0.1
    pad = _PAD
    plot_h = h - pad - pad // 2
    plot_w = w - 2 * pad
    y_range = y_max - y_min or 1.0
    zero_y = pad // 2 + plot_h * (1 - (0 - y_min) / y_range)
    bar_w = max(4, plot_w // max(len(values), 1) - 2)

    parts = [_svg_open(w, h)]
    if title:
        parts.append(
            f'<text x="{w // 2}" y="14" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">{title}</text>'
        )
    parts.append(_svg_axes(w, h, pad, labels, y_min, y_max))
    parts.append(
        f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{w - pad}" y2="{zero_y:.1f}" '
        f'stroke="#AAAAAA" stroke-width="1"/>'
    )

    for i, (lbl, val) in enumerate(zip(labels, values)):
        cx = pad + plot_w * (i + 0.5) / len(values)
        bar_x = cx - bar_w / 2
        y_val = pad // 2 + plot_h * (1 - (val - y_min) / y_range)
        bar_top = min(y_val, zero_y)
        bar_ht = abs(y_val - zero_y)
        fill = color if val >= 0 else neg_color
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{bar_top:.1f}" width="{bar_w}" height="{max(1.0, bar_ht):.1f}" '
            f'fill="{fill}" opacity="0.85" rx="2"/>'
        )

    parts.append(_svg_close())
    return "".join(parts)


def _svg_calibration(
    p_model: List[float],
    p_actual: List[float],
    ns: List[int],
    title: str = "Calibration",
    w: int = _SVG_W,
    h: int = _SVG_H,
) -> str:
    pad = _PAD
    plot_w = w - 2 * pad
    plot_h = h - pad - pad // 2

    parts = [_svg_open(w, h)]
    if title:
        parts.append(
            f'<text x="{w // 2}" y="14" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">{title}</text>'
        )
    parts.append(_svg_axes(w, h, pad, [f"{i / 10:.1f}" for i in range(11)], 0.0, 1.0))

    x0, y0 = pad, pad // 2 + plot_h
    x1, y1 = w - pad, pad // 2
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
        f'stroke="#AAAAAA" stroke-width="1" stroke-dasharray="6,3"/>'
    )

    max_n = max(ns) if ns else 1
    for pm, pa, n in zip(p_model, p_actual, ns):
        if pa is None:
            continue
        cx = pad + plot_w * pm
        cy = pad // 2 + plot_h * (1 - pa)
        radius = max(3, 4 + 8 * n / max_n)
        color = (
            "#2ECC71"
            if abs(pm - pa) < 0.05
            else ("#E74C3C" if abs(pm - pa) > 0.15 else "#F39C12")
        )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
            f'fill="{color}" opacity="0.75" stroke="white" stroke-width="1"/>'
        )

    parts.append(_svg_close())
    return "".join(parts)


def _svg_scatter(
    xs: List[float],
    ys: List[float],
    title: str = "",
    x_label: str = "x",
    y_label: str = "y",
    w: int = _SVG_W,
    h: int = _SVG_H,
) -> str:
    if not xs:
        return f"{_svg_open(w, h)}<text x='50%' y='50%' text-anchor='middle'>No data</text>{_svg_close()}"
    pad = _PAD
    plot_w = w - 2 * pad
    plot_h = h - pad - pad // 2
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    parts = [_svg_open(w, h)]
    if title:
        parts.append(
            f'<text x="{w // 2}" y="14" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">{title}</text>'
        )

    x_labels = [f"{x_min + i * x_range / 5:.2f}" for i in range(6)]
    parts.append(_svg_axes(w, h, pad, x_labels, y_min, y_max))

    for xv, yv in zip(xs, ys):
        cx = pad + plot_w * (xv - x_min) / x_range
        cy = pad // 2 + plot_h * (1 - (yv - y_min) / y_range)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#4A90D9" opacity="0.6"/>'
        )

    parts.append(
        f'<text x="{w // 2}" y="{h - 2}" text-anchor="middle" font-size="9" fill="#666">{x_label}</text>'
    )
    parts.append(_svg_close())
    return "".join(parts)


def _make_table(headers: List[str], rows: List[List[str]]) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    t.horizontalHeader().setStretchLastSection(True)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setItem(r_idx, c_idx, item)
    t.resizeColumnsToContents()
    return t
