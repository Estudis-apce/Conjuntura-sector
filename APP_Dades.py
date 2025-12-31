# ---------------------------
# Standard library
# ---------------------------
from datetime import datetime
from typing import List, Tuple, Optional, Iterable, Union
import base64
import io
import json
import re

# ---------------------------
# Third-party libraries
# ---------------------------
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Importante: antes de pyplot en entornos sin display (ej. Streamlit)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly.express as px
import plotly.graph_objects as go

import geopandas as gpd

import yaml
from yaml.loader import SafeLoader

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import streamlit_authenticator as stauth

# ---------------------------
# ReportLab (PDF)
# ---------------------------
from reportlab.lib.pagesizes import A4, landscape as rl_landscape
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

from reportlab.platypus import (
    SimpleDocTemplate,
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,           # Nota: Image de platypus
    KeepTogether,
    PageBreak,
    NextPageTemplate,
)
# Alias útil para diferenciar imágenes si lo prefieres en tu código:
from reportlab.platypus import Image as RLImage

from reportlab.platypus.flowables import CondPageBreak


# ========== COLORES / CONFIG ==========
CSS_COLORS = {
    "bg": "#fcefdc",
    "primary": "#de7207",
    "accent": "#fac678",
    "text": "#7a3601",
    "brand_dark": "#2e5d34"
}

GLOBAL_PALETTE = {
    "total": "#2d538f",
    "segunda_ma": "#de7207",
    "nou": "#1b7f3a",
    "unifamiliar": "#de7207",
    "plurifamiliar": "#1b7f3a",
}

TABLE_TRIM_START_YEAR = 2023
TABLE_ANNUAL_START_YEAR = 2014
SERIES_START_YEAR = 2014
TITLE_SPACING_CM = 0.6
BLOCK_SPACING_CM = 0.8  

# ========== FORMATEO ==========
def _mpl_finish(fig) -> bytes:
    buf = io.BytesIO()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(buf, format="png", dpi=190, bbox_inches="tight", facecolor=CSS_COLORS["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _format_thousands(x, pos=None):
    try:
        s = f"{x:,.0f}"
        return s.replace(",", ".")
    except Exception:
        return str(x)

def _format_df_thousands(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(lambda v: f"{v:,.0f}".replace(",", ".") if pd.notnull(v) else "")
    return df2

def _delta_fmt(delta_str: Optional[str]) -> str:
    if not delta_str:
        return ""
    try:
        val = float(str(delta_str).replace("%", "").replace(",", "."))
    except Exception:
        return f"{delta_str}"
    arrow = "▲" if val >= 0 else "▼"
    color = "#1b7f3a" if val >= 0 else "#b00020"
    return f"<font color='{color}'>{arrow} {abs(val):.1f}%</font>"

# ========== ÍNDICES / PERIODOS / FILTROS ==========
def _flatten_period_token(token) -> str:
    if isinstance(token, str):
        s = token.upper().replace("Q", "T")
        if "T" in s:
            return s if s.startswith("T") else ("T" + s.split("T")[-1])
        if s.isdigit():
            return f"T{int(s)}"
        return s
    if isinstance(token, (int, np.integer)):
        return f"T{int(token)}"
    return str(token)

def _flatten_period_index(idx: Iterable) -> list:
    if isinstance(idx, pd.MultiIndex) and len(idx.levels) == 2:
        out = []
        for (y, t) in idx:
            try:
                y_str = str(int(y))
            except Exception:
                y_str = str(y)
            t_str = _flatten_period_token(t)
            out.append(f"{y_str}{t_str if t_str else ''}")
        return out

    if isinstance(idx, pd.DatetimeIndex):
        try:
            periods = idx.to_period("Q")
            return [f"{p.year}T{p.quarter}" for p in periods]
        except Exception:
            return [str(d.year) for d in idx]

    return [str(x) for x in idx]

def _infer_year_from_label(label: str) -> Optional[int]:
    try:
        return int(str(label)[:4])
    except Exception:
        return None

def _filter_df_by_year(df: pd.DataFrame, start_year: int = SERIES_START_YEAR) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    data = df.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        return data[data.index.year >= start_year]
    if isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 1:
        try:
            years = data.index.get_level_values(0).astype(int)
            return data[years >= start_year]
        except Exception:
            pass
    try:
        idx_str = [str(i) for i in data.index]
        mask = []
        for lab in idx_str:
            y = _infer_year_from_label(lab)
            mask.append((y is None) or (y >= start_year))
        return data[np.array(mask)]
    except Exception:
        return data

# ========== MATPLOTLIB HELPERS ==========
def _mpl_base(figsize=(13, 6), dpi=190):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(CSS_COLORS["bg"])
    ax.set_facecolor(CSS_COLORS["bg"])
    ax.tick_params(colors=CSS_COLORS["text"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    return fig, ax

def _tune_axes(ax, max_xticks=6, force_all_xticks=False):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.grid(False)
    if not force_all_xticks:
        xs = ax.get_xticks()
        if len(xs) > max_xticks and max_xticks > 0:
            step = max(1, len(xs) // max_xticks)
            try:
                ax.set_xticks(xs[::step])
            except Exception:
                pass
    for label in ax.get_xticklabels():
        label.set_color(CSS_COLORS["text"])
        label.set_fontsize(9)
    for label in ax.get_yticklabels():
        label.set_color(CSS_COLORS["text"])
        label.set_fontsize(9)

def _annotate_last(ax, x_labels: list, y: np.ndarray):
    try:
        if len(y) == 0 or np.all(np.isnan(y)):
            return
        ax.annotate(
            f"{y[-1]:,.0f}".replace(",", "."),
            xy=(len(x_labels) - 1, y[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=CSS_COLORS["text"]
        )
    except Exception:
        pass

def mpl_line(df: pd.DataFrame, cols: list, title: str, ylab: str,
             xlab: str = "Període", start_year: int = SERIES_START_YEAR,
             palette: Optional[List[str]] = None,
             force_all_xticks: bool = False) -> bytes:
    data = _filter_df_by_year(df, start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax = _mpl_base()
    if palette is None:
        palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"], "#727375"]

    sel = [c for c in cols if c in data.columns]
    x_labels = _flatten_period_index(data.index)

    plotted = False
    for i, c in enumerate(sel):
        y = pd.to_numeric(data[c], errors="coerce").values
        ax.plot(x_labels, y, label=c, linewidth=1.8, color=palette[i % len(palette)])
        _annotate_last(ax, x_labels, y)
        plotted = True

    ax.set_ylabel(ylab, color=CSS_COLORS["text"])
    ax.set_xlabel(xlab, color=CSS_COLORS["text"])
    ax.grid(False)
    if plotted:
        ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3
    )
    _tune_axes(ax, max_xticks=6, force_all_xticks=force_all_xticks)
    return _mpl_finish(fig)

def mpl_bar(df: pd.DataFrame, cols: list, title: str, ylab: str,
            start_year: int = SERIES_START_YEAR,
            palette: Optional[List[str]] = None,
            force_all_xticks: bool = False) -> bytes:
    data = _filter_df_by_year(df.copy(), start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax = _mpl_base()
    if palette is None:
        palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"], "#727375"]

    sel = [c for c in cols if c in data.columns]
    x_labels = _flatten_period_index(data.index)
    n = len(sel)
    if n == 0 or len(x_labels) == 0:
        return _mpl_finish(fig)

    width = 0.8 / n
    x_idx = np.arange(len(x_labels))
    cur = 0
    for i, c in enumerate(sel):
        y = pd.to_numeric(data[c], errors="coerce").values
        offs = x_idx + cur * width
        bars = ax.bar(
            offs, y, width=width, label=c,
            color=palette[i % len(palette)],
            edgecolor="white", linewidth=0.3
        )
        # Etiqueta de valor en cada barra
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:,.0f}".replace(",", "."),
                    ha="center", va="bottom", fontsize=8, color=CSS_COLORS["text"]
                )
        cur += 1

    ax.set_xticks(x_idx + (n - 1) * width / 2)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel(ylab, color=CSS_COLORS["text"])
    ax.grid(False)
    if n > 1:  # leyenda solo si hay >1 serie
        ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3
    )
    _tune_axes(ax, max_xticks=6, force_all_xticks=force_all_xticks)
    return _mpl_finish(fig)

def mpl_area(df: pd.DataFrame, cols: List[str], title: str, ylab: str,
             xlab: str = "Període", start_year: int = SERIES_START_YEAR,
             palette: Optional[List[str]] = None,
             force_all_xticks: bool = False) -> bytes:
    data = _filter_df_by_year(df, start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax = _mpl_base()
    if palette is None:
        palette = ["#2d538f", "#1b7f3a", "#de7207", "#6a3d9a", "#b15928", "#727375", "#9aa0a6"]

    sel = [c for c in cols if c in data.columns]
    x_labels = _flatten_period_index(data.index)
    if sel:
        ys = [pd.to_numeric(data[c], errors="coerce").values for c in sel]
        ax.stackplot(x_labels, ys, labels=sel, colors=[palette[i % len(palette)] for i in range(len(sel))], alpha=0.95)

    ax.set_ylabel(ylab, color=CSS_COLORS["text"])
    ax.set_xlabel(xlab, color=CSS_COLORS["text"])
    ax.grid(False)
    if sel:
        ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3
    )
    _tune_axes(ax, max_xticks=6, force_all_xticks=force_all_xticks)
    return _mpl_finish(fig)

def mpl_dual_line(df: pd.DataFrame, left_col: str, right_col: str,
                  left_label: str, right_label: str,
                  left_ylab: str, right_ylab: str,
                  start_year: int = SERIES_START_YEAR,
                  left_color: str = GLOBAL_PALETTE["total"],
                  right_color: str = GLOBAL_PALETTE["segunda_ma"],
                  force_all_xticks: bool = False) -> bytes:
    data = _filter_df_by_year(df, start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax1 = _mpl_base()
    x_labels = _flatten_period_index(data.index)

    y1 = pd.to_numeric(data.get(left_col, pd.Series(index=data.index)), errors="coerce").values
    y2 = pd.to_numeric(data.get(right_col, pd.Series(index=data.index)), errors="coerce").values

    ax1.plot(x_labels, y1, label=left_label, color=left_color, linewidth=1.8)
    _annotate_last(ax1, x_labels, y1)
    ax1.set_ylabel(left_ylab, color=CSS_COLORS["text"])

    ax2 = ax1.twinx()
    ax2.plot(x_labels, y2, label=right_label, color=right_color, linewidth=1.8, linestyle="--")
    ax2.set_ylabel(right_ylab, color=CSS_COLORS["text"])

    ax1.set_xlabel("Període", color=CSS_COLORS["text"])
    ax1.grid(False)
    _tune_axes(ax1, max_xticks=6, force_all_xticks=force_all_xticks)

    lines, labels = [], []
    for a in (ax1, ax2):
        ln, lb = a.get_legend_handles_labels()
        lines.extend(ln); labels.extend(lb)
    ax1.legend(lines, labels, frameon=False, fontsize=9, loc="upper left", ncol=2)

    return _mpl_finish(fig)

def mpl_dual_bar(df: pd.DataFrame, left_col: str, right_col: str,
                 left_label: str, right_label: str,
                 left_ylab: str, right_ylab: str,
                 start_year: int = SERIES_START_YEAR,
                 left_color: str = GLOBAL_PALETTE["total"],
                 right_color: str = GLOBAL_PALETTE["segunda_ma"],
                 force_all_xticks: bool = True) -> bytes:
    data = _filter_df_by_year(df, start_year=start_year).replace([np.inf, -np.inf], np.nan)
    fig, ax1 = _mpl_base()
    x_labels = _flatten_period_index(data.index)
    x_idx = np.arange(len(x_labels))

    y1 = pd.to_numeric(data.get(left_col, pd.Series(index=data.index)), errors="coerce").values
    y2 = pd.to_numeric(data.get(right_col, pd.Series(index=data.index)), errors="coerce").values

    ax1.bar(x_idx, y1, color=left_color, width=0.6, label=left_label, alpha=0.9, edgecolor="white", linewidth=0.3)
    ax1.set_ylabel(left_ylab, color=CSS_COLORS["text"])
    ax1.set_xticks(x_idx); ax1.set_xticklabels(x_labels)
    _tune_axes(ax1, max_xticks=6, force_all_xticks=force_all_xticks)

    ax2 = ax1.twinx()
    ax2.plot(x_labels, y2, label=right_label, color=right_color, linewidth=1.8, linestyle="--")
    ax2.set_ylabel(right_ylab, color=CSS_COLORS["text"])

    lines, labels = [], []
    for a in (ax1, ax2):
        ln, lb = a.get_legend_handles_labels()
        lines.extend(ln); labels.extend(lb)
    ax1.legend(lines, labels, frameon=False, fontsize=9, loc="upper left", ncol=2)

    return _mpl_finish(fig)

# ========== TABLAS ==========
def _hex_to_rl(hexstr: str):
    return rl_colors.HexColor(hexstr)

def _maybe_flatten_index_and_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out.index = _flatten_period_index(out.index)
    except Exception:
        pass
    if isinstance(out.columns, pd.MultiIndex):
        new_cols = []
        for tpl in out.columns:
            lab = " ".join([str(x) for x in tpl if (x is not None and str(x) != "")])
            new_cols.append(lab)
        out.columns = new_cols
    return out

def _styled_table_from_df(df, max_rows: Optional[int] = None, max_cols: int = 12) -> Table:
    if isinstance(df, str) and "<table" in df.lower():
        try:
            lst = pd.read_html(df)
            if lst: df = lst[0]
        except Exception:
            pass
    if hasattr(df, "data"):
        try:
            df = df.data
        except Exception:
            pass

    df = _maybe_flatten_index_and_cols(pd.DataFrame(df))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = _format_df_thousands(df)


    if max_rows is not None:
        df = df.iloc[:max_rows, :max_cols]
    else:
        df = df.iloc[:, :max_cols]

    data = [[""] + [str(c) for c in df.columns]]
    for idx, row in df.iterrows():
        data.append([str(idx)] + [str(v) for v in row.values])

    tbl = Table(data, repeatRows=1)

    try:
        total_width = 1.15  # 15% más ancha
        tbl._argW = [w * total_width if w else None for w in tbl._argW]
    except Exception:
        pass

    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), _hex_to_rl(CSS_COLORS["primary"])),
        ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('TEXTCOLOR', (0,1), (-1,-1), _hex_to_rl(CSS_COLORS["text"])),
        ('ALIGN', (0,1), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.25, rl_colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [_hex_to_rl(CSS_COLORS["bg"]), _hex_to_rl(CSS_COLORS["accent"])]),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    return tbl


def _header_footer(canvas, doc):
    """
    Cabecera y pie para todas las páginas excepto:
    - Portada (usa _header_footer_cover)
    - Última página (no muestra número ni fuente de datos)
    """
    canvas.saveState()

    # Fondo
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["bg"]))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], stroke=0, fill=1)

    # Determinar si es la última página
    # -------------------------------------------------
    # ReportLab no sabe "en tiempo real" el total de páginas,
    # pero podemos marcar la última desde el flujo.
    # Truco: añadiremos una marca en el story.
    is_last_page = getattr(doc, "_is_last_page", False)
    # -------------------------------------------------

    # Título de sección (solo si existe)
    try:
        if getattr(doc, "section_title", ""):
            canvas.setFont("Helvetica-Bold", 12)
            canvas.setFillColor(_hex_to_rl(CSS_COLORS["brand_dark"]))
            x = doc.leftMargin
            y = doc.pagesize[1] - doc.topMargin + 0.5 * cm
            canvas.drawString(x, y, doc.section_title)
    except Exception:
        pass

    # Logo (solo si NO es la última página)
    if not is_last_page:
        try:
            logo_path = "APCE_mod.png"
            logo_width, logo_height = 3.5 * cm, 3.5 * cm
            x_position = doc.pagesize[0] - logo_width - 0.25 * cm
            y_position = doc.pagesize[1] - logo_height
            canvas.drawImage(logo_path, x_position, y_position,
                             width=logo_width, height=logo_height,
                             preserveAspectRatio=True, mask='auto')
        except Exception:
            pass

    # Pie de página y número (solo si NO es la última página)
    if not is_last_page:
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.setFillColor(_hex_to_rl(CSS_COLORS["text"]))
        canvas.drawString(
            1.2 * cm, 0.8 * cm,
            "Font de les dades: APCE, Agència de l'Habitatge de Catalunya, INCASÒL, INE."
        )
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(doc.pagesize[0] - 1.2 * cm, 0.8 * cm, f"Pàgina {doc.page}")

    canvas.restoreState()


def _header_footer_cover(canvas, doc):
    """
    Cabecera/pie para la portada:
    sin logo superior derecho ni texto de fuentes.
    """
    canvas.saveState()
    # Fondo igual que el resto para mantener consistencia
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["bg"]))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], stroke=0, fill=1)
    canvas.restoreState()

    # (Opcional) si quieres número de página en portada, descomenta esto:
    # canvas.setFont("Helvetica", 8)
    # canvas.drawRightString(doc.pagesize[0] - 1.2 * cm, 0.8 * cm, f"Pàgina {doc.page}")





def _header_footer_normal(canvas, doc):
    canvas.saveState()

    W, H = doc.pagesize
    margin_x = 1.2*cm
    footer_h = 1.35*cm
    bar_h    = 0.25*cm
    y0       = 0
    y_text   = y0 + bar_h + 0.55*cm

    # Fons
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["bg"]))
    canvas.rect(0, 0, W, H, stroke=0, fill=1)

    # Franja inferior
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["primary"]))
    canvas.rect(0, y0, W, bar_h, stroke=0, fill=1)

    # Línia divisòria
    canvas.setStrokeColor(_hex_to_rl(CSS_COLORS["accent"]))
    canvas.setLineWidth(0.6)
    canvas.line(margin_x, y0 + bar_h + 0.35*cm, W - margin_x, y0 + bar_h + 0.35*cm)

    # Colors/textos
    txt_color  = _hex_to_rl(CSS_COLORS["text"])
    link_color = _hex_to_rl(CSS_COLORS["primary"])

    # Esquerra: fonts
    canvas.setFillColor(txt_color)
    canvas.setFont("Helvetica-Oblique", 9)
    left_text = "Font de les dades: APCE, Agència de l'Habitatge de Catalunya, INCASÒL, INE."
    canvas.drawString(margin_x, y_text, left_text)

    # Centre: web clicable
    center_text = "www.apcebcn.cat"
    canvas.setFont("Helvetica-Bold", 9)
    tw_center = canvas.stringWidth(center_text, "Helvetica-Bold", 9)
    cx = W/2 - tw_center/2
    canvas.setFillColor(link_color)
    canvas.drawString(cx, y_text, center_text)
    try:
        canvas.linkURL("https://apcebcn.cat/", (cx, y_text-1, cx+tw_center, y_text+10), relative=0, thickness=0)
    except Exception:
        pass

    # Dreta: badge només amb número de pàgina, mida i amplada adaptativa
    page_text = str(doc.page)
    font_name = "Helvetica-Bold"
    font_size = 13  # número més gran
    canvas.setFont(font_name, font_size)

    # Amplada del text (punts)
    tw = canvas.stringWidth(page_text, font_name, font_size)

    # Padding en punts (adaptatius)
    pad_x = 8   # ~2.8 mm
    pad_y = 4   # ~1.4 mm

    badge_w = tw + 2*pad_x
    badge_h = font_size + 2*pad_y  # suficient per encabir l’altura del text

    bx = W - margin_x - badge_w
    # Vertical: alineem amb la línia base del text central
    by = y_text - (badge_h - font_size)/2 - 1

    # Pastilla arrodonida
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["accent"]))
    try:
        canvas.roundRect(bx, by, badge_w, badge_h, 6, stroke=0, fill=1)
    except Exception:
        canvas.rect(bx, by, badge_w, badge_h, stroke=0, fill=1)

    # Número centrat
    canvas.setFillColor(txt_color)
    tx = bx + (badge_w - tw)/2
    ty = by + (badge_h - font_size)/2 - 1  # petit ajust òptic
    canvas.drawString(tx, ty, page_text)

    canvas.restoreState()



def _header_footer_minimal(canvas, doc):
    # Solo fondo, sin logo ni número ni fuente
    canvas.saveState()
    canvas.setFillColor(_hex_to_rl(CSS_COLORS["bg"]))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], stroke=0, fill=1)
    canvas.restoreState()









def build_location_pdf_ordered(
    location_name: str,
    kpis: List[Tuple[str, str, Optional[str]]],
    sections: List[Tuple[str, List[Tuple[str, object]]]],  # [(titulo_seccion, [("table", (titulo, df)) o ("fig", (titulo, png_bytes)) , ...])]
) -> bytes:
    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer,
        pagesize=rl_landscape(A4),
        rightMargin=1.75 * cm,
        leftMargin=1.75 * cm,
        topMargin=2.5 * cm,
        bottomMargin=1.75 * cm
    )

    # === Crear el frame común ===
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='frame')

    # === Añadir las tres plantillas de página ===
    tpl_cover   = PageTemplate(id='Cover',   frames=frame, onPage=_header_footer_cover)
    tpl_normal  = PageTemplate(id='Normal',  frames=frame, onPage=_header_footer_normal)
    tpl_minimal = PageTemplate(id='Minimal', frames=frame, onPage=_header_footer_minimal)
    doc.addPageTemplates([tpl_cover, tpl_normal, tpl_minimal])
    doc.title = f"Informe de mercat residencial (APCE) — {location_name}"
    doc.author = "APCE CATALUNYA"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBrand", parent=styles["Title"], fontSize=18,
                              textColor=_hex_to_rl(CSS_COLORS["brand_dark"])))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=14,
                              textColor=_hex_to_rl(CSS_COLORS["brand_dark"]), spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9,
                              textColor=_hex_to_rl(CSS_COLORS["text"])))
    styles.add(ParagraphStyle(name="KPI", parent=styles["BodyText"], fontSize=14, leading=17,
                              textColor=_hex_to_rl(CSS_COLORS["text"])))
    styles.add(ParagraphStyle(
        name="SectionBand", parent=styles["Heading2"], fontSize=14, leading=16,
        textColor=_hex_to_rl("#ffffff"), backColor=_hex_to_rl(CSS_COLORS["primary"]),
        leftIndent=0, rightIndent=0, spaceBefore=8, spaceAfter=6, alignment=0
    ))
    story = []



    def append_cover_page(story, styles, location_name, logo_path="APCE_mod.png"):
        story.append(Spacer(1, 2 * cm))  # margen superior

        # Logo grande centrado
        try:
            logo = RLImage(logo_path, width=10 * cm, height=5 * cm)
            logo.hAlign = 'CENTER'
            story.append(logo)
        except Exception:
            story.append(Spacer(1, 6 * cm))

        story.append(Spacer(1, 1.0 * cm))

        # Título principal
        story.append(Paragraph(
            "INFORME DE MERCAT RESIDENCIAL",
            ParagraphStyle(
                "CoverTitle",
                parent=styles["Title"],
                fontSize=28,
                leading=32,
                alignment=1,  # centrado
                textColor=_hex_to_rl(CSS_COLORS["brand_dark"]),
                spaceAfter=12
            )
        ))

        # Subtítulo con nombre del municipio
        story.append(Paragraph(
            f"<b>{location_name.upper()}</b>",
            ParagraphStyle(
                "CoverSub",
                parent=styles["BodyText"],
                fontSize=20,
                alignment=1,
                textColor=_hex_to_rl(CSS_COLORS["primary"]),
                spaceAfter=20
            )
        ))

        # Línea divisoria fina (opcional)
        story.append(Spacer(1, 0.3 * cm))
        tbl = Table([[""]], colWidths=[16 * cm], rowHeights=[0.05 * cm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), _hex_to_rl("#e0e0e0")),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.8 * cm))

        # Texto de autoría
        story.append(Paragraph(
            "Elaborat per <b>APCE Catalunya</b>",
            ParagraphStyle(
                "CoverMeta",
                parent=styles["BodyText"],
                alignment=1,
                fontSize=12,
                textColor=_hex_to_rl(CSS_COLORS["text"]),
                spaceAfter=6
            )
        ))

        # Fecha de generación
        story.append(Paragraph(
            f"Data de generació de l'informe: {datetime.now():%d/%m/%Y}",
            ParagraphStyle(
                "CoverMeta2",
                parent=styles["BodyText"],
                alignment=1,
                fontSize=11,
                textColor=_hex_to_rl("#777777")
            )
        ))

        # Empujar hacia el final
        story.append(Spacer(1, 0.5 * cm))

        # Enlace institucional (clicable)
        story.append(Paragraph(
            f'<link href="https://apcebcn.cat/es/" color="{CSS_COLORS["primary"]}">www.apcebcn.cat</link>',
            ParagraphStyle(
                "CoverLink",
                parent=styles["BodyText"],
                alignment=1,
                fontSize=12,
                textColor=_hex_to_rl(CSS_COLORS["primary"])
            )
        ))
        # ⬇⬇⬇ AÑADIR ESTAS DOS LÍNEAS ANTES DEL SALTO ⬇⬇⬇

        story.append(NextPageTemplate('Normal'))
        story.append(PageBreak())







    # === estilos extra para la página final ===
    styles.add(ParagraphStyle(name="CenterBig", parent=styles["BodyText"],
                            alignment=1, fontSize=14,
                            textColor=_hex_to_rl(CSS_COLORS["brand_dark"])))
    styles.add(ParagraphStyle(name="Center", parent=styles["BodyText"],
                            alignment=1, fontSize=11,
                            textColor=_hex_to_rl(CSS_COLORS["text"])))
    styles.add(ParagraphStyle(name="SmallCorner", parent=styles["BodyText"],
                            alignment=2, fontSize=8,
                            textColor=_hex_to_rl("#777777")))

    def append_closing_page(story, styles, logo_path="APCE_serveis1.png"):
        try:
            logo = RLImage(logo_path, width=24*cm, height=13.5*cm)
            logo.hAlign = 'CENTER'
            story.append(Spacer(1, 0*cm))
            story.append(logo)
        except Exception:
            story.append(Spacer(1, 10.0*cm))


    # === Portada ===
    append_cover_page(story, styles, location_name=selected_mun, logo_path="APCE_mod.png")


    story.append(Paragraph(
        f"Informe de mercat residencial (APCE): Municipi de {location_name}",
        styles["TitleBrand"]
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Darreres dades dels indicadors socioeconòmics i del mercat residencial a {selected_mun}", styles["Section"]))

    if kpis and len(kpis) > 0:
        # Preparamos las celdas como Paragraphs
        kpi_paragraphs = []
        for label, val, delta in kpis:
            line = f"<b>{label}:</b> {val} " + (_delta_fmt(delta) if delta else "")
            kpi_paragraphs.append(Paragraph(line, styles["KPI"]))

        # Calculamos nº de filas (mitad superior redondeada)
        n = len(kpi_paragraphs)
        half = (n + 1) // 2

        # Dividimos en dos columnas
        col1 = kpi_paragraphs[:half]
        col2 = kpi_paragraphs[half:]

        # Igualamos alturas (rellenamos con vacío si hace falta)
        while len(col1) < len(col2):
            col1.append(Paragraph("", styles["KPI"]))
        while len(col2) < len(col1):
            col2.append(Paragraph("", styles["KPI"]))

        # Creamos tabla 2 columnas
        kpi_data = list(zip(col1, col2))
        kpi_tbl = Table(kpi_data, colWidths=[doc.width/2, doc.width/2])  # ajusta al ancho del PDF
        kpi_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), _hex_to_rl(CSS_COLORS["accent"])),
            ('BOX', (0,0), (-1,-1), 0.25, rl_colors.lightgrey),
            ('INNERGRID', (0,0), (-1,-1), 0.25, rl_colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(kpi_tbl)

    else:
        story.append(Paragraph("No hi ha KPIs disponibles.", styles["Small"]))

    story.append(Spacer(1, 0.1*cm))
    story.append(PageBreak())
    # ====== SECCIONES con salto de página entre ellas ======
    for si, (section_title, items) in enumerate(sections):
        # Cabecera de sección (opcional, si quieres un gran título por bloque)
        # story.append(Paragraph(section_title, styles["Section"]))
        # story.append(Spacer(1, 2*cm))

        for kind, payload in items:
            if kind == "table":
                title, df = payload
                hdr = Paragraph(title, styles["Section"])
                hdr.keepWithNext = True
                story.append(hdr)
                try:
                    df_disp = df.data if hasattr(df, "data") else df
                    tbl = _styled_table_from_df(df_disp, max_rows=None, max_cols=12)
                    story.append(tbl)
                except Exception:
                    story.append(Paragraph("[No s'ha pogut mostrar la taula]", styles["Small"]))
                story.append(Spacer(1, 1*cm))

            elif kind == "fig":
                title, png_bytes = payload
                hdr = Paragraph(title, styles["Section"])
                hdr.keepWithNext = True
                img = Image(io.BytesIO(png_bytes), width=23*cm, height=11*cm)
                story.append(KeepTogether([hdr, img]))
                story.append(Spacer(1, 1*cm))

        if si < len(sections) - 1:
            story.append(Spacer(1, 0.5*cm))
            story.append(CondPageBreak(8*cm))  # rompe si quedan < 8 cm libres
    story.append(NextPageTemplate('Minimal'))
    append_closing_page(story, styles, logo_path="APCE_serveis1.png")
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ========== HELPERS EXTRA (ALTRES INDICADORS) ==========
def _latest_value(df: pd.DataFrame, col: str):
    try:
        dfx = df[["Any", col]].dropna().copy().sort_values("Any")
        return int(dfx["Any"].iloc[-1]), dfx[col].iloc[-1]
    except Exception:
        return None, None

def mpl_donut(labels, values) -> bytes:
    fig, ax = _mpl_base()
    donut_colors = ["#2d538f", "#de7207", "#1b7f3a", "#6a3d9a", "#b15928", "#727375"][:len(labels)]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, startangle=90, colors=donut_colors,
        wedgeprops=dict(width=0.45, edgecolor=CSS_COLORS["bg"]),
        autopct="%1.1f%%", pctdistance=0.75
    )
    for t in texts:
        t.set_fontsize(9)
        t.set_color(CSS_COLORS["text"])
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(9)
        at.set_weight("bold")
    ax.axis("equal")
    ax.grid(False)
    return _mpl_finish(fig)

def _map_df_mun_idescat_basic(df_mun_idescat: pd.DataFrame, selected_mun: str) -> Optional[pd.DataFrame]:
    """
    Devuelve un DataFrame con una columna adicional 'nombre_largo'
    que mapea las variables internas (de df_mun_idescat) a sus nombres
    descriptivos en catalán, eliminando el sufijo del municipio.

    Se usa para extraer valores agregados (last_year, last_value) de
    indicadores demográficos, económicos o laborales.
    """

    if df_mun_idescat is None or "variable" not in df_mun_idescat.columns:
        return None

    # Diccionario maestro de variables reconocidas
    name_map = {
        # ECONOMIA I RENDA
        "IRPF_Base_imposable": "Base imposable mitjana de l’IRPF (€)",
        "Pensionistes_Total": "Nombre de pensionistes",
        "Parc_vehicles_Total": "Parc total de vehicles",
        "Residus_mun_per_capita": "Residus municipals per càpita (kg/hab/dia)",

        # MERCAT LABORAL
        "AfiliatSS_Agricultura": "Afiliats a la Seguretat Social – Agricultura",
        "AfiliatSS_Construcció": "Afiliats a la Seguretat Social – Construcció",
        "AfiliatSS_Indústria": "Afiliats a la Seguretat Social – Indústria",
        "AfiliatSS_Serveis": "Afiliats a la Seguretat Social – Serveis",
        "AfiliatSS_Total": "Afiliats a la Seguretat Social – Total",
        "Atur registrat_Agricultura": "Atur registrat – Agricultura",
        "Atur registrat_Construcció": "Atur registrat – Construcció",
        "Atur registrat_Indústria": "Atur registrat – Indústria",
        "Atur registrat_Serveis": "Atur registrat – Serveis",
        "Atur registrat_Total": "Atur registrat – Total",
        "poblacio_activa": "Població activa",
        "poblacio_ocupada": "Població ocupada",
        "poblacio_desocupada": "Població desocupada",
        "poblacio_inactiva": "Població inactiva",

        # DEMOGRAFIA
        "Matrimonis_Total": "Nombre de matrimonis",
        "Naixements_Total": "Nombre de naixements",
    }

    df = df_mun_idescat.copy()
    df["variable_sin_municipi"] = df["variable"].astype(str).str.replace(
        f"_{selected_mun}$", "", regex=True
    )
    df["nombre_largo"] = df["variable_sin_municipi"].map(name_map)
    return df


def _pick_last_val(df_long: pd.DataFrame, long_label: str):
    try:
        row = df_long.loc[df_long["nombre_largo"] == long_label].iloc[0]
        return int(row["last_year"]), float(row["last_value"])
    except Exception:
        return None, None

# ========== CONSTRUCTOR ORDENADO DE SECCIONES ==========
# Para mantener el orden, rellenamos "blocks" en generar_pdf_municipi_tot en el orden deseado.
def append_altres_indicadors_blocks(
    selected_mun: str,
    kpis_pdf: list,
    blocks: list,
    censo_2021=None, DT_mun_y=None, idescat_muns=None, rentaneta_mun=None,
    df_mun_idescat: Optional[pd.DataFrame] = None,   # <-- NUEVO
    df_pob_ine: Optional[pd.DataFrame] = None, 
    start_year_series: int = 2014
):
    # --- Población (tabla + línea)
    try:
        pop_col = f"poptottine_{selected_mun}"
        if DT_mun_y is not None and pop_col in DT_mun_y.columns:
            df_pop = DT_mun_y.loc[:, ["Fecha", pop_col]].dropna().copy()
            df_pop["Fecha"] = pd.to_numeric(df_pop["Fecha"], errors="coerce").astype("Int64")
            df_pop = df_pop.dropna(subset=["Fecha"]).copy()
            df_pop["Fecha"] = df_pop["Fecha"].astype(int)
            df_pop = df_pop.sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last")
            df_pop = df_pop.set_index("Fecha")
            df_pop[pop_col] = pd.to_numeric(df_pop[pop_col], errors="coerce")
            df_pop = df_pop[df_pop.index >= 2000]
            df_pop = df_pop.rename(columns={pop_col: "Població"})

            try:
                last_year = int(df_pop.index[-1])
                last_val = int(df_pop["Població"].iloc[-1])
                prev_year = last_year - 5
                if prev_year in df_pop.index:
                    prev_val = float(df_pop.loc[prev_year, "Població"])
                    delta = f"{(100.0 * (last_val/prev_val - 1)):.1f}%"
                else:
                    delta = None
                    if len(df_pop) >= 6:
                        prev_val = float(df_pop["Població"].iloc[-6])
                        delta = f"{(100.0 * (last_val/prev_val - 1)):.1f}%"
                kpis_pdf.append(("Població (últim any)", f"{last_val:,.0f}".replace(",", "."), delta))
            except Exception:
                pass

            blocks.append(("table", f"Evolució anual de la població al municipi de {selected_mun}", df_pop[df_pop.index>2014].T))
            blocks.append(("figure",
                           f"Evolució de la població al municipi de {selected_mun}",
                           mpl_line(df_pop, ["Població"], "", "Persones", "Any", start_year=2008, force_all_xticks=True)))
    except Exception:
        pass

    # --- Tamaño hogar (donut + KPIs)
    try:
        if censo_2021 is not None:
            row = censo_2021[censo_2021["Municipi"] == selected_mun].iloc[0]
            labels = ["1", "2", "3", "4", "5 o más"]
            vals = [row.get("1", 0), row.get("2", 0), row.get("3", 0), row.get("4", 0), row.get("5 o más", 0)]
            blocks.append(("figure",
                           f"Demografia — Distribució per grandària de llar (Cens 2021) — {selected_mun}",
                           mpl_donut(labels, vals)))
            try:
                kpis_pdf.append(("Grandària de la llar més freqüent", f"{row['Tamaño_hogar_frecuente']} llars", None))
                kpis_pdf.append(("Grandària mitjà de la llar", f"{float(row['Tamaño medio del hogar']):.2f}", None))
                kpis_pdf.append(("Població nacional", f"{(100.0 - float(row['Perc_extranjera'])):.1f}%", None))
                kpis_pdf.append(("Població estrangera", f"{float(row['Perc_extranjera']):.1f}%", None))
            except Exception:
                pass
    except Exception:
        pass

    # --- Renta neta por hogar (tabla + barras)
    try:
        if rentaneta_mun is not None:
            df_rn = rentaneta_mun.rename(columns={"Año": "Any"}).copy()
            col_rn = f"rentanetahogar_{selected_mun}"
            if col_rn in df_rn.columns:
                df_rn = df_rn[["Any", col_rn]].dropna().rename(columns={col_rn: "Renda neta per llar"})
                df_rn = df_rn.set_index("Any")
                try:
                    any_rn = int(df_rn.index[-1])
                    val_rn = float(df_rn.iloc[-1, 0])
                    kpis_pdf.append((f"Renda neta per llar ({any_rn})", f"{val_rn:,.0f}".replace(",", "."), None))
                except Exception:
                    pass
                blocks.append(("table",
                               f"Economia — Renda mitjana neta per llar (anual) — {selected_mun}",
                               df_rn.T))
                blocks.append(("figure",
                               f"Economia — Evolució de la renda neta per llar ({selected_mun})",
                               mpl_bar(df_rn, ["Renda neta per llar"], "", "€ per llar",
                                       start_year=max(start_year_series, 2015), force_all_xticks=True)))
    except Exception:
        pass


# ========== GENERADOR — MUNICIPI (ORDEN COHERENTE) ==========
def generar_pdf_municipi_tot(
    selected_mun: str,
    # --- Producció
    table_mun_prod: pd.DataFrame, table_mun_prod_y: pd.DataFrame,
    table_mun_prod_pluri: pd.DataFrame, table_mun_prod_uni: pd.DataFrame,
    selected_columns_ini: List[str], selected_columns_fin: List[str],
    # --- Compravendes
    table_mun_tr: pd.DataFrame, table_mun_tr_y: pd.DataFrame,
    # --- Preus
    table_mun_pr: pd.DataFrame, table_mun_pr_y: pd.DataFrame,
    # --- Superfície
    table_mun_sup: pd.DataFrame, table_mun_sup_y: pd.DataFrame,
    # --- Lloguer
    table_mun_llog: pd.DataFrame, table_mun_llog_y: pd.DataFrame,
    # --- Altres indicadors (dataframes globales ya cargados)
    censo_2021=None, DT_mun_y=None, idescat_muns=None, rentaneta_mun=None, tabla_estudi_oferta=None
):
    """Genera el PDF del municipi con secciones ordenadas (tabla(s) → gráfico(s)) y salto de página entre indicadores."""
    # ==========================
    # 1) KPIs
    # ==========================
    kpis_pdf = []

    def _safe_add_kpi(table_y, table_q, col, label):
        try:
            year = str(datetime.now().year)
            val = indicator_year(table_y, table_q, year, col, "level")
            var = indicator_year(table_y, table_q, year, col, "var")
        except Exception:
            try:
                last_year = str(table_y.index[-1])
                val = indicator_year(table_y, table_q, last_year, col, "level")
                var = indicator_year(table_y, table_q, last_year, col, "var")
            except Exception:
                kpis_pdf.append((label, "No disponible", None))
                return
        kpis_pdf.append((label, f"{val:,.0f}".replace(",", "."), f"{var}%"))

    # Producció — totals + tipologies
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges iniciats", "Habitatges iniciats")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges acabats", "Habitatges acabats")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges iniciats plurifamiliars", "Iniciats plurifamiliars")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges iniciats unifamiliars", "Iniciats unifamiliars")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges acabats plurifamiliars", "Acabats plurifamiliars")
    _safe_add_kpi(table_mun_prod_y, table_mun_prod, "Habitatges acabats unifamiliars", "Acabats unifamiliars")
    try:
        if DT_mun_y is not None:
            col_prov = f"calprovgene_{selected_mun}"
            col_def  = f"caldefgene_{selected_mun}"
            cols_ok = [c for c in [col_prov, col_def] if c in DT_mun_y.columns]
            if cols_ok:
                # Base limpia (desde 2000)
                df_vpo = (
                    DT_mun_y.loc[:, ["Fecha"] + cols_ok]
                    .dropna(how="all", subset=cols_ok)
                    .assign(Fecha=lambda d: pd.to_numeric(d["Fecha"], errors="coerce").astype("Int64"))
                    .dropna(subset=["Fecha"])
                    .assign(Fecha=lambda d: d["Fecha"].astype(int))
                    .sort_values("Fecha")
                    .drop_duplicates(subset=["Fecha"], keep="last")
                )
                df_vpo = df_vpo[df_vpo["Fecha"] >= 2000]

                for col, label in [
                    (col_prov, "Qualificacions provisionals d'HPO"),
                    (col_def,  "Qualificacions definitives d'HPO"),
                ]:
                    if col in df_vpo.columns and not df_vpo[col].dropna().empty:
                        df_col = df_vpo.dropna(subset=[col])
                        last_year = int(df_col["Fecha"].iloc[-1])
                        last_val  = float(df_col[col].iloc[-1])
                        # delta vs. año anterior (si existe y no es 0)
                        prev = df_col.loc[df_col["Fecha"] == last_year - 1, col]
                        delta = None
                        if not prev.empty and float(prev.iloc[0]) != 0:
                            delta = f"{(100.0 * (last_val / float(prev.iloc[0]) - 1)):.1f}%"

                        kpis_pdf.append((
                            f"{label} ({last_year})",
                            f"{last_val:,.0f}".replace(",", "."),
                            delta
                        ))
    except Exception:
        pass

    # Compravendes
    _safe_add_kpi(table_mun_tr_y, table_mun_tr, "Compravendes d'habitatge total", "Compravendes")
    _safe_add_kpi(table_mun_tr_y, table_mun_tr, "Compravendes d'habitatge de segona mà", "Compravendes segona mà")
    _safe_add_kpi(table_mun_tr_y, table_mun_tr, "Compravendes d'habitatge nou", "Compravendes habitatge nou")

    # Preus
    _safe_add_kpi(table_mun_pr_y, table_mun_pr, "Preu d'habitatge total", "Preu €/m²")
    _safe_add_kpi(table_mun_pr_y, table_mun_pr, "Preu d'habitatge de segona mà", "Preu €/m² segona mà")
    _safe_add_kpi(table_mun_pr_y, table_mun_pr, "Preu d'habitatge nou", "Preu €/m² nou")

    # Superfície
    _safe_add_kpi(table_mun_sup_y, table_mun_sup, "Superfície mitjana total", "Superfície mitjana (m² construït)")
    _safe_add_kpi(table_mun_sup_y, table_mun_sup, "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana segona mà (m² construït)")
    _safe_add_kpi(table_mun_sup_y, table_mun_sup, "Superfície mitjana d'habitatge nou", "Superfície mitjana nou (m² construït)")

    # Lloguer
    _safe_add_kpi(table_mun_llog_y, table_mun_llog, "Nombre de contractes de lloguer", "Contractes de lloguer")
    _safe_add_kpi(table_mun_llog_y, table_mun_llog, "Rendes mitjanes de lloguer", "Renda mitjana lloguer (€/mes)")
    # === Altres indicadors -> IRPF al bloque de KPIs ===
    # === BLOQUE UNIFICADO: Altres indicadors (df_mun_idescat) ===
    try:
        if censo_2021 is not None:
            row = censo_2021[censo_2021["Municipi"] == selected_mun].iloc[0]

            # evita duplicados si este bloque se ejecuta más de una vez
            parc_labels = {
                "Propietat", "Habitatges principals", "Habitatges no principals",
                "Habitatges en lloguer", "Edat mitjana habitatges", "Superfície mitjana (m²)"
            }
            kpis_pdf = [k for k in kpis_pdf if k[0] not in parc_labels]

            parc_kpis = [
                ("Habitatges en propietat",                f"{float(row['Perc_propiedad']):.1f}%", None),
                ("Habitatges principals",    f"{(100.0 - float(row['Perc_noprincipales_y'])):.1f}%", None),
                ("Habitatges no principals", f"{float(row['Perc_noprincipales_y']):.1f}%", None),
                ("Habitatges en lloguer",    f"{float(row['Perc_alquiler']):.1f}%", None),
                ("Edat mitjana habitatges",  f"{float(row['Edad media']):.1f}", None),
                ("Superfície mitjana (m²)",  f"{float(row['Superficie media']):.1f}", None),
            ]

            # Añadirlos al final del listado de KPIs
            kpis_pdf.extend(parc_kpis)
    except Exception:
        pass
    try:
        df_long = _map_df_mun_idescat_basic(df_mun_idescat, selected_mun)
        if df_long is not None and not df_long.empty:

            def _append_if_ok(nombre_largo: str, label_fmt: Optional[str] = None, fmt: str = "int"):
                yr, val = _pick_last_val(df_long, nombre_largo)
                if yr is not None and pd.notnull(val):
                    label = label_fmt.format(yr) if label_fmt else f"{nombre_largo} ({yr})"
                    if fmt == "int":
                        val_str = f"{int(round(val)):,}".replace(",", ".")
                    elif fmt == "float1":
                        val_str = f"{float(val):.1f}"
                    elif fmt == "float2":
                        val_str = f"{float(val):.2f}"
                    else:
                        val_str = str(val)
                    kpis_pdf.append((label, val_str, None))

            # --- MERCAT LABORAL ---
            _append_if_ok("Afiliats a la Seguretat Social – Agricultura")
            _append_if_ok("Afiliats a la Seguretat Social – Construcció")
            _append_if_ok("Afiliats a la Seguretat Social – Indústria")
            _append_if_ok("Afiliats a la Seguretat Social – Serveis")
            _append_if_ok("Afiliats a la Seguretat Social – Total")

            _append_if_ok("Atur registrat – Agricultura")
            _append_if_ok("Atur registrat – Construcció")
            _append_if_ok("Atur registrat – Indústria")
            _append_if_ok("Atur registrat – Serveis")
            _append_if_ok("Atur registrat – Total")

            _append_if_ok("Població activa")
            _append_if_ok("Població ocupada")
            _append_if_ok("Població desocupada")
            _append_if_ok("Població inactiva")
            # --- ECONOMIA I RENDA ---
            _append_if_ok("Base imposable mitjana de l’IRPF (€)")
            _append_if_ok("Nombre de pensionistes")
            _append_if_ok("Parc total de vehicles")
            _append_if_ok("Residus municipals per càpita (kg/hab/dia)", fmt="float2")

            # --- DEMOGRAFIA ---
            _append_if_ok("Nombre de matrimonis")
            _append_if_ok("Nombre de naixements")

    except Exception as e:
        print(f"⚠️ Error al afegir Altres indicadors al bloc de KPIs: {e}")
        pass



    # ==========================
    # 2) SECCIONES (tabla(s) → gráfico(s))
    # ==========================
    sections: List[Tuple[str, List[Tuple[str, Tuple[str, object]]]]] = []

    # --------- PRODUCCIÓ ---------
    items_produccio = []
    try:
        items_produccio.append((
            "table",
            (f"Evolució trimestral de la producció d'habitatges al municipi de {selected_mun}",
             table_trim(table_mun_prod, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_produccio.append((
            "table",
            (f"Evolució anual de la producció d'habitatges al municipi de {selected_mun}",
             table_year(table_mun_prod_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    try:
        items_produccio.append((
            "fig",
            (f"Evolució trimestral dels habitatges iniciats i acabats al municipi de {selected_mun}",
             mpl_line(table_mun_prod, ["Habitatges iniciats", "Habitatges acabats"], "", "Habitatges",
                      start_year=SERIES_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_produccio.append((
            "fig",
            (f"Evolució anual dels habitatges iniciats i acabats al municipi de {selected_mun}",
             mpl_bar(table_mun_prod_y, ["Habitatges iniciats", "Habitatges acabats"], "", "Habitatges",
                     start_year=SERIES_START_YEAR, force_all_xticks=True))
        ))
    except Exception:
        pass
    # Tipologies
    try:
        typ_ini_cols = selected_columns_ini
        typ_ini_palette = [GLOBAL_PALETTE["unifamiliar"] if "unifam" in c.lower()
                           else GLOBAL_PALETTE["plurifamiliar"] if "plurifam" in c.lower()
                           else GLOBAL_PALETTE["total"] for c in typ_ini_cols]
        items_produccio.append((
            "fig",
            (f"Evolució dels habitatges iniciats per tipologia al municipi de {selected_mun}",
             mpl_area(table_mun_prod[typ_ini_cols], typ_ini_cols, "", "Habitatges iniciats",
                      start_year=SERIES_START_YEAR, palette=typ_ini_palette))
        ))
    except Exception:
        pass

    try:
        typ_fin_cols = selected_columns_fin
        typ_fin_palette = [GLOBAL_PALETTE["unifamiliar"] if "unifam" in c.lower()
                           else GLOBAL_PALETTE["plurifamiliar"] if "plurifam" in c.lower()
                           else GLOBAL_PALETTE["total"] for c in typ_fin_cols]
        items_produccio.append((
            "fig",
            (f"Evolució dels habitatges acabats per tipologia al municipi de {selected_mun}",
             mpl_area(table_mun_prod[typ_fin_cols], typ_fin_cols, "", "Habitatges acabats",
                      start_year=SERIES_START_YEAR, palette=typ_fin_palette))
        ))
    except Exception:
        pass

    # Per superfície
    try:
        items_produccio.append((
            "fig",
            (f"Habitatges iniciats plurifamiliars per superfície al municipi de {selected_mun}",
             mpl_area(table_mun_prod_pluri, table_mun_prod_pluri.columns.tolist(), "", "Habitatges iniciats",
                      start_year=SERIES_START_YEAR,
                      palette=["#2d538f", "#1b7f3a", "#de7207", "#6a3d9a", "#b15928", "#727375", "#9aa0a6"]))
        ))
    except Exception:
        pass
    try:
        items_produccio.append((
            "fig",
            (f"Habitatges iniciats unifamiliars per superfície al municipi de {selected_mun}",
             mpl_area(table_mun_prod_uni, table_mun_prod_uni.columns.tolist(), "", "Habitatges iniciats",
                      start_year=SERIES_START_YEAR,
                      palette=["#2d538f", "#1b7f3a", "#de7207", "#6a3d9a", "#b15928", "#727375", "#9aa0a6"]))
        ))
    except Exception:
        pass
    if items_produccio:
        sections.append(("Producció", items_produccio))
# --------- HABITATGE PROTEGIT (HPO) ---------
    items_vpo = []
    try:
        if DT_mun_y is not None:
            col_prov = f"calprovgene_{selected_mun}"
            col_def  = f"caldefgene_{selected_mun}"
            cols_ok = [c for c in [col_prov, col_def] if c in DT_mun_y.columns]
            if cols_ok:
                df_vpo = DT_mun_y.loc[:, ["Fecha"] + cols_ok].dropna(how="all", subset=cols_ok).copy()
                df_vpo["Fecha"] = pd.to_numeric(df_vpo["Fecha"], errors="coerce").astype("Int64")
                df_vpo = df_vpo.dropna(subset=["Fecha"]).copy()
                df_vpo["Fecha"] = df_vpo["Fecha"].astype(int)
                df_vpo = df_vpo.sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last")
                df_vpo = df_vpo[df_vpo["Fecha"] >= 2000].copy()  # desde 2000
                df_vpo = df_vpo.rename(columns={
                    col_prov: "Qualificacions provisionals HPO",
                    col_def:  "Qualificacions definitives HPO"
                })
                df_vpo = df_vpo.set_index("Fecha")

                # Tabla (transpuesta para encajar con tu estilo)
                items_vpo.append((
                    "table",
                    (f"Evolució de les qualificacions anuals d'habitatge protegit (HPO) al municipi de {selected_mun}",
                    df_vpo[df_vpo.index>2012].T)
                ))

                # Gráfico de líneas (desde 2000)
                items_vpo.append((
                    "fig",
                    (f"",
                    mpl_line(
                        df_vpo,
                        [c for c in ["Qualificacions provisionals HPO", "Qualificacions definitives HPO"] if c in df_vpo.columns],
                        title="",
                        ylab="Habitatges",
                        xlab="Any",
                        start_year=2000,
                        force_all_xticks=True
                    ))
                ))
    except Exception:
        pass

    if items_vpo:
        sections.append(("Habitatge protegit (HPO)", items_vpo))

    # --------- COMPRAVENDES ---------
    items_comp = []
    try:
        items_comp.append((
            "table",
            (f"Evolució trimestral de les compravendes al municipi de {selected_mun}",
             table_trim(table_mun_tr, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_comp.append((
            "table",
            (f"Evolució anual de les compravendes al municipi de  {selected_mun}",
             table_year(table_mun_tr_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    comp_cols = [
        "Compravendes d'habitatge total",
        "Compravendes d'habitatge de segona mà",
        "Compravendes d'habitatge nou"
    ]
    comp_palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"]]
    try:
        items_comp.append((
            "fig",
            (f"Evolució trimestral de les compravendes al municipi de {selected_mun}",
             mpl_line(table_mun_tr, comp_cols, "", "Operacions",
                      start_year=SERIES_START_YEAR, palette=comp_palette))
        ))
    except Exception:
        pass
    try:
        items_comp.append((
            "fig",
            (f"Evolució anual de les compravendes al municipi de {selected_mun}",
             mpl_bar(table_mun_tr_y, comp_cols, "", "Operacions",
                     start_year=SERIES_START_YEAR, palette=comp_palette, force_all_xticks=True))
        ))
    except Exception:
        pass
    if items_comp:
        sections.append(("Compravendes", items_comp))

    # --------- PREUS ---------
    items_preus = []
    try:
        items_preus.append((
            "table",
            (f"Evolució trimestral dels preus al municipi de {selected_mun}",
             table_trim(table_mun_pr, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_preus.append((
            "table",
            (f"Evolució anual dels preus €/m² al municipi de {selected_mun}",
             table_year(table_mun_pr_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    preus_cols = [
        "Preu d'habitatge total",
        "Preu d'habitatge de segona mà",
        "Preu d'habitatge nou"
    ]
    preus_palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"]]
    try:
        items_preus.append((
            "fig",
            (f"Evolució trimestral dels preus €/m² al municipi de {selected_mun}",
             mpl_line(table_mun_pr, preus_cols, "", "€/m²",
                      start_year=SERIES_START_YEAR, palette=preus_palette))
        ))
    except Exception:
        pass
    try:
        items_preus.append((
            "fig",
            (f"Evolució anual dels preus €/m² al municipi de {selected_mun}",
             mpl_bar(table_mun_pr_y, preus_cols, "", "€/m²",
                     start_year=SERIES_START_YEAR, palette=preus_palette, force_all_xticks=True))
        ))
    except Exception:
        pass
    if items_preus:
        sections.append(("Preus", items_preus))

    # --------- SUPERFÍCIE ---------
    items_sup = []
    try:
        items_sup.append((
            "table",
            (f"Evolució trimestral de la superfície en m² construïts al municipi de {selected_mun}",
             table_trim(table_mun_sup, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_sup.append((
            "table",
            (f"Evolució anual de la superfície en m² construïts al municipi de {selected_mun}",
             table_year(table_mun_sup_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    sup_cols = [
        "Superfície mitjana total",
        "Superfície mitjana d'habitatge de segona mà",
        "Superfície mitjana d'habitatge nou"
    ]
    sup_palette = [GLOBAL_PALETTE["total"], GLOBAL_PALETTE["segunda_ma"], GLOBAL_PALETTE["nou"]]
    try:
        items_sup.append((
            "fig",
            (f"Evolució trimestral de la superfície en m² construïts al municipi de {selected_mun}",
             mpl_line(table_mun_sup, sup_cols, "", "m²",
                      start_year=SERIES_START_YEAR, palette=sup_palette))
        ))
    except Exception:
        pass
    try:
        items_sup.append((
            "fig",
            (f"Evolució anual de la superfície en m² construïts al municipi de {selected_mun}",
             mpl_bar(table_mun_sup_y, sup_cols, "", "m²",
                     start_year=SERIES_START_YEAR, palette=sup_palette, force_all_xticks=True))
        ))
    except Exception:
        pass
    if items_sup:
        sections.append(("Superfície", items_sup))

    # --------- LLOGUER ---------
    items_llog = []
    try:
        items_llog.append((
            "table",
            (f"Evolució trimestral del mercat de lloguer al municipi de {selected_mun}",
             table_trim(table_mun_llog, TABLE_TRIM_START_YEAR))
        ))
    except Exception:
        pass
    try:
        items_llog.append((
            "table",
            (f"Evolució anual del mercat de lloguer al municipi de {selected_mun}",
             table_year(table_mun_llog_y, TABLE_ANNUAL_START_YEAR, rounded=False))
        ))
    except Exception:
        pass
    # Doble eje (trimestral)
    if ("Nombre de contractes de lloguer" in getattr(table_mun_llog, "columns", [])) and \
       ("Rendes mitjanes de lloguer" in getattr(table_mun_llog, "columns", [])):
        try:
            items_llog.append((
                "fig",
                (f"Evolució del mercat de lloguer al municipi de {selected_mun}",
                 mpl_dual_line(table_mun_llog,
                               left_col="Nombre de contractes de lloguer",
                               right_col="Rendes mitjanes de lloguer",
                               left_label="Contractes", right_label="Renda mitjana",
                               left_ylab="Contractes", right_ylab="€ / mes",
                               start_year=SERIES_START_YEAR,
                               left_color=GLOBAL_PALETTE["total"],
                               right_color=GLOBAL_PALETTE["segunda_ma"],
                               force_all_xticks=False))
            ))
        except Exception:
            pass
    # Doble eje (anual)
    if ("Nombre de contractes de lloguer" in getattr(table_mun_llog_y, "columns", [])) and \
       ("Rendes mitjanes de lloguer" in getattr(table_mun_llog_y, "columns", [])):
        try:
            items_llog.append((
                "fig",
                (f"Evolució del mercat de lloguer al municipi de {selected_mun}",
                 mpl_dual_bar(table_mun_llog_y,
                              left_col="Nombre de contractes de lloguer",
                              right_col="Rendes mitjanes de lloguer",
                              left_label="Contractes", right_label="Renda mitjana",
                              left_ylab="Contractes", right_ylab="€ / mes",
                              start_year=SERIES_START_YEAR,
                              left_color=GLOBAL_PALETTE["total"],
                              right_color=GLOBAL_PALETTE["segunda_ma"],
                              force_all_xticks=True))
            ))
        except Exception:
            pass
    if items_llog:
        sections.append(("Lloguer", items_llog))

    # --------- DEMOGRAFIA: Població ---------
    items_demo_pop = []
    try:
        pop_col = f"poptottine_{selected_mun}"
        if DT_mun_y is not None and pop_col in DT_mun_y.columns:
            df_pop = DT_mun_y.loc[:, ["Fecha", pop_col]].dropna().copy()
            df_pop["Fecha"] = pd.to_numeric(df_pop["Fecha"], errors="coerce").astype("Int64")
            df_pop = df_pop.dropna(subset=["Fecha"]).copy()
            df_pop["Fecha"] = df_pop["Fecha"].astype(int)
            df_pop = df_pop.sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last")
            df_pop = df_pop.set_index("Fecha")
            df_pop[pop_col] = pd.to_numeric(df_pop[pop_col], errors="coerce")
            df_pop = df_pop[df_pop.index >= 2000]
            df_pop = df_pop.rename(columns={pop_col: "Població"})

            # KPI población
            try:
                last_year = int(df_pop.index[-1])
                last_val = int(df_pop["Població"].iloc[-1])
                prev_year = last_year - 5
                if prev_year in df_pop.index:
                    prev_val = float(df_pop.loc[prev_year, "Població"])
                    delta = f"{(100.0 * (last_val/prev_val - 1)):.1f}%"
                else:
                    delta = None
                    if len(df_pop) >= 6:
                        prev_val = float(df_pop["Població"].iloc[-6])
                        delta = f"{(100.0 * (last_val/prev_val - 1)):.1f}%"
                kpis_pdf.append(("Població (últim any)", f"{last_val:,.0f}".replace(",", "."), delta))
            except Exception:
                pass

            # Tabla (transpuesta)
            items_demo_pop.append((
                "table",
                (f"Evolució anual de la població al municipi de {selected_mun}", df_pop[df_pop.index>=2015].T)
            ))
            # Gráfico línea
            items_demo_pop.append((
                "fig",
                (f"",
                 mpl_line(df_pop, ["Població"], title="", ylab="Persones", xlab="Any",
                          start_year=2000, force_all_xticks=True))
            ))
    except Exception:
        pass
    if items_demo_pop:
        sections.append(("Demografia — Població", items_demo_pop))

    # --------- DEMOGRAFIA: Tamaño de llar (Censo 2021) ---------
    items_demo_llar = []
    try:
        if censo_2021 is not None:
            row = censo_2021[censo_2021["Municipi"] == selected_mun].iloc[0]
            labels = ["1", "2", "3", "4", "5 o más"]
            vals = [row.get("1", 0), row.get("2", 0), row.get("3", 0), row.get("4", 0), row.get("5 o más", 0)]
            items_demo_llar.append((
                "fig",
                (f"Distribució per grandària de llar al municipi de {selected_mun} (Cens 2021)",
                 mpl_donut(labels, vals))
            ))
            # KPIs adicionales
            try:
                kpis_pdf.append(("Grandària de la llar més freqüent", f"{row['Tamaño_hogar_frecuente']} llars", None))
                kpis_pdf.append(("Grandària mitjà de la llar", f"{float(row['Tamaño medio del hogar']):.2f}", None))
                kpis_pdf.append(("Població nacional", f"{(100.0 - float(row['Perc_extranjera'])):.1f}%", None))
                kpis_pdf.append(("Població estrangera", f"{float(row['Perc_extranjera']):.1f}%", None))
            except Exception:
                pass
    except Exception:
        pass
    if items_demo_llar:
        sections.append(("Demografia — Llar", items_demo_llar))

    # --------- ECONOMIA: Renda neta per llar ---------
    items_renda = []
    try:
        if rentaneta_mun is not None:
            df_rn = rentaneta_mun.rename(columns={"Año": "Any"}).copy()
            col_rn = f"rentanetahogar_{selected_mun}"
            if col_rn in df_rn.columns:
                df_rn = df_rn[["Any", col_rn]].dropna().rename(columns={col_rn: "Renda neta per llar"})
                df_rn = df_rn.set_index("Any")
                try:
                    any_rn = int(df_rn.index[-1])
                    val_rn = float(df_rn.iloc[-1, 0])
                    kpis_pdf.append((f"Renda neta per llar ({any_rn})", f"{val_rn:,.0f}".replace(",", "."), None))
                except Exception:
                    pass

                # Tabla (transpuesta)
                items_renda.append((
                    "table",
                    (f"Evolució anual de la renda mitjana neta per llar al municipi de {selected_mun}", df_rn.T)
                ))
                # Gráfico (barras con etiquetas, sin leyenda si 1 serie)
                items_renda.append((
                    "fig",
                    (f"",
                     mpl_bar(df_rn, ["Renda neta per llar"], title="", ylab="€ per llar",
                             start_year=max(SERIES_START_YEAR, 2015), force_all_xticks=True))
                ))
                
    except Exception:
        pass
    if items_renda:
        sections.append(("Economia — Renda", items_renda))

    # --------- OFERTA DE NOVA CONSTRUCCIÓ (APCE) ---------

    items_oferta = []  # <- important: sempre es reinicia

    if (
        tabla_estudi_oferta is not None
        and isinstance(tabla_estudi_oferta, (list, tuple))
        and len(tabla_estudi_oferta) >= 3
    ):
        try:
            oferta_tables = [
                (
                    f"Habitatges totals a l'estudi d'oferta de nova construcció APCE 2024 — {selected_mun}",
                    tabla_estudi_oferta[0].set_index("Variable")
                ),
                (
                    f"Habitatges unifamiliars a l'estudi d'oferta de nova construcció APCE 2024 — {selected_mun}",
                    tabla_estudi_oferta[1].set_index("Variable")
                ),
                (
                    f"Habitatges plurifamiliars a l'estudi d'oferta de nova construcció APCE 2024 — {selected_mun}",
                    tabla_estudi_oferta[2].set_index("Variable")
                ),
            ]

            for titulo_tab, df_tab in oferta_tables:
                if df_tab is not None and not df_tab.empty:
                    items_oferta.append(("table", (titulo_tab, df_tab)))

            if len(items_oferta) > 0:
                sections.append(("Oferta de nova construcció", items_oferta))

        except Exception:
            pass




    # ==========================
    # 3) Construir y descargar PDF
    # ==========================
    try:
        pdf_bytes = build_location_pdf_ordered(
            location_name=f"{selected_mun}",
            kpis=kpis_pdf,
            sections=sections
        )
        st.download_button(
            label=f"Descarregar informe de mercat — {selected_mun}",
            data=pdf_bytes,
            file_name=f"Informe_{selected_mun}_{datetime.now():%Y%m%d}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"No s'ha pogut generar el PDF per a {selected_mun}: {e}")


#Funciones parte de indicadores idescat


def detect_and_coerce_years(df):
    years = sorted({str(c) for c in df.columns if re.fullmatch(r"\d{4}", str(c))}, reverse=True)
    for y in years:
        df[y] = pd.to_numeric(df[y], errors="coerce")
    return years


def add_last_cols(df):
    present = [y for y in YEARS if y in df.columns]
    if not present:
        df["last_year"] = None
        df["last_value"] = np.nan
        return df
    arr = df[present].to_numpy(copy=False)
    mask = ~np.isnan(arr)
    idx = mask.argmax(1)
    vals = arr[np.arange(len(df)), idx]
    yrs  = np.array(present)[idx]
    none_mask = ~mask.any(1)
    vals[none_mask] = np.nan
    yrs[none_mask]  = None
    df["last_year"] = yrs
    df["last_value"] = vals
    return df

def fmt_int(x):  return "—" if pd.isna(x) else f"{int(round(float(x))):,}".replace(",",".")
def fmt_pct(x):  return "—" if pd.isna(x) else f"{float(x):.2f}".replace(".",",")+" %"

def get_year_val(df, vars_, year):
    if not year: return np.nan
    for v in vars_:
        row = df.loc[df["variable"]==v]
        if not row.empty and year in row.columns and pd.notnull(row.iloc[0][year]):
            return float(row.iloc[0][year])
    return np.nan

def latest_year_value(df, vars_):
    """(año, valor) más reciente con dato para vars_."""
    for y in YEARS:
        v = get_year_val(df, vars_, y)
        if pd.notnull(v): return y, v
    return None, np.nan

def prev_year_value(df, vars_, cur_year):
    """(año, valor) inmediatamente anterior con dato a cur_year para vars_."""
    if not cur_year or cur_year not in YEARS: return None, np.nan
    start = YEARS.index(cur_year) + 1
    for y in YEARS[start:]:
        v = get_year_val(df, vars_, y)
        if pd.notnull(v): return y, v
    return None, np.nan

def sum_age(year, groups):
    s=0.0; ok=False
    for cand_es,cand_cat in groups:
        v=get_year_val(df_pob_ine,[cand_es,cand_cat],year)
        if pd.notnull(v): s+=v; ok=True
    return s if ok else np.nan

def latest_year_sum_age(groups):
    """(año, suma) más reciente con dato para la suma de grupos."""
    for y in YEARS:
        s = sum_age(y, groups)
        if pd.notnull(s): return y, s
    return None, np.nan

path = ""

st.set_page_config(
    page_title="Conjuntura de sector",
    page_icon="""data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAA1VBMVEVHcEylpKR6eHaBgH9GREGenJxRT06op6evra2Qj49kYWCbmpqdnJyWlJS+vb1CPzyurKyHhYWMiYl7eXgOCgiPjY10cnJZV1WEgoKCgYB9fXt
    /fHyzsrGUk5OTkZGlo6ONioqko6OLioq7urqysbGdnJuurazCwcHLysp+fHx9fHuDgYGJh4Y4NTJcWVl9e3uqqalcWlgpJyacm5q7urrJyMizsrLS0tKIhoaMioqZmJiTkpKgn5+Bf36WlZWdnJuFg4O4t7e2tbXFxMR3dXTg39/T0dLqKxxpAAAAOHRSTlMA/WCvR6hq/
    v7+OD3U9/1Fpw+SlxynxXWZ8yLp+IDo2ufp9s3oUPII+jyiwdZ1vczEli7waWKEmIInp28AAADMSURBVBiVNczXcsIwEAVQyQZLMrYhQOjV1DRKAomKJRkZ+P9PYpCcfbgze+buAgDA5nf1zL8TcLNamssiPG/
    vt2XbwmA8Rykqton/XVZAbYKTSxzVyvVlPMc4no2KYhFaePvU8fDHmGT93i47Xh8ijPrB/0lTcA3lcGQO7otPmZJfgwhhoytPeKX5LqxOPA9i7oDlwYwJ3p0iYaEqWDdlRB2nkDjgJPA7nX0QaVq3kPGPZq/V6qUqt9BAmVaCUcqEdACzTBFCpcyvFfAAxgMYYVy1sTwAAAAASUVORK5CYII=""",
    layout="wide"
)
def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css_file(path + "main.css")


with open(path + "APCE_mod.png", "rb") as f:
    data_uri = base64.b64encode(f.read()).decode("utf-8")
markdown = f"""
<div class="image-apce-container">
<img src="data:image/png;base64, {data_uri}" alt="image" class="image-apce">
</div>
"""
st.markdown(markdown, unsafe_allow_html=True)


left_col, right_col, margin_right = st.columns((0.15, 1, 0.15))
with right_col:
    selected = option_menu(
        menu_title=None,  # required
        options=["Espanya","Catalunya","Províncies i àmbits", "Comarques", "Municipis", "Districtes de Barcelona", "Informe de mercat"],  # Dropdown menu
        icons=[None, None, "map", "map","house-fill", "house-fill",  "file-earmark-text"],  # Icons for dropdown menu
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0px important!", "background-color": "#fcefdc", "align":"justify", "overflow":"hidden"},
            "icon": {"color": "#bf6002", "font-size": "1.1em"},
            "nav-link": {
                "font-size": "1.1em",
                "text-align": "center",
                "font-weight": "bold",
                "color":"#363534",
                "padding": "5px",
                "--hover-color": "#fcefdc",
                "background-color": "#fcefdc",
                "overflow":"hidden"},
            "nav-link-selected": {"background-color": "#de7207"}
            })

#Trimestre lloguer. Única variable que introduce 0s en lugar de NaNs
max_trim_lloguer= "2025-07-01"
date_max_hipo_aux = "2025-10-01"
date_max_ciment_aux = "2025-11-01"
date_max_euribor = "2025-11-01"
date_max_ipc = "2025-11-01"
##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def import_data(trim_limit, month_limit):
    with open(path + 'DT_oferta_conjuntura.json', 'r') as outfile:
        list_estudi = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    with open('Idescat.json', 'r') as outfile:
        list_idescat_mun = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
        idescat_muns= list_idescat_mun[0].copy()
    with open('Censo2021.json', 'r') as outfile:
        list_censo = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    with open('Indicadors_mun.json', 'r', encoding="latin-1") as outfile:
        list_mun_idescat = json.load(outfile)
    df_mun_idescat = pd.DataFrame(list_mun_idescat["municipis"])
    df_pob_ine  = pd.DataFrame(list_mun_idescat.get("poblacio_edat_nacionalitat", []))
    censo_2021= list_censo[0].copy()
    censo_2021['Municipi'] = censo_2021['Municipi'].str.replace("L'", "l'", regex=False)
    rentaneta_mun= list_censo[1].copy()
    rentaneta_mun = rentaneta_mun.applymap(lambda x: x.replace(".", "") if isinstance(x, str) else x)
    rentaneta_mun = rentaneta_mun.apply(pd.to_numeric, errors='ignore')
    rentaneta_mun.columns = rentaneta_mun.columns.str.replace("L'", "l'", regex=False)
    censo_2021_dis= list_censo[2].copy()
    rentaneta_dis = list_censo[3].copy()
    rentaneta_dis = rentaneta_dis.applymap(lambda x: x.replace(".", "") if isinstance(x, str) else x)
    rentaneta_dis = rentaneta_dis.apply(pd.to_numeric, errors='ignore')
    with open('DT_simple.json', 'r') as outfile:
        list_of_df = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    DT_terr= list_of_df[0].copy()
    DT_mun= list_of_df[1].copy()
    DT_mun_aux= list_of_df[2].copy()
    DT_mun_aux2= list_of_df[3].copy()
    DT_mun_aux3= list_of_df[4].copy()
    DT_dis= list_of_df[5].copy()
    DT_terr_y= list_of_df[6].copy()
    DT_mun_y= list_of_df[7].copy()
    DT_mun_y_aux= list_of_df[8].copy()
    DT_mun_y_aux2= list_of_df[9].copy()
    DT_mun_y_aux3= list_of_df[10].copy()
    DT_dis_y= list_of_df[11].copy()
    DT_monthly= list_of_df[12].copy()
    DT_monthly["Fecha"] = DT_monthly["Fecha"].astype("datetime64[ns]")
    maestro_mun= list_of_df[13].copy()
    maestro_dis= list_of_df[14].copy()


    DT_monthly = DT_monthly[DT_monthly["Fecha"]<=month_limit]
    DT_terr = DT_terr[DT_terr["Fecha"]<=trim_limit]
    DT_mun = DT_mun[DT_mun["Fecha"]<=trim_limit]
    DT_mun_aux = DT_mun_aux[DT_mun_aux["Fecha"]<=trim_limit]
    DT_mun_aux2 = DT_mun_aux2[DT_mun_aux2["Fecha"]<=trim_limit]
    DT_mun_aux3 = DT_mun_aux3[DT_mun_aux3["Fecha"]<=trim_limit]
    DT_mun_pre = pd.merge(DT_mun, DT_mun_aux, how="left", on=["Trimestre","Fecha"])
    DT_mun_pre2 = pd.merge(DT_mun_pre, DT_mun_aux2, how="left", on=["Trimestre","Fecha"])
    DT_mun_def = pd.merge(DT_mun_pre2, DT_mun_aux3, how="left", on=["Trimestre","Fecha"])
    mun_list_aux = list(map(str, maestro_mun.loc[maestro_mun["ADD"] == "SI", "Municipi"].tolist()))
    mun_list = ["Trimestre", "Fecha"] + mun_list_aux
    muns_list = '|'.join(mun_list)
    DT_mun_def = DT_mun_def[[col for col in DT_mun_def.columns if any(mun in col for mun in mun_list)]]
    DT_dis = DT_dis[DT_dis["Fecha"]<=trim_limit]
    DT_mun_y_pre = pd.merge(DT_mun_y, DT_mun_y_aux, how="left", on="Fecha")
    DT_mun_y_pre2 = pd.merge(DT_mun_y_pre, DT_mun_y_aux2, how="left", on="Fecha")
    DT_mun_y_def = pd.merge(DT_mun_y_pre2, DT_mun_y_aux3, how="left", on="Fecha")    
    DT_mun_y_def = DT_mun_y_def[[col for col in DT_mun_y_def.columns if any(mun in col for mun in mun_list)]]

    return([DT_monthly, DT_terr, DT_terr_y, DT_mun_def, DT_mun_y_def, DT_dis, DT_dis_y, maestro_mun, maestro_dis, censo_2021, rentaneta_mun, censo_2021_dis, rentaneta_dis, idescat_muns, df_mun_idescat, df_pob_ine, list_estudi])

DT_monthly, DT_terr, DT_terr_y, DT_mun, DT_mun_y, DT_dis, DT_dis_y, maestro_mun, maestro_dis, censo_2021, rentaneta_mun, censo_2021_dis, rentaneta_dis, idescat_muns, df_mun_idescat, df_pob_ine, list_estudi = import_data("2025-10-01", "2025-11-01")


@st.cache_resource
def import_hist_mun(list_estudi):
    mun_2018_2019 = list_estudi[0].copy()
    mun_2020_2021 = list_estudi[1].copy()
    mun_2022 = list_estudi[2].copy()
    mun_2023 = list_estudi[3].copy()
    mun_2024 = list_estudi[4].copy()
    mun_2025 = list_estudi[5].copy()
    maestro_estudi = list_estudi[6].copy()

    mun_2019 = mun_2018_2019.iloc[:,14:27]
    mun_2020 = mun_2020_2021.iloc[:,:13]
    mun_2020 = mun_2020.dropna(how ='all',axis=0)
    mun_2021 = mun_2020_2021.iloc[:,14:27]
    mun_2021 = mun_2021.dropna(how ='all',axis=0)

    mun_2022 = mun_2022.iloc[:,14:27]
    mun_2022 = mun_2022.dropna(how ='all',axis=0)

    mun_2023 = mun_2023.iloc[:,14:27]
    mun_2023 = mun_2023.dropna(how ='all',axis=0)

    mun_2024 = mun_2024.iloc[:,14:27]
    mun_2024 = mun_2024.dropna(how ='all',axis=0)

    mun_2025 = mun_2025.iloc[:,14:27]
    mun_2025 = mun_2025.dropna(how ='all',axis=0)

    return([mun_2019, mun_2020, mun_2021, mun_2022, mun_2023, mun_2024, mun_2025, maestro_estudi])
mun_2019, mun_2020, mun_2021, mun_2022, mun_2023, mun_2024, mun_2025, maestro_estudi = import_hist_mun(list_estudi)
@st.cache_resource
def tidy_data(mun_year, year):
    df =mun_year.T
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,:].reset_index()
    df.columns.values[:3] = ['Any', 'Tipologia', "Variable"]
    df['Tipologia'] = df['Tipologia'].ffill()
    df['Any'] = year
    geo = df.columns[3:].values
    df_melted = pd.melt(df, id_vars=['Any', 'Tipologia', 'Variable'], value_vars=geo, value_name='Valor')
    df_melted.columns.values[3] = 'GEO'
    return(df_melted)

def table_mun_oferta(Municipi, any_ini, any_fin):
    df_vf_aux = pd.DataFrame()

    for df_frame, year in zip(["mun_2019", "mun_2020", "mun_2021", "mun_2022", "mun_2023", "mun_2024", "mun_2025"], [2019, 2020, 2021, 2022, 2023, 2024, 2025]):
        df_vf_aux = pd.concat([df_vf_aux, tidy_data(eval(df_frame), year)], axis=0)


    df_vf_aux['Variable']= np.where(df_vf_aux['Variable']=="Preu de     venda per      m² útil (€)", "Preu de venda per m² útil (€)", df_vf_aux['Variable'])
    df_vf_aux['Valor'] = pd.to_numeric(df_vf_aux['Valor'], errors='coerce')
    df_vf_aux['GEO'] = np.where(df_vf_aux['GEO']=="Municipis de Catalunya", "Catalunya", df_vf_aux['GEO'])
    df_vf_aux = df_vf_aux[~df_vf_aux['GEO'].str.contains("província|Província|Municipis")]

    df_vf_merged = pd.merge(df_vf_aux, maestro_estudi, how="left", on="GEO")
    df_vf_merged = df_vf_merged[~df_vf_merged["Província"].isna()].dropna(axis=1, how="all")
    df_vf = df_vf_merged[df_vf_merged["Variable"]!="Unitats"]
    df_unitats = df_vf_merged[df_vf_merged["Variable"]=="Unitats"].drop("Variable", axis=1)
    df_unitats = df_unitats.rename(columns={"Valor": "Unitats"})
    df_final_cat = pd.merge(df_vf, df_unitats, how="left")
    df_final = df_final_cat[df_final_cat["GEO"]!="Catalunya"]
    df_mun_filtered = df_final[(df_final["GEO"]==Municipi) & (df_final["Any"]>=any_ini) & (df_final["Any"]<=any_fin)].drop(["Àmbits territorials","Corones","Comarques","Província", "codiine"], axis=1).pivot(index=["Any"], columns=["Tipologia", "Variable"], values="Valor")
    df_mun_unitats = df_final[(df_final["GEO"]==Municipi) & (df_final["Any"]>=any_ini) & (df_final["Any"]<=any_fin)].drop(["Àmbits territorials","Corones","Comarques","Província", "codiine"], axis=1).drop_duplicates(["Any","Tipologia","Unitats"]).pivot(index=["Any"], columns=["Tipologia"], values="Unitats")
    df_mun_unitats.columns= [("HABITATGES PLURIFAMILIARS", "Unitats"), ("HABITATGES UNIFAMILIARS", "Unitats"), ("TOTAL HABITATGES", "Unitats")]
    df_mun_n = pd.concat([df_mun_filtered, df_mun_unitats], axis=1)
    # df_mun_n[("HABITATGES PLURIFAMILIARS", "Unitats %")] = (df_mun_n[("HABITATGES PLURIFAMILIARS", "Unitats")]/df_mun_n[("TOTAL HABITATGES", "Unitats")])*100
    # df_mun_n[("HABITATGES UNIFAMILIARS", "Unitats %")] = (df_mun_n[("HABITATGES UNIFAMILIARS", "Unitats")] /df_mun_n[("TOTAL HABITATGES", "Unitats")])*100
    df_mun_n = df_mun_n.sort_index(axis=1, level=[0,1])
    num_cols = df_mun_n.select_dtypes(include=['float64', 'Int64']).columns
    df_mun_n[num_cols] = df_mun_n[num_cols].round(0)
    df_mun_n[num_cols] = df_mun_n[num_cols].astype("Int64")
    num_cols = df_mun_n.select_dtypes(include=['float64', 'Int64']).columns
    df_mun_n[num_cols] = df_mun_n[num_cols].map(lambda x: '{:,.0f}'.format(x).replace(',', '#').replace('.', ',').replace('#', '.'))
    return(df_mun_n)

def table_mun_oferta_aux(Municipi, any_ini):
    df_vf_aux = pd.DataFrame()

    for df_frame, year in zip(["mun_2019", "mun_2020", "mun_2021", "mun_2022", "mun_2023", "mun_2024", "mun_2025"], [2019, 2020, 2021, 2022, 2023, 2024, 2025]):
        df_vf_aux = pd.concat([df_vf_aux, tidy_data(eval(df_frame), year)], axis=0)


    df_vf_aux['Variable']= np.where(df_vf_aux['Variable']=="Preu de     venda per      m² útil (€)", "Preu de venda per m² útil (€)", df_vf_aux['Variable'])
    df_vf_aux['Valor'] = pd.to_numeric(df_vf_aux['Valor'], errors='coerce')
    df_vf_aux['GEO'] = np.where(df_vf_aux['GEO']=="Municipis de Catalunya", "Catalunya", df_vf_aux['GEO'])
    df_vf_aux = df_vf_aux[~df_vf_aux['GEO'].str.contains("província|Província|Municipis")]

    df_vf_merged = pd.merge(df_vf_aux, maestro_estudi, how="left", on="GEO")
    df_vf_merged = df_vf_merged[~df_vf_merged["Província"].isna()].dropna(axis=1, how="all")
    df_vf_merged = df_vf_merged[(df_vf_merged["GEO"]==Municipi) & (df_vf_merged["Any"]==any_ini)].drop(["GEO","Àmbits territorials","Corones","Comarques","Província", "codiine", "Any"], axis=1)
    num_cols = df_vf_merged.select_dtypes(include=['float64', 'Int64']).columns
    df_vf_merged[num_cols] = df_vf_merged[num_cols].round(0)
    df_vf_merged_total = df_vf_merged[df_vf_merged["Tipologia"]=="TOTAL HABITATGES"].drop("Tipologia", axis=1)
    df_vf_merged_uni = df_vf_merged[df_vf_merged["Tipologia"]=="HABITATGES UNIFAMILIARS"].drop("Tipologia", axis=1)
    df_vf_merged_pluri = df_vf_merged[df_vf_merged["Tipologia"]=="HABITATGES PLURIFAMILIARS"].drop("Tipologia", axis=1)
    return([df_vf_merged_total, df_vf_merged_uni, df_vf_merged_pluri])


##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_m(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Fecha"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Fecha"] + columns_output
    output_data["Month"] = output_data['Fecha'].dt.month
    output_data = output_data.dropna()
    output_data = output_data[(output_data["Month"]<=output_data['Month'].iloc[-1])]
    return(output_data.drop(["Data", "Month"], axis=1))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Trimestre"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Trimestre"] + columns_output

    return(output_data.set_index("Trimestre").drop("Data", axis=1))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_anual(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = columns_output
    output_data["Any"] = output_data["Any"].astype(str)
    return(output_data.set_index("Any"))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_mensual(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Fecha"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Fecha"] + columns_output
    output_data["Fecha"] = output_data["Fecha"].astype(str)
    return(output_data.set_index("Fecha"))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present(data_ori, columns_sel, year):
    output_data = data_ori[data_ori[columns_sel]!=0][["Trimestre"] + [columns_sel]].dropna()
    output_data["Trimestre_aux"] = output_data["Trimestre"].str[-1]
    output_data = output_data[(output_data["Trimestre_aux"]<=output_data['Trimestre_aux'].iloc[-1])]
    output_data["Any"] = output_data["Trimestre"].str[0:4]
    output_data = output_data.drop(["Trimestre", "Trimestre_aux"], axis=1)
    output_data = output_data.groupby("Any").mean().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==str(year)]
    output_data = output_data.set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + [columns_sel]]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha", columns_sel])
    output_data = output_data.set_index("Fecha").groupby("Any").sum().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly_aux(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + columns_sel].dropna(axis=0)
    output_data["month_aux"] = output_data["Fecha"].dt.month
    output_data = output_data[(output_data["month_aux"]<=output_data['month_aux'].iloc[-1])]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha"] + columns_sel)
    output_data = output_data.set_index("Fecha").groupby("Any").sum().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly_diff(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + columns_sel].dropna(axis=0)
    output_data["month_aux"] = output_data["Fecha"].dt.month
    output_data = output_data[(output_data["month_aux"]<=output_data['month_aux'].iloc[-1])]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha"] + columns_sel)
    output_data = output_data.set_index("Fecha").groupby("Any").mean().diff().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def indicator_year(df, df_aux, year, variable, tipus, frequency=None):
    if (year==str(datetime.now().year) and (frequency=="month") and ((tipus=="var") or (tipus=="diff"))):
        return(round(tidy_present_monthly(df_aux, variable, year),2))
    if (year==str(datetime.now().year) and (frequency=="month_aux") and (tipus=="var")):
        return(round(tidy_present_monthly_aux(df_aux, variable, year),2))
    if (year==str(datetime.now().year) and (frequency=="month_aux") and ((tipus=="diff"))):
        return(round(tidy_present_monthly_diff(df_aux, variable, year),2))
    if (year==str(datetime.now().year) and ((tipus=="var") or (tipus=="diff"))):
        return(round(tidy_present(df_aux.reset_index(), variable, year),2))
    if tipus=="level":
        df = df[df.index==year][variable]
        return(round(df.values[0],2))
    if tipus=="var":
        df = df[variable].pct_change().mul(100)
        df = df[df.index==year]
        return(round(df.values[0],2))
    if tipus=="diff":
        df = df[variable].diff().mul(100)
        df = df[df.index==year]
        return(round(df.values[0],2))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def concatenate_lists(list1, list2):
    result_list = []
    for i in list1:
        result_element = i+ list2
        result_list.append(result_element)
    return(result_list)


def filedownload(df, filename):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=True, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode("latin-1")
    href = f"""<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">
    <button class="download-button">Descarregar</button></a>"""
    return href

#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)

@st.cache_resource
def line_plotly_pob(df, col, title_main, title_y, title_x="Any"):
    fig = px.line(
        df,
        x="Fecha",
        y=col,
        title=title_main,
        labels={"Fecha": title_x, col: title_y},
        color_discrete_sequence=['#2d538f'],
        markers=True
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#fcefdc",
        plot_bgcolor="#fcefdc",
        title=dict(font=dict(size=13))
    )
    fig.update_yaxes(tickformat=",d")
    return fig


@st.cache_resource
def line_plotly(table_n, selection_n, title_main, title_y, title_x="Trimestre", replace_0=False):
    plot_cat = table_n[selection_n]
    if replace_0==True:
        plot_cat = plot_cat.replace(0, np.nan)
    colors = ['#2d538f', '#de7207', '#385723', "#727375"]
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Scatter(
            x=plot_cat.index,
            y=plot_cat[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title=title_x),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = ['#2d538f', '#de7207', '#385723']
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            marker=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title="Any"),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig
#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def stacked_bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = ['#2d538f', '#de7207', '#385723']
    
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            marker=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title="Any"),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        barmode='stack',
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig
#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def area_plotly(table_n, selection_n, title_main, title_y, trim):
    plot_cat = table_n[table_n.index>=trim][selection_n]
    fig = px.area(plot_cat, x=plot_cat.index, y=plot_cat.columns, title=title_main)
    fig.for_each_trace(lambda trace: trace.update(fillcolor = trace.line.color))
    fig.update_layout(xaxis_title="Trimestre", yaxis=dict(title=title_y, tickformat=",d"), barmode='stack')
    fig.update_traces(opacity=0.4)  # Change opacity to 0.8
    fig.update_layout(legend_title_text="")
    fig.update_layout(
        title=dict(text=title_main, font=dict(size=13), y=0.97),
        legend=dict(x=-0.08, y=1.25, orientation="h"),  # Adjust the x and y values for the legend position
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    return fig

@st.cache_resource
def bar_plotly_demografia(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7"]
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col]/1000,
            name=col,
            text=plot_cat[col], 
            textfont=dict(color="white"),
            marker=dict(color=colors[i % len(colors)]),
        )
        traces.append(trace)
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title="Año"),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

@st.cache_resource
def donut_plotly_demografia(table_n, selection_n, title_main, title_y):
    plot_cat = table_n[selection_n]
    plot_cat = plot_cat.set_index("Tamany").sort_index()
    colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7", "#FFA07A"]
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Pie(
            labels=plot_cat.index,
            values=plot_cat[col],
            name=col,
            hole=0.5,
            marker=dict(colors=colors)
        )
        traces.append(trace)
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        yaxis=dict(title=title_y),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def table_monthly(data_ori, year_ini, rounded=True):
    data_ori = data_ori.reset_index()
    month_mapping_catalan = {
        1: 'Gener',
        2: 'Febrer',
        3: 'Març',
        4: 'Abril',
        5: 'Maig',
        6: 'Juny',
        7: 'Juliol',
        8: 'Agost',
        9: 'Setembre',
        10: 'Octubre',
        11: 'Novembre',
        12: 'Desembre'
    }

    try:
        output_data = data_ori[data_ori["Data"]>=pd.to_datetime(str(year_ini)+"/01/01", format="%Y/%m/%d")]
        output_data['Mes'] = output_data['Data'].dt.month.map(month_mapping_catalan)
        if rounded==True:
            numeric_columns = output_data.select_dtypes(include=['float64', 'int64']).columns
            output_data[numeric_columns] = output_data[numeric_columns].applymap(lambda x: round(x, 1))
        output_data = output_data.drop(["Fecha", "Data"], axis=1).set_index("Mes").reset_index().T
        output_data.columns = output_data.iloc[0,:]
        output_data = output_data.iloc[1:,:]
    except KeyError:
        output_data = data_ori[data_ori["Fecha"]>=pd.to_datetime(str(year_ini)+"/01/01", format="%Y/%m/%d")]
        output_data['Mes'] = output_data['Fecha'].dt.month.map(month_mapping_catalan)
        if rounded==True:
            numeric_columns = output_data.select_dtypes(include=['float64', 'int64']).columns
            output_data[numeric_columns] = output_data[numeric_columns].applymap(lambda x: round(x, 1))
        output_data = output_data.drop(["Fecha", "index"], axis=1).set_index("Mes").reset_index().T
        output_data.columns = output_data.iloc[0,:]
        output_data = output_data.iloc[1:,:]
    return(output_data)

def format_dataframes(df, style_n):
    if style_n==True:
        return(df.style.format("{:,.0f}"))
    else:
        return(df.style.format("{:,.1f}"))



def table_trim(data_ori, year_ini, rounded=False, formated=True):
    data_ori = data_ori.reset_index()
    data_ori["Any"] = data_ori["Trimestre"].str.split("T").str[0]
    data_ori["Trimestre"] = data_ori["Trimestre"].str.split("T").str[1]
    data_ori["Trimestre"] = data_ori["Trimestre"] + "T"
    data_ori = data_ori[data_ori["Any"]>=str(year_ini)]
    data_ori = data_ori.replace(0, np.nan)
    if rounded==True:
        numeric_columns = data_ori.select_dtypes(include=['float64', 'int64']).columns
        data_ori[numeric_columns] = data_ori[numeric_columns].applymap(lambda x: round(x, 1))
    output_data = data_ori.set_index(["Any", "Trimestre"]).T#.dropna(axis=1, how="all")
    last_column_contains_all_nans = output_data.iloc[:, -1].isna().all()
    if last_column_contains_all_nans:
        output_data = output_data.iloc[:, :-1]
    else:
        output_data = output_data.copy()
    
    if formated==True:   
        return(format_dataframes(output_data, True))
    else:
        return(format_dataframes(output_data, False))


def table_year(data_ori, year_ini, rounded=False, formated=True):
    data_ori = data_ori.reset_index()
    if rounded==True:
        numeric_columns = data_ori.select_dtypes(include=['float64', 'int64']).columns
        data_ori[numeric_columns] = data_ori[numeric_columns].applymap(lambda x: round(x, 1))
    data_output = data_ori[data_ori["Any"]>=str(year_ini)].T
    data_output.columns = data_output.iloc[0,:]
    data_output = data_output.iloc[1:,:]
    if formated==True:   
        return(format_dataframes(data_output, True))
    else:
        return(format_dataframes(data_output, False))
    
#Defining years
max_year= 2026
available_years = list(range(2018,max_year))
index_year = 2025

###################################################################### SCRIPT PESTAÑAS ##########################################################################
if selected == "Espanya":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Selecciona un tipus d'indicador**", ("Sector residencial","Indicadors econòmics"), horizontal=True)
    with center:
        if selected_type=="Indicadors econòmics":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Índex de Preus al Consum (IPC)", "Consum de ciment","Tipus d'interès", "Hipoteques"], key=101)
        if selected_type=="Sector residencial":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus"], key=201)
    with right:
        selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=102)

    if selected_type=="Indicadors econòmics":
        if selected_index=="Índex de Preus al Consum (IPC)":
            st.subheader("ÍNDEX DE PREUS AL CONSUM (IPC)")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2002
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "IPC_Nacional_x", "IPC_subyacente", "IGC_Nacional"], f"{str(min_year)}-01-01", date_max_ipc,["Data","IPC (Base 2021)","IPC subjacent", "IGC"])

            table_espanya_m["Inflació"] = table_espanya_m["IPC (Base 2021)"].pct_change(12).mul(100)
            table_espanya_m["Inflació subjacent"] = round(table_espanya_m["IPC subjacent"],1)
            table_espanya_m["Índex de Garantia de Competitivitat (IGC)"] = round(table_espanya_m["IGC"],1)
            table_espanya_m = table_espanya_m.drop(["IPC subjacent", "IGC"], axis=1)
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","IPC_Nacional_x", "IPC_subyacente", "IGC_Nacional"], min_year, max_year,["Any", "IPC (Base 2021)","IPC subjacent", "IGC"])
            table_espanya_y["Inflació"] = table_espanya_y["IPC (Base 2021)"].pct_change(1).mul(100)
            table_espanya_y["Inflació subjacent"] = round(table_espanya_y["IPC subjacent"],1)
            table_espanya_y["Índex de Garantia de Competitivitat (IGC)"] = round(table_espanya_y["IGC"],1)
            table_espanya_y = table_espanya_y.drop(["IPC subjacent", "IGC"], axis=1)

            if selected_year_n==max_year:
                left, center, right= st.columns((1,1,1))
                with left:
                    st.metric(label="**Inflació** (var. anual)", value=f"""{round(table_espanya_m["Inflació"][-1],1)}%""")
                with center:
                    st.metric(label="**Inflació subjacent** (var. anual)", value=f"""{round(table_espanya_m["Inflació subjacent"][-1],1)}%""")
                with right:
                    st.metric(label="**Índex de Garantia de Competitivitat** (var. anual)", value=f"""{round(table_espanya_m["Índex de Garantia de Competitivitat (IGC)"][-1],1)}%""")
            if selected_year_n!=max_year:
                left, center, right= st.columns((1,1,1))
                with left:
                    st.metric(label="**Inflació** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Inflació"].values[0], 1)}%""")
                with center:
                    st.metric(label="**Inflació subjacent** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Inflació subjacent"].values[0], 1)}%""")
                with right:
                    st.metric(label="**Índex de Garantia de Competitivitat** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Índex de Garantia de Competitivitat (IGC)"].values[0], 1)}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_monthly(table_espanya_m[(table_espanya_m["Data"]>=f"{str(selected_year_n)}-01-01") & (table_espanya_m["Data"]<f"{str(selected_year_n+1)}-01-01")], selected_year_n).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_monthly(table_espanya_m, 2023), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2008, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            st.plotly_chart(line_plotly(table_espanya_m[table_espanya_m.index>="2015-01-01"], ["Inflació", "Inflació subjacent", "Índex de Garantia de Competitivitat (IGC)"], "Evolució mensual de la inflació (variació anual del IPC) i l'IGC (Índex de Garantia de Competitivitat)", "%",  "Any"), use_container_width=True, responsive=True)
        if selected_index=="Consum de ciment":
            st.subheader("CONSUM DE CIMENT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            table_espanya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["cons_ciment_Espanya"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Consum de ciment"])
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha","cons_ciment_Espanya"],  f"{str(min_year)}-01-01", f"{date_max_ciment_aux}",["Data", "Consum de ciment"])
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","cons_ciment_Espanya"], min_year, max_year,["Any", "Consum de ciment"])
            table_espanya_q = table_espanya_q.dropna(axis=0).div(1000)
            table_espanya_y = table_espanya_y.dropna(axis=0).div(1000)
            st.metric(label="**Consum de ciment** (Milers de tones)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Consum de ciment", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Consum de ciment", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_espanya_q, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_espanya_q, 2012), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2008, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_espanya_q, ["Consum de ciment"], "Consum de ciment (Milers T.)", "Milers de T."), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_espanya_y.pct_change(1).mul(100).dropna(axis=0), ["Consum de ciment"], "Variació anual del consum de ciment (%)", "%", 2012), use_container_width=True, responsive=True)     
        if selected_index=="Tipus d'interès":
            min_year=2008
            st.subheader("TIPUS D'INTERÈS I POLÍTICA MONETÀRIA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "Euribor_1m", "Euribor_3m",	"Euribor_6m", "Euribor_1y", "tipo_hipo"], f"{str(min_year)}-01-01", date_max_euribor,["Data","Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_m = table_espanya_m[["Data","Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]].reset_index(drop=True).rename(columns={"Data":"Fecha"})
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha", "Euribor_1m", "Euribor_3m","Euribor_6m", "Euribor_1y", "tipo_hipo"],  f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_q = table_espanya_q[["Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"]]
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "Euribor_1m", "Euribor_3m","Euribor_6m", "Euribor_1y", "tipo_hipo"], min_year, max_year,["Any", "Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_y = table_espanya_y[["Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]]

            if selected_year_n==max_year:
                left, left_center, right_center, right = st.columns((1,1,1,1))
                with left:
                    st.metric(label="**Euríbor a 3 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 3 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 3 mesos"], "diff", "month_aux")} p.b.""")
                with left_center:
                    st.metric(label="**Euríbor a 6 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 6 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 6 mesos"], "diff", "month_aux")} p.b.""")
                with right_center:
                    st.metric(label="**Euríbor a 1 any** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 1 any", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 1 any"], "diff", "month_aux")} p.b.""")
                with right:
                    st.metric(label="**Tipus d'interès d'hipoteques** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Tipus d'interès d'hipoteques", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Tipus d'interès d'hipoteques"], "diff", "month_aux")} p.b.""")
            if selected_year_n!=max_year:
                left, left_center, right_center, right = st.columns((1,1,1,1))
                with left:
                    st.metric(label="**Euríbor a 3 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 3 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 3 mesos", "diff", "month")} p.b.""")
                with left_center:
                    st.metric(label="**Euríbor a 6 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 6 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 6 mesos", "diff", "month")} p.b.""")
                with right_center:
                    st.metric(label="**Euríbor a 1 any** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 1 any", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 1 any", "diff", "month")} p.b.""")
                with right:
                    st.metric(label="**Tipus d'interès d'hipoteques** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Tipus d'interès d'hipoteques", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Tipus d'interès d'hipoteques", "diff", "month")} p.b.""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_monthly(table_espanya_m[(table_espanya_m["Fecha"]>=f"{str(selected_year_n)}-01-01") & (table_espanya_m["Fecha"]<f"{str(selected_year_n+1)}-01-01")], selected_year_n).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_monthly(table_espanya_m, 2024), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2014, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            selected_columns = ["Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_espanya_m.set_index("Fecha"), selected_columns, "Evolució mensual dels tipus d'interès (%)", "Tipus d'interès (%)",  "Fecha"), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_espanya_y, ["Euríbor a 1 any", "Tipus d'interès d'hipoteques"], "Evolució anual dels tipus d'interès (%)", "Tipus d'interès (%)",  2005), use_container_width=True, responsive=True)
        if selected_index=="Hipoteques":
            st.subheader("IMPORT I NOMBRE D'HIPOTEQUES INSCRITES EN ELS REGISTRES DE PROPIETAT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "hipon_Nacional", "hipoimp_Nacional"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data","Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_m = table_espanya_m[["Data", "Nombre d'hipoteques", "Import d'hipoteques"]].rename(columns={"Data":"Fecha"})
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha", "hipon_Nacional", "hipoimp_Nacional"],  f"{str(min_year)}-01-01", f"{date_max_hipo_aux}",["Data", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_q = table_espanya_q[["Nombre d'hipoteques", "Import d'hipoteques"]]
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","hipon_Nacional", "hipoimp_Nacional"], min_year, max_year,["Any", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_y = table_espanya_y[["Nombre d'hipoteques", "Import d'hipoteques"]]
            if selected_year_n==max_year-1:
                left, right = st.columns((1,1))
                with left:
                    st.metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Nombre d'hipoteques"], "var", "month_aux")}%""")
                with right:
                    st.metric(label="**Import d'hipoteques** (Milers d'euros)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Import d'hipoteques"], "var", "month_aux")}%""")
            if selected_year_n!=max_year-1:
                left, right = st.columns((1,1))
                with left:
                    st.metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Nombre d'hipoteques", "var")}%""")
                with right:
                    st.metric(label="**Import d'hipoteques** (Milers d'euros)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Import d'hipoteques", "var")}%""")

            selected_columns = ["Nombre d'hipoteques", "Import d'hipoteques"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_espanya_q, 2022).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_espanya_q, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2009, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, rounded=False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_espanya_m, ["Nombre d'hipoteques"], "Evolució mensual del nombre d'hipoteques", "Nombre d'hipoteques",  "Data"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_espanya_m, ["Import d'hipoteques"], "Evolució mensual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  "Data"), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_espanya_y, ["Nombre d'hipoteques"], "Evolució anual del nombre d'hipoteques", "Nombre d'hipoteques",  2005), use_container_width=True, responsive=True)
                st.plotly_chart(bar_plotly(table_espanya_y, ["Import d'hipoteques"], "Evolució anual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  2005), use_container_width=True, responsive=True)

    if selected_type=="Sector residencial":
        if selected_index=="Producció":
            min_year=2008
            st.subheader("PRODUCCIÓ D'HABITATGES A ESPANYA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_esp_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional"), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])                                                                                                                                                                                                                                                                                                                     
            table_esp = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Espanya"), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats", 
                                                                                                                                                                                                                                                                                            "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                            "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
            table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional")+ concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Espanya"), min_year, max_year,["Any", "Habitatges iniciats", "Habitatges acabats", "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
            left, right = st.columns((1,1))
            with left:
                st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
            with right:
                st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Habitatges acabats", "var","month")}%""")

            left, right = st.columns((1,1))    
            with left:
                try:
                    st.metric(label="**Qualificacions provisionals d'HPO**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO", "var")}%""")
                except IndexError:
                    st.metric(label="**Qualificacions provisionals d'HPO**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Qualificacions definitives d'HPO**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO", "var")}%""")
                except IndexError:
                    st.metric(label="**Qualificacions definitives d'HPO**", value="No disponible")

            left, right = st.columns((1,1))
            with left:
                try:
                    st.metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "var")}%""")
                except IndexError:
                    st.metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value="No disponible")

            with right:
                try:
                    st.metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "var")}%""")
                except IndexError:
                    st.metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value="No disponible")
            left, right = st.columns((1,1))
            with left:
                try:
                    st.metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "var")}%""")
                except IndexError:
                    st.metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value="No disponible")
            with right:
                try:
                    st.metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "var")}%""")
                except IndexError:
                    st.metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value="No disponible")

            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            selected_columns_aux1 = ["Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)"]
            selected_columns_aux2 = ["Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_esp, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_esp, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_esp_y, 2014).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_esp_y, 2008), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_esp, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                st.plotly_chart(stacked_bar_plotly(table_esp_y, selected_columns_aux1, "Qualificacions provisionals de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_esp_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(stacked_bar_plotly(table_esp_y, selected_columns_aux2, "Qualificacions definitives de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2008
            st.subheader("COMPRAVENDES D'HABITATGES A ESPANYA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_esp_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["trvivses", "trvivnes"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp_m["Compravendes d'habitatge total"] = table_esp_m["Compravendes d'habitatge de segona mà"] + table_esp_m["Compravendes d'habitatge nou"]
            table_esp = tidy_Catalunya(DT_terr, ["Fecha", "trvivses", "trvivnes"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp["Compravendes d'habitatge total"] = table_esp["Compravendes d'habitatge de segona mà"] + table_esp["Compravendes d'habitatge nou"]
            table_esp = table_esp[["Compravendes d'habitatge total","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]]
            table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivses", "trvivnes"], min_year, max_year,["Any", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp_y["Compravendes d'habitatge total"] = table_esp_y["Compravendes d'habitatge de segona mà"] + table_esp_y["Compravendes d'habitatge nou"]
            table_esp_y = table_esp_y[["Compravendes d'habitatge total","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]]

            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge total", "var", "month")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
            with center:
                try:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var", "month")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge nou", "var", "month")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge nou**", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_esp, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_esp, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_esp_y, 2014).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_esp_y, 2008), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_esp[table_esp.notna()], table_esp.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia d'habitatge", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(stacked_bar_plotly(table_esp_y[table_esp_y.notna()], table_esp.columns.tolist()[1:3], "Evolució anual de les compravendes d'habitatge per tipologia d'habitatge", "Nombre de compravendes", 2008), use_container_width=True, responsive=True)
        if selected_index=="Preus":
                min_year=2008
                st.subheader("VALOR TASAT MITJÀ D'HABITATGE LLIURE €/M\u00b2 (MITMA)")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_esp = tidy_Catalunya(DT_terr, ["Fecha", "prvivlfom_Nacional", "prvivlnfom_Nacional"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu de l'habitatge lliure", "Preu de l'habitatge lliure nou"])
                table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "prvivlfom_Nacional", "prvivlnfom_Nacional"], min_year, max_year,["Any", "Preu de l'habitatge lliure", "Preu de l'habitatge lliure nou"])
                left, right = st.columns((1,1))
                with left:
                    try:
                        st.metric(label=f"""**Preu de l'habitatge lliure** (€/m\u00b2)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu de l'habitatge lliure", "level"):,.0f}""")
                    except IndexError:
                        st.metric(label="**Preu de l'habitatge lliure** (€/m\u00b2)", value="No disponible")
                with right:
                    try:
                        st.metric(label=f"""**Preu de l'habitatge lliure nou** (€/m\u00b2)""", value=f"""{round(indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu de l'habitatge lliure nou", "level"),1):,.0f}""")
                    except IndexError:
                        st.metric(label="**Preu de l'habitatge lliure nou** (€/m\u00b2)", value="No disponible")

                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_esp, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_esp, 2008, True, False), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_esp_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_esp_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_esp, table_esp.columns.tolist(), "Preus per m\u00b2 de tasació per tipologia d'habitatge", "€/m\u00b2"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_esp_y, table_esp.columns.tolist(), "Preus per m\u00b2 de tasació per tipologia d'habitatge", "€/m\u00b2", 2010), use_container_width=True, responsive=True)
                st.subheader("VARIACIONS ANUALS DE L'ÍNDEX DEL PREU DE L'HABITATGE (INE)")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_esp = tidy_Catalunya(DT_terr, ["Fecha", "ipves", "ipvses", "ipvnes"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "ipves", "ipvses", "ipvnes"], min_year, max_year,["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label=f"""**Preu d'habitatge total** (var. anual)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu d'habitatge total", "level")} %""")
                    except IndexError:
                        st.metric(label="**Preu d'habitatge total** (var. anual)", value="No disponible")
                with center:
                    try:
                        st.metric(label=f"""**Preu d'habitatge de segona mà** (var. anual)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preus d'habitatge de segona mà", "level")} %""")
                    except IndexError:
                        st.metric(label="**Preu d'habitatge de segona mà** (var. anual)", value="No disponible")
                with right:
                    try:
                        st.metric(label=f"""**Preu d'habitatge nou** (var. anual)""", value=f"""{round(indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preus d'habitatge nou", "level"),1)} %""")
                    except IndexError:
                        st.metric(label="**Preu d'habitatge nou** (var. anual)", value="No disponible")

                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_esp, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_esp, 2008, True, False), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_esp_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_esp_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_esp, table_esp.columns.tolist(), "Índex trimestral de preus per tipologia d'habitatge (variació anual %)", "%"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_esp_y, table_esp.columns.tolist(), "Índex anual de preus per tipologia d'habitatge (variació anual %)", "%", 2007), use_container_width=True, responsive=True)

if selected == "Catalunya":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_indicator = st.radio("**Selecciona un tipus d'indicador**", ("Sector residencial","Indicadors econòmics"), horizontal=True, key=301)
        if selected_indicator=="Sector residencial":
            selected_type = st.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"), horizontal=True)
    with center:
        if (selected_indicator=="Indicadors econòmics"):
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Costos de construcció", "Mercat laboral", "Consum de Ciment", "Hipoteques"], key=302)
        if ((selected_indicator=="Sector residencial")):
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=303)
        # if (selected_type=="Lloguer") and (selected_indicator=="Sector residencial"):
        #     st.write("")
        
    with right:
        selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=305)

    if selected_indicator=="Indicadors econòmics":
        if selected_index=="Mercat laboral":
            st.subheader("MERCAT LABORAL DEL SECTOR DE LA CONSTRUCCIÓ")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            table_catalunya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["ssunempcons_Catalunya", "aficons_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "emptot_Catalunya", "empcons_Catalunya", "ssunempcons_Catalunya", "aficons_Catalunya"],  f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Total població ocupada", "Ocupació del sector de la construcció","Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","emptot_Catalunya", "empcons_Catalunya", "ssunempcons_Catalunya", "aficons_Catalunya"], min_year, max_year,["Any", "Total població ocupada", "Ocupació del sector de la construcció","Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_q = table_catalunya_q.dropna(axis=0)
            table_catalunya_y = table_catalunya_y.dropna(axis=0)
            left, right = st.columns((1,1))
            with left:
                st.metric(label="**Total població ocupada** (Milers)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Total població ocupada", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Total població ocupada", "var")}%""")
                st.metric(label="**Atur registrat del sector de la construcció**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Atur registrat del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Atur registrat del sector de la construcció", "var", "month")}%""")
            with right:
                st.metric(label="**Ocupació del sector de la construcció** (Milers)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Ocupació del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Ocupació del sector de la construcció", "var")}%""")
                st.metric(label="**Afiliats del sector de la construcció**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Afiliats del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Afiliats del sector de la construcció", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2021, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2012, rounded=True), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2008, rounded=True), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

            
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(stacked_bar_plotly(table_catalunya_y, ["Total població ocupada", "Ocupació del sector de la construcció"], "Ocupats totals i del sector de la construcció (milers)", "Milers de persones", 2014), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_catalunya_y, ["Afiliats del sector de la construcció", "Atur registrat del sector de la construcció"], "Afiliats i aturats del sector de la construcció", "Persones", 2014), use_container_width=True, responsive=True)

        if selected_index=="Costos de construcció":
            st.subheader("COSTOS DE CONSTRUCCIÓ PER TIPOLOGIA EDIFICATÒRIA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2013
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "Costos_edificimitjaneres", "Costos_Unifamiliar2plantes", "Costos_nauind", "Costos_edificioficines"],  f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","Costos_edificimitjaneres", "Costos_Unifamiliar2plantes", "Costos_nauind", "Costos_edificioficines"], min_year, max_year,["Any", "Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"])
            table_catalunya_q = table_catalunya_q.dropna(axis=0)
            table_catalunya_y = table_catalunya_y.dropna(axis=0)
            left, right = st.columns((1,1))
            with left:
                st.metric(label="**Edifici renda normal entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici renda normal entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici renda normal entre mitjaneres", "var")}%""")
                st.metric(label="**Nau industrial** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nau industrial", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nau industrial", "var")}%""")
            with right:
                st.metric(label="**Unifamiliar de dos plantes entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Unifamiliar de dos plantes entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Unifamiliar de dos plantes entre mitjaneres", "var")}%""")
                st.metric(label="**Edifici d’oficines entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici d’oficines entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici d’oficines entre mitjaneres", "var")}%""")
            desc_bec_aux = """Els preus per m² construït inclouen l’estudi de seguretat i salut, els honoraris tècnics i permisos d’obra amb un benefici industrial del 20% i despeses generals. Addicionalment, 
            cal comentar que aquests preus fan referència a la província de Barcelona. Si la ubicació de l'obra es troba en una província diferent, la disminució dels preus serà d'un 6% a 8% a Girona, 8% a 10% a Tarragona i del 12% a 15% a Lleida."""
            # desc_bec = f'<div style="text-align: justify">{desc_bec_aux}</div>'
            # st.markdown(desc_bec, unsafe_allow_html=True)
            # st.markdown("")
            # st.markdown(f"""<a href="https://drive.google.com/file/d/1ArRHGTPnDjI2gq9iaGhL4MQK7SbIDNDb/" target="_blank"><button class="download-button">Descarregar BEC</button></a>""", unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2013), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2013, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2013, rounded=True), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_catalunya_q, ["Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"], "Costos de construcció per tipologia (€/m\u00b2)", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(line_plotly(table_catalunya_q.pct_change(4).mul(100).iloc[4:,:], ["Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"], "Costos de construcció per tipologia (% var. anual)", "%"), use_container_width=True, responsive=True)

        if selected_index=="Consum de Ciment":
            st.subheader("CONSUM DE CIMENT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2012
            table_catalunya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["cons_ciment_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Consum de ciment"])
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha","cons_ciment_Catalunya"],  f"{str(min_year)}-01-01", f"{date_max_ciment_aux}",["Data", "Consum de ciment"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","cons_ciment_Catalunya"], min_year, max_year,["Any", "Consum de ciment"])

            table_catalunya_q = table_catalunya_q.dropna(axis=0).div(1000)
            table_catalunya_y = table_catalunya_y.dropna(axis=0).div(1000)
            st.metric(label="**Consum de ciment** (Milers de tones)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Consum de ciment", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Consum de ciment", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2018).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2014), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2014, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2014, True), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_catalunya_q, ["Consum de ciment"], "Consum de ciment (Milers T.)", "Milers de T."), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_catalunya_y.pct_change(1).mul(100).dropna(axis=0), ["Consum de ciment"], "Variació anual del consum de ciment (Milers T.)", "%", 2012), use_container_width=True, responsive=True)
        if selected_index=="Hipoteques":
            st.subheader("IMPORT I NOMBRE D'HIPOTEQUES INSCRITES EN ELS REGISTRES DE PROPIETAT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            table_catalunya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "hipon_Catalunya", "hipoimp_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data","Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_m = table_catalunya_m[["Data","Nombre d'hipoteques", "Import d'hipoteques"]].rename(columns={"Data":"Fecha"})
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "hipon_Catalunya", "hipoimp_Catalunya"],  f"{str(min_year)}-01-01", f"{date_max_hipo_aux}",["Data", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_q = table_catalunya_q[["Nombre d'hipoteques", "Import d'hipoteques"]]
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","hipon_Catalunya", "hipoimp_Catalunya"], min_year, max_year,["Any", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_y = table_catalunya_y[["Nombre d'hipoteques", "Import d'hipoteques"]]
            if selected_year_n==max_year-1:
                left, right = st.columns((1,1))
                with left:
                    st.metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), ["Nombre d'hipoteques"], "var", "month_aux")}%""")
                with right:
                    st.metric(label="**Import d'hipoteques** (Milers €)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), ["Import d'hipoteques"], "var", "month_aux")}%""")
            if selected_year_n!=max_year-1:
                left, right = st.columns((1,1))
                with left:
                    st.metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Nombre d'hipoteques", "var")}%""")
                with right:
                    st.metric(label="**Import d'hipoteques** (Milers €)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Import d'hipoteques", "var")}%""")
            selected_columns = ["Nombre d'hipoteques", "Import d'hipoteques"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2022).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2014), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2014, rounded=False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_catalunya_m, ["Nombre d'hipoteques"], "Evolució mensual del nombre d'hipoteques", "Nombre d'hipoteques",  "Data"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_catalunya_m, ["Import d'hipoteques"], "Evolució mensual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  "Data"), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_catalunya_y, ["Nombre d'hipoteques"], "Evolució anual del nombre d'hipoteques", "Nombre d'hipoteques",  2005), use_container_width=True, responsive=True)
                st.plotly_chart(bar_plotly(table_catalunya_y, ["Import d'hipoteques"], "Evolució anual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  2005), use_container_width=True, responsive=True)

    if selected_indicator=="Sector residencial":
        if selected_type=="Venda":
            if selected_index=="Producció":
                min_year=2008
                st.subheader("PRODUCCIÓ D'HABITATGES A CATALUNYA")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_cat_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])     
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], "Catalunya") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Cataluña"), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars",
                                                                                                                                                                                                                                                                                                                                   "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                                                                    "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], "Catalunya") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Cataluña"), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars",
                                                                                                                                                                                                                                                                                                                                              "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                                                                                "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
                table_Catalunya_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_Catalunya_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_cat_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
                with center:
                    try:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_cat_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                with center:
                    try:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Qualificacions provisionals d'HPO**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions provisionals d'HPO**", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Qualificacions definitives d'HPO**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions definitives d'HPO**", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n),  "Qualificacions definitives d'HPO (Promotor públic)", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n),  "Qualificacions definitives d'HPO (Promotor privat)", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value="No disponible")
                # st.markdown("La producció d'habitatge a Catalunya al 2022")
                
                # selected_columns = st.multiselect("**Selecció d'indicadors:**", table_Catalunya.columns.tolist(), default=table_Catalunya.columns.tolist())
                selected_columns_ini = [col for col in table_Catalunya.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_Catalunya.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                selected_columns_aux1 = ["Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)"]
                selected_columns_aux2 = ["Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2008), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2008), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_Catalunya, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st.plotly_chart(stacked_bar_plotly(table_Catalunya_y, selected_columns_aux1, "Qualificacions provisionals de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_Catalunya[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_Catalunya_pluri, table_Catalunya_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_Catalunya_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True) 
                    st.plotly_chart(stacked_bar_plotly(table_Catalunya_y, selected_columns_aux2, "Qualificacions definitives de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_Catalunya[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_Catalunya_uni, table_Catalunya_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            if selected_index=="Compravendes":
                min_year=2014
                st.subheader("COMPRAVENDES D'HABITATGES A CATALUNYA")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "trvivt_Catalunya", "trvivs_Catalunya", "trvivn_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivt_Catalunya", "trvivs_Catalunya", "trvivn_Catalunya"], min_year, max_year,["Any", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge nou**", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_Catalunya,  table_Catalunya.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(stacked_bar_plotly(table_Catalunya_y,  table_Catalunya.columns.tolist()[1:3], "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2014), use_container_width=True, responsive=True)
            if selected_index=="Preus":
                min_year=2014
                st.subheader("PREUS PER M\u00b2 CONSTRUÏT")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "prvivt_Catalunya", "prvivs_Catalunya", "prvivn_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "prvivt_Catalunya", "prvivs_Catalunya", "prvivn_Catalunya"], min_year, max_year,["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")  
                with center:
                    try:
                        st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")  
                with right:
                    try:
                        st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge nou", "var")}%""")
                    except IndexError:
                        st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value="No disponible")  
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014, True, False), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True, False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_Catalunya, table_Catalunya.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_Catalunya_y, table_Catalunya.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2014), use_container_width=True, responsive=True)
            if selected_index=="Superfície":
                min_year=2014
                st.subheader("SUPERFÍCIE EN M\u00b2 CONSTRUÏTS")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "supert_Catalunya", "supers_Catalunya", "supern_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "supert_Catalunya", "supers_Catalunya", "supern_Catalunya"], min_year, max_year,["Any", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")  
                with center:
                    try:
                        st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")  
                with right:
                    try:
                        st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")   
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014, True, False), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True, False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_Catalunya, table_Catalunya.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construïts"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_Catalunya_y, table_Catalunya.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construïts", 2014), use_container_width=True, responsive=True)   
        if selected_type=="Lloguer":
            st.subheader("MERCAT DE LLOGUER")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2014
            table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "trvivalq_Catalunya", "pmvivalq_Catalunya"], f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivalq_Catalunya",  "pmvivalq_Catalunya"], min_year, max_year,["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                try:
                    st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
                except IndexError:
                    st.metric(label="**Nombre de contractes de lloguer**", value="No disponible")
            with right_col:
                try:
                    st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
                except IndexError:
                    st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_Catalunya, 2021, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_Catalunya, 2014, True), f"{selected_type}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_Catalunya_y, 2014, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True), f"{selected_type}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_Catalunya, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer a Catalunya", "€/mes"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_Catalunya, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer a Catalunya", "Nombre de contractes de lloguer"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_Catalunya_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer a Catalunya", "€/mes", 2005), use_container_width=True, responsive=True)   
                st.plotly_chart(bar_plotly(table_Catalunya_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer a Catalunya", "Nombre de contractes de lloguer", 2005), use_container_width=True, responsive=True)  
if selected == "Províncies i àmbits":
    prov_names = ["Barcelona", "Girona", "Tarragona", "Lleida"]
    ambit_names = ["Alt Pirineu i Aran","Camp de Tarragona","Comarques centrals","Comarques gironines","Metropolità","Penedès","Ponent","Terres de l'Ebre"]
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"), horizontal=True, key=400)
        selected_option = st.radio("**Selecciona un tipus d'àrea geogràfica:**", ["Províncies", "Àmbits territorials"], key=401)
    with center:
        if selected_option=="Províncies":
            selected_geo = st.selectbox('**Selecciona una província:**', prov_names, index= prov_names.index("Barcelona"))
        if selected_option=="Àmbits territorials":
            selected_geo = st.selectbox('**Selecciona un àmbit territorial:**', ambit_names, index= ambit_names.index("Metropolità"), key=402)
        if selected_type=="Venda":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=403)
    with right:
        selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=404)
    if selected_type=="Venda":
        if selected_option=="Àmbits territorials":
            if selected_index=="Producció":
                min_year=2008
                st.subheader(f"PRODUCCIÓ D'HABITATGES A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_province_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats**", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats**", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
                selected_columns_ini = [col for col in table_province.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_province.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2008), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2008, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_pluri, table_province_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True) 
                    st.plotly_chart(area_plotly(table_province[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_uni, table_province_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

            if selected_index=="Compravendes":
                min_year=2014
                st.subheader(f"COMPRAVENDES D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), min_year, max_year,["Any", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge nou**", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True) 
            if selected_index=="Preus":
                min_year=2014
                st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), min_year, max_year,["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "var")}%""") 
                    except IndexError:
                        st.metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2005), use_container_width=True, responsive=True) 
            if selected_index=="Superfície":
                min_year=2014
                st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de la superfície mitjana en m\u00b2 construïts per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de la superfície mitjana en m\u00b2 construïts per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True) 
        if selected_option=="Províncies":
            if selected_index=="Producció":
                min_year=2008
                st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])     
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_", "calprovgene_","finviv_","finviv_uni_", "finviv_pluri_", "caldefgene_"], selected_geo), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Qualificacions provisionals d'HPO", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars", "Qualificacions definitives d'HPO"])
                table_province_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_province_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats**", value="No disponible")          
                with center:
                    try:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats**", value="No disponible")      
                with center:
                    try:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats unifamiliars**", value="No disponible")

                selected_columns_ini = [col for col in table_province.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_province.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2008), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2008, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_pluri, table_province_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_uni, table_province_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

            if selected_index=="Compravendes":
                min_year=2014
                st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                    except IndexError:
                        st.metric(label="**Compravendes d'habitatge nou**", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)     
            if selected_index=="Preus":
                min_year=2014
                st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), min_year, max_year,["Any","Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "var")}%""")
                    except IndexError:
                        st.metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2005), use_container_width=True, responsive=True)     
                
            if selected_index=="Superfície":
                min_year=2014
                st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    try:
                        st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")
                with center:
                    try:
                        st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")
                with right:
                    try:
                        st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                    except IndexError:
                        st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2021, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True)

    if selected_type=="Lloguer":
        if selected_option=="Àmbits territorials":
            min_year=2014
            st.subheader(f"MERCAT DE LLOGUER A L'ÀMBIT: {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), min_year, max_year,["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                try:
                    st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
                except IndexError:
                    st.metric(label="**Nombre de contractes de lloguer**", value="No disponible")
            with right_col:
                try:
                    st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
                except IndexError:
                    st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2021, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)
        if selected_option=="Províncies":
            min_year=2014
            st.subheader(f"MERCAT DE LLOGUER A {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), min_year, max_year,["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                try:
                    st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
                except IndexError:
                    st.metric(label="**Nombre de contractes de lloguer**", value="No disponible")
            with right_col:
                try:
                    st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
                except IndexError:
                    st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2021, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)

if selected=="Comarques":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"), horizontal=True, key=501)
    with center:
        selected_com = st.selectbox("**Selecciona una comarca:**", sorted(maestro_mun["Comarca"].unique().tolist()), index= sorted(maestro_mun["Comarca"].unique().tolist()).index("Barcelonès"), key=502)
        if selected_type=="Venda":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=503)
    with right:
        selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=504)
    if selected_type=="Venda":
        if selected_index=="Producció":
            min_year=2008
            st.subheader(f"PRODUCCIÓ D'HABITATGES A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_com_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])     
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_", "calprovgene_","finviv_","finviv_uni_", "finviv_pluri_", "caldefgene_"], selected_com), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Qualificacions provisionals d'HPO", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars", "Qualificacions definitives d'HPO"])
            table_com_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            table_com_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats**", value="No disponible")
            with center:
                try:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats**", value="No disponible")          
            with center:
                try:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
            selected_columns_ini = [col for col in table_com.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_com.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2008), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2008, rounded=False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_com[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_com[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_com_pluri, table_com_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_com_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la produció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_com[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_com_uni, table_com_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_com), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
                
            with center:
                try:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge nou**", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, rounded=False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_com), min_year, max_year,["Any","Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")

            with center:
                try:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")
            with right:
                try:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2021,True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014, True, False), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, True, False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", 2005), use_container_width=True, responsive=True)
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_com), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")
            with center:
                try:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")
            with right:
                try:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014, True, False), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, True, False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)

            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        min_year=2014
        st.subheader(f"MERCAT DE LLOGUER A LA COMARCA: {selected_com.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_com), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_com), min_year, max_year,["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            try:
                st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Nombre de contractes de lloguer**", value="No disponible")
        with right_col:
            try:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_province, 2021, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_com}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
            st.plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st.plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)
if selected=="Municipis":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Selecciona un tipus d'indicador**", ("Venda", "Lloguer", "Altres indicadors"), key=601, horizontal=False)
    with center:
        selected_mun = st.selectbox("**Selecciona un municipi:**", maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].unique(), index= maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].tolist().index("Barcelona"), key=602)
        if selected_type=="Venda":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=603)
    with right:
        if (selected_type=="Venda") or (selected_type=="Lloguer"):
            selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=604)
    if selected_type=="Venda":
        if selected_index=="Producció":
            min_year=2008
            st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_", "calprovgene_", "finviv_","finviv_uni_", "finviv_pluri_", "caldefgene_"], selected_mun), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Qualificacions provisionals d'HPO", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars", "Qualificacions definitives d'HPO"])
            table_mun_pluri = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            table_mun_uni = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats**", value="No disponible")
            with center:
                try:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats**", value="No disponible")
            with center:
                try:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
            selected_columns_ini = [col for col in table_mun.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_mun.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2008), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2008, rounded=False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun_pluri, table_mun_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la producció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun_uni, table_mun_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
            with center:
                try:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value="No disponible") 
            with right:
                try:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge nou**", value="No disponible") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, rounded=False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_mun = table_mun.replace(0, np.nan)
            table_mun_y = table_mun.reset_index().copy()
            table_mun_y["Any"] = table_mun_y["Trimestre"].str[:4]
            table_mun_y = table_mun_y.drop("Trimestre", axis=1)
            table_mun_y = table_mun_y.groupby("Any").mean()
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2 construït)", value="No disponible") 
            with center:
                try:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2 construït)", value="No disponible") 
            with right:
                try:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2 construït)", value="No disponible") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014, True, False), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, True, False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", 2005), use_container_width=True, responsive=True)
            try:
                tabla_estudi_oferta = table_mun_oferta(selected_mun, 2019, 2025)
                st.subheader("Estudi d'Oferta de Nova Construcció (APCE). Municipi de " + selected_mun.split(',')[0].strip())
                st.markdown(tabla_estudi_oferta.to_html(), unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
                        <a href="https://estudi-oferta.apcebcn.cat/" 
                        class="button" 
                        target="_blank" 
                        rel="noopener noreferrer">
                        Accedir a l'Estudi d'Oferta
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception:
                pass
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")
            with center:
                try:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")
            with right:
                try:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014, True, False), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, True, False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", 2005), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        min_year=2014
        st.subheader(f"MERCAT DE LLOGUER A {selected_mun.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun), min_year, max_year,["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            try:
                st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Nombre de contractes de lloguer**", value="No disponible")
        with right_col:
            try:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")
                st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_mun, 2021, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_mun, 2014, rounded=True), f"{selected_type}_{selected_mun}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_mun_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_mun_y, 2014, rounded=True), f"{selected_type}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(line_plotly(table_mun, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes", True), use_container_width=True, responsive=True)
            st.plotly_chart(line_plotly(table_mun, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(bar_plotly(table_mun_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st.plotly_chart(bar_plotly(table_mun_y, ["Nombre de contractes de lloguer"],  "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)

    if selected_type=="Altres indicadors":
        st.markdown('<div class="custom-box">DEMOGRAFIA (2021)</div>', unsafe_allow_html=True)
        years_mun = detect_and_coerce_years(df_mun_idescat)
        years_pe  = detect_and_coerce_years(df_pob_ine)
        YEARS = sorted(set(years_mun + years_pe), reverse=True)  # orden descendente global

        df_mun_idescat = add_last_cols(df_mun_idescat)
        df_pob_ine  = add_last_cols(df_pob_ine)

        nombre_variables = {
            "AfiliatSS_Agricultura": "Afiliats a la Seguretat Social – Agricultura",
            "AfiliatSS_Construcció": "Afiliats a la Seguretat Social – Construcció",
            "AfiliatSS_Indústria": "Afiliats a la Seguretat Social – Indústria",
            "AfiliatSS_Serveis": "Afiliats a la Seguretat Social – Serveis",
            "AfiliatSS_Total": "Afiliats a la Seguretat Social – Total",
            "Atur registrat_Agricultura": "Atur registrat – Agricultura",
            "Atur registrat_Construcció": "Atur registrat – Construcció",
            "Atur registrat_Indústria": "Atur registrat – Indústria",
            "Atur registrat_Serveis": "Atur registrat – Serveis",
            "Atur registrat_Total": "Atur registrat – Total",
            "IRPF_Base_imposable": "Base imposable mitjana de l’IRPF (€)",
            "Matrimonis_Total": "Nombre de matrimonis",
            "Naixements_Total": "Nombre de naixements",
            "Parc_vehicles_Total": "Parc total de vehicles",
            "Pensionistes_Total": "Nombre de pensionistes",
            "Residus_mun_per_capita": "Residus municipals per càpita (kg/hab/dia)",
            "poblacio_activa": "Població activa",
            "poblacio_ocupada": "Població ocupada",
            "poblacio_desocupada": "Població desocupada",
            "poblacio_inactiva": "Població inactiva",
            "Població total": "Població total",
            "Creixement població interanual": "Creixement interanual de la població",
            "Població 25–34 anys (% sobre total)": "Població de 25 a 34 anys (% sobre total)",
            "Població 35–44 anys (% sobre total)": "Població de 35 a 44 anys (% sobre total)",
            "Naixements sobre població": "Naixements sobre població total (%)",
            "Matrimonis sobre població": "Matrimonis sobre població total (%)",
        }
        df_mun_idescat["variable_sin_municipi"] = df_mun_idescat["variable"].str.replace(f"_{selected_mun}$", "", regex=True)
        df_mun_idescat["nombre_largo"] = df_mun_idescat["variable_sin_municipi"].map(nombre_variables)
        sel = (
            df_mun_idescat[df_mun_idescat["variable"].astype(str).str.endswith("_"+selected_mun, na=False)]
            .sort_values("variable")
        )


        POP_KEYS=[f"{selected_mun}_Total_Todas las edades", f"{selected_mun}_Total_Totes les edats"]
        AGE_25_34=[[f"{selected_mun}_Total_De 25 a 29 años", f"{selected_mun}_Total_De 25 a 29 anys"],
                [f"{selected_mun}_Total_De 30 a 34 años", f"{selected_mun}_Total_De 30 a 34 anys"]]
        AGE_35_44=[[f"{selected_mun}_Total_De 35 a 39 años", f"{selected_mun}_Total_De 35 a 39 anys"],
                [f"{selected_mun}_Total_De 40 a 44 años", f"{selected_mun}_Total_De 40 a 44 anys"]]


        # Población total y crecimiento
        pop_year, pop_val = latest_year_value(df_pob_ine, POP_KEYS)
        prev_pop_year, pop_prev = prev_year_value(df_pob_ine, POP_KEYS, pop_year)
        creix = (pop_val/pop_prev - 1)*100 if pd.notnull(pop_val) and pd.notnull(pop_prev) and pop_prev>0 else np.nan

        # Estructura por edades (cada tramo con su año válido y mismo denominador)
        age2534_year, p2534 = latest_year_sum_age(AGE_25_34)
        den2534 = get_year_val(df_pob_ine, POP_KEYS, age2534_year)
        pct2534 = (p2534/den2534*100) if pd.notnull(p2534) and pd.notnull(den2534) and den2534>0 else np.nan

        age3544_year, p3544 = latest_year_sum_age(AGE_35_44)
        den3544 = get_year_val(df_pob_ine, POP_KEYS, age3544_year)
        pct3544 = (p3544/den3544*100) if pd.notnull(p3544) and pd.notnull(den3544) and den3544>0 else np.nan

        # Nacimientos y matrimonios (año propio y mismo denominador)
        naix_year, naix_val = latest_year_value(df_mun_idescat, [f"Naixements_Total_{selected_mun}"])
        naix_pop = get_year_val(df_pob_ine, POP_KEYS, naix_year)
        naix_pct = (naix_val/naix_pop*100) if pd.notnull(naix_val) and pd.notnull(naix_pop) and naix_pop>0 else np.nan

        matr_year, matr_val = latest_year_value(df_mun_idescat, [f"Matrimonis_Total_{selected_mun}"])
        matr_pop = get_year_val(df_pob_ine, POP_KEYS, matr_year)
        matr_pct = (matr_val/matr_pop*100) if pd.notnull(matr_val) and pd.notnull(matr_pop) and matr_pop>0 else np.nan

        subset_tamaño_mun = censo_2021[censo_2021["Municipi"] == selected_mun][["1", "2", "3", "4", "5 o más"]]
        subset_tamaño_mun_aux = subset_tamaño_mun.T.reset_index()
        subset_tamaño_mun_aux.columns = ["Tamany", "Llars"]
        left, right = st.columns((1,1))
        with left:
            st.metric("Grandària de la llar més freqüent", value=censo_2021[censo_2021["Municipi"]==selected_mun]["Tamaño_hogar_frecuente"].values[0])
            st.metric("Proporció de població nacional", value=f"""{round(100 - censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_extranjera"].values[0],2):,.0f}%""")
            st.metric(label=f"Població total ({pop_year})", value=fmt_int(pop_val))
            st.metric(label=f"Població 25–34 anys (% sobre total) ({age2534_year})", value=fmt_pct(pct2534))
            st.metric(
                label=f"Nombre de naixements ({int(sel[sel['nombre_largo']=='Nombre de naixements']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Nombre de naixements']['last_value'].values[0])
            )
            st.metric(
                label=f"Nombre de matrimonis ({int(sel[sel['nombre_largo']=='Nombre de matrimonis']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Nombre de matrimonis']['last_value'].values[0])
            )

        with right:
            st.metric("Grandària mitjà de la llar", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Tamaño medio del hogar"].values[0],2)}""")
            st.metric("Proporció de població estrangera", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_extranjera"].values[0],1)}%""")
            st.metric("Proporció de població amb educació superior", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Porc_Edu_superior"].values[0],1)}%""")
            st.metric(label=f"Població 35–44 anys (% sobre total) ({age3544_year})", value=fmt_pct(pct3544))
            st.metric(label=f"Naixements sobre població ({naix_year})", value=fmt_pct(naix_pct))
            st.metric(label=f"Matrimonis sobre població ({matr_year})", value=fmt_pct(matr_pct))
        if f"poptottine_{selected_mun}" in DT_mun_y.columns and not DT_mun_y[f"poptottine_{selected_mun}"].isna().all():
            st.plotly_chart(
                line_plotly_pob(
                    DT_mun_y[["Fecha", f"poptottine_{selected_mun}"]],
                    f"poptottine_{selected_mun}",
                    f"Evolució anual de la població a {selected_mun}",
                    "Habitants"
                ),
                use_container_width=True
            )
        st.markdown("<div class='custom-box'>ECONOMIA, RENDA I ALTRES</div>", unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            st.metric("Renda neta per llar (2021)", value=f"""{(rentaneta_mun["rentanetahogar_" + selected_mun].values[-1]):,.0f}""")
            st.metric(
                label=f"Nombre de pensionistes ({int(sel[sel['nombre_largo']=='Nombre de pensionistes']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Nombre de pensionistes']['last_value'].values[0])
            )
            st.metric(
                label=f"Parc total de vehicles ({int(sel[sel['nombre_largo']=='Parc total de vehicles']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Parc total de vehicles']['last_value'].values[0])
            )
            st.plotly_chart(bar_plotly_demografia(rentaneta_mun.rename(columns={"Año":"Any"}).set_index("Any"), ["rentanetahogar_" + selected_mun], "Evolució anual de la renda mitjana neta", "€", 2015), use_container_width=True, responsive=True)
        with right:
            st.metric(
                label=f"Base imposable mitjana de l’IRPF (€) ({int(sel[sel['nombre_largo']=='Base imposable mitjana de l’IRPF (€)']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Base imposable mitjana de l’IRPF (€)']['last_value'].values[0])
            )
            st.metric("Quota integra del Impost sobre Bèns Immobles (IBI) (" + str(idescat_muns[["Any", "IBI_quota_" + selected_mun]].dropna()["Any"].values[0]) + ")", value=f"""{int(idescat_muns["IBI_quota_" + selected_mun].dropna().values[0]):,.0f}""")
            st.metric(
                label=f"Residus municipals per càpita (kg/hab/dia) ({int(sel[sel['nombre_largo']=='Residus municipals per càpita (kg/hab/dia)']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Residus municipals per càpita (kg/hab/dia)']['last_value'].values[0])
            )
            st.plotly_chart(donut_plotly_demografia(subset_tamaño_mun_aux,["Tamany", "Llars"], "Distribució del nombre de membres per llar", "Llars"), use_container_width=True, responsive=True)
        st.markdown("<div class='custom-box'>CARACTERÍSTIQUES DEL PARC D'HABITATGE (2021)</div>", unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            st.metric("Proporció d'habitatges en propietat", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_propiedad"].values[0],1)}%""")
            st.metric("Proporció d'habitatges principals", value=f"""{round(100 - censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_noprincipales_y"].values[0],1)}%""")
            st.metric("Edat mitjana dels habitatges", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Edad media"].values[0],1)}""")

        with right:
            st.metric("Proporció d'habitatges en lloguer", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_alquiler"].values[0], 1)}%""")
            st.metric("Proporció d'habitatges no principals", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_noprincipales_y"].values[0],1)}%""")
            st.metric("Superfície mitjana dels habitatges", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Superficie media"].values[0],1)}""")
        st.markdown("<div class='custom-box'>MERCAT LABORAL</div>", unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            st.metric(
                label=f"Afiliats a la Seguretat Social – Agricultura ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Agricultura']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Agricultura']['last_value'].values[0])
            )
            st.metric(
                label=f"Afiliats a la Seguretat Social – Construcció ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Construcció']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Construcció']['last_value'].values[0])
            )
            st.metric(
                label=f"Atur registrat – Agricultura ({int(sel[sel['nombre_largo']=='Atur registrat – Agricultura']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Agricultura']['last_value'].values[0])
            )
            st.metric(
                label=f"Atur registrat – Construcció ({int(sel[sel['nombre_largo']=='Atur registrat – Construcció']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Construcció']['last_value'].values[0])
            )

            st.metric(
                label=f"Atur registrat – Indústria ({int(sel[sel['nombre_largo']=='Atur registrat – Indústria']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Indústria']['last_value'].values[0])
            )
            st.metric(
                label=f"Població activa ({int(sel[sel['nombre_largo']=='Població activa']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Població activa']['last_value'].values[0])
            )
            st.metric(
                label=f"Població inactiva ({int(sel[sel['nombre_largo']=='Població inactiva']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Població inactiva']['last_value'].values[0])
            )
        with right:
            st.metric(
                label=f"Afiliats a la Seguretat Social – Indústria ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Indústria']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Indústria']['last_value'].values[0])
            )
            st.metric(
                label=f"Afiliats a la Seguretat Social – Serveis ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Serveis']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Serveis']['last_value'].values[0])
            )
            st.metric(
                label=f"Afiliats a la Seguretat Social – Total ({int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Total']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Afiliats a la Seguretat Social – Total']['last_value'].values[0])
            )

            st.metric(
                label=f"Atur registrat – Serveis ({int(sel[sel['nombre_largo']=='Atur registrat – Serveis']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Serveis']['last_value'].values[0])
            )
            st.metric(
                label=f"Atur registrat – Total ({int(sel[sel['nombre_largo']=='Atur registrat – Total']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Atur registrat – Total']['last_value'].values[0])
            )
            st.metric(
                label=f"Població ocupada ({int(sel[sel['nombre_largo']=='Població ocupada']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Població ocupada']['last_value'].values[0])
            )
            st.metric(
                label=f"Població desocupada ({int(sel[sel['nombre_largo']=='Població desocupada']['last_year'].values[0])})",
                value=int(sel[sel['nombre_largo']=='Població desocupada']['last_value'].values[0])
            )
if selected=="Districtes de Barcelona":
    left, center, right= st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Selecciona un tipus d'indicador**", ("Venda", "Lloguer", "Demografia i parc d'habitatge"), key=701, horizontal=False)
    with center:
        selected_dis = st.selectbox("**Selecciona un districte de Barcelona:**", maestro_dis["Districte"].unique(), key=702)
        if selected_type=="Venda":
            selected_index = st.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus", "Superfície"], key=703)
    with right:
        if (selected_type=="Venda") or (selected_type=="Lloguer"):
            selected_year_n = st.selectbox("**Selecciona un any:**", available_years, available_years.index(index_year), key=704)
    if selected_type=="Venda":
        if selected_index=="Producció":
            min_year=2011
            st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_dis), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            # table_dis_pluri = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            # table_dis_uni = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats**", value="0")
            with center:
                try:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value="No disponible")
            with left:
                try:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats**", value="0")
            with center:
                try:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value="No disponible")           
            with right:
                try:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value="No disponible")
            selected_columns_ini = [col for col in table_dis.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_dis.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2014), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2014, rounded=False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_dis[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la producció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_dis[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_dis), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
            with center:
                try:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
            with right:
                try:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge total**", value="No disponible")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2021).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, rounded=False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis_y.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2017), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_dis), min_year, max_year,["Any","Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value="No disponible")
            with center:
                try:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value="No disponible")
            with right:
                try:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value="No disponible")  
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017, True, False), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, True, False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m2 útil", 2017), use_container_width=True, responsive=True)
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_dis), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value="No disponible")  
            with center:
                try:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="No disponible")  
            with right:
                try:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="No disponible")  
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2021, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017, True, False), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, True, False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m2 útil", 2017), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        st.subheader(f"MERCAT DE LLOGUER A {selected_dis.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        min_year=2014
        table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_dis), f"{str(min_year)}-01-01", max_trim_lloguer,["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_dis), min_year, max_year,["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            try:
                st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Nombre de contractes de lloguer**", value="No disponible")   
        with right_col:
            try:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="No disponible")   
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_dis, 2021, True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_dis, 2014, True), f"{selected_type}_{selected_dis}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_dis_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_dis_y, 2014, rounded=True), f"{selected_type}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(line_plotly(table_dis, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes", True), use_container_width=True, responsive=True)
            st.plotly_chart(line_plotly(table_dis, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(bar_plotly(table_dis_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st.plotly_chart(bar_plotly(table_dis_y, ["Nombre de contractes de lloguer"],  "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)

    if selected_type=="Demografia i parc d'habitatge":
        st.markdown(f'<div class="custom-box">DEMOGRAFIA Y RENDA (2021)</div>', unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            subset_tamaño_dis = censo_2021_dis[censo_2021_dis["Distrito"] == selected_dis][["1", "2", "3", "4", "5 o más"]]
            subset_tamaño_dis_aux = subset_tamaño_dis.T.reset_index()
            subset_tamaño_dis_aux.columns = ["Tamany", "Llars"]
            max_column = subset_tamaño_dis.idxmax(axis=1).values[0]
            st.metric("Grandària de la llar més freqüent", value=max_column)
            st.metric("Proporció de població nacional", value=f"""{round(100 - censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_extranjera"].values[0]*100,0)}%""")
            st.metric("Renda neta per llar", value=f"""{(rentaneta_dis["rentahogar_" + selected_dis].values[-1]):,.0f}""")
        with right:
            st.metric("Grandària mitjà de la llar", value=f"""{censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Tamaño medio del hogar"].values[0]}""")
            st.metric("Proporció de població estrangera", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_extranjera"].values[0],2)*100}%""")
            st.metric("Proporció de població amb educació superior", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_edusuperior"].values[0]*100,1)}%""")

        st.markdown(f"<div class='custom-box'>CARACTERÍSTIQUES DEL PARC D'HABITATGE (2021)</div>", unsafe_allow_html=True)
        left, right = st.columns((1,1))
        with left:
            st.metric("Proporció d'habitatges en propietat", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_propiedad"].values[0],1)}%""")
            st.metric("Proporció d'habitatges principals", value=f"""{round(100 - censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_noprincipales"].values[0],1)}%""")
            st.metric("Edat mitjana dels habitatges", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Edad media"].values[0],1)}""")
            st.plotly_chart(bar_plotly_demografia(rentaneta_dis.rename(columns={"Año":"Any"}).set_index("Any"), ["rentahogar_" + selected_dis], "Evolución anual de la renta media neta anual", "€", 2015), use_container_width=True, responsive=True)
        with right:
            st.metric("Proporció d'habitatges en lloguer", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_alquiler"].values[0], 1)}%""")
            st.metric("Proporció d'habitatges no principals", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_noprincipales"].values[0],1)}%""")
            st.metric("Superfície mitjana dels habitatges", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Superficie Media"].values[0],1)}""")
            st.plotly_chart(donut_plotly_demografia(subset_tamaño_dis_aux,["Tamany", "Llars"], "Distribució del nombre de membres per llar", "Llars"), use_container_width=True, responsive=True)



if selected=="Informe de mercat":
    left, center, right = st.columns((1,1,1))
    with left:
        selected_mun = st.selectbox("**Selecciona un municipi:**", maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].unique(), index= maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].tolist().index("Barcelona"), key=602)

        st.write("**Descarrega el informe complet del municipi seleccionat:**")
        if st.button("📄 Descarregar informe PDF"):
            with st.spinner(f"Generant informe per a {selected_mun}..."):
                min_year=2008
                #Producció
                table_mun_prod = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_mun_prod_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_", "calprovgene_", "finviv_","finviv_uni_", "finviv_pluri_", "caldefgene_"], selected_mun), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Qualificacions provisionals d'HPO", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars", "Qualificacions definitives d'HPO"])
                table_mun_prod_pluri = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_mun_prod_uni = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                selected_columns_ini = [col for col in table_mun_prod.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_mun_prod.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                # --- Compravendes ---
                table_mun_tr = tidy_Catalunya(
                    DT_mun,
                    ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun),
                    f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",
                    ["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]
                )
                table_mun_tr_y = tidy_Catalunya_anual(
                    DT_mun_y,
                    ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun),
                    min_year, max_year,
                    ["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]
                )

                # --- Preus (no sobreescribir table_mun) ---
                table_mun_pr = tidy_Catalunya(
                    DT_mun,
                    ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_mun),
                    f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",
                    ["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"]
                ).replace(0, np.nan)
                table_mun_pr_y = table_mun_pr.reset_index().copy()
                table_mun_pr_y["Any"] = table_mun_pr_y["Trimestre"].str[:4]
                table_mun_pr_y = table_mun_pr_y.drop("Trimestre", axis=1).groupby("Any").mean()

                # --- Superfície ---
                table_mun_sup = tidy_Catalunya(
                    DT_mun,
                    ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun),
                    f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",
                    ["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"]
                )
                table_mun_sup_y = tidy_Catalunya_anual(
                    DT_mun_y,
                    ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun),
                    min_year, max_year,
                    ["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"]
                )

                # --- Lloguer ---
                table_mun_llog = tidy_Catalunya(
                    DT_mun,
                    ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun),
                    f"{str(min_year)}-01-01", max_trim_lloguer,
                    ["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"]
                )
                table_mun_llog_y = tidy_Catalunya_anual(
                    DT_mun_y,
                    ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun),
                    min_year, max_year,
                    ["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"]
                )

                years_mun = detect_and_coerce_years(df_mun_idescat)
                years_pe  = detect_and_coerce_years(df_pob_ine)
                YEARS = sorted(set(years_mun + years_pe), reverse=True)  # orden descendente global

                df_mun_idescat = add_last_cols(df_mun_idescat)
                df_pob_ine  = add_last_cols(df_pob_ine)


                df_mun_idescat = add_last_cols(df_mun_idescat)
                df_pob_ine  = add_last_cols(df_pob_ine)

                try:
                    tabla_estudi_oferta = table_mun_oferta_aux(selected_mun, 2024)
                except:
                    tabla_estudi_oferta = None
                nombre_variables = {
                    "AfiliatSS_Agricultura": "Afiliats a la Seguretat Social – Agricultura",
                    "AfiliatSS_Construcció": "Afiliats a la Seguretat Social – Construcció",
                    "AfiliatSS_Indústria": "Afiliats a la Seguretat Social – Indústria",
                    "AfiliatSS_Serveis": "Afiliats a la Seguretat Social – Serveis",
                    "AfiliatSS_Total": "Afiliats a la Seguretat Social – Total",
                    "Atur registrat_Agricultura": "Atur registrat – Agricultura",
                    "Atur registrat_Construcció": "Atur registrat – Construcció",
                    "Atur registrat_Indústria": "Atur registrat – Indústria",
                    "Atur registrat_Serveis": "Atur registrat – Serveis",
                    "Atur registrat_Total": "Atur registrat – Total",
                    "IRPF_Base_imposable": "Base imposable mitjana de l’IRPF (€)",
                    "Matrimonis_Total": "Nombre de matrimonis",
                    "Naixements_Total": "Nombre de naixements",
                    "Parc_vehicles_Total": "Parc total de vehicles",
                    "Pensionistes_Total": "Nombre de pensionistes",
                    "Residus_mun_per_capita": "Residus municipals per càpita (kg/hab/dia)",
                    "poblacio_activa": "Població activa",
                    "poblacio_ocupada": "Població ocupada",
                    "poblacio_desocupada": "Població desocupada",
                    "poblacio_inactiva": "Població inactiva",
                    "Població total": "Població total",
                    "Creixement població interanual": "Creixement interanual de la població",
                    "Població 25–34 anys (% sobre total)": "Població de 25 a 34 anys (% sobre total)",
                    "Població 35–44 anys (% sobre total)": "Població de 35 a 44 anys (% sobre total)",
                    "Naixements sobre població": "Naixements sobre població total (%)",
                    "Matrimonis sobre població": "Matrimonis sobre població total (%)",
                }
                df_mun_idescat["variable_sin_municipi"] = df_mun_idescat["variable"].str.replace(f"_{selected_mun}$", "", regex=True)
                df_mun_idescat["nombre_largo"] = df_mun_idescat["variable_sin_municipi"].map(nombre_variables)
                sel = (
                    df_mun_idescat[df_mun_idescat["variable"].astype(str).str.endswith("_"+selected_mun, na=False)]
                    .sort_values("variable")
                )


                generar_pdf_municipi_tot(
                    selected_mun=selected_mun,
                    # Producció
                    table_mun_prod=table_mun_prod,
                    table_mun_prod_y=table_mun_prod_y,
                    table_mun_prod_pluri=table_mun_prod_pluri,
                    table_mun_prod_uni=table_mun_prod_uni,
                    selected_columns_ini=selected_columns_ini,
                    selected_columns_fin=selected_columns_fin,
                    # Compravendes
                    table_mun_tr=table_mun_tr,
                    table_mun_tr_y=table_mun_tr_y,
                    # Preus
                    table_mun_pr=table_mun_pr,
                    table_mun_pr_y=table_mun_pr_y,
                    # Superfície
                    table_mun_sup=table_mun_sup,
                    table_mun_sup_y=table_mun_sup_y,
                    # Lloguer
                    table_mun_llog=table_mun_llog,
                    table_mun_llog_y=table_mun_llog_y,
                    # Altres indicadors (ya los cargas en tu app)
                    censo_2021=censo_2021,
                    DT_mun_y=DT_mun_y,
                    idescat_muns=idescat_muns,
                    rentaneta_mun=rentaneta_mun,
                    tabla_estudi_oferta = tabla_estudi_oferta
                )

# if selected=="Contacte":
#     load_css_file(path + "main.css")
#     CONTACT_EMAIL = "estudis@apcecat.cat"
#     st.write("")
#     st.subheader(":mailbox: Contacteu-nos!")
#     contact_form = f"""
#     <form action="https://formsubmit.co/{CONTACT_EMAIL}" method="POST">
#         <input type="hidden" class="Contacte" name="_captcha" value="false">
#         <input type="text" class="Contacte" name="name" placeholder="Nom" required>
#         <input type="email" class="Contacte" name="email" placeholder="Correu electrónic" required>
#         <textarea class="Contacte" name="message" placeholder="La teva consulta aquí"></textarea>
#         <button type="submit" class="button">Enviar ✉</button>
#     </form>
#     """
#     st.markdown(contact_form, unsafe_allow_html=True)
