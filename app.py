# =============================================================================
# GIPEX — Geospatial Indicators for Proxy Environmental eXposure
# CHEAQI-MNCH  ·  v2.1  ·  2026
# Dash application — port 8087
# =============================================================================

import io
import base64
import json
import os
import sqlite3
import tempfile
import importlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from extraction import runner, VARIABLE_GROUPS, DERIVED_VARS, ROAD_VARS

# ── App identity ──────────────────────────────────────────────────────────────
APP_NAME    = "GIPEX"
APP_FULL    = "Geospatial Indicators for Proxy Environmental eXposure"
APP_VERSION = "2.1.0"
APP_YEAR    = "2026"
APP_PROJECT = "CHEAQI-MNCH"

# ── Module availability ───────────────────────────────────────────────────────
def _mod_ok(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

MOD_STATUS = {
    "requests":        _mod_ok("requests"),
    "openpyxl":        _mod_ok("openpyxl"),
    "xlrd":            _mod_ok("xlrd"),
    "pyarrow":         _mod_ok("pyarrow"),
    "geopandas":       _mod_ok("geopandas"),
    "earthengine-api": _mod_ok("ee"),
}

# ── Roads SQLite DB ───────────────────────────────────────────────────────────
_APP_DIR       = os.path.dirname(os.path.abspath(__file__))
OSM_ROADS_PATH = os.path.join(_APP_DIR, "OSM_Roads.shp")
DB_PATH        = os.path.join(_APP_DIR, "pixel.db")

_MAJOR_ROADS = {"motorway", "motorway_link", "trunk", "primary",
                "primary_link", "secondary", "secondary_link", "tertiary"}
_ALL_ROADS   = _MAJOR_ROADS | {"residential", "unclassified", "service"}


def _roads_db_ok():
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        ok   = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='roads'"
        ).fetchone() is not None
        conn.close()
        return ok
    except Exception:
        return False


_ROADS_READY = _roads_db_ok()


def load_osm_roads(bbox, major_only=True, max_features=3000):
    """Return (lats, lons) for Scattermap from the SQLite roads DB."""
    if not _ROADS_READY:
        return None, None
    lon_min, lat_min, lon_max, lat_max = bbox
    fclass_set  = _MAJOR_ROADS if major_only else _ALL_ROADS
    placeholders = ",".join("?" * len(fclass_set))
    sql = f"""
        SELECT wkt FROM roads
        WHERE fclass IN ({placeholders})
          AND lon_max >= ? AND lon_min <= ?
          AND lat_max >= ? AND lat_min <= ?
        LIMIT ?
    """
    params = list(fclass_set) + [lon_min, lon_max, lat_min, lat_max, max_features * 2]
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        from shapely import wkt as swkt
        lats, lons = [], []
        count = 0
        for (wkt_str,) in rows:
            if count >= max_features:
                break
            try:
                geom = swkt.loads(wkt_str)
                xs, ys = geom.xy
                lons.extend(list(xs)); lats.extend(list(ys))
                lons.append(None);    lats.append(None)
                count += 1
            except Exception:
                pass
        return (lats, lons) if lats else (None, None)
    except Exception:
        return None, None


# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800"
        "&family=JetBrains+Mono:wght@400;500&display=swap",
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = f"{APP_NAME} — {APP_FULL}"
server = app.server

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(
    bg     = "#080E1A",
    surf   = "#0D1B2E",
    card   = "#112240",
    atm    = "#0C1F3F",
    border = "#1E3A5F",
    cyan   = "#00E5FF",
    green  = "#00FF9F",
    orange = "#FF9F00",
    red    = "#FF4C5E",
    purple = "#B07FFF",
    pink   = "#FF7FBF",
    text   = "#E8EDF5",
    muted  = "#7A8BAA",
)
CHART_COLORS = [C["cyan"], C["green"], C["orange"], C["red"],
                C["purple"], C["pink"], "#FFD700", "#00BFFF", "#ADFF2F"]

PLOT_LAYOUT = dict(
    plot_bgcolor  = C["surf"],
    paper_bgcolor = C["bg"],
    font          = dict(color=C["text"], family="Inter, sans-serif", size=12),
    xaxis         = dict(gridcolor=C["border"], zerolinecolor=C["border"],
                         linecolor=C["border"], tickfont_color=C["text"]),
    yaxis         = dict(gridcolor=C["border"], zerolinecolor=C["border"],
                         linecolor=C["border"], tickfont_color=C["text"]),
    margin        = dict(l=55, r=25, t=55, b=55),
    legend        = dict(bgcolor=C["card"], bordercolor=C["border"],
                         borderwidth=1, font_color=C["text"]),
    hoverlabel    = dict(bgcolor=C["card"], bordercolor=C["cyan"],
                         font_color=C["text"]),
    title_font    = dict(color=C["cyan"], size=15),
)

META_COLS = {
    "_task_id", "grid_id", "cell_id",
    "lat_center", "lon_center", "lat", "lon", "latitude", "longitude",
    "date", "source", "country",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_special_cols(df):
    cl   = {c.lower(): c for c in df.columns}
    lat  = next((cl[k] for k in ("lat_center", "lat", "latitude")  if k in cl), None)
    lon  = next((cl[k] for k in ("lon_center", "lon", "longitude") if k in cl), None)
    date = next((cl[k] for k in ("date",)                          if k in cl), None)
    gid  = next((cl[k] for k in ("grid_id", "cell_id")            if k in cl), None)
    return lat, lon, date, gid


def get_indicator_cols(df):
    lat, lon, date, gid = get_special_cols(df)
    skip = META_COLS | {c for c in [lat, lon, date, gid] if c}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]


def _coerce_dtypes(df):
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    return df


def parse_bytes(content_bytes, filename):
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "csv"
    try:
        if ext in ("csv", "txt"):
            df = pd.read_csv(io.BytesIO(content_bytes))
        elif ext == "tsv":
            df = pd.read_csv(io.BytesIO(content_bytes), sep="\t")
        elif ext in ("xlsx",):
            df = pd.read_excel(io.BytesIO(content_bytes), engine="openpyxl")
        elif ext in ("xls",):
            df = pd.read_excel(io.BytesIO(content_bytes))
        elif ext == "json":
            df = pd.read_json(io.BytesIO(content_bytes))
        elif ext == "parquet":
            df = pd.read_parquet(io.BytesIO(content_bytes))
        else:
            df = pd.read_csv(io.BytesIO(content_bytes))
        return _coerce_dtypes(df)
    except Exception:
        return pd.read_csv(io.BytesIO(content_bytes))


def parse_upload(contents, filename):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = parse_bytes(decoded, filename)
        _, _, date, _ = get_special_cols(df)
        if date:
            df[date] = pd.to_datetime(df[date], errors="coerce")
        return df.to_json(date_format="iso", orient="split"), None
    except Exception as exc:
        return None, str(exc)


def df_from_store(data):
    df = pd.read_json(io.StringIO(data), orient="split")
    _, _, date, _ = get_special_cols(df)
    if date:
        df[date] = pd.to_datetime(df[date], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Parallel stats
# ─────────────────────────────────────────────────────────────────────────────
def _col_stats(args):
    df, col = args
    s = df[col].dropna()
    return {
        "Indicator": col,
        "N":         int(s.count()),
        "Missing %": f"{df[col].isna().mean()*100:.1f}",
        "Mean":      f"{s.mean():.4g}"   if len(s) else "—",
        "Std":       f"{s.std():.4g}"    if len(s) else "—",
        "Min":       f"{s.min():.4g}"    if len(s) else "—",
        "Median":    f"{s.median():.4g}" if len(s) else "—",
        "Max":       f"{s.max():.4g}"    if len(s) else "—",
    }


def compute_stats_parallel(df, indicators):
    workers = min(8, max(1, len(indicators)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_col_stats, [(df, c) for c in indicators]))


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────
def fig_map(df, indicator=None, show_roads=True, major_roads_only=True):
    lat, lon, date, gid = get_special_cols(df)
    if not lat or not lon:
        return go.Figure().update_layout(**PLOT_LAYOUT, title="No lat/lon columns found")

    grp     = [c for c in [lat, lon, gid] if c]
    plot_df = df[grp + ([indicator] if indicator else [])].copy()

    if indicator:
        plot_df = plot_df.groupby(grp, as_index=False)[indicator].mean()
        title   = f"Spatial Distribution — {indicator} (mean)"
        color   = indicator
        cscale  = "Turbo"
    else:
        plot_df = plot_df[grp].drop_duplicates()
        plot_df["_pts"] = 1
        color   = "_pts"
        cscale  = [[0, C["cyan"]], [1, C["cyan"]]]
        title   = "Grid Cell Locations"

    lat_min = float(plot_df[lat].min())
    lat_max = float(plot_df[lat].max())
    lon_min = float(plot_df[lon].min())
    lon_max = float(plot_df[lon].max())
    center  = {"lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2}
    span    = max(lat_max - lat_min, lon_max - lon_min)
    zoom    = max(2, min(14, int(8.5 - np.log2(span + 0.001))))

    fig = px.scatter_map(
        plot_df, lat=lat, lon=lon,
        color=color, color_continuous_scale=cscale,
        zoom=zoom, center=center, height=540, title=title,
    )
    fig.update_traces(marker_size=9, marker_opacity=0.88)

    if show_roads and _ROADS_READY:
        buf  = max(0.05, span * 0.05)
        bbox = (lon_min - buf, lat_min - buf, lon_max + buf, lat_max + buf)
        road_lats, road_lons = load_osm_roads(bbox, major_only=major_roads_only)
        if road_lats and road_lons:
            fig.add_trace(go.Scattermap(
                lat=road_lats, lon=road_lons, mode="lines",
                line=dict(color=C["orange"], width=1.4),
                name="OSM Major Roads" if major_roads_only else "OSM Roads",
                opacity=0.65, hoverinfo="skip",
            ))

    fig.add_trace(go.Scattermap(
        lat=[lat_min, lat_min, lat_max, lat_max, lat_min],
        lon=[lon_min, lon_max, lon_max, lon_min, lon_min],
        mode="lines", line=dict(color=C["cyan"], width=1.5, dash="dot"),
        name="Extent", hoverinfo="skip",
    ))
    fig.update_layout(
        map_style="carto-darkmatter", paper_bgcolor=C["bg"],
        font=dict(color=C["text"], family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=44, b=0),
        title_font=dict(color=C["cyan"], size=15),
        legend=dict(bgcolor=C["card"], bordercolor=C["border"],
                    font_color=C["text"], x=0.01, y=0.99),
        coloraxis_colorbar=dict(tickfont_color=C["text"], title_font_color=C["text"],
                                bgcolor=C["card"], outlinecolor=C["border"]),
    )
    return fig


def fig_timeseries(df, indicators, date_range=None):
    lat, lon, date, gid = get_special_cols(df)
    if not date or not indicators:
        return go.Figure().update_layout(**PLOT_LAYOUT, title="Select at least one indicator")
    sub = df.copy()
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        sub = sub[(sub[date] >= date_range[0]) & (sub[date] <= date_range[1])]
    fig = go.Figure()
    for i, ind in enumerate(indicators):
        ts  = sub.groupby(date)[ind].mean().reset_index().sort_values(date)
        col = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(
            x=ts[date], y=ts[ind], name=ind,
            mode="lines+markers",
            line=dict(color=col, width=2),
            marker=dict(size=4, color=col),
        ))
    fig.update_layout(**PLOT_LAYOUT, title="Time Series — Daily Spatial Mean",
                      xaxis_title="Date", yaxis_title="Value",
                      hovermode="x unified", height=460)
    return fig


def fig_correlation(df, indicators):
    if len(indicators) < 2:
        return go.Figure().update_layout(**PLOT_LAYOUT, title="Select ≥ 2 indicators")
    corr = df[indicators].corr().round(3)
    n    = len(indicators)
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        textfont=dict(size=max(6, min(11, int(130 / n))), color=C["text"]),
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont_color=C["text"], title_font_color=C["text"],
                      title="r", bgcolor=C["card"], outlinecolor=C["border"]),
    ))
    fig.update_layout(**PLOT_LAYOUT, title="Correlation Matrix",
                      xaxis=dict(tickangle=-45, tickfont=dict(size=10, color=C["text"]),
                                 gridcolor=C["border"]),
                      yaxis=dict(tickfont=dict(size=10, color=C["text"]),
                                 gridcolor=C["border"]),
                      height=max(420, n * 36 + 80))
    return fig


def fig_boxplots(df, indicators):
    if not indicators:
        return go.Figure().update_layout(**PLOT_LAYOUT, title="Select indicators")
    fig = go.Figure()
    for i, ind in enumerate(indicators):
        col = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Box(
            y=df[ind].dropna(), name=ind,
            marker_color=col, line_color=col,
            fillcolor=col + "28", boxmean="sd", showlegend=False,
        ))
    fig.update_layout(**PLOT_LAYOUT, title="Distribution — Box Plots",
                      yaxis_title="Value",
                      xaxis=dict(tickangle=-35, gridcolor=C["border"],
                                 tickfont=dict(size=10, color=C["text"])),
                      height=500)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────
def make_report(df):
    indicators = get_indicator_cols(df)
    lat, lon, date, gid = get_special_cols(df)

    rows = compute_stats_parallel(df, indicators)
    tbl  = pd.DataFrame(rows).to_html(index=False, border=0, classes="report-table")

    top   = indicators[:6]
    fmap  = fig_map(df, indicators[0] if indicators else None)
    fts   = fig_timeseries(df, top)
    fcorr = fig_correlation(df, indicators[:16])
    fbox  = fig_boxplots(df, indicators[:14])

    dr = "N/A"
    if date:
        d0, d1 = df[date].min(), df[date].max()
        if pd.notna(d0):
            dr = f"{d0.strftime('%Y-%m-%d')} → {d1.strftime('%Y-%m-%d')}"

    cells = df[gid].nunique() if gid else "N/A"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><title>{APP_NAME} Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root{{--cyan:#00E5FF;--green:#00FF9F;--bg:#080E1A;--surf:#0D1B2E;
           --card:#112240;--atm:#0C1F3F;--border:#1E3A5F;--text:#E8EDF5;--muted:#7A8BAA;}}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:var(--bg);color:var(--text);font-family:Inter,sans-serif;padding:40px 48px;line-height:1.6;}}
    h1{{color:var(--cyan);font-size:26px;border-bottom:2px solid var(--cyan);padding-bottom:12px;margin-bottom:6px;}}
    h2{{color:var(--green);font-size:18px;margin:40px 0 16px;border-left:3px solid var(--green);padding-left:10px;}}
    .meta{{color:var(--muted);font-size:13px;margin-bottom:28px;}}
    .meta span{{color:var(--text);font-weight:500;margin-left:4px;}}
    .chart-wrap{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px;margin-bottom:28px;}}
    .report-table{{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border-radius:8px;overflow:hidden;}}
    .report-table th{{background:var(--atm);color:var(--cyan);padding:9px 14px;text-align:left;font-weight:600;}}
    .report-table td{{padding:8px 14px;border-bottom:1px solid var(--border);}}
    .report-table tr:last-child td{{border-bottom:none;}}
    .report-table tr:hover td{{background:var(--surf);}}
    .footer{{color:var(--muted);font-size:12px;margin-top:56px;border-top:1px solid var(--border);padding-top:14px;}}
  </style>
</head>
<body>
<h1>{APP_NAME} — Indicators Report</h1>
<div class="meta">
  Generated: <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span> &nbsp;·&nbsp;
  Records: <span>{len(df):,}</span> &nbsp;·&nbsp;
  Grid Cells: <span>{cells}</span> &nbsp;·&nbsp;
  Indicators: <span>{len(indicators)}</span> &nbsp;·&nbsp;
  Date Range: <span>{dr}</span>
</div>
<h2>Summary Statistics</h2>
<div class="chart-wrap">{tbl}</div>
<h2>Spatial Extent</h2>
<div class="chart-wrap">{fmap.to_html(full_html=False, include_plotlyjs=False)}</div>
<h2>Time Series (top 6)</h2>
<div class="chart-wrap">{fts.to_html(full_html=False, include_plotlyjs=False)}</div>
<h2>Correlation Matrix</h2>
<div class="chart-wrap">{fcorr.to_html(full_html=False, include_plotlyjs=False)}</div>
<h2>Distributions</h2>
<div class="chart-wrap">{fbox.to_html(full_html=False, include_plotlyjs=False)}</div>
<div class="footer">{APP_PROJECT} · {APP_NAME} {APP_FULL} · {datetime.now().year}</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(value, label, color=None):
    return html.Div([
        html.Div(str(value), className="metric-value", style={"color": color or C["cyan"]}),
        html.Div(label, className="metric-label"),
    ], className="metric-card")


def section_header(title, sub=None):
    return html.Div([
        html.Div(title, className="section-title"),
        html.Div(sub, style={"color": C["muted"], "fontSize": "12px",
                              "marginTop": "-10px", "marginBottom": "16px"}) if sub else None,
    ])


def card_wrap(*children, mb=20):
    return html.Div(list(children), className="chart-card",
                    style={"marginBottom": f"{mb}px"})


def ctrl_label(text):
    return html.Label(text, className="ctrl-label")


# ─────────────────────────────────────────────────────────────────────────────
# Top horizontal tab bar
# ─────────────────────────────────────────────────────────────────────────────
ALL_PAGES = ["config", "run", "overview", "map", "timeseries",
             "correlations", "distributions", "report", "about"]

_TAB_GROUPS = [
    ("Pipeline", C["orange"], [
        ("⚙", "config",  "Configure"),
        ("▶", "run",     "Run"),
    ]),
    ("Dashboard", C["cyan"], [
        ("▦", "overview",      "Overview"),
        ("🗺", "map",           "Map"),
        ("📈", "timeseries",    "Time Series"),
        ("⊞", "correlations",  "Correlations"),
        ("⊡", "distributions", "Distributions"),
        ("⎙", "report",        "Report"),
    ]),
    ("", C["purple"], [
        ("ℹ", "about", "About"),
    ]),
]


def top_tab_bar(active_page="overview"):
    groups = []
    for group_label, group_color, tabs in _TAB_GROUPS:
        btns = []
        for icon, pid, label in tabs:
            is_active = (pid == active_page)
            btns.append(html.Button(
                [html.Span(icon, className="tab-icon"), label],
                id={"type": "nav-btn", "index": pid},
                className="top-tab top-tab-active" if is_active else "top-tab",
                n_clicks=0,
                **{"data-color": group_color},
            ))
        if group_label:
            groups.append(html.Div([
                html.Span(group_label, className="tab-group-label",
                          style={"color": group_color}),
                html.Div(btns, className="tab-group-btns"),
            ], className="tab-group"))
        else:
            groups.append(html.Div(btns, className="tab-group tab-group-solo"))
    return html.Div(groups, className="top-tab-bar", id="top-tab-bar")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (data ingestion only — nav moved to top)
# ─────────────────────────────────────────────────────────────────────────────
sidebar = html.Div([
    # Brand
    html.Div([
        html.Div([
            html.Div("⬡", style={
                "fontSize": "26px",
                "background": f"linear-gradient(135deg, {C['cyan']}, {C['green']})",
                "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
            }),
            html.Div([
                html.Div(APP_NAME, style={
                    "fontSize": "22px", "fontWeight": "800",
                    "background": f"linear-gradient(90deg, {C['cyan']}, {C['green']})",
                    "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                    "letterSpacing": "0.08em",
                }),
                html.Div(APP_FULL[:28] + "…", style={
                    "fontSize": "7.5px", "color": C["muted"],
                    "textTransform": "uppercase", "letterSpacing": "0.08em",
                }),
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                  "padding": "16px 18px 8px"}),
        html.Div([
            html.Span(f"v{APP_VERSION}", className="version-badge"),
            html.Span("  ◉", style={
                "color": C["green"] if _ROADS_READY else C["muted"],
                "fontSize": "8px", "marginLeft": "6px",
            }),
            html.Span(" Roads DB" if _ROADS_READY else " Roads —",
                      style={"fontSize": "9px",
                             "color": C["green"] if _ROADS_READY else C["muted"]}),
        ], style={"padding": "0 18px 14px", "display": "flex", "alignItems": "center"}),
    ], className="sidebar-brand"),

    html.Hr(style={"borderColor": C["border"], "margin": "0 14px 12px"}),

    # File upload
    html.Div([
        html.Div("Upload Dataset", className="sidebar-section-label"),
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                html.Div("⬆", style={"fontSize": "16px", "marginBottom": "2px"}),
                html.Div("CSV · TSV · Excel · JSON · Parquet",
                         style={"fontSize": "10px", "lineHeight": "1.5"}),
                html.Div("drop or click", style={"fontSize": "9px", "opacity": "0.6"}),
            ]),
            className="upload-zone", multiple=False,
        ),
        html.Div(id="upload-status",
                 style={"fontSize": "10px", "marginTop": "5px",
                        "color": C["muted"], "textAlign": "center", "minHeight": "13px"}),
    ], style={"padding": "0 14px 10px"}),

    html.Hr(style={"borderColor": C["border"], "margin": "0 14px 10px"}),

    # URL upload
    html.Div([
        html.Div("Load from URL", className="sidebar-section-label"),
        html.Div([
            dbc.Input(
                id="url-input", placeholder="https://…/data.csv", debounce=False,
                style={"background": C["atm"], "color": C["text"],
                       "border": f"1px solid {C['border']}", "borderRadius": "6px",
                       "fontSize": "10.5px", "padding": "5px 8px", "flex": "1"},
            ),
            dbc.Button("↓", id="btn-url-load", size="sm", style={
                "background": C["cyan"], "border": "none", "color": C["bg"],
                "fontWeight": "800", "borderRadius": "6px",
                "padding": "5px 11px", "fontSize": "12px", "flexShrink": "0",
            }),
        ], style={"display": "flex", "gap": "5px"}),
        html.Div(id="url-status",
                 style={"fontSize": "10px", "marginTop": "5px",
                        "color": C["muted"], "minHeight": "13px"}),
    ], style={"padding": "0 14px 12px"}),

    html.Hr(style={"borderColor": C["border"], "margin": "0 14px 0"}),

    html.Div(id="sidebar-info", style={"padding": "10px 14px 0"}),

    # Footer
    html.Div([
        html.Hr(style={"borderColor": C["border"], "margin": "0 0 8px"}),
        html.Div([
            html.Span(APP_NAME + " ", style={
                "color": C["cyan"], "fontWeight": "700",
                "fontFamily": "JetBrains Mono, monospace", "fontSize": "10px",
            }),
            html.Span(f"v{APP_VERSION}", style={"color": C["muted"], "fontSize": "9px"}),
        ]),
        html.Div(f"{APP_PROJECT} · {APP_YEAR}",
                 style={"fontSize": "9px", "color": C["muted"], "marginTop": "2px"}),
    ], className="sidebar-footer"),
], id="sidebar")


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="store-data"),
    dcc.Store(id="store-page", data="overview"),
    dcc.Store(id="store-runner-state"),
    dcc.Store(id="store-grid-csv"),   # raw bytes b64 of the grid CSV for pipeline
    dcc.Download(id="download-report"),
    dcc.Download(id="download-result-csv"),
    dcc.Interval(id="poll-interval", interval=2000, disabled=True),

    sidebar,

    html.Div([
        top_tab_bar("overview"),
        html.Div(id="main-content", style={"padding": "22px 28px"}),
    ], style={
        "marginLeft": "220px",
        "minHeight": "100vh",
        "background": "transparent",
    }),
], style={"fontFamily": "Inter, sans-serif"})


# ─────────────────────────────────────────────────────────────────────────────
# Page renderers
# ─────────────────────────────────────────────────────────────────────────────

def page_welcome():
    features = [
        (C["cyan"],   "◈", "40+ GEE Variables",     "Sentinel-2, MODIS, ERA5, S5P, VIIRS NTL"),
        (C["green"],  "⬡", "Multi-format Ingestion", "CSV · TSV · Excel · JSON · Parquet · URL"),
        (C["orange"], "▦", "Interactive Mapping",    "Spatial distribution with OSM roads"),
        (C["purple"], "⊞", "Analytics Suite",        "Time series · Correlations · Distributions"),
    ]
    feat_cards = html.Div([
        html.Div([
            html.Div(icon, style={"fontSize": "22px", "color": color, "marginBottom": "8px"}),
            html.Div(title, style={"color": color, "fontWeight": "600",
                                   "fontSize": "13px", "marginBottom": "4px"}),
            html.Div(sub,   style={"color": C["muted"], "fontSize": "11px", "lineHeight": "1.5"}),
        ], className="feature-card")
        for color, icon, title, sub in features
    ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginTop": "32px"})

    return html.Div([html.Div([
        html.Div([
            html.Div([
                html.Span(APP_NAME, style={
                    "fontSize": "52px", "fontWeight": "800",
                    "background": f"linear-gradient(90deg, {C['cyan']}, {C['green']})",
                    "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                    "letterSpacing": "0.02em", "lineHeight": "1",
                }),
            ]),
            html.Div(APP_FULL, style={
                "fontSize": "13px", "color": C["muted"], "marginTop": "6px",
                "textTransform": "uppercase", "letterSpacing": "0.10em",
            }),
            html.Div(
                "Upload a dataset via the sidebar or run the GEE extraction pipeline. "
                "Supports CSV, Excel, JSON, Parquet and direct URL ingestion.",
                style={"color": C["muted"], "fontSize": "13px",
                       "marginTop": "18px", "maxWidth": "520px", "lineHeight": "1.8"},
            ),
            feat_cards,
        ], style={"position": "relative", "zIndex": "1"}),
    ], className="welcome-hero")])


def page_config(grid_csv_name=None):
    grp_checks = []
    for gk, gv in VARIABLE_GROUPS.items():
        var_str = ", ".join(gv["vars"])
        grp_checks.append(
            html.Div([
                dbc.Checkbox(id={"type": "grp-check", "index": gk}, value=True, className="me-2"),
                html.Div([
                    html.Span(gv["label"], style={"color": C["text"], "fontSize": "13px",
                                                   "fontWeight": "500"}),
                    html.Span(f"  ({len(gv['vars'])} vars)",
                              style={"color": C["muted"], "fontSize": "11px"}),
                    html.Div(var_str, style={"color": C["muted"], "fontSize": "11px",
                                             "fontFamily": "JetBrains Mono, monospace",
                                             "marginTop": "2px"}),
                ]),
            ], style={"display": "flex", "alignItems": "flex-start", "gap": "10px",
                      "padding": "10px 14px", "background": C["surf"],
                      "borderRadius": "6px", "marginBottom": "6px",
                      "border": f"1px solid {C['border']}"}),
        )

    def field(label, id_, placeholder, value=""):
        return dbc.Col([
            ctrl_label(label),
            dbc.Input(id=id_, placeholder=placeholder, value=value,
                      style={"background": C["card"], "color": C["text"],
                             "border": f"1px solid {C['border']}",
                             "borderRadius": "6px", "fontSize": "13px"}),
        ], md=6, className="mb-3")

    csv_status = html.Div([
        html.Span("◉ ", style={"color": C["green"]}),
        html.Span(grid_csv_name or "No grid CSV loaded",
                  style={"color": C["green"] if grid_csv_name else C["muted"],
                         "fontSize": "12px"}),
    ], style={
        "background": "rgba(0,255,159,0.06)", "border": "1px solid rgba(0,255,159,0.18)",
        "borderRadius": "8px", "padding": "8px 14px", "marginBottom": "4px",
        "display": "inline-flex", "alignItems": "center", "gap": "6px",
    }) if grid_csv_name else html.Div(
        html.Span("⚠  No grid CSV uploaded yet", style={"color": C["orange"], "fontSize": "12px"}),
        style={"background": "rgba(255,159,0,0.06)", "border": "1px solid rgba(255,159,0,0.18)",
               "borderRadius": "8px", "padding": "8px 14px", "marginBottom": "4px"}
    )

    return html.Div([
        section_header("Pipeline Configuration",
                       "Set up and run the Google Earth Engine extraction"),

        card_wrap(
            html.Div([
                html.Div("Grid Input CSV", className="config-section-header"),
                csv_status,
                html.Div(style={"height": "10px"}),
                dcc.Upload(
                    id="upload-grid-csv",
                    children=html.Div([
                        html.Span("⬆  Upload grid_cells CSV",
                                  style={"fontSize": "12px", "color": C["muted"]}),
                        html.Span("  (CSV with lat/lon/cell_id columns)",
                                  style={"fontSize": "11px", "color": C["border"]}),
                    ]),
                    style={
                        "border": f"1px dashed {C['border']}", "borderRadius": "8px",
                        "padding": "12px 18px", "cursor": "pointer",
                        "background": C["surf"], "transition": "all 0.2s",
                    },
                    multiple=False,
                ),
            ], style={"padding": "8px 4px"}),
        ),

        card_wrap(
            html.Div([
                html.Div("GEE & Connection", className="config-section-header"),
                dbc.Row([
                    field("GEE Project", "cfg-gee-project", "ee-your-project", "ee-cheaqi"),
                ]),
            ], style={"padding": "8px 4px"}),
        ),

        card_wrap(
            html.Div([
                html.Div("Column Mapping", className="config-section-header"),
                dbc.Row([
                    field("Cell ID column",  "cfg-col-id",   "e.g. cell_id", "cell_id"),
                    field("Latitude column", "cfg-col-lat",  "e.g. lat",     "lat"),
                    field("Longitude col.",  "cfg-col-lon",  "e.g. lon",     "lon"),
                    field("Date column",     "cfg-col-date",
                          "e.g. date_only (blank = use Date Range below)", "date_only"),
                    field("Source/ID col. (optional)", "cfg-col-src", "e.g. PID", "PID"),
                ]),
            ], style={"padding": "8px 4px"}),
        ),

        card_wrap(
            html.Div([
                html.Div("Date Range", className="config-section-header"),
                html.Div("Used only when the grid has NO per-row date column.",
                         style={"color": C["muted"], "fontSize": "11px", "marginBottom": "14px"}),
                dbc.Row([
                    dbc.Col([ctrl_label("From"),
                             dbc.Input(id="cfg-date-from", type="date", value="2022-04-07",
                                       style={"background": C["card"], "color": C["text"],
                                              "border": f"1px solid {C['border']}",
                                              "borderRadius": "6px", "fontSize": "13px"})],
                            md=3, className="mb-3"),
                    dbc.Col([ctrl_label("To"),
                             dbc.Input(id="cfg-date-to", type="date", value="2023-01-12",
                                       style={"background": C["card"], "color": C["text"],
                                              "border": f"1px solid {C['border']}",
                                              "borderRadius": "6px", "fontSize": "13px"})],
                            md=3, className="mb-3"),
                    dbc.Col([ctrl_label("Task workers (outer)"),
                             dbc.Input(id="cfg-max-workers", type="number", value=50,
                                       min=1, max=200,
                                       style={"background": C["card"], "color": C["text"],
                                              "border": f"1px solid {C['border']}",
                                              "borderRadius": "6px", "fontSize": "13px"})],
                            md=3, className="mb-3"),
                    dbc.Col([ctrl_label("Var workers (inner)"),
                             dbc.Input(id="cfg-var-workers", type="number", value=6,
                                       min=1, max=40,
                                       style={"background": C["card"], "color": C["text"],
                                              "border": f"1px solid {C['border']}",
                                              "borderRadius": "6px", "fontSize": "13px"})],
                            md=3, className="mb-3"),
                ]),
            ], style={"padding": "8px 4px"}),
        ),

        card_wrap(
            html.Div([
                html.Div("Variable Groups", className="config-section-header"),
                html.Div("40+ GEE variables + derived met (WS, WD10, RH) + road metrics",
                         style={"color": C["muted"], "fontSize": "11px", "marginBottom": "14px"}),
                *grp_checks,
            ], style={"padding": "8px 4px"}),
        ),

        html.Div([
            dbc.Button("▶  Start Extraction", id="btn-start",
                       style={"background": C["green"], "border": "none", "color": C["bg"],
                              "fontWeight": "700", "borderRadius": "6px",
                              "padding": "11px 28px", "fontSize": "14px"}),
            html.Div(id="start-msg",
                     style={"color": C["muted"], "fontSize": "12px",
                            "marginLeft": "16px", "display": "inline-block"}),
        ], style={"marginTop": "8px"}),
    ])


def page_run(runner_state=None):
    st       = runner_state or runner.get_state()
    status   = st["status"]
    progress = st["progress"]
    total    = st["total"]
    pct      = st["pct"]
    logs     = st["logs"]
    elapsed  = st.get("elapsed", "")
    result   = st.get("result_path", "")

    status_color = {
        "idle": C["muted"], "running": C["cyan"],
        "done": C["green"], "error": C["red"], "stopped": C["orange"],
    }.get(status, C["muted"])

    status_label = {
        "idle": "⬤  Idle", "running": "⬤  Running …",
        "done": "⬤  Done", "error": "⬤  Error", "stopped": "⬤  Stopped",
    }.get(status, status)

    progress_bar = html.Div([html.Div(style={
        "height": "8px", "width": f"{pct}%",
        "background": f"linear-gradient(90deg, {C['cyan']}, {C['green']})",
        "borderRadius": "4px", "transition": "width 0.4s ease",
        "boxShadow": "0 0 10px rgba(0,229,255,0.5)" if pct > 0 else "none",
    })], style={"height": "8px", "width": "100%",
                "background": C["border"], "borderRadius": "4px", "marginBottom": "8px"})

    log_text = "\n".join(logs[-60:]) if logs else "No log yet."

    action_btns = [
        dbc.Button("■  Stop", id="btn-stop", disabled=(status != "running"),
                   style={"background": C["red"], "border": "none", "color": C["text"],
                          "fontWeight": "600", "borderRadius": "6px",
                          "padding": "7px 18px", "fontSize": "12px",
                          "opacity": "1" if status == "running" else "0.4"}),
    ]
    if status == "done" and result and os.path.exists(result):
        action_btns += [
            dbc.Button("⬆  Load into Dashboard", id="btn-load-results",
                       style={"background": C["cyan"], "border": "none", "color": C["bg"],
                              "fontWeight": "700", "borderRadius": "6px",
                              "padding": "7px 18px", "fontSize": "12px",
                              "marginLeft": "8px"}),
            dbc.Button("⬇  Download CSV", id="btn-download-result",
                       style={"background": C["green"], "border": "none", "color": C["bg"],
                              "fontWeight": "700", "borderRadius": "6px",
                              "padding": "7px 18px", "fontSize": "12px",
                              "marginLeft": "8px"}),
        ]

    return html.Div([
        section_header("Run Extraction", "Live progress of the GEE extraction thread"),
        card_wrap(
            dbc.Row([
                dbc.Col(html.Div([
                    html.Span(status_label,
                              style={"color": status_color, "fontSize": "14px",
                                     "fontWeight": "600"}),
                    html.Span(f"  {elapsed}",
                              style={"color": C["muted"], "fontSize": "12px"}) if elapsed else None,
                ]), md=4),
                dbc.Col(html.Div(
                    f"{progress:,} / {total:,} tasks   ({pct}%)",
                    style={"color": C["text"], "fontSize": "13px",
                           "fontFamily": "JetBrains Mono, monospace"},
                ), md=4),
                dbc.Col(html.Div(action_btns,
                                 style={"display": "flex", "alignItems": "center"}), md=4),
            ], className="g-3 align-items-center", style={"padding": "8px 4px"}),
        ),
        card_wrap(
            html.Div([
                progress_bar,
                html.Div(f"{pct}% complete",
                         style={"color": C["muted"], "fontSize": "11px",
                                "textAlign": "right", "marginBottom": "12px"}),
                html.Div("Extraction Log", style={
                    "color": C["muted"], "fontSize": "10px",
                    "textTransform": "uppercase", "letterSpacing": "0.08em",
                    "marginBottom": "8px",
                }),
                html.Pre(log_text, id="log-pre"),
            ], style={"padding": "4px"}),
        ),
    ])


def page_overview(df):
    indicators       = get_indicator_cols(df)
    lat, lon, date, gid = get_special_cols(df)
    dr = "N/A"
    if date:
        d0, d1 = df[date].min(), df[date].max()
        if pd.notna(d0):
            dr = f"{d0.strftime('%b %d %Y')} – {d1.strftime('%b %d %Y')}"

    rows    = compute_stats_parallel(df, indicators)
    summary = pd.DataFrame(rows)

    tbl = dash_table.DataTable(
        data=summary.to_dict("records"),
        columns=[{"name": c, "id": c} for c in summary.columns],
        sort_action="native", filter_action="native", page_size=30,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": C["atm"], "color": C["cyan"],
                      "fontWeight": "600", "border": f"1px solid {C['border']}",
                      "fontSize": "12px"},
        style_cell={"backgroundColor": C["card"], "color": C["text"],
                    "border": f"1px solid {C['border']}",
                    "fontFamily": "Inter, sans-serif", "fontSize": "13px",
                    "padding": "9px 14px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": C["surf"]},
            {"if": {"filter_query": "{Missing %} > 20"}, "color": C["orange"]},
        ],
    )

    return html.Div([
        section_header("Overview"),
        dbc.Row([
            dbc.Col(metric_card(len(indicators), "Indicators",   C["cyan"]),   md=3),
            dbc.Col(metric_card(df[gid].nunique() if gid else "—",
                                "Grid Cells",    C["green"]),                   md=3),
            dbc.Col(metric_card(f"{len(df):,}",  "Total Records", C["orange"]), md=3),
            dbc.Col(metric_card(dr,              "Date Range",    C["purple"]), md=3),
        ], className="g-3 mb-4"),
        section_header("Per-Indicator Summary Statistics"),
        card_wrap(tbl),
    ])


def page_map(df):
    indicators       = get_indicator_cols(df)
    lat, lon, date, gid = get_special_cols(df)
    lat_min = float(df[lat].min()) if lat else 0
    lat_max = float(df[lat].max()) if lat else 0
    lon_min = float(df[lon].min()) if lon else 0
    lon_max = float(df[lon].max()) if lon else 0

    opts = [{"label": "Cell Locations (no colour)", "value": "__none__"}] + \
           [{"label": c, "value": c} for c in indicators]

    roads_badge = html.Div([
        html.Span("◉ ", style={"color": C["green"]}),
        html.Span("OSM Roads DB loaded  ·  2.8 M segments indexed",
                  style={"color": C["green"], "fontSize": "12px"}),
    ], style={
        "background": "rgba(0,255,159,0.06)", "border": "1px solid rgba(0,255,159,0.18)",
        "borderRadius": "7px", "padding": "8px 14px", "marginBottom": "14px",
    }) if _ROADS_READY else html.Div(
        html.Span("⚠ OSM Roads DB not found", style={"color": C["orange"], "fontSize": "12px"}),
        style={"padding": "8px 0", "marginBottom": "10px"},
    )

    return html.Div([
        section_header("Map View — Spatial Distribution"),
        roads_badge,
        dbc.Row([
            dbc.Col([ctrl_label("Colour by indicator"),
                     dcc.Dropdown(id="map-indicator", options=opts,
                                  value=indicators[0] if indicators else "__none__",
                                  clearable=False, className="dark-dropdown")], md=5),
            dbc.Col([ctrl_label("Roads overlay"),
                     html.Div([dbc.Switch(id="map-show-roads", label="Show roads",
                                          value=True, style={"color": C["text"]})],
                              style={"paddingTop": "6px"})], md=3),
            dbc.Col([ctrl_label("Road detail"),
                     dbc.RadioItems(
                         id="map-road-detail",
                         options=[{"label": "Major", "value": "major"},
                                  {"label": "All",   "value": "all"}],
                         value="major", inline=True,
                         style={"fontSize": "12px", "color": C["text"]},
                     )], md=4),
        ], className="g-3 mb-3"),
        html.Div([
            html.Span("Extent  ", style={"color": C["muted"], "fontSize": "12px"}),
            html.Span(
                f"Lat {lat_min:.4f} → {lat_max:.4f}   "
                f"Lon {lon_min:.4f} → {lon_max:.4f}   "
                f"{df[gid].nunique() if gid else '?'} grid cells",
                style={"color": C["cyan"], "fontFamily": "JetBrains Mono, monospace",
                       "fontSize": "12px"},
            ),
        ], className="info-row", style={"marginBottom": "10px"}),
        card_wrap(dcc.Graph(id="map-chart", config={"scrollZoom": True, "displayModeBar": True})),
    ])


def page_timeseries(df):
    indicators       = get_indicator_cols(df)
    lat, lon, date, gid = get_special_cols(df)
    dr_picker = []
    if date:
        d0, d1 = df[date].min().date(), df[date].max().date()
        dr_picker = [dbc.Col([
            ctrl_label("Date range"),
            dcc.DatePickerRange(id="ts-date-range",
                                min_date_allowed=str(d0), max_date_allowed=str(d1),
                                start_date=str(d0), end_date=str(d1),
                                display_format="YYYY-MM-DD", className="dark-date-picker"),
        ], md=6)]
    return html.Div([
        section_header("Time Series — Daily Spatial Mean"),
        dbc.Row([
            dbc.Col([ctrl_label("Indicators"),
                     dcc.Dropdown(id="ts-indicators",
                                  options=[{"label": c, "value": c} for c in indicators],
                                  value=indicators[:4], multi=True,
                                  placeholder="Select indicators…",
                                  className="dark-dropdown")], md=6),
        ] + dr_picker, className="g-3 mb-3"),
        card_wrap(dcc.Graph(id="ts-chart")),
    ])


def page_correlations(df):
    indicators = get_indicator_cols(df)
    return html.Div([
        section_header("Correlation Matrix"),
        dbc.Row([dbc.Col([
            ctrl_label("Select indicators (≥ 2)"),
            dcc.Dropdown(id="corr-indicators",
                         options=[{"label": c, "value": c} for c in indicators],
                         value=indicators[:14], multi=True,
                         placeholder="Select indicators…",
                         className="dark-dropdown"),
        ], md=8)], className="g-3 mb-3"),
        card_wrap(dcc.Graph(id="corr-chart")),
    ])


def page_distributions(df):
    indicators = get_indicator_cols(df)
    return html.Div([
        section_header("Distributions — Box Plots"),
        dbc.Row([dbc.Col([
            ctrl_label("Select indicators"),
            dcc.Dropdown(id="box-indicators",
                         options=[{"label": c, "value": c} for c in indicators],
                         value=indicators[:12], multi=True,
                         placeholder="Select indicators…",
                         className="dark-dropdown"),
        ], md=8)], className="g-3 mb-3"),
        card_wrap(dcc.Graph(id="box-chart")),
    ])


def page_report(df):
    indicators       = get_indicator_cols(df)
    lat, lon, date, gid = get_special_cols(df)
    dr = "N/A"
    if date:
        d0, d1 = df[date].min(), df[date].max()
        if pd.notna(d0):
            dr = f"{d0.strftime('%Y-%m-%d')} → {d1.strftime('%Y-%m-%d')}"

    return html.Div([
        section_header("Report Generator"),
        card_wrap(dbc.Row([dbc.Col([
            html.Div("Report Preview", style={"color": C["muted"], "fontSize": "12px",
                                              "textTransform": "uppercase",
                                              "letterSpacing": "0.08em",
                                              "marginBottom": "16px"}),
            dbc.Row([
                dbc.Col(metric_card(len(indicators), "Indicators",  C["cyan"]),   md=4),
                dbc.Col(metric_card(df[gid].nunique() if gid else "—",
                                    "Grid Cells",  C["green"]),                    md=4),
                dbc.Col(metric_card(f"{len(df):,}", "Records",      C["orange"]),  md=4),
            ], className="g-2 mb-3"),
            html.Div([
                html.Span("Date Range  ", style={"color": C["muted"], "fontSize": "13px"}),
                html.Span(dr, style={"color": C["cyan"], "fontSize": "13px"}),
            ], style={"marginBottom": "24px"}),
            dbc.Button("⎙  Download HTML Report", id="btn-report",
                       style={"background": C["cyan"], "border": "none",
                              "color": C["bg"], "fontWeight": "600",
                              "borderRadius": "6px", "padding": "10px 24px"}),
            html.Div(id="report-status",
                     style={"color": C["muted"], "fontSize": "12px", "marginTop": "10px"}),
        ], md=8)])),
    ])


def page_about():
    def mod_badge(name, ok):
        return html.Div([
            html.Span("●", style={"color": C["green"] if ok else C["orange"],
                                   "fontSize": "9px", "marginRight": "6px"}),
            html.Span(name, style={"fontFamily": "JetBrains Mono, monospace",
                                    "fontSize": "12px", "color": C["text"]}),
            html.Span(" ✓" if ok else " not installed",
                      style={"fontSize": "10px",
                             "color": C["green"] if ok else C["orange"]}),
        ], className="mod-badge")

    supported_formats = [
        ("CSV",     C["cyan"],   "Comma-separated values"),
        ("TSV",     C["green"],  "Tab-separated values"),
        ("Excel",   C["orange"], ".xlsx / .xls spreadsheets"),
        ("JSON",    C["purple"], "Flat or records-oriented"),
        ("Parquet", C["pink"],   "Columnar binary format"),
        ("URL",     C["cyan"],   "Direct HTTP/HTTPS ingestion"),
    ]

    data_sources = [
        "Sentinel-2 (10 m optical)",
        "MODIS Terra/Aqua",
        "ERA5-Land (hourly reanalysis)",
        "Sentinel-5P (trace gases)",
        "VIIRS Nighttime Lights",
        "Dynamic World (LULC)",
        "SRTM Digital Elevation",
        "OpenStreetMap Roads (SQLite DB)",
    ]

    return html.Div([
        html.Div([
            html.Div([
                html.Span(APP_NAME, style={
                    "fontSize": "52px", "fontWeight": "800",
                    "background": f"linear-gradient(90deg, {C['cyan']}, {C['green']})",
                    "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                    "letterSpacing": "0.04em", "lineHeight": "1",
                }),
                html.Div(APP_FULL, style={
                    "fontSize": "13px", "color": C["muted"], "marginTop": "6px",
                    "textTransform": "uppercase", "letterSpacing": "0.10em",
                }),
                html.Div([
                    html.Span(f"Version {APP_VERSION}", className="version-badge-lg"),
                    html.Span(APP_PROJECT, style={
                        "fontSize": "11px", "color": C["orange"], "fontWeight": "600",
                        "background": "rgba(255,159,0,0.10)",
                        "border": "1px solid rgba(255,159,0,0.25)",
                        "borderRadius": "20px", "padding": "3px 10px", "marginLeft": "8px",
                    }),
                ], style={"marginTop": "14px", "display": "flex", "alignItems": "center"}),
                html.Div(
                    f"{APP_NAME} extracts, visualises and reports on 40+ environmental and "
                    "climate proxy indicators from Google Earth Engine across custom grid cells. "
                    f"Part of the {APP_PROJECT} research programme on child health and "
                    "environmental exposure.",
                    style={"color": C["muted"], "fontSize": "13px", "maxWidth": "600px",
                           "lineHeight": "1.8", "marginTop": "18px"},
                ),
            ]),
        ], className="about-hero"),

        html.Hr(style={"borderColor": C["border"], "margin": "24px 0"}),

        dbc.Row([
            dbc.Col([card_wrap(html.Div([
                html.Div("Developed By", className="about-section-label"),
                html.Div("CHEAQI-MNCH Research Team", style={
                    "color": C["text"], "fontSize": "14px", "fontWeight": "600",
                    "marginBottom": "4px",
                }),
                html.Div("Child Health, Environment & Air Quality Index",
                         style={"color": C["muted"], "fontSize": "12px"}),
                html.Div("Maternal, Neonatal & Child Health Programme",
                         style={"color": C["muted"], "fontSize": "12px",
                                "marginBottom": "14px"}),
                html.Div("Contact", className="about-section-label"),
                html.A("hbnyoni@gmail.com", href="mailto:hbnyoni@gmail.com",
                       style={"color": C["cyan"], "fontSize": "12px",
                              "textDecoration": "none"}),
                html.Hr(style={"borderColor": C["border"], "margin": "14px 0"}),
                html.Div("Release Info", className="about-section-label"),
                html.Div([
                    html.Div([html.Span("Version  ", style={"color": C["muted"]}),
                              html.Span(APP_VERSION, style={"color": C["cyan"],
                                        "fontFamily": "JetBrains Mono, monospace"})]),
                    html.Div([html.Span("Year     ", style={"color": C["muted"]}),
                              html.Span(APP_YEAR,    style={"color": C["text"],
                                        "fontFamily": "JetBrains Mono, monospace"})]),
                    html.Div([html.Span("Port     ", style={"color": C["muted"]}),
                              html.Span("8087",      style={"color": C["text"],
                                        "fontFamily": "JetBrains Mono, monospace"})]),
                    html.Div([html.Span("License  ", style={"color": C["muted"]}),
                              html.Span("MIT",       style={"color": C["green"],
                                        "fontFamily": "JetBrains Mono, monospace"})]),
                ], style={"fontSize": "12px", "lineHeight": "2.2"}),
            ], style={"padding": "4px"}))], md=4),

            dbc.Col([card_wrap(html.Div([
                html.Div("Supported Input Formats", className="about-section-label"),
                html.Div([
                    html.Div([
                        html.Span(fmt, style={"color": col, "fontWeight": "700",
                                               "fontFamily": "JetBrains Mono, monospace",
                                               "fontSize": "12px", "width": "60px",
                                               "display": "inline-block"}),
                        html.Span(desc, style={"color": C["muted"], "fontSize": "11px"}),
                    ], style={"marginBottom": "6px"})
                    for fmt, col, desc in supported_formats
                ], style={"marginBottom": "20px"}),
                html.Div("GEE Data Sources", className="about-section-label"),
                html.Div([
                    html.Div([
                        html.Span("◈ ", style={"color": C["cyan"], "fontSize": "10px"}),
                        html.Span(src, style={"color": C["muted"], "fontSize": "11px"}),
                    ], style={"marginBottom": "5px"})
                    for src in data_sources
                ]),
            ], style={"padding": "4px"}))], md=4),

            dbc.Col([card_wrap(html.Div([
                html.Div("Module Status", className="about-section-label"),
                html.Div("Core", style={"color": C["muted"], "fontSize": "10px",
                                         "textTransform": "uppercase",
                                         "letterSpacing": "0.08em",
                                         "marginBottom": "6px"}),
                *[mod_badge(m, True) for m in
                  ["dash", "pandas", "numpy", "plotly", "dash_bootstrap_components"]],
                html.Div("Optional", style={"color": C["muted"], "fontSize": "10px",
                                             "textTransform": "uppercase",
                                             "letterSpacing": "0.08em",
                                             "marginTop": "12px", "marginBottom": "6px"}),
                *[mod_badge(m, ok) for m, ok in MOD_STATUS.items()],
            ], style={"padding": "4px"}))], md=4),
        ], className="g-3"),

        html.Hr(style={"borderColor": C["border"], "margin": "8px 0 18px"}),
        html.Div("Built With", className="about-section-label"),
        html.Div([
            html.Span(t, className="tech-badge")
            for t in ["Python 3", "Dash", "Plotly", "Pandas", "NumPy",
                      "Google Earth Engine", "GeoPandas", "SQLite", "Bootstrap 5"]
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginTop": "10px"}),

        html.Div(
            f"© {APP_YEAR} {APP_PROJECT} Research Team  ·  {APP_NAME} v{APP_VERSION}  ·  MIT License",
            style={"color": C["border"], "fontSize": "11px", "marginTop": "28px",
                   "textAlign": "center", "paddingBottom": "16px"},
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-data",    "data"),
    Output("upload-status", "children"),
    Input("upload-data",    "contents"),
    State("upload-data",    "filename"),
    prevent_initial_call=True,
)
def cb_upload(contents, filename):
    if not contents:
        return no_update, no_update
    data, err = parse_upload(contents, filename)
    if err:
        return no_update, html.Span(f"✗ {err[:50]}", style={"color": C["red"]})
    df    = df_from_store(data)
    n_ind = len(get_indicator_cols(df))
    return data, html.Span(
        f"✓ {filename}  ·  {len(df):,} rows  ·  {n_ind} vars",
        style={"color": C["green"]},
    )


@app.callback(
    Output("store-data",    "data",     allow_duplicate=True),
    Output("upload-status", "children", allow_duplicate=True),
    Output("url-status",    "children"),
    Input("btn-url-load",   "n_clicks"),
    State("url-input",      "value"),
    prevent_initial_call=True,
)
def cb_url_load(n, url):
    if not n or not url:
        return no_update, no_update, no_update
    if not MOD_STATUS["requests"]:
        return no_update, no_update, html.Span(
            "pip install requests", style={"color": C["orange"]}
        )
    try:
        import requests as _req
        resp  = _req.get(url.strip(), timeout=30)
        resp.raise_for_status()
        fname = url.strip().split("?")[0].rstrip("/").split("/")[-1] or "data.csv"
        if "." not in fname:
            fname += ".csv"
        df = parse_bytes(resp.content, fname)
        _, _, date, _ = get_special_cols(df)
        if date:
            df[date] = pd.to_datetime(df[date], errors="coerce")
        data  = df.to_json(date_format="iso", orient="split")
        n_ind = len(get_indicator_cols(df))
        msg   = html.Span(f"✓ {fname}  ·  {len(df):,} rows  ·  {n_ind} vars",
                          style={"color": C["green"]})
        return data, msg, html.Span("✓ Loaded", style={"color": C["green"]})
    except Exception as exc:
        return no_update, no_update, html.Span(
            f"✗ {str(exc)[:55]}", style={"color": C["red"]}
        )


@app.callback(
    Output("store-grid-csv", "data"),
    Output("start-msg",      "children", allow_duplicate=True),
    Input("upload-grid-csv", "contents"),
    State("upload-grid-csv", "filename"),
    prevent_initial_call=True,
)
def cb_grid_csv_upload(contents, filename):
    if not contents:
        return no_update, no_update
    return contents, html.Span(f"✓ Grid CSV: {filename}", style={"color": C["green"]})


@app.callback(
    Output("store-page", "data"),
    Input({"type": "nav-btn", "index": dash.ALL}, "n_clicks"),
    State("store-page", "data"),
    prevent_initial_call=True,
)
def cb_nav(_, current):
    if not ctx.triggered_id:
        return current
    return ctx.triggered_id["index"]


@app.callback(
    Output({"type": "nav-btn", "index": dash.ALL}, "className"),
    Input("store-page", "data"),
)
def cb_nav_style(page):
    return ["top-tab top-tab-active" if p == page else "top-tab"
            for p in ALL_PAGES]


@app.callback(
    Output("store-runner-state", "data"),
    Output("poll-interval",      "disabled"),
    Input("poll-interval",       "n_intervals"),
)
def cb_poll(_):
    st      = runner.get_state()
    disable = st["status"] not in ("running",)
    return st, disable


@app.callback(
    Output("main-content",      "children"),
    Input("store-page",         "data"),
    Input("store-data",         "data"),
    Input("store-runner-state", "data"),
    Input("store-grid-csv",     "data"),
)
def cb_page(page, data, runner_state, grid_csv):
    trigger = ctx.triggered_id
    if trigger == "store-runner-state" and page != "run":
        return no_update

    if page == "config":
        name = None
        return page_config(name)
    if page == "run":
        return page_run(runner_state)
    if page == "about":
        return page_about()

    if not data:
        return page_welcome()

    df = df_from_store(data)
    dispatch = {
        "overview":      page_overview,
        "map":           page_map,
        "timeseries":    page_timeseries,
        "correlations":  page_correlations,
        "distributions": page_distributions,
        "report":        page_report,
    }
    return dispatch.get(page, page_overview)(df)


@app.callback(
    Output("sidebar-info", "children"),
    Input("store-data",    "data"),
)
def cb_sidebar_info(data):
    if not data:
        return html.Div("No data loaded",
                        style={"color": C["border"], "fontSize": "11px",
                               "textAlign": "center", "padding": "8px 0"})
    df              = df_from_store(data)
    lat, lon, date, gid = get_special_cols(df)
    ind   = get_indicator_cols(df)
    items = [("Rows",  f"{len(df):,}"),
             ("Cells", str(df[gid].nunique()) if gid else "—"),
             ("Vars",  str(len(ind)))]
    if date:
        d0, d1 = df[date].min(), df[date].max()
        if pd.notna(d0):
            items += [("From", d0.strftime("%Y-%m-%d")),
                      ("To",   d1.strftime("%Y-%m-%d"))]
    return html.Div([
        html.Div([
            html.Span(k + "  ", style={"color": C["muted"], "fontSize": "10.5px"}),
            html.Span(v, style={"color": C["text"], "fontSize": "10.5px",
                                "fontFamily": "JetBrains Mono, monospace"}),
        ], style={"marginBottom": "4px"})
        for k, v in items
    ], style={"background": C["atm"], "border": f"1px solid {C['border']}",
               "borderRadius": "6px", "padding": "9px 12px"})


@app.callback(
    Output("start-msg",     "children"),
    Output("store-page",    "data",    allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Input("btn-start",      "n_clicks"),
    State("cfg-gee-project", "value"),
    State("store-grid-csv",  "data"),
    State("cfg-col-id",      "value"),
    State("cfg-col-lat",     "value"),
    State("cfg-col-lon",     "value"),
    State("cfg-col-date",    "value"),
    State("cfg-col-src",     "value"),
    State("cfg-date-from",   "value"),
    State("cfg-date-to",     "value"),
    State("cfg-max-workers", "value"),
    State("cfg-var-workers", "value"),
    State({"type": "grp-check", "index": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def cb_start(n, gee_proj, grid_csv_b64, col_id, col_lat, col_lon,
             col_date, col_src, date_from, date_to, max_workers, var_workers, grp_vals):
    if not n:
        return no_update, no_update, no_update
    if not gee_proj:
        return html.Span("GEE project is required.", style={"color": C["red"]}), no_update, no_update
    if not grid_csv_b64:
        return html.Span("Upload a grid CSV first.", style={"color": C["red"]}), no_update, no_update

    # Write grid CSV to temp file
    try:
        _, content_str = grid_csv_b64.split(",")
        csv_bytes = base64.b64decode(content_str)
        tmp_dir  = tempfile.mkdtemp(prefix="gipex_")
        grid_csv = os.path.join(tmp_dir, "grid_cells.csv")
        with open(grid_csv, "wb") as f:
            f.write(csv_bytes)
    except Exception as exc:
        return html.Span(f"CSV error: {exc}", style={"color": C["red"]}), no_update, no_update

    grp_keys   = list(VARIABLE_GROUPS.keys())
    sel_groups = [grp_keys[i] for i, v in enumerate(grp_vals) if v]
    if not sel_groups:
        return html.Span("Select at least one variable group.", style={"color": C["red"]}), no_update, no_update

    output_dir = tempfile.mkdtemp(prefix="gipex_out_")
    config = dict(
        gee_project = gee_proj,
        grid_csv    = grid_csv,
        output_dir  = output_dir,
        roads_shp   = OSM_ROADS_PATH if os.path.exists(OSM_ROADS_PATH) else "",
        col_id      = col_id   or "cell_id",
        col_lat     = col_lat  or "lat",
        col_lon     = col_lon  or "lon",
        col_date    = col_date or "",
        col_src     = col_src  or "",
        date_from   = date_from or "2022-01-01",
        date_to     = date_to   or "2023-01-01",
        max_workers = int(max_workers or 50),
        var_workers = int(var_workers or 6),
        var_groups  = sel_groups,
    )
    result = runner.start(config)
    if result == "already_running":
        return html.Span("Already running.", style={"color": C["orange"]}), no_update, no_update
    return (html.Span("Started — switching to Run page …", style={"color": C["green"]}),
            "run", False)


@app.callback(
    Output("btn-stop", "disabled"),
    Input("btn-stop",  "n_clicks"),
    prevent_initial_call=True,
)
def cb_stop(n):
    if n:
        runner.stop()
    return True


@app.callback(
    Output("store-data",      "data",     allow_duplicate=True),
    Output("upload-status",   "children", allow_duplicate=True),
    Output("store-page",      "data",     allow_duplicate=True),
    Input("btn-load-results", "n_clicks"),
    prevent_initial_call=True,
)
def cb_load_results(n):
    if not n:
        return no_update, no_update, no_update
    st   = runner.get_state()
    path = st.get("result_path", "")
    if not path or not os.path.exists(path):
        return no_update, html.Span("Result file not found.", style={"color": C["red"]}), no_update
    try:
        df = pd.read_csv(path)
        _, _, date, _ = get_special_cols(df)
        if date:
            df[date] = pd.to_datetime(df[date], errors="coerce")
        data  = df.to_json(date_format="iso", orient="split")
        n_ind = len(get_indicator_cols(df))
        msg   = html.Span(f"✓ {len(df):,} rows  ·  {n_ind} vars",
                          style={"color": C["green"]})
        return data, msg, "overview"
    except Exception as exc:
        return no_update, html.Span(str(exc), style={"color": C["red"]}), no_update


@app.callback(
    Output("download-result-csv", "data"),
    Input("btn-download-result",  "n_clicks"),
    prevent_initial_call=True,
)
def cb_download_result(n):
    if not n:
        return no_update
    st   = runner.get_state()
    path = st.get("result_path", "")
    if not path or not os.path.exists(path):
        return no_update
    with open(path, "rb") as f:
        content = f.read()
    fname = f"{APP_NAME}_indicators_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return dict(content=base64.b64encode(content).decode(),
                filename=fname, type="text/csv", base64=True)


@app.callback(
    Output("map-chart",      "figure"),
    Input("map-indicator",   "value"),
    Input("map-show-roads",  "value"),
    Input("map-road-detail", "value"),
    State("store-data",      "data"),
    prevent_initial_call=True,
)
def cb_map(indicator, show_roads, road_detail, data):
    if not data:
        return go.Figure().update_layout(**PLOT_LAYOUT)
    return fig_map(df_from_store(data),
                   None if indicator == "__none__" else indicator,
                   show_roads=bool(show_roads),
                   major_roads_only=(road_detail != "all"))


@app.callback(
    Output("ts-chart",     "figure"),
    Input("ts-indicators", "value"),
    Input("ts-date-range", "start_date"),
    Input("ts-date-range", "end_date"),
    State("store-data",    "data"),
    prevent_initial_call=True,
)
def cb_ts(indicators, start, end, data):
    if not data or not indicators:
        return go.Figure().update_layout(**PLOT_LAYOUT, title="Select indicators")
    return fig_timeseries(df_from_store(data), indicators, date_range=[start, end])


@app.callback(
    Output("corr-chart",     "figure"),
    Input("corr-indicators", "value"),
    State("store-data",      "data"),
    prevent_initial_call=True,
)
def cb_corr(indicators, data):
    if not data or not indicators:
        return go.Figure().update_layout(**PLOT_LAYOUT)
    return fig_correlation(df_from_store(data), indicators)


@app.callback(
    Output("box-chart",     "figure"),
    Input("box-indicators", "value"),
    State("store-data",     "data"),
    prevent_initial_call=True,
)
def cb_box(indicators, data):
    if not data or not indicators:
        return go.Figure().update_layout(**PLOT_LAYOUT)
    return fig_boxplots(df_from_store(data), indicators)


@app.callback(
    Output("download-report", "data"),
    Output("report-status",   "children"),
    Input("btn-report",       "n_clicks"),
    State("store-data",       "data"),
    prevent_initial_call=True,
)
def cb_report(n, data):
    if not data:
        return no_update, html.Span("No data.", style={"color": C["red"]})
    try:
        html_str = make_report(df_from_store(data))
        fname    = f"{APP_NAME}_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        return (dict(content=html_str, filename=fname, type="text/html"),
                html.Span("Downloading …", style={"color": C["green"]}))
    except Exception as exc:
        return no_update, html.Span(str(exc), style={"color": C["red"]})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8087, debug=False)
