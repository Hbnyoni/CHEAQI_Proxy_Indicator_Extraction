# =============================================================================
# GIPEX — Geospatial Indicators for Proxy Environmental eXposure
# CHEAQI-MNCH  ·  v2.1  ·  2026
# Dash application — port 8087
# =============================================================================

import io
import base64
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
    "xarray":          _mod_ok("xarray"),
    "netCDF4":         _mod_ok("netCDF4"),
    "folium":          _mod_ok("folium"),
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

# ── Colour palette — deep black environmental theme ───────────────────────────
C = dict(
    bg     = "#000000",
    surf   = "#060A10",
    card   = "#0A1520",
    atm    = "#071018",
    border = "#152840",
    cyan   = "#00E5FF",   # air / water quality
    green  = "#39D353",   # vegetation / NDVI
    orange = "#FF8C00",   # heat / temperature
    red    = "#FF3B3B",   # pollution / fire
    purple = "#9B59B6",   # night lights
    pink   = "#FF6B9D",
    gold   = "#FFD700",   # solar radiation
    teal   = "#00C9A7",   # atmospheric
    text   = "#DDE6F0",
    muted  = "#526880",
)
CHART_COLORS = [C["cyan"], C["green"], C["orange"], C["red"],
                C["purple"], C["pink"], C["gold"], C["teal"], "#00BFFF", "#ADFF2F"]

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

RESOLUTION_OPTIONS = [
    {"label": "Point extraction (no buffer)",        "value": "0"},
    {"label": "100 m  — local / street scale",       "value": "100"},
    {"label": "250 m  — neighbourhood scale",        "value": "250"},
    {"label": "500 m  — district scale",             "value": "500"},
    {"label": "1 km   — city block / ERA5-Land",     "value": "1000"},
    {"label": "2 km   — intra-urban scale",          "value": "2000"},
    {"label": "5 km   — regional scale",             "value": "5000"},
    {"label": "10 km  — national / ERA5 scale",      "value": "10000"},
]

# ── Product resolution reference ──────────────────────────────────────────────
PRODUCT_RESOLUTIONS = [
    ("Sentinel-2",      "NDVI, NDBI, NDWI, MNDWI, SAVI, MSAVI, GCI, ARVI, EVI2", "10–20 m",  C["green"]),
    ("Dynamic World",   "DW_label, BuiltUp",                                       "10 m",     C["green"]),
    ("SRTM Terrain",    "Elevation, Slope",                                         "30 m",     C["teal"]),
    ("Landsat 8 C2",    "NDII (imperviousness)",                                    "30 m",     C["teal"]),
    ("VIIRS NTL",       "VIIRS_NTL (night-time lights)",                            "~500 m",   C["purple"]),
    ("MODIS",           "EVI, LST, ET, Fire, BurnedArea, Soil_Moist, NDVI_MO",     "500 m–1 km", C["orange"]),
    ("MODIS MCD19A2",   "SPM25 (PM2.5 proxy via AOD)",                              "1 km",     C["red"]),
    ("Sentinel-5P",     "NO2, AOD_S5P (aerosol index)",                            "~7 km",    C["red"]),
    ("ERA5-Land",       "T2M, DEW, TP, SP, U10, V10, SSR, Soil_Moist",            "~11 km (0.1°)", C["cyan"]),
    ("ERA5 Hourly",     "BLH, MSLP",                                               "~28 km (0.25°)", C["cyan"]),
    ("Derived (post)",  "WS, WD10 (wind), RH (humidity)",                          "inherits ERA5", C["teal"]),
    ("OSM Roads",       "EM_m, EH_m, WRND_km_km2",                                "vector", C["gold"]),
]

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
        elif ext in ("nc", "nc4", "netcdf"):
            import xarray as xr
            ds = xr.open_dataset(io.BytesIO(content_bytes))
            df = ds.to_dataframe().reset_index()
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


def df_to_netcdf_bytes(df):
    try:
        import xarray as xr
    except ImportError:
        return None
    lat, lon, date, gid = get_special_cols(df)
    indicators = get_indicator_cols(df)
    if not indicators:
        return None
    try:
        ds_vars = {}
        obs = np.arange(len(df))
        for ind in indicators:
            ds_vars[ind] = xr.DataArray(
                df[ind].values.astype(float),
                dims=["obs"],
                attrs={"long_name": ind},
            )
        coords = {"obs": obs}
        if lat:
            coords["latitude"]  = ("obs", df[lat].values.astype(float))
        if lon:
            coords["longitude"] = ("obs", df[lon].values.astype(float))
        if date:
            coords["time"] = ("obs", pd.to_datetime(df[date]).values)
        if gid:
            coords["grid_id"] = ("obs", df[gid].astype(str).values)
        ds = xr.Dataset(ds_vars, coords=coords, attrs={
            "title":       f"{APP_NAME} — Environmental Indicators",
            "institution": APP_PROJECT,
            "source":      f"{APP_NAME} v{APP_VERSION}",
            "history":     f"Created {datetime.now().isoformat()}",
            "Conventions": "CF-1.8",
        })
        buf = io.BytesIO()
        ds.to_netcdf(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Parallel stats
# ─────────────────────────────────────────────────────────────────────────────
def _col_stats(args):
    df, col = args
    s = df[col].dropna()
    return {
        "Indicator":  col,
        "N":          int(s.count()),
        "Coverage %": f"{(1 - df[col].isna().mean()) * 100:.1f}",
        "Mean":       f"{s.mean():.4g}"   if len(s) else "—",
        "Std":        f"{s.std():.4g}"    if len(s) else "—",
        "Min":        f"{s.min():.4g}"    if len(s) else "—",
        "P25":        f"{s.quantile(.25):.4g}" if len(s) else "—",
        "Median":     f"{s.median():.4g}" if len(s) else "—",
        "P75":        f"{s.quantile(.75):.4g}" if len(s) else "—",
        "Max":        f"{s.max():.4g}"    if len(s) else "—",
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
# Folium interactive map
# ─────────────────────────────────────────────────────────────────────────────
def generate_folium_html(df, indicator=None):
    if not MOD_STATUS["folium"]:
        return ("<html><body style='background:#000;color:#526880;font-family:sans-serif;"
                "padding:60px;text-align:center;'>"
                "<h3 style='color:#FF8C00;'>folium not installed</h3>"
                "<p>Run: <code style='color:#00E5FF;'>pip install folium branca</code></p>"
                "</body></html>")
    import folium
    try:
        import branca.colormap as cm_branca
        _have_branca = True
    except ImportError:
        _have_branca = False

    lat_col, lon_col, date_col, gid_col = get_special_cols(df)
    if not lat_col or not lon_col:
        return "<html><body style='background:#000;color:#526880;padding:40px;'>No lat/lon columns.</body></html>"

    grp = [c for c in [lat_col, lon_col, gid_col] if c]
    if indicator and indicator in df.columns:
        plot_df = (df.groupby(grp, as_index=False)[indicator].mean()
                     .dropna(subset=[lat_col, lon_col]))
    else:
        plot_df = df[grp].drop_duplicates().dropna(subset=[lat_col, lon_col])

    if plot_df.empty:
        return "<html><body style='background:#000;color:#526880;padding:40px;'>No valid coordinates.</body></html>"

    center_lat = float(plot_df[lat_col].mean())
    center_lon = float(plot_df[lon_col].mean())
    span       = max(
        float(plot_df[lat_col].max() - plot_df[lat_col].min()),
        float(plot_df[lon_col].max() - plot_df[lon_col].min()),
    )
    zoom = max(4, min(14, int(8.5 - np.log2(span + 0.001))))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap © CARTO",
        name="CartoDB Dark",
        subdomains="abcd",
    ).add_to(m)

    colormap = None
    if indicator and indicator in plot_df.columns and _have_branca:
        vals = plot_df[indicator].dropna()
        if len(vals) > 1:
            vmin = float(vals.quantile(0.02))
            vmax = float(vals.quantile(0.98))
            if vmin == vmax:
                vmin, vmax = float(vals.min()), float(vals.max())
            ind_l = indicator.lower()
            if any(k in ind_l for k in ["temp", "lst", "t2m", "heat", "dew"]):
                colors = ["#0022FF", "#00AAFF", "#FFFF00", "#FF6600", "#FF0000"]
            elif any(k in ind_l for k in ["ndvi", "evi", "savi", "msavi", "gci", "arvi"]):
                colors = ["#8B4513", "#FFEE58", "#66BB6A", "#1B5E20"]
            elif any(k in ind_l for k in ["no2", "aod", "pm", "spm", "pollut", "air"]):
                colors = ["#00FF88", "#FFFF00", "#FF8800", "#FF0000", "#880088"]
            elif any(k in ind_l for k in ["ntl", "viirs", "light", "ntlight"]):
                colors = ["#000033", "#003388", "#0077FF", "#FFCC00", "#FFFFFF"]
            elif any(k in ind_l for k in ["ndbi", "builtup", "ndii", "imperv"]):
                colors = ["#4CAF50", "#FFEB3B", "#FF9800", "#F44336", "#880000"]
            elif any(k in ind_l for k in ["tp", "precip", "rain", "ndwi", "mndwi"]):
                colors = ["#FFF9C4", "#81D4FA", "#0288D1", "#01579B", "#1A237E"]
            else:
                colors = ["#00E5FF", "#0088FF", "#8800FF", "#FF0088", "#FF4400"]
            try:
                colormap = cm_branca.LinearColormap(
                    colors=colors, vmin=vmin, vmax=vmax,
                    caption=f"{indicator} (mean per grid cell)",
                )
                colormap.add_to(m)
            except Exception:
                colormap = None

    for _, row in plot_df.iterrows():
        val = row.get(indicator) if indicator else None
        if indicator and pd.isna(val):
            continue
        try:
            if colormap and val is not None:
                fill_color = colormap(float(val))
            else:
                fill_color = "#00E5FF"
        except Exception:
            fill_color = "#00E5FF"

        tip_val = f"{float(val):.4g}" if val is not None and not pd.isna(val) else "—"
        popup_html = (
            f"<div style='font-family:sans-serif;font-size:13px;'>"
            + (f"<b>Cell:</b> {row[gid_col]}<br>" if gid_col and gid_col in row.index else "")
            + f"<b>Lat:</b> {float(row[lat_col]):.5f}<br>"
            + f"<b>Lon:</b> {float(row[lon_col]):.5f}<br>"
            + (f"<b>{indicator}:</b> {tip_val}" if indicator else "")
            + "</div>"
        )
        folium.CircleMarker(
            location=[float(row[lat_col]), float(row[lon_col])],
            radius=7,
            color="rgba(0,0,0,0.25)",
            weight=0.5,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{indicator}: {tip_val}" if indicator else f"({float(row[lat_col]):.4f}, {float(row[lon_col]):.4f})",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()


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

    cells    = df[gid].nunique() if gid else "N/A"
    lat_rng  = f"{float(df[lat].min()):.4f} → {float(df[lat].max()):.4f}" if lat else "N/A"
    lon_rng  = f"{float(df[lon].min()):.4f} → {float(df[lon].max()):.4f}" if lon else "N/A"
    cov_mean = np.mean([(1 - df[c].isna().mean()) * 100 for c in indicators]) if indicators else 0
    missing  = [c for c in indicators if df[c].isna().mean() > 0.2]
    miss_str = ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "") if missing else "None"

    quality_rows = [
        ("Total records",          f"{len(df):,}"),
        ("Grid cells",             str(cells)),
        ("Indicators extracted",   str(len(indicators))),
        ("Date range",             dr),
        ("Latitude extent",        lat_rng),
        ("Longitude extent",       lon_rng),
        ("Mean coverage",          f"{cov_mean:.1f}%"),
        ("High-missing vars (>20%)", miss_str),
    ]
    qual_html = "".join(
        f"<tr><td style='color:var(--muted);padding:7px 14px;border-bottom:1px solid var(--border);'>{k}</td>"
        f"<td style='padding:7px 14px;border-bottom:1px solid var(--border);'>{v}</td></tr>"
        for k, v in quality_rows
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><title>{APP_NAME} Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root{{--cyan:#00E5FF;--green:#39D353;--orange:#FF8C00;--bg:#000000;--surf:#060A10;
           --card:#0A1520;--atm:#071018;--border:#152840;--text:#DDE6F0;--muted:#526880;}}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:var(--bg);color:var(--text);font-family:Inter,sans-serif;padding:40px 48px;line-height:1.6;}}
    h1{{color:var(--cyan);font-size:26px;border-bottom:2px solid var(--cyan);padding-bottom:12px;margin-bottom:6px;}}
    h2{{color:var(--green);font-size:18px;margin:40px 0 16px;border-left:3px solid var(--green);padding-left:10px;}}
    h3{{color:var(--orange);font-size:14px;margin:24px 0 12px;}}
    .meta{{color:var(--muted);font-size:13px;margin-bottom:28px;}}
    .meta span{{color:var(--text);font-weight:500;margin-left:4px;}}
    .chart-wrap{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px;margin-bottom:28px;}}
    .report-table{{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border-radius:8px;overflow:hidden;}}
    .report-table th{{background:var(--atm);color:var(--cyan);padding:9px 14px;text-align:left;font-weight:600;}}
    .report-table td{{padding:8px 14px;border-bottom:1px solid var(--border);}}
    .report-table tr:last-child td{{border-bottom:none;}}
    .report-table tr:hover td{{background:var(--surf);}}
    table.qual{{width:100%;border-collapse:collapse;background:var(--card);border-radius:8px;overflow:hidden;font-size:13px;}}
    .footer{{color:var(--muted);font-size:12px;margin-top:56px;border-top:1px solid var(--border);padding-top:14px;}}
  </style>
</head>
<body>
<h1>{APP_NAME} — Environmental Exposure Indicators Report</h1>
<div class="meta">
  Generated: <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span> &nbsp;·&nbsp;
  Records: <span>{len(df):,}</span> &nbsp;·&nbsp;
  Grid Cells: <span>{cells}</span> &nbsp;·&nbsp;
  Indicators: <span>{len(indicators)}</span> &nbsp;·&nbsp;
  Date Range: <span>{dr}</span>
</div>
<h2>Dataset Quality Summary</h2>
<div class="chart-wrap">
  <table class="qual">{qual_html}</table>
</div>
<h2>Per-Indicator Summary Statistics</h2>
<div class="chart-wrap">{tbl}</div>
<h2>Spatial Distribution — {indicators[0] if indicators else 'Grid cells'}</h2>
<div class="chart-wrap">{fmap.to_html(full_html=False, include_plotlyjs=False)}</div>
<h2>Time Series (top 6 indicators)</h2>
<div class="chart-wrap">{fts.to_html(full_html=False, include_plotlyjs=False)}</div>
<h2>Correlation Matrix</h2>
<div class="chart-wrap">{fcorr.to_html(full_html=False, include_plotlyjs=False)}</div>
<h2>Distributions</h2>
<div class="chart-wrap">{fbox.to_html(full_html=False, include_plotlyjs=False)}</div>
<div class="footer">{APP_PROJECT} · {APP_NAME} {APP_FULL} · {datetime.now().year} · MIT License</div>
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
ALL_PAGES = ["pipeline", "overview", "map", "envmap",
             "timeseries", "correlations", "distributions", "report", "about"]

_TAB_GROUPS = [
    ("Pipeline", C["orange"], [
        ("⚙", "pipeline", "Pipeline"),
    ]),
    ("Dashboard", C["cyan"], [
        ("▦", "overview",      "Overview"),
        ("🗺", "map",           "Map"),
        ("🌍", "envmap",        "Env Map"),
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
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
sidebar = html.Div([
    html.Div([
        html.Div([
            html.Div("🌍", style={"fontSize": "22px", "marginBottom": "2px"}),
            html.Div([
                html.Div(APP_NAME, style={
                    "fontSize": "22px", "fontWeight": "800",
                    "background": f"linear-gradient(90deg, {C['cyan']}, {C['green']})",
                    "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                    "letterSpacing": "0.08em",
                }),
                html.Div("Environmental Exposure Platform", style={
                    "fontSize": "7px", "color": C["muted"],
                    "textTransform": "uppercase", "letterSpacing": "0.06em",
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

    html.Div([
        html.Div("Load Results / Dataset", className="sidebar-section-label"),
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                html.Div("⬆", style={"fontSize": "16px", "marginBottom": "2px"}),
                html.Div("CSV · TSV · Excel · JSON · Parquet · NC",
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
# Pipeline run-progress and output-section sub-renderers
# ─────────────────────────────────────────────────────────────────────────────
def _run_progress_content(st):
    if not st:
        return None
    status   = st["status"]
    progress = st["progress"]
    total    = st["total"]
    pct      = st["pct"]
    logs     = st.get("logs", [])
    elapsed  = st.get("elapsed", "")

    if status == "idle":
        return None

    status_color = {
        "running": C["cyan"], "done": C["green"],
        "error": C["red"],    "stopped": C["orange"],
    }.get(status, C["muted"])
    status_label = {
        "running": "⬤  Running …", "done": "⬤  Done",
        "error":   "⬤  Error",     "stopped": "⬤  Stopped",
    }.get(status, status)

    progress_bar = html.Div([html.Div(style={
        "height": "8px", "width": f"{pct}%",
        "background": f"linear-gradient(90deg, {C['cyan']}, {C['green']})",
        "borderRadius": "4px", "transition": "width 0.4s ease",
        "boxShadow": "0 0 10px rgba(0,229,255,0.5)" if pct > 0 else "none",
    })], style={"height": "8px", "width": "100%",
                "background": C["border"], "borderRadius": "4px", "marginBottom": "8px"})

    return card_wrap(
        dbc.Row([
            dbc.Col(html.Div([
                html.Span(status_label, style={"color": status_color, "fontSize": "14px",
                                               "fontWeight": "600"}),
                html.Span(f"  {elapsed}", style={"color": C["muted"], "fontSize": "12px"}) if elapsed else None,
            ]), md=4),
            dbc.Col(html.Div(
                f"{progress:,} / {total:,} tasks  ({pct}%)",
                style={"color": C["text"], "fontSize": "13px",
                       "fontFamily": "JetBrains Mono, monospace"},
            ), md=5),
            dbc.Col(html.Div([
                dbc.Button("■  Stop", id="btn-stop", disabled=(status != "running"),
                           style={"background": C["red"], "border": "none", "color": C["text"],
                                  "fontWeight": "600", "borderRadius": "6px",
                                  "padding": "7px 18px", "fontSize": "12px",
                                  "opacity": "1" if status == "running" else "0.4"}),
            ]), md=3),
        ], className="g-3 align-items-center", style={"padding": "8px 4px"}),
        html.Div([
            progress_bar,
            html.Div(f"{pct}% complete",
                     style={"color": C["muted"], "fontSize": "11px",
                            "textAlign": "right", "marginBottom": "12px"}),
            html.Div("Extraction Log", style={
                "color": C["muted"], "fontSize": "10px", "textTransform": "uppercase",
                "letterSpacing": "0.08em", "marginBottom": "8px",
            }),
            html.Pre("\n".join(logs[-60:]) if logs else "No log yet.", id="log-pre"),
        ], style={"padding": "4px"}),
    )


def _output_section_content(st):
    if not st:
        return None
    status = st.get("status")
    result = st.get("result_path", "")
    if status != "done" or not result or not os.path.exists(result):
        return None

    nc_avail = MOD_STATUS["xarray"]
    return card_wrap(
        html.Div([
            html.Div("Output & Download", className="config-section-header"),
            html.Div([
                html.Span("◉ Extraction complete  ", style={"color": C["green"], "fontSize": "12px"}),
                html.Span(os.path.basename(result),
                          style={"color": C["muted"], "fontSize": "11px",
                                 "fontFamily": "JetBrains Mono, monospace"}),
            ], style={"marginBottom": "16px"}),
            html.Div([
                dbc.Button("⬆  Load into Dashboard", id="btn-load-results",
                           style={"background": C["cyan"], "border": "none", "color": C["bg"],
                                  "fontWeight": "700", "borderRadius": "6px",
                                  "padding": "9px 20px", "fontSize": "13px",
                                  "marginRight": "10px", "marginBottom": "8px"}),
                dbc.Button("⬇  Download CSV", id="btn-download-result",
                           style={"background": C["green"], "border": "none", "color": C["bg"],
                                  "fontWeight": "700", "borderRadius": "6px",
                                  "padding": "9px 20px", "fontSize": "13px",
                                  "marginRight": "10px", "marginBottom": "8px"}),
                dbc.Button(
                    "⬇  Download NetCDF",
                    id="btn-download-nc",
                    disabled=not nc_avail,
                    title="" if nc_avail else "pip install xarray netcdf4",
                    style={"background": C["purple"] if nc_avail else C["muted"],
                           "border": "none", "color": C["bg"] if nc_avail else C["surf"],
                           "fontWeight": "700", "borderRadius": "6px",
                           "padding": "9px 20px", "fontSize": "13px",
                           "marginBottom": "8px",
                           "opacity": "1" if nc_avail else "0.5"},
                ),
            ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"}),
            html.Div(
                "NetCDF requires xarray + netcdf4: pip install xarray netcdf4" if not nc_avail else "",
                style={"color": C["orange"], "fontSize": "11px", "marginTop": "6px"},
            ),
        ], style={"padding": "8px 4px"}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="store-data"),
    dcc.Store(id="store-page", data="overview"),
    dcc.Store(id="store-runner-state"),
    dcc.Store(id="store-grid-csv"),
    dcc.Store(id="store-grid-csv-name"),
    dcc.Download(id="download-report"),
    dcc.Download(id="download-result-csv"),
    dcc.Download(id="download-result-nc"),
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
        (C["orange"], "🌡", "Heat & Temperature",     "LST, T2M, ERA5 thermal metrics"),
        (C["red"],    "💨", "Air Quality",             "NO₂, AOD, PM₂.₅ proxy (TROPOMI/MODIS)"),
        (C["green"],  "🌿", "Vegetation & LULC",       "NDVI, EVI, SAVI, Dynamic World"),
        (C["cyan"],   "💧", "Water & Precipitation",   "NDWI, MNDWI, ET, soil moisture"),
        (C["purple"], "🌃", "Urban & Night Lights",    "NDBI, NDII, BuiltUp, VIIRS NTL"),
        (C["teal"],   "🏔", "Terrain & Atmosphere",   "Elevation, Slope, BLH, MSLP, RH"),
    ]
    feat_cards = html.Div([
        html.Div([
            html.Div(icon, style={"fontSize": "22px", "marginBottom": "8px"}),
            html.Div(title, style={"color": color, "fontWeight": "600",
                                   "fontSize": "13px", "marginBottom": "4px"}),
            html.Div(sub,   style={"color": C["muted"], "fontSize": "11px", "lineHeight": "1.5"}),
        ], className="feature-card")
        for color, icon, title, sub in features
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "32px"})

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
                "A geospatial platform for extracting proxy environmental exposure indicators "
                "— heat, air quality, vegetation, water, urban intensity — across custom grids. "
                "Designed for settings without ground monitoring stations.",
                style={"color": C["muted"], "fontSize": "13px",
                       "marginTop": "18px", "maxWidth": "560px", "lineHeight": "1.8"},
            ),
            feat_cards,
        ], style={"position": "relative", "zIndex": "1"}),
    ], className="welcome-hero")])


def page_pipeline(runner_state=None, grid_csv_name=None):
    st = runner_state or runner.get_state()

    grp_checks = []
    for gk, gv in VARIABLE_GROUPS.items():
        grp_checks.append(
            html.Div([
                dbc.Checkbox(id={"type": "grp-check", "index": gk}, value=True, className="me-2"),
                html.Div([
                    html.Span(gv["label"], style={"color": C["text"], "fontSize": "13px",
                                                   "fontWeight": "500"}),
                    html.Span(f"  ({len(gv['vars'])} vars)",
                              style={"color": C["muted"], "fontSize": "11px"}),
                    html.Div(", ".join(gv["vars"]),
                             style={"color": C["muted"], "fontSize": "11px",
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

    csv_badge = html.Div([
        html.Span("◉ ", style={"color": C["green"]}),
        html.Span(grid_csv_name or "No grid file loaded yet",
                  style={"color": C["green"] if grid_csv_name else C["muted"],
                         "fontSize": "12px"}),
    ], style={
        "background": f"rgba(0,255,159,0.06)" if grid_csv_name else f"rgba(255,140,0,0.06)",
        "border": f"1px solid {'rgba(0,255,159,0.18)' if grid_csv_name else 'rgba(255,140,0,0.18)'}",
        "borderRadius": "8px", "padding": "8px 14px", "marginBottom": "10px",
        "display": "inline-flex", "alignItems": "center", "gap": "6px",
    })

    return html.Div([
        section_header("Pipeline",
                       "Upload grid data · configure GEE extraction · run · download results"),

        # ── Section 1: Data Input ──────────────────────────────────────────────
        card_wrap(html.Div([
            html.Div("1 · Input Data", className="config-section-header"),
            csv_badge,
            dcc.Upload(
                id="upload-grid-csv",
                children=html.Div([
                    html.Span("⬆  Upload grid file",
                              style={"fontSize": "12px", "color": C["muted"]}),
                    html.Span("  CSV (lat/lon/cell_id) · NC · Parquet",
                              style={"fontSize": "11px", "color": C["border"]}),
                ]),
                style={"border": f"1px dashed {C['border']}", "borderRadius": "8px",
                       "padding": "12px 18px", "cursor": "pointer",
                       "background": C["surf"], "transition": "all 0.2s", "marginBottom": "14px"},
                multiple=False,
            ),
            dbc.Row([
                dbc.Col([
                    ctrl_label("Input data type"),
                    dbc.RadioItems(
                        id="cfg-input-type",
                        options=[
                            {"label": "Point data (lat/lon columns)", "value": "point"},
                            {"label": "Gridded / NC file (preserve source grid)", "value": "grid"},
                        ],
                        value="point", inline=False,
                        style={"fontSize": "13px", "color": C["text"]},
                    ),
                ], md=6),
                dbc.Col([
                    ctrl_label("Grid resolution (point data only)"),
                    dcc.Dropdown(
                        id="cfg-grid-resolution",
                        options=RESOLUTION_OPTIONS,
                        value="0",
                        clearable=False,
                        className="dark-dropdown",
                    ),
                    html.Div(
                        "Buffer radius applied around each point for GEE extraction.",
                        style={"color": C["muted"], "fontSize": "10px", "marginTop": "5px"},
                    ),
                ], md=6),
            ], className="g-3"),
        ], style={"padding": "8px 4px"}), mb=14),

        # ── Section 2: GEE & Columns ──────────────────────────────────────────
        card_wrap(html.Div([
            html.Div("2 · GEE & Column Mapping", className="config-section-header"),
            dbc.Row([
                field("GEE Project", "cfg-gee-project", "ee-your-project", "ee-cheaqi"),
            ]),
            dbc.Row([
                field("Cell ID column",  "cfg-col-id",   "e.g. cell_id", "cell_id"),
                field("Latitude column", "cfg-col-lat",  "e.g. lat",     "lat"),
                field("Longitude col.",  "cfg-col-lon",  "e.g. lon",     "lon"),
                field("Date column",     "cfg-col-date",
                      "e.g. date_only (blank → use date range)", "date_only"),
                field("Source/ID col. (optional)", "cfg-col-src", "e.g. PID", "PID"),
            ]),
        ], style={"padding": "8px 4px"}), mb=14),

        # ── Section 3: Date Range & Workers ───────────────────────────────────
        card_wrap(html.Div([
            html.Div("3 · Date Range & Workers", className="config-section-header"),
            html.Div("Date range is used only when the grid has NO per-row date column.",
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
                         dbc.Input(id="cfg-max-workers", type="number", value=50, min=1, max=200,
                                   style={"background": C["card"], "color": C["text"],
                                          "border": f"1px solid {C['border']}",
                                          "borderRadius": "6px", "fontSize": "13px"})],
                        md=3, className="mb-3"),
                dbc.Col([ctrl_label("Var workers (inner)"),
                         dbc.Input(id="cfg-var-workers", type="number", value=6, min=1, max=40,
                                   style={"background": C["card"], "color": C["text"],
                                          "border": f"1px solid {C['border']}",
                                          "borderRadius": "6px", "fontSize": "13px"})],
                        md=3, className="mb-3"),
            ]),
        ], style={"padding": "8px 4px"}), mb=14),

        # ── Section 4: Variable Groups ────────────────────────────────────────
        card_wrap(html.Div([
            html.Div("4 · Variable Groups", className="config-section-header"),
            html.Div(
                "40+ satellite & reanalysis variables across heat, air quality, "
                "vegetation, water, urban and terrain domains.",
                style={"color": C["muted"], "fontSize": "11px", "marginBottom": "14px"},
            ),
            *grp_checks,
        ], style={"padding": "8px 4px"}), mb=14),

        # ── Start ─────────────────────────────────────────────────────────────
        html.Div([
            dbc.Button("▶  Start Extraction", id="btn-start",
                       style={"background": C["green"], "border": "none", "color": C["bg"],
                              "fontWeight": "700", "borderRadius": "6px",
                              "padding": "11px 28px", "fontSize": "14px"}),
            html.Div(id="start-msg",
                     style={"color": C["muted"], "fontSize": "12px",
                            "marginLeft": "16px", "display": "inline-block"}),
        ], style={"marginTop": "8px", "marginBottom": "20px"}),

        # ── Run progress (updated by cb_run_progress, not full page re-render) ─
        html.Div(_run_progress_content(st), id="run-progress-box"),

        # ── Output section (shown after completion) ───────────────────────────
        html.Div(_output_section_content(st), id="output-section-box"),
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
            {"if": {"filter_query": "{Coverage %} < 80"}, "color": C["orange"]},
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
        "background": "rgba(57,211,83,0.06)", "border": "1px solid rgba(57,211,83,0.18)",
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


def page_envmap(df):
    indicators = get_indicator_cols(df)
    opts = [{"label": "Grid cell locations (no indicator)", "value": "__none__"}] + \
           [{"label": c, "value": c} for c in indicators]

    folium_status = html.Div([
        html.Span("◉ ", style={"color": C["green"]}),
        html.Span("folium available — interactive map ready",
                  style={"color": C["green"], "fontSize": "12px"}),
    ], style={
        "background": "rgba(57,211,83,0.06)", "border": "1px solid rgba(57,211,83,0.18)",
        "borderRadius": "7px", "padding": "8px 14px", "marginBottom": "14px",
    }) if MOD_STATUS["folium"] else html.Div([
        html.Span("⚠ ", style={"color": C["orange"]}),
        html.Span("folium not installed — run: pip install folium branca",
                  style={"color": C["orange"], "fontSize": "12px"}),
    ], style={"padding": "8px 0", "marginBottom": "10px"})

    initial_html = generate_folium_html(df, indicators[0] if indicators else None)

    return html.Div([
        section_header("Environmental Exposure Map",
                       "Interactive Folium map — each grid cell coloured by selected indicator"),
        folium_status,
        dbc.Row([
            dbc.Col([
                ctrl_label("Environmental exposure indicator"),
                dcc.Dropdown(
                    id="folium-indicator",
                    options=opts,
                    value=indicators[0] if indicators else "__none__",
                    clearable=False,
                    className="dark-dropdown",
                ),
            ], md=6),
            dbc.Col(html.Div(
                "Colours adapt to indicator type (heat → thermal scale, "
                "vegetation → green, air quality → red/purple, etc.)",
                style={"color": C["muted"], "fontSize": "11px",
                       "paddingTop": "28px", "lineHeight": "1.6"},
            ), md=6),
        ], className="g-3 mb-3"),
        html.Div(
            html.Iframe(
                id="folium-map-frame",
                srcDoc=initial_html or "",
                style={"width": "100%", "height": "650px", "border": "none",
                       "borderRadius": "10px"},
            ),
            style={"border": f"1px solid {C['border']}", "borderRadius": "10px",
                   "overflow": "hidden"},
        ),
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
    cov_mean = np.mean([(1 - df[c].isna().mean()) * 100 for c in indicators]) if indicators else 0
    lat_rng  = (f"{float(df[lat].min()):.4f} → {float(df[lat].max()):.4f}") if lat else "N/A"
    lon_rng  = (f"{float(df[lon].min()):.4f} → {float(df[lon].max()):.4f}") if lon else "N/A"

    return html.Div([
        section_header("Report Generator",
                       "Comprehensive HTML report with statistics, maps and charts"),
        card_wrap(dbc.Row([dbc.Col([
            html.Div("Report Summary", style={"color": C["muted"], "fontSize": "12px",
                                              "textTransform": "uppercase",
                                              "letterSpacing": "0.08em",
                                              "marginBottom": "16px"}),
            dbc.Row([
                dbc.Col(metric_card(len(indicators), "Indicators",  C["cyan"]),   md=3),
                dbc.Col(metric_card(df[gid].nunique() if gid else "—",
                                    "Grid Cells",   C["green"]),                   md=3),
                dbc.Col(metric_card(f"{len(df):,}", "Records",      C["orange"]),  md=3),
                dbc.Col(metric_card(f"{cov_mean:.0f}%", "Coverage", C["teal"]),    md=3),
            ], className="g-2 mb-3"),
            html.Div([
                html.Div([html.Span("Date Range  ", style={"color": C["muted"], "fontSize": "13px"}),
                          html.Span(dr, style={"color": C["cyan"], "fontSize": "13px"})],
                         style={"marginBottom": "6px"}),
                html.Div([html.Span("Lat Extent  ", style={"color": C["muted"], "fontSize": "13px"}),
                          html.Span(lat_rng, style={"color": C["text"], "fontSize": "13px",
                                                     "fontFamily": "JetBrains Mono, monospace"})],
                         style={"marginBottom": "6px"}),
                html.Div([html.Span("Lon Extent  ", style={"color": C["muted"], "fontSize": "13px"}),
                          html.Span(lon_rng, style={"color": C["text"], "fontSize": "13px",
                                                     "fontFamily": "JetBrains Mono, monospace"})]),
            ], style={"marginBottom": "24px"}),
            dbc.Button("⎙  Download HTML Report", id="btn-report",
                       style={"background": C["cyan"], "border": "none",
                              "color": C["bg"], "fontWeight": "600",
                              "borderRadius": "6px", "padding": "10px 24px"}),
            html.Div(id="report-status",
                     style={"color": C["muted"], "fontSize": "12px", "marginTop": "10px"}),
        ], md=10)])),
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
                        "background": "rgba(255,140,0,0.10)",
                        "border": "1px solid rgba(255,140,0,0.25)",
                        "borderRadius": "20px", "padding": "3px 10px", "marginLeft": "8px",
                    }),
                ], style={"marginTop": "14px", "display": "flex", "alignItems": "center"}),
                html.Div(
                    f"{APP_NAME} is a geospatial platform for extracting and analysing "
                    "proxy environmental exposure indicators — including surface heat, air pollution "
                    "(NO₂, PM₂.₅, aerosol), vegetation indices, water indicators, built-up intensity, "
                    "night-time lights, and atmospheric conditions — across custom grid cells or study areas. "
                    "It is designed for use in settings where no ground-based environmental monitoring "
                    "stations exist, providing satellite-derived and reanalysis-based proxies suitable "
                    "for epidemiological and public health exposure assessment.",
                    style={"color": C["muted"], "fontSize": "13px", "maxWidth": "680px",
                           "lineHeight": "1.8", "marginTop": "18px"},
                ),
                html.Div([
                    html.Div([
                        html.Span(icon, style={"fontSize": "15px", "marginRight": "8px"}),
                        html.Span(label, style={"color": col, "fontWeight": "600",
                                                "fontSize": "12px", "marginRight": "6px"}),
                        html.Span(desc, style={"color": C["muted"], "fontSize": "11px"}),
                    ], style={"display": "flex", "alignItems": "center",
                              "background": C["surf"], "borderRadius": "6px",
                              "padding": "8px 12px", "border": f"1px solid {C['border']}",
                              "flex": "1 1 260px"})
                    for icon, label, desc, col in [
                        ("🌡", "Heat",        "LST, T2M, ERA5 thermal",         C["orange"]),
                        ("💨", "Air Quality", "NO₂, AOD, PM₂.₅ proxy",         C["red"]),
                        ("🌿", "Vegetation",  "NDVI, EVI, SAVI, DW land cover", C["green"]),
                        ("💧", "Water",       "NDWI, MNDWI, ET, precipitation", C["cyan"]),
                        ("🏙", "Urban",       "NDBI, NDII, Built-up, NTL",      C["purple"]),
                        ("🌤", "Atmosphere",  "BLH, MSLP, RH, wind speed",      C["teal"]),
                    ]
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginTop": "20px"}),
            ]),
        ], className="about-hero"),

        html.Hr(style={"borderColor": C["border"], "margin": "24px 0"}),

        card_wrap(html.Div([
            html.Div("Product Resolutions", className="about-section-label"),
            html.Div(
                "Spatial resolution of each satellite/reanalysis product extracted via Google Earth Engine.",
                style={"color": C["muted"], "fontSize": "11px", "marginBottom": "14px"},
            ),
            dash_table.DataTable(
                data=[
                    {"Dataset": src, "Indicators": inds, "Resolution": res}
                    for src, inds, res, _ in PRODUCT_RESOLUTIONS
                ],
                columns=[
                    {"name": "Dataset",     "id": "Dataset"},
                    {"name": "Indicators",  "id": "Indicators"},
                    {"name": "Resolution",  "id": "Resolution"},
                ],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": C["atm"], "color": C["cyan"],
                              "fontWeight": "600", "border": f"1px solid {C['border']}",
                              "fontSize": "12px"},
                style_cell={"backgroundColor": C["card"], "color": C["text"],
                            "border": f"1px solid {C['border']}",
                            "fontFamily": "Inter, sans-serif", "fontSize": "12px",
                            "padding": "8px 14px", "textAlign": "left"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": C["surf"]},
                ],
                page_size=15,
            ),
        ], style={"padding": "4px"})),

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
                       style={"color": C["cyan"], "fontSize": "12px", "textDecoration": "none"}),
                html.Hr(style={"borderColor": C["border"], "margin": "14px 0"}),
                html.Div("Release", className="about-section-label"),
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
                html.Div("Module Status", className="about-section-label"),
                html.Div("Core", style={"color": C["muted"], "fontSize": "10px",
                                         "textTransform": "uppercase",
                                         "letterSpacing": "0.08em", "marginBottom": "6px"}),
                *[mod_badge(m, True) for m in
                  ["dash", "pandas", "numpy", "plotly", "dash_bootstrap_components"]],
                html.Div("Optional", style={"color": C["muted"], "fontSize": "10px",
                                             "textTransform": "uppercase",
                                             "letterSpacing": "0.08em",
                                             "marginTop": "12px", "marginBottom": "6px"}),
                *[mod_badge(m, ok) for m, ok in MOD_STATUS.items()],
            ], style={"padding": "4px"}))], md=4),

            dbc.Col([card_wrap(html.Div([
                html.Div("Tech Stack", className="about-section-label"),
                html.Div([
                    html.Span(t, className="tech-badge", style={"marginBottom": "6px"})
                    for t in ["Python 3", "Dash", "Plotly", "Pandas", "NumPy",
                              "Google Earth Engine", "GeoPandas", "xarray", "Folium",
                              "SQLite", "Bootstrap 5", "NetCDF4"]
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}),
            ], style={"padding": "4px"}))], md=4),
        ], className="g-3"),

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
    Output("store-grid-csv",      "data"),
    Output("store-grid-csv-name", "data"),
    Output("start-msg",           "children", allow_duplicate=True),
    Input("upload-grid-csv",      "contents"),
    State("upload-grid-csv",      "filename"),
    prevent_initial_call=True,
)
def cb_grid_csv_upload(contents, filename):
    if not contents:
        return no_update, no_update, no_update
    return contents, filename, html.Span(f"✓ Grid file: {filename}",
                                          style={"color": C["green"]})


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
    Output("store-page",         "data",    allow_duplicate=True),
    Input("poll-interval",       "n_intervals"),
    State("store-runner-state",  "data"),
    State("store-page",          "data"),
    prevent_initial_call=True,
)
def cb_poll(_, prev_state, current_page):
    st      = runner.get_state()
    disable = st["status"] not in ("running",)
    new_page = no_update
    if (st["status"] == "done"
            and prev_state is not None
            and prev_state.get("status") == "running"
            and current_page == "pipeline"):
        new_page = "overview"
    return st, disable, new_page


@app.callback(
    Output("run-progress-box",   "children"),
    Output("output-section-box", "children"),
    Input("store-runner-state",  "data"),
    State("store-page",          "data"),
    prevent_initial_call=True,
)
def cb_pipeline_live(runner_state, page):
    if page != "pipeline":
        return no_update, no_update
    return _run_progress_content(runner_state), _output_section_content(runner_state)


@app.callback(
    Output("main-content",        "children"),
    Input("store-page",           "data"),
    Input("store-data",           "data"),
    Input("store-grid-csv-name",  "data"),
)
def cb_page(page, data, grid_csv_name):
    if page == "pipeline":
        return page_pipeline(runner.get_state(), grid_csv_name)
    if page == "about":
        return page_about()

    if not data:
        return page_welcome()

    df = df_from_store(data)
    dispatch = {
        "overview":      page_overview,
        "map":           page_map,
        "envmap":        page_envmap,
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
    Output("start-msg",     "children",  allow_duplicate=True),
    Output("poll-interval", "disabled",  allow_duplicate=True),
    Input("btn-start",      "n_clicks"),
    State("cfg-gee-project",    "value"),
    State("store-grid-csv",     "data"),
    State("cfg-col-id",         "value"),
    State("cfg-col-lat",        "value"),
    State("cfg-col-lon",        "value"),
    State("cfg-col-date",       "value"),
    State("cfg-col-src",        "value"),
    State("cfg-date-from",      "value"),
    State("cfg-date-to",        "value"),
    State("cfg-max-workers",    "value"),
    State("cfg-var-workers",    "value"),
    State("cfg-grid-resolution","value"),
    State({"type": "grp-check", "index": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def cb_start(n, gee_proj, grid_csv_b64, col_id, col_lat, col_lon,
             col_date, col_src, date_from, date_to, max_workers, var_workers,
             grid_resolution, grp_vals):
    if not n:
        return no_update, no_update
    if not gee_proj:
        return html.Span("GEE project is required.", style={"color": C["red"]}), no_update
    if not grid_csv_b64:
        return html.Span("Upload a grid file first.", style={"color": C["red"]}), no_update

    try:
        _, content_str = grid_csv_b64.split(",")
        csv_bytes = base64.b64decode(content_str)
        tmp_dir   = tempfile.mkdtemp(prefix="gipex_")
        grid_csv  = os.path.join(tmp_dir, "grid_cells.csv")
        with open(grid_csv, "wb") as f:
            f.write(csv_bytes)
    except Exception as exc:
        return html.Span(f"File error: {exc}", style={"color": C["red"]}), no_update

    grp_keys   = list(VARIABLE_GROUPS.keys())
    sel_groups = [grp_keys[i] for i, v in enumerate(grp_vals) if v]
    if not sel_groups:
        return html.Span("Select at least one variable group.", style={"color": C["red"]}), no_update

    output_dir = tempfile.mkdtemp(prefix="gipex_out_")
    config = dict(
        gee_project       = gee_proj,
        grid_csv          = grid_csv,
        output_dir        = output_dir,
        roads_shp         = OSM_ROADS_PATH if os.path.exists(OSM_ROADS_PATH) else "",
        col_id            = col_id   or "cell_id",
        col_lat           = col_lat  or "lat",
        col_lon           = col_lon  or "lon",
        col_date          = col_date or "",
        col_src           = col_src  or "",
        date_from         = date_from or "2022-01-01",
        date_to           = date_to   or "2023-01-01",
        max_workers       = int(max_workers or 50),
        var_workers       = int(var_workers or 6),
        var_groups        = sel_groups,
        grid_resolution_m = int(grid_resolution or 0),
    )
    result = runner.start(config)
    if result == "already_running":
        return html.Span("Already running.", style={"color": C["orange"]}), no_update
    return (html.Span("Extraction started …", style={"color": C["green"]}), False)


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
    Output("download-result-nc", "data"),
    Input("btn-download-nc",     "n_clicks"),
    prevent_initial_call=True,
)
def cb_download_nc(n):
    if not n:
        return no_update
    st   = runner.get_state()
    path = st.get("result_path", "")
    if not path or not os.path.exists(path):
        return no_update
    try:
        df = pd.read_csv(path)
        nc_bytes = df_to_netcdf_bytes(df)
        if not nc_bytes:
            return no_update
        fname = f"{APP_NAME}_indicators_{datetime.now().strftime('%Y%m%d_%H%M')}.nc"
        return dict(content=base64.b64encode(nc_bytes).decode(),
                    filename=fname, type="application/octet-stream", base64=True)
    except Exception:
        return no_update


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
    Output("folium-map-frame", "srcDoc"),
    Input("folium-indicator",  "value"),
    State("store-data",        "data"),
    prevent_initial_call=True,
)
def cb_folium(indicator, data):
    if not data:
        return ("<html><body style='background:#000000;color:#526880;"
                "font-family:sans-serif;padding:60px;text-align:center;'>"
                "<p>No data loaded — upload a dataset first.</p></body></html>")
    return generate_folium_html(
        df_from_store(data),
        None if indicator == "__none__" else indicator,
    ) or ""


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
