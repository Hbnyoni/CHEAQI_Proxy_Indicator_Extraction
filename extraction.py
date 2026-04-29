# =============================================================================
# GIPEX — Geospatial Indicators for Proxy Environmental eXposure
# GEE Extraction Engine  ·  CHEAQI-MNCH  ·  v2.1  ·  2026
#
# Variables extracted (40+):
#   Sentinel-2   : NDVI, NDBI, NDWI, MNDWI, SAVI, MSAVI, GCI, ARVI, EVI2
#   MODIS        : EVI, NDVI_MO, NDWI_MO, LST_C, LSTN_C, ET, FRP, FireMask,
#                  BurnedArea, Soil_Moist
#   ERA5-Land    : T2M, DEW, TP, SP, U10, V10, SSR
#   ERA5 Hourly  : BLH, MSLP
#   Sentinel-5P  : NO2, AOD_S5P, SPM25 (MODIS MCD19A2 AOD proxy)
#   Dynamic World: DW_label, BuiltUp
#   Impervious   : NDII (Normalised Difference Impervious Index, Landsat 8 C2 L2)
#   SRTM terrain : Elevation, Slope
#   VIIRS NTL    : VIIRS_NTL
#   Derived (post): WS, WD10, RH   (from U10/V10/T2M/DEW)
#   Road metrics  : EM_m, EH_m, WRND_km_km2  (from OSM shapefile, optional)
#
# Parallelism:
#   Outer pool  — cfg['max_workers']  tasks (cell × date) in parallel (default 50)
#   Inner pool  — cfg['var_workers']  variables per task in parallel   (default 6)
# =============================================================================

import os
import threading
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# GEE is imported lazily so the app starts even without earthengine-api
_ee = None


def _get_ee():
    global _ee
    if _ee is None:
        import ee as _ee_mod
        _ee = _ee_mod
    return _ee


def init_gee(project: str) -> str:
    """Initialise GEE. Returns '' on success, error string on failure."""
    ee = _get_ee()
    try:
        ee.Initialize(project=project)
        return ''
    except Exception as e1:
        try:
            ee.Authenticate()
            ee.Initialize(project=project)
            return ''
        except Exception as e2:
            return str(e2)


# ── Scale / offset helpers ────────────────────────────────────────────────────
def _scl(img, band, factor=1, offset=0, new_name=None):
    return img.select(band).multiply(factor).add(offset).rename(new_name or band)


def _build_scale_fns():
    return {
        'EVI'       : lambda im: _scl(im, 'EVI',               0.0001,            new_name='EVI'),
        'LST'       : lambda im: _scl(im, 'LST_Day_1km',       0.02,   -273.15,  new_name='LST_C'),
        'LSTN'      : lambda im: _scl(im, 'LST_Night_1km',     0.02,   -273.15,  new_name='LSTN_C'),
        'AOD'       : lambda im: _scl(im, 'Optical_Depth_047', 0.001,             new_name='AOD'),
        'ET'        : lambda im: _scl(im, 'ET',                0.1,               new_name='ET'),
        'FRP'       : lambda im: _scl(im, 'MaxFRP',            0.1,               new_name='FRP'),
        'T2M'       : lambda im: _scl(im, 'temperature_2m',                       1, -273.15, new_name='T2M'),
        'DEW'       : lambda im: _scl(im, 'dewpoint_temperature_2m',              1, -273.15, new_name='DEW'),
        'TP'        : lambda im: _scl(im, 'total_precipitation_sum',           1000,          new_name='TP'),
        'SP'        : lambda im: _scl(im, 'surface_pressure',                 0.01,           new_name='SP'),
        'SSR'       : lambda im: _scl(im, 'surface_solar_radiation_downwards_sum', 1/3600,    new_name='SSR'),
        'U10'       : lambda im: _scl(im, 'u_component_of_wind_10m',                          new_name='U10'),
        'V10'       : lambda im: _scl(im, 'v_component_of_wind_10m',                          new_name='V10'),
        'Soil_Moist': lambda im: _scl(im, 'volumetric_soil_water_layer_1',                    new_name='Soil_Moist'),
        'BLH'       : lambda im: _scl(im, 'boundary_layer_height',            1,              new_name='BLH'),
        'MSLP'      : lambda im: _scl(im, 'mean_sea_level_pressure',          0.01,           new_name='MSLP'),
    }


# ── Core extraction helpers ───────────────────────────────────────────────────
def _extract(col_id, band, date, geom, scale_m, fn=None):
    ee = _get_ee()
    img = (ee.ImageCollection(col_id)
             .filterDate(ee.Date(date), ee.Date(date).advance(1, 'day'))
             .filterBounds(geom))
    if fn:
        img = img.map(fn)
    img = img.mean()
    try:
        return img.select(band).reduceRegion(
            ee.Reducer.mean(), geom, scale=scale_m, maxPixels=1e13
        ).get(band).getInfo()
    except Exception:
        return None


def _extract_fallback(col_id, band, date, geom, scale_m, fn=None, max_days=8):
    """Try ±max_days around date — handles cloud gaps in Sentinel-2 / MODIS."""
    date_dt = pd.to_datetime(date)
    for offset in range(max_days + 1):
        for sign in [+1, -1]:
            try_date = (date_dt + pd.DateOffset(days=sign * offset)).strftime('%Y-%m-%d')
            val = _extract(col_id, band, try_date, geom, scale_m, fn)
            if val is not None:
                return val
    return None


def _extract_monthly(col_id, band, date, geom, scale_m, fn=None):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = f'{dt.year}-{dt.month:02d}-01'
    end   = (dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    img   = ee.ImageCollection(col_id).filterDate(start, end).filterBounds(geom)
    if fn:
        img = img.map(fn)
    img = img.mean()
    try:
        return img.select(band).reduceRegion(
            ee.Reducer.mean(), geom, scale=scale_m, maxPixels=1e13
        ).get(band).getInfo()
    except Exception:
        return None


def _era5_land(date, geom, band, fn=None):
    return _extract('ECMWF/ERA5_LAND/DAILY_AGGR', band, date, geom, 11132, fn)


def _era5_hourly(date, geom, band, fn=None):
    return _extract('ECMWF/ERA5/HOURLY', band, date, geom, 27830, fn)


def _s2_cloudmask(img):
    qa   = img.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return img.updateMask(mask).divide(10000)


def _nd_index(date, geom, band1, band2, name):
    return _extract_fallback(
        'COPERNICUS/S2_SR_HARMONIZED', name, date, geom, 20,
        lambda im: _s2_cloudmask(im).normalizedDifference([band1, band2]).rename(name),
    )


def _s2_expr(date, geom, expr_str, var_bands, out_name):
    def fn(im):
        m = _s2_cloudmask(im)
        return m.expression(expr_str, {k: m.select(v) for k, v in var_bands.items()}).rename(out_name)
    return _extract_fallback('COPERNICUS/S2_SR_HARMONIZED', out_name, date, geom, 20, fn)


def _get_elevation(geom):
    ee = _get_ee()
    return ee.Image('USGS/SRTMGL1_003').reduceRegion(
        ee.Reducer.mean(), geom, 30).get('elevation').getInfo()


def _get_slope(geom):
    ee = _get_ee()
    return ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).reduceRegion(
        ee.Reducer.mean(), geom, 30).get('slope').getInfo()


def _viirs_ntl(date, geom):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = f'{dt.year}-{dt.month:02d}-01'
    end   = (dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    img   = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterDate(start, end).first()
    try:
        return img.reduceRegion(ee.Reducer.mean(), geom, 500).get('avg_rad').getInfo()
    except Exception:
        return None


def _ndii(date, geom):
    """NDII = (Red − TIR_norm) / (Red + TIR_norm), Landsat 8 C2 L2, ±16-day window.
    Red = SR_B4 (reflectance [0,1]); TIR = ST_B10 normalised from 250–350 K → [0,1].
    """
    ee = _get_ee()
    date_dt = pd.to_datetime(date)

    def _prep(img):
        qa   = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        red  = img.select('SR_B4').multiply(0.0000275).add(-0.2).rename('red')
        tir  = (img.select('ST_B10').multiply(0.00341802).add(149.0)
                   .subtract(250).divide(100).rename('tir'))
        return red.addBands(tir).updateMask(mask)

    for offset in range(17):
        for sign in [+1, -1]:
            try_date = (date_dt + pd.DateOffset(days=sign * offset)).strftime('%Y-%m-%d')
            try:
                col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                         .filterDate(ee.Date(try_date), ee.Date(try_date).advance(1, 'day'))
                         .filterBounds(geom)
                         .map(_prep))
                img = col.mean()
                ndii_img = img.normalizedDifference(['red', 'tir']).rename('NDII')
                val = ndii_img.reduceRegion(
                    ee.Reducer.mean(), geom, scale=30, maxPixels=1e13
                ).get('NDII').getInfo()
                if val is not None:
                    return val
            except Exception:
                continue
    return None


def _burned_area(date, geom):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = f'{dt.year}-{dt.month:02d}-01'
    end   = (dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    img   = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(start, end).select('BurnDate').mean()
    try:
        return img.reduceRegion(ee.Reducer.mean(), geom, 500).get('BurnDate').getInfo()
    except Exception:
        return None


def _dw_label(date, geom):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = (dt - pd.DateOffset(days=15)).strftime('%Y-%m-%d')
    end   = (dt + pd.DateOffset(days=15)).strftime('%Y-%m-%d')
    img   = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
               .filterDate(start, end).filterBounds(geom).select('label').mode())
    try:
        return img.reduceRegion(ee.Reducer.mode(), geom, 10).get('label').getInfo()
    except Exception:
        return None


def _dw_band(date, geom, band):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = (dt - pd.DateOffset(days=15)).strftime('%Y-%m-%d')
    end   = (dt + pd.DateOffset(days=15)).strftime('%Y-%m-%d')
    img   = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
               .filterDate(start, end).filterBounds(geom).select(band).mean())
    try:
        return img.reduceRegion(ee.Reducer.mean(), geom, 10).get(band).getInfo()
    except Exception:
        return None


# ── Variable catalogue ────────────────────────────────────────────────────────
VARIABLE_GROUPS = {
    's2': {
        'label': 'Sentinel-2 Spectral Indices (10-20 m)',
        'vars' : ['NDVI', 'NDBI', 'NDWI', 'MNDWI', 'SAVI', 'MSAVI', 'GCI', 'ARVI', 'EVI2'],
        'desc' : 'Cloud-masked surface reflectance. ±8-day fallback for cloud gaps.',
    },
    'modis': {
        'label': 'MODIS Composites (500 m – 1 km)',
        'vars' : ['EVI', 'NDVI_MO', 'NDWI_MO', 'LST_C', 'LSTN_C',
                  'ET', 'FRP', 'FireMask', 'BurnedArea', 'Soil_Moist'],
        'desc' : 'Vegetation, land surface temperature, evapotranspiration, fire.',
    },
    'era5_land': {
        'label': 'ERA5-Land Meteorology (daily aggregates)',
        'vars' : ['T2M', 'DEW', 'TP', 'SP', 'U10', 'V10', 'SSR'],
        'desc' : '2 m temperature/dew point, precipitation, pressure, wind, solar radiation.',
    },
    'era5_hourly': {
        'label': 'ERA5 Hourly → daily mean',
        'vars' : ['BLH', 'MSLP'],
        'desc' : 'Boundary layer height, mean sea-level pressure.',
    },
    's5p': {
        'label': 'Sentinel-5P / TROPOMI (7 km)',
        'vars' : ['NO2', 'AOD_S5P', 'SPM25'],
        'desc' : 'NO₂ column, aerosol index, PM2.5 proxy (MODIS MCD19A2 AOD).',
    },
    'impervious': {
        'label': 'Impervious Surface (Landsat 8 C2 L2, 30 m)',
        'vars' : ['NDII'],
        'desc' : 'Normalised Difference Impervious Index: (Red − TIR_norm) / (Red + TIR_norm). ±16-day window.',
    },
    'dynworld': {
        'label': 'Dynamic World Land Cover (10 m)',
        'vars' : ['DW_label', 'BuiltUp'],
        'desc' : 'Google DW dominant class and built-up probability.',
    },
    'terrain': {
        'label': 'SRTM Terrain (30 m, static)',
        'vars' : ['Elevation', 'Slope'],
        'desc' : 'Elevation (m) and slope (°) — extracted once per grid cell.',
    },
    'viirs': {
        'label': 'VIIRS Night-time Lights (monthly, ~500 m)',
        'vars' : ['VIIRS_NTL'],
        'desc' : 'Average radiance from NOAA/VIIRS VCMSLCFG monthly composite.',
    },
}

# Columns added post-extraction (not via GEE getters)
DERIVED_VARS = ['WS', 'WD10', 'RH']   # from U10/V10/T2M/DEW
ROAD_VARS    = ['EM_m', 'EH_m', 'WRND_km_km2']  # from OSM shapefile


def build_getters(var_groups=None):
    """
    Return the extraction getters dict, optionally filtered to selected groups.
    var_groups: list of group keys (e.g. ['s2', 'era5_land']) or None for all.
    """
    sc = _build_scale_fns()

    all_getters = {
        # ── Sentinel-2 ──────────────────────────────────────────────────────
        'NDVI'      : lambda d, g: _nd_index(d, g, 'B8',  'B4',  'NDVI'),
        'NDBI'      : lambda d, g: _nd_index(d, g, 'B11', 'B8',  'NDBI'),
        'NDWI'      : lambda d, g: _nd_index(d, g, 'B3',  'B8',  'NDWI'),
        'MNDWI'     : lambda d, g: _nd_index(d, g, 'B3',  'B11', 'MNDWI'),
        'SAVI'      : lambda d, g: _s2_expr(d, g,
                          '1.5*(NIR-RED)/(NIR+RED+0.5)',
                          {'NIR': 'B8', 'RED': 'B4'}, 'SAVI'),
        'MSAVI'     : lambda d, g: _s2_expr(d, g,
                          '(2.0*NIR+1-sqrt((2.0*NIR+1)*(2.0*NIR+1)-8*(NIR-RED)))/2',
                          {'NIR': 'B8', 'RED': 'B4'}, 'MSAVI'),
        'GCI'       : lambda d, g: _s2_expr(d, g,
                          'NIR/GREEN-1', {'NIR': 'B7', 'GREEN': 'B3'}, 'GCI'),
        'ARVI'      : lambda d, g: _s2_expr(d, g,
                          '(NIR-(2*RED-BLUE))/(NIR+(2*RED-BLUE))',
                          {'NIR': 'B8', 'RED': 'B4', 'BLUE': 'B2'}, 'ARVI'),
        'EVI2'      : lambda d, g: _s2_expr(d, g,
                          '2.5*(NIR-RED)/(NIR+2.4*RED+1)',
                          {'NIR': 'B8', 'RED': 'B4'}, 'EVI2'),
        # ── MODIS ───────────────────────────────────────────────────────────
        'EVI'       : lambda d, g: _extract_fallback('MODIS/061/MOD13A1', 'EVI', d, g, 500, sc['EVI']),
        'NDVI_MO'   : lambda d, g: _extract_monthly('MODIS/061/MOD13A3', 'NDVI', d, g, 1000,
                          lambda im: im.select('NDVI').multiply(0.0001).rename('NDVI_MO')),
        'NDWI_MO'   : lambda d, g: _extract_fallback('MODIS/061/MOD09A1', 'NDWI_MO', d, g, 500,
                          lambda im: im.normalizedDifference(['sur_refl_b04', 'sur_refl_b06'])
                                        .multiply(0.0001).rename('NDWI_MO')),
        'LST_C'     : lambda d, g: _extract_fallback('MODIS/061/MOD11A1', 'LST_C',  d, g, 1000, sc['LST']),
        'LSTN_C'    : lambda d, g: _extract_fallback('MODIS/061/MOD11A1', 'LSTN_C', d, g, 1000, sc['LSTN']),
        'ET'        : lambda d, g: _extract_fallback('MODIS/061/MOD16A2', 'ET',     d, g, 500,  sc['ET']),
        'FRP'       : lambda d, g: _extract('MODIS/061/MOD14A1', 'FRP',      d, g, 1000, sc['FRP']),
        'FireMask'  : lambda d, g: _extract('MODIS/061/MOD14A1', 'FireMask', d, g, 1000),
        'BurnedArea': lambda d, g: _burned_area(d, g),
        'Soil_Moist': lambda d, g: _era5_land(d, g, 'Soil_Moist', sc['Soil_Moist']),
        # ── Dynamic World ────────────────────────────────────────────────────
        'DW_label'  : lambda d, g: _dw_label(d, g),
        'BuiltUp'   : lambda d, g: _dw_band(d, g, 'built'),
        # ── SRTM terrain ─────────────────────────────────────────────────────
        'Elevation' : lambda d, g: _get_elevation(g),
        'Slope'     : lambda d, g: _get_slope(g),
        # ── ERA5-Land ────────────────────────────────────────────────────────
        'T2M'       : lambda d, g: _era5_land(d, g, 'T2M', sc['T2M']),
        'DEW'       : lambda d, g: _era5_land(d, g, 'DEW', sc['DEW']),
        'TP'        : lambda d, g: _era5_land(d, g, 'TP',  sc['TP']),
        'SP'        : lambda d, g: _era5_land(d, g, 'SP',  sc['SP']),
        'U10'       : lambda d, g: _era5_land(d, g, 'U10', sc['U10']),
        'V10'       : lambda d, g: _era5_land(d, g, 'V10', sc['V10']),
        'SSR'       : lambda d, g: _era5_land(d, g, 'SSR', sc['SSR']),
        # ── ERA5 Hourly ──────────────────────────────────────────────────────
        'BLH'       : lambda d, g: _era5_hourly(d, g, 'BLH',  sc['BLH']),
        'MSLP'      : lambda d, g: _era5_hourly(d, g, 'MSLP', sc['MSLP']),
        # ── Sentinel-5P / TROPOMI ────────────────────────────────────────────
        'NO2'       : lambda d, g: _extract('COPERNICUS/S5P/OFFL/L3_NO2',
                          'tropospheric_NO2_column_number_density', d, g, 7000),
        'AOD_S5P'   : lambda d, g: _extract('COPERNICUS/S5P/OFFL/L3_AER_AI',
                          'absorbing_aerosol_index', d, g, 7000),
        'SPM25'     : lambda d, g: _extract_fallback('MODIS/061/MCD19A2_GRANULES',
                          'AOD', d, g, 1000, sc['AOD']),
        # ── VIIRS NTL ────────────────────────────────────────────────────────
        'VIIRS_NTL' : lambda d, g: _viirs_ntl(d, g),
        # ── Impervious ───────────────────────────────────────────────────────
        'NDII'      : lambda d, g: _ndii(d, g),
    }

    if var_groups is None:
        return all_getters

    keep = set()
    for gk in var_groups:
        keep.update(VARIABLE_GROUPS[gk]['vars'])
    return {k: v for k, v in all_getters.items() if k in keep}


# ── Post-processing helpers ───────────────────────────────────────────────────
def add_derived_met(df: pd.DataFrame) -> pd.DataFrame:
    """Derive WS (m/s), WD10 (°), RH (%) from ERA5 wind/temperature columns."""
    df = df.copy()
    if 'U10' in df.columns and 'V10' in df.columns:
        df['WS']   = np.sqrt(df['U10'] ** 2 + df['V10'] ** 2)
        df['WD10'] = (270 - np.degrees(np.arctan2(df['V10'], df['U10']))) % 360
    if 'T2M' in df.columns and 'DEW' in df.columns:
        def sat_vp(T):
            return 6.1078 * 10 ** (7.5 * T / (237.3 + T))
        df['RH'] = (sat_vp(df['DEW']) / sat_vp(df['T2M']) * 100).clip(0, 100)
    return df


def gap_fill(df: pd.DataFrame, col_id: str, date_col: str,
             var_cols: list, window: int = 7) -> pd.DataFrame:
    """Centered 7-day rolling mean gap fill, per grid cell."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = (df.sort_values([col_id, date_col])
            .drop_duplicates(subset=[col_id, date_col], keep='last')
            .reset_index(drop=True))

    def _fill(s):
        if s.isna().all():
            return s
        return s.fillna(s.rolling(window, center=True, min_periods=1).mean())

    for col in var_cols:
        if col in df.columns:
            df[col] = df.groupby(col_id)[col].transform(_fill)
    return df


def add_road_metrics(df: pd.DataFrame, roads_shp: str,
                     col_id: str, col_lat: str, col_lon: str) -> pd.DataFrame:
    """
    Compute EM_m (dist to main road), EH_m (dist to highway),
    WRND_km_km2 (road density in 1 km buffer) from OSM roads shapefile.
    Requires 'fclass' column in the shapefile (standard OSM format).
    """
    import geopandas as gpd
    grid_u = df[[col_id, col_lat, col_lon]].drop_duplicates(subset=[col_id]).copy()
    gdf_pts = gpd.GeoDataFrame(
        grid_u,
        geometry=gpd.points_from_xy(grid_u[col_lon], grid_u[col_lat]),
        crs='EPSG:4326',
    ).to_crs(epsg=3857)

    gdf_roads = gpd.read_file(roads_shp).to_crs(epsg=3857)
    if 'fclass' not in gdf_roads.columns:
        raise ValueError("Roads shapefile must have an 'fclass' column (OSM format).")

    main_types = ['trunk', 'primary', 'secondary', 'tertiary']
    hw_types   = ['motorway', 'motorway_link', 'trunk']
    gdf_main = gdf_roads[gdf_roads['fclass'].isin(main_types)]
    gdf_hw   = gdf_roads[gdf_roads['fclass'].isin(hw_types)]
    si_main  = gdf_main.sindex
    si_hw    = gdf_hw.sindex

    results = []
    for pt in gdf_pts.geometry:
        cm = gdf_main.iloc[list(si_main.intersection(pt.buffer(5000).bounds))]
        em = float(cm.distance(pt).min()) if not cm.empty else None
        ch = gdf_hw.iloc[list(si_hw.intersection(pt.buffer(5000).bounds))]
        eh = float(ch.distance(pt).min()) if not ch.empty else None
        buf  = pt.buffer(1000)
        cb   = gdf_main.iloc[list(si_main.intersection(buf.bounds))]
        ri   = cb[cb.intersects(buf)]
        wrnd = (ri.geometry.length.sum() / 1000) / (buf.area / 1e6)
        results.append((em, eh, wrnd))

    gdf_pts[['EM_m', 'EH_m', 'WRND_km_km2']] = pd.DataFrame(
        results, index=gdf_pts.index
    )
    road_df = gdf_pts[[col_id, 'EM_m', 'EH_m', 'WRND_km_km2']].copy()
    return df.merge(road_df, on=col_id, how='left')


# ── Extraction runner (background thread with live progress) ─────────────────
class ExtractionRunner:
    """
    Manages the GEE extraction background thread.
    Thread-safe state is polled by Dash callbacks every 2 s.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._thread = None
        self._reset_state()

    def _reset_state(self):
        self.state = {
            'status'     : 'idle',   # idle | running | done | error | stopped
            'progress'   : 0,
            'total'      : 0,
            'pct'        : 0,
            'logs'       : [],
            'result_path': None,
            'error'      : None,
            'started_at' : None,
            'elapsed'    : '',
        }

    def get_state(self) -> dict:
        with self._lock:
            st = dict(self.state)
            st['logs'] = list(st['logs'])
            if st['started_at']:
                elapsed = (datetime.now() - datetime.fromisoformat(st['started_at'])).seconds
                h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
                st['elapsed'] = f'{h:02d}:{m:02d}:{s:02d}'
            return st

    def stop(self):
        with self._lock:
            if self.state['status'] == 'running':
                self.state['status'] = 'stopped'

    def reset(self):
        with self._lock:
            if self.state['status'] not in ('running',):
                self._reset_state()

    def start(self, config: dict):
        with self._lock:
            if self.state['status'] == 'running':
                return 'already_running'
            self._reset_state()
            self.state['status']     = 'running'
            self.state['started_at'] = datetime.now().isoformat()
        self._config = config
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return 'started'

    def _log(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        entry = f'[{ts}]  {msg}'
        with self._lock:
            self.state['logs'].append(entry)
            if len(self.state['logs']) > 600:
                self.state['logs'] = self.state['logs'][-600:]

    def _run(self):
        try:
            self._do_extraction(self._config)
        except Exception as exc:
            self._log(f'FATAL: {exc}')
            self._log(traceback.format_exc())
            with self._lock:
                self.state['status'] = 'error'
                self.state['error']  = str(exc)

    def _do_extraction(self, cfg: dict):
        # 1 ── GEE init ────────────────────────────────────────────────
        self._log(f"GEE initialising (project: {cfg['gee_project']}) …")
        err = init_gee(cfg['gee_project'])
        if err:
            raise RuntimeError(f'GEE init failed: {err}')
        self._log('GEE ready ✓')
        ee = _get_ee()

        # 2 ── Load grid CSV ───────────────────────────────────────────
        self._log(f"Loading grid: {cfg['grid_csv']}")
        grid_df  = pd.read_csv(cfg['grid_csv'])
        col_id   = cfg['col_id']
        col_lat  = cfg['col_lat']
        col_lon  = cfg['col_lon']
        col_date = cfg.get('col_date') or ''
        col_src  = cfg.get('col_src')  or ''

        if col_date and col_date in grid_df.columns:
            grid_df[col_date] = pd.to_datetime(grid_df[col_date], errors='coerce')
            grid_df = grid_df.dropna(subset=[col_date])
            keep = [col_id, col_lat, col_lon, col_date]
            if col_src and col_src in grid_df.columns:
                keep.append(col_src)
            tasks_df = grid_df[keep].copy()
        else:
            date_range = pd.date_range(cfg['date_from'], cfg['date_to'], freq='D')
            keep = [col_id, col_lat, col_lon]
            if col_src and col_src in grid_df.columns:
                keep.append(col_src)
            base = grid_df[keep].copy()
            base['_j'] = 1
            dr_df = pd.DataFrame({'date': date_range, '_j': 1})
            tasks_df = base.merge(dr_df, on='_j').drop(columns='_j')
            col_date = 'date'

        tasks_df['_task_id'] = (
            tasks_df[col_id].astype(str) + '_' +
            pd.to_datetime(tasks_df[col_date]).dt.strftime('%Y%m%d')
        )
        self._log(f'Grid: {grid_df[col_id].nunique()} cells × {len(tasks_df):,} tasks')

        # 3 ── Checkpoint ──────────────────────────────────────────────
        os.makedirs(cfg['output_dir'], exist_ok=True)
        chk_path = os.path.join(cfg['output_dir'], 'grid_pixels_checkpoint.csv')
        if os.path.exists(chk_path):
            chk_df   = pd.read_csv(chk_path)
            done_ids = set(chk_df['_task_id'])
            self._log(f'Checkpoint: {len(done_ids):,} tasks already done — resuming.')
        else:
            chk_df   = pd.DataFrame()
            done_ids = set()

        tasks_todo = tasks_df[~tasks_df['_task_id'].isin(done_ids)]
        n_total    = len(tasks_df)
        n_done     = len(done_ids)

        with self._lock:
            self.state['total']    = n_total
            self.state['progress'] = n_done
            self.state['pct']      = int(n_done / n_total * 100) if n_total else 0

        self._log(f'Tasks remaining: {len(tasks_todo):,} / {n_total:,}')

        # 4 ── Build getters ───────────────────────────────────────────
        getters = build_getters(cfg.get('var_groups'))
        self._log(
            f'Variables ({len(getters)}): {", ".join(getters.keys())}'
        )
        self._log(
            f'Post-extraction: WS, WD10, RH (derived met)'
            + (f', EM_m / EH_m / WRND (road metrics)' if cfg.get('roads_shp') else '')
        )

        # 5 ── Extraction loop ─────────────────────────────────────────
        chk_write_lock = threading.Lock()

        def extract_one(i_row):
            i, row = i_row
            with self._lock:
                if self.state['status'] == 'stopped':
                    return None

            date_str = pd.to_datetime(row[col_date]).strftime('%Y-%m-%d')
            resolution_m = cfg.get('grid_resolution_m', 0)
            _pt = ee.Geometry.Point([float(row[col_lon]), float(row[col_lat])])
            geom = _pt.buffer(resolution_m / 2).bounds() if resolution_m and resolution_m > 0 else _pt

            rec = {
                '_task_id': row['_task_id'],
                col_id    : row[col_id],
                col_lat   : row[col_lat],
                col_lon   : row[col_lon],
                'date'    : date_str,
            }
            if col_src and col_src in row.index:
                rec[col_src] = row[col_src]

            var_workers = cfg.get('var_workers', 6)

            def _extract_var(kf):
                k, f = kf
                try:
                    return k, f(date_str, geom)
                except Exception:
                    return k, None

            with ThreadPoolExecutor(max_workers=var_workers) as var_ex:
                for key, val in var_ex.map(_extract_var, list(getters.items())):
                    rec[key] = val

            # Write to checkpoint immediately
            with chk_write_lock:
                pd.DataFrame([rec]).to_csv(
                    chk_path, mode='a',
                    header=not os.path.exists(chk_path),
                    index=False,
                )

            with self._lock:
                self.state['progress'] += 1
                prog = self.state['progress']
                self.state['pct'] = int(prog / n_total * 100) if n_total else 0

            self._log(
                f'[{prog}/{n_total}] {row[col_id]}  {date_str}  '
                f'({self.state["pct"]}%)'
            )
            return rec

        new_records = []
        with ThreadPoolExecutor(max_workers=cfg.get('max_workers', 50)) as ex:
            for result in ex.map(extract_one, list(tasks_todo.iterrows())):
                with self._lock:
                    if self.state['status'] == 'stopped':
                        break
                if result is not None:
                    new_records.append(result)

        with self._lock:
            if self.state['status'] == 'stopped':
                self._log('Extraction stopped by user — checkpoint preserved.')
                return

        # 6 ── Combine & gap-fill ──────────────────────────────────────
        self._log('Combining records …')
        parts = ([chk_df] if not chk_df.empty else []) + \
                ([pd.DataFrame(new_records)] if new_records else [])
        if not parts:
            raise RuntimeError('No records — nothing to combine.')
        raw_df = pd.concat(parts, ignore_index=True)

        meta = {'_task_id', col_id, col_lat, col_lon, 'date'}
        if col_src:
            meta.add(col_src)
        var_cols = [c for c in raw_df.columns
                    if c not in meta and pd.api.types.is_numeric_dtype(raw_df[c])]

        self._log(f'Gap-filling {len(var_cols)} variable columns (7-day rolling) …')
        final_df = gap_fill(raw_df, col_id, 'date', var_cols)

        self._log('Computing derived meteorology: WS, WD10, RH …')
        final_df = add_derived_met(final_df)

        # 7 ── Optional road metrics ───────────────────────────────────
        roads_shp = cfg.get('roads_shp', '')
        if roads_shp and os.path.exists(roads_shp):
            self._log(f'Road metrics from {os.path.basename(roads_shp)} …')
            try:
                final_df = add_road_metrics(final_df, roads_shp, col_id, col_lat, col_lon)
                self._log('EM_m, EH_m, WRND_km_km2 merged ✓')
            except Exception as exc:
                self._log(f'Road metrics skipped: {exc}')

        # 8 ── Export ──────────────────────────────────────────────────
        out_csv = os.path.join(cfg['output_dir'], 'grid_pixels.csv')
        core    = ['_task_id', col_id, col_lat, col_lon, 'date']
        if col_src and col_src in final_df.columns:
            core.append(col_src)
        ordered = core + [c for c in final_df.columns if c not in core]
        final_df[ordered].to_csv(out_csv, index=False)

        if os.path.exists(chk_path):
            os.remove(chk_path)

        self._log('─' * 56)
        self._log(f'DONE  →  {out_csv}')
        self._log(f'Rows: {len(final_df):,}   Cells: {final_df[col_id].nunique():,}'
                  f'   Variables: {len(var_cols) + len([c for c in ["WS","WD10","RH"] if c in final_df.columns])}')

        with self._lock:
            self.state['status']      = 'done'
            self.state['result_path'] = out_csv
            self.state['progress']    = n_total
            self.state['pct']         = 100


# Module-level singleton — imported by app.py
runner = ExtractionRunner()
