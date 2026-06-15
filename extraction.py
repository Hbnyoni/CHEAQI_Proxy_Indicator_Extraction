# =============================================================================
# GIPEX — Geospatial Indicators for Proxy Environmental eXposure
# GEE Extraction Engine  ·  CHEAQI-MNCH  ·  v2.1  ·  2026
#
# Variables extracted (40+):
#   Sentinel-2   : NDVI, NDBI, NDWI, MNDWI, SAVI, MSAVI, GCI, ARVI, EVI2
#   MODIS        : EVI, NDVI_MO, NDWI_MO, LST_C, LSTN_C, ET, FRP, FireMask,
#                  BurnedArea, Soil_Moist
#   ERA5-Land    : T2M, DEW, TP, SP, U10, V10, SSR
#   ERA5 Daily   : MSLP; BLH is daily mean from ERA5 hourly source
#   Sentinel-5P  : NO2, AOD_S5P
#   Dynamic World: DW_label, BuiltUp
#   Impervious   : NDII (Normalised Difference Impervious Index, Landsat 8 C2 L2)
#   SRTM terrain : Elevation, Slope
#   VIIRS NTL    : VIIRS_NTL
#   Derived (post): WS, WD10, RH   (from U10/V10/T2M/DEW)
#                   NO2_ugm3, HCHO_ugm3  (tropospheric S5P columns / BLH -> µg/m³)
#   Road metrics  : EM_m, EH_m, WRND_km_km2  (from OSM shapefile, optional)
#
# Parallelism:
#   Outer pool  — cfg['max_workers']  tasks (cell × date) in parallel (default 24)
#   Inner pool  — cfg['var_workers']  variables per task in parallel   (default 4)
# =============================================================================

import os
import threading
import traceback
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# GEE is imported lazily so the app starts even without earthengine-api
_ee = None

LEGACY_BUFFER_M = 250
LEGACY_S2_SCALE = 100
LEGACY_ERA5_SCALE = 10000
LEGACY_MODIS_SCALE = 250
LEGACY_MODIS_LST_SCALE = 1000
LEGACY_MODIS_AOD_SCALE = 1000

# Optical compositing half-windows (days). Cloud-prone optical/thermal data
# needs a wide cloud-masked composite or points fall on masked pixels -> blank.
S2_WINDOW_DAYS = 45         # Sentinel-2 cloud-masked median composite
MODIS_OPT_WINDOW_DAYS = 32  # MODIS optical/thermal (EVI, LST, ET, NDWI_MO)
DW_WINDOW_DAYS = 90         # Dynamic World land cover (slowly changing)
FALLBACK_WINDOW_DAYS = 180  # Bounded gap-fill window when the primary window is empty
FALLBACK_MAX_IMAGES = 40    # Cap fallback composite size (dense reanalysis -> OOM otherwise)


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
    # Band math (.multiply/.add) drops system:time_start, which breaks the later
    # filterDate (-> empty collection -> null image). Re-attach the timestamp.
    out = img.select(band).multiply(factor).add(offset).rename(new_name or band)
    return out.set('system:time_start', img.get('system:time_start'))


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
    img = _collection_image_with_fallback(
        col_id, ee.Date(date), ee.Date(date).advance(1, 'day'), geom, fn=fn
    )
    try:
        return img.select(band).reduceRegion(
            ee.Reducer.mean(), geom, scale=scale_m, maxPixels=1e13
        ).get(band).getInfo()
    except Exception:
        return None


def _extract_fallback(col_id, band, date, geom, scale_m, fn=None, max_days=8):
    """Use one GEE query for a ±window composite, then fall back to earliest imagery."""
    ee = _get_ee()
    date_dt = pd.to_datetime(date)
    start = (date_dt - pd.DateOffset(days=max_days)).strftime('%Y-%m-%d')
    end = (date_dt + pd.DateOffset(days=max_days + 1)).strftime('%Y-%m-%d')
    img = _collection_image_with_fallback(col_id, start, end, geom, fn=fn)
    try:
        return img.select(band).reduceRegion(
            ee.Reducer.mean(), geom, scale=scale_m, maxPixels=1e13
        ).get(band).getInfo()
    except Exception:
        return None


def _fallback_dates(center_date, fallback_days):
    dt = pd.to_datetime(center_date)
    return [
        (dt + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        for offset in range(-fallback_days, fallback_days + 1)
    ]


def _extract_first_fallback(col_id, band, date, geom, scale_m, fn=None, fallback_days=0):
    """CHEAQI legacy style: scan daily images and return first non-null reduced value."""
    ee = _get_ee()
    for try_date in _fallback_dates(date, fallback_days):
        try:
            end = (pd.to_datetime(try_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            img = (ee.ImageCollection(col_id)
                     .filterBounds(geom)
                     .filterDate(try_date, end)
                     .first())
            if fn:
                img = fn(img)
            val = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=scale_m,
                maxPixels=1e13,
            ).get(band).getInfo()
            if val is not None:
                return val
        except Exception:
            continue
    return None


def _reduce_collection(col, reducer):
    if reducer == 'median':
        return col.median()
    if reducer == 'max':
        return col.max()
    if reducer == 'min':
        return col.min()
    return col.mean()


def _collection_image_with_fallback(col_id, start, end, geom, fn=None, select=None,
                                    reducer='mean'):
    """Return window composite; if empty, composite a ±FALLBACK_WINDOW_DAYS window."""
    ee = _get_ee()
    col = ee.ImageCollection(col_id).filterBounds(geom)
    if select:
        col = col.select(select)

    # Filter by date BEFORE mapping. Band math (multiply/normalizedDifference/
    # expression) drops system:time_start, which would make a post-map filterDate
    # return an empty collection -> null image. Date-filter on the raw timestamps,
    # then apply the transform to the already-selected images.
    primary = col.filterDate(start, end)
    # Bounded fallback window. Sorting the WHOLE archive (1970-2100) is both
    # semantically wrong (earliest image ever, not the nearest) and can blow the
    # GEE compute memory limit on dense collections like CAMS NRT.
    fb_start = ee.Date(start).advance(-FALLBACK_WINDOW_DAYS, 'day')
    fb_end = ee.Date(end).advance(FALLBACK_WINDOW_DAYS, 'day')
    fallback = col.filterDate(fb_start, fb_end).limit(FALLBACK_MAX_IMAGES)
    if fn:
        primary = primary.map(fn)
        fallback = fallback.map(fn)
    primary_img = _reduce_collection(primary, reducer)
    fallback_img = _reduce_collection(fallback, reducer)
    return ee.Image(ee.Algorithms.If(primary.size().gt(0), primary_img, fallback_img))


def _masked_constant_band(out_name):
    ee = _get_ee()
    return ee.Image.constant(0).updateMask(ee.Image.constant(0)).rename(out_name)


def _select_first_available(img, candidates, out_name, scale=1, offset=0):
    """Select the first available band from an image, otherwise return a masked band."""
    ee = _get_ee()
    names = img.bandNames()
    out = _masked_constant_band(out_name)
    for band in reversed(candidates):
        selected = img.select([band]).multiply(scale).add(offset).rename(out_name)
        out = ee.Image(ee.Algorithms.If(names.contains(band), selected, out))
    return out


def _image_collection_with_fallback(col, start, end, reducer='mean'):
    """Same fallback as above for callers that already built/mapped a collection.
    Mapped collections must keep system:time_start (see _with_time) for filterDate."""
    ee = _get_ee()
    primary = col.filterDate(start, end)
    fb_start = ee.Date(start).advance(-FALLBACK_WINDOW_DAYS, 'day')
    fb_end = ee.Date(end).advance(FALLBACK_WINDOW_DAYS, 'day')
    fallback = col.filterDate(fb_start, fb_end).limit(FALLBACK_MAX_IMAGES)
    primary_img = _reduce_collection(primary, reducer)
    fallback_img = _reduce_collection(fallback, reducer)
    return ee.Image(ee.Algorithms.If(primary.size().gt(0), primary_img, fallback_img))


def _extract_monthly(col_id, band, date, geom, scale_m, fn=None):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = f'{dt.year}-{dt.month:02d}-01'
    end   = (dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    img = _collection_image_with_fallback(col_id, start, end, geom, fn=fn)
    try:
        return img.select(band).reduceRegion(
            ee.Reducer.mean(), geom, scale=scale_m, maxPixels=1e13
        ).get(band).getInfo()
    except Exception:
        return None


def _reduce_image_dict(img, geom, scale_m, bands):
    ee = _get_ee()
    try:
        vals = img.select(bands).reduceRegion(
            ee.Reducer.mean(), geom, scale=scale_m, maxPixels=1e13
        ).getInfo()
        return {band: vals.get(band) for band in bands}
    except Exception:
        return {band: None for band in bands}


def _era5_land(date, geom, band, fn=None):
    return _extract('ECMWF/ERA5_LAND/DAILY_AGGR', band, date, geom, 11132, fn)


def _era5_hourly(date, geom, band, fn=None):
    return _extract('ECMWF/ERA5/HOURLY', band, date, geom, 27830, fn)


def _era5_daily(date, geom, band, fn=None):
    return _extract('ECMWF/ERA5/DAILY', band, date, geom, 27830, fn)


def _era5_land_batch(date, geom, bands):
    ee = _get_ee()
    sc = _build_scale_fns()
    img = _collection_image_with_fallback(
        'ECMWF/ERA5_LAND/DAILY_AGGR', ee.Date(date), ee.Date(date).advance(1, 'day'), geom
    )
    out = None
    for band in bands:
        one = sc[band](img)
        out = one if out is None else out.addBands(one)
    return _reduce_image_dict(out, geom, 11132, bands)


def _era5_hourly_batch(date, geom, bands):
    ee = _get_ee()
    sc = _build_scale_fns()
    img = _collection_image_with_fallback(
        'ECMWF/ERA5/HOURLY', ee.Date(date), ee.Date(date).advance(1, 'day'), geom
    )
    out = None
    for band in bands:
        one = sc[band](img)
        out = one if out is None else out.addBands(one)
    return _reduce_image_dict(out, geom, 27830, bands)


def _era5_daily_batch(date, geom, bands):
    ee = _get_ee()
    sc = _build_scale_fns()
    img = _collection_image_with_fallback(
        'ECMWF/ERA5/DAILY', ee.Date(date), ee.Date(date).advance(1, 'day'), geom
    )
    out = None
    for band in bands:
        one = sc[band](img)
        out = one if out is None else out.addBands(one)
    return _reduce_image_dict(out, geom, 27830, bands)


def _with_time(fn):
    """Wrap a per-image map fn so band math keeps system:time_start (else a later
    filterDate drops every image and the fallback yields a null image)."""
    def _wrapped(img):
        ee = _get_ee()
        return ee.Image(fn(img)).set('system:time_start', img.get('system:time_start'))
    return _wrapped


def _s2_cloudmask(img):
    qa   = img.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    # .divide() is band math and drops system:time_start; re-stamp it so the
    # downstream filterDate window keeps the cloud-masked images.
    out  = img.updateMask(mask).divide(10000)
    return out.set('system:time_start', img.get('system:time_start'))


def _nd_index(date, geom, band1, band2, name):
    return _extract_first_fallback(
        'COPERNICUS/S2_SR_HARMONIZED',
        name,
        date,
        geom,
        LEGACY_S2_SCALE,
        lambda im: _s2_cloudmask(im).normalizedDifference([band1, band2]).rename(name),
        fallback_days=7,
    )


def _half_month_bounds(date):
    dt = pd.to_datetime(date)
    start_day = 1 if dt.day <= 15 else 16
    start = dt.replace(day=start_day)
    if start_day == 1:
        end = dt.replace(day=16)
    else:
        end = start + pd.offsets.MonthBegin(1)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def _date_range_for_frequency(date_from, date_to, frequency='daily'):
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    if pd.isna(start) or pd.isna(end) or end < start:
        return pd.DatetimeIndex([])

    if frequency == 'daily':
        return pd.date_range(start, end, freq='D')
    if frequency == 'monthly':
        dates = pd.date_range(start.normalize().replace(day=1), end, freq='MS')
        if start.day != 1:
            dates = dates[dates >= start]
        if len(dates) == 0 or dates[0] != start.normalize():
            dates = pd.DatetimeIndex([start.normalize()]).append(dates)
        return dates[dates <= end]

    first_month = start.normalize().replace(day=1)
    dates = []
    for month_start in pd.date_range(first_month, end, freq='MS'):
        for day in (1, 16):
            dt = month_start.replace(day=day)
            if start <= dt <= end:
                dates.append(dt)
    if not dates or dates[0] != start.normalize():
        dates.insert(0, start.normalize())
    return pd.DatetimeIndex(sorted(set(dates)))


def _s2_ndvi_bimonthly(date, geom):
    return _nd_index(date, geom, 'B8', 'B4', 'NDVI')


def _s2_expr(date, geom, expr_str, var_bands, out_name):
    def fn(im):
        m = _s2_cloudmask(im)
        return m.expression(expr_str, {k: m.select(v) for k, v in var_bands.items()}).rename(out_name)
    return _extract_first_fallback(
        'COPERNICUS/S2_SR_HARMONIZED',
        out_name,
        date,
        geom,
        LEGACY_S2_SCALE,
        fn,
        fallback_days=7,
    )


def _get_elevation(geom):
    ee = _get_ee()
    return ee.Image('USGS/SRTMGL1_003').reduceRegion(
        ee.Reducer.mean(), geom, 90).get('elevation').getInfo()


def _get_slope(geom):
    ee = _get_ee()
    return ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).reduceRegion(
        ee.Reducer.mean(), geom, 90).get('slope').getInfo()


def _viirs_ntl(date, geom):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = f'{dt.year}-{dt.month:02d}-01'
    end   = (dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    img = _collection_image_with_fallback(
        'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG', start, end, geom, select='avg_rad'
    )
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

    start = (date_dt - pd.DateOffset(days=S2_WINDOW_DAYS)).strftime('%Y-%m-%d')
    end = (date_dt + pd.DateOffset(days=S2_WINDOW_DAYS + 1)).strftime('%Y-%m-%d')
    try:
        col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                 .filterBounds(geom)
                 .map(_prep))
        img = _image_collection_with_fallback(col, start, end)
        ndii_img = img.normalizedDifference(['red', 'tir']).rename('NDII')
        return ndii_img.reduceRegion(
            ee.Reducer.mean(), geom, scale=30, maxPixels=1e13
        ).get('NDII').getInfo()
    except Exception:
        return None


def _burned_area(date, geom):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = f'{dt.year}-{dt.month:02d}-01'
    end   = (dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    img = _collection_image_with_fallback(
        'MODIS/061/MCD64A1', start, end, geom, select='BurnDate'
    )
    try:
        return img.reduceRegion(ee.Reducer.mean(), geom, 500).get('BurnDate').getInfo()
    except Exception:
        return None


def _burned_area_cci(date, geom):
    ee = _get_ee()
    try:
        dt = pd.to_datetime(date)
        start = f'{dt.year}-{dt.month:02d}-01'
        end = (dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        img = (ee.ImageCollection('ESA/CCI/FireCCI/5_1')
                 .filterDate(start, end)
                 .select('BurnDate')
                 .mean())
        return img.reduceRegion(
            ee.Reducer.mean(), geom, scale=500, maxPixels=1e13
        ).get('BurnDate').getInfo()
    except Exception:
        return None


def _dw_label(date, geom):
    return _extract_first_fallback(
        'GOOGLE/DYNAMICWORLD/V1',
        'label',
        date,
        geom,
        30,
        lambda im: im.select('label'),
        fallback_days=15,
    )


def _dw_band(date, geom, band):
    ee = _get_ee()
    dt    = pd.to_datetime(date)
    start = (dt - pd.DateOffset(days=15)).strftime('%Y-%m-%d')
    end   = (dt + pd.DateOffset(days=15)).strftime('%Y-%m-%d')
    img = _collection_image_with_fallback('GOOGLE/DYNAMICWORLD/V1', start, end, geom, select=band)
    try:
        return img.reduceRegion(ee.Reducer.mean(), geom, 10).get(band).getInfo()
    except Exception:
        return None


def _modis_evi(date, geom):
    return _extract_first_fallback(
        'MODIS/061/MOD13Q1',
        'EVI',
        date,
        geom,
        LEGACY_MODIS_SCALE,
        lambda im: im.select('EVI').multiply(0.0001).rename('EVI'),
        fallback_days=8,
    )


def _modis_ndvi(date, geom):
    return _extract_first_fallback(
        'MODIS/061/MOD13Q1',
        'NDVI_MO',
        date,
        geom,
        LEGACY_MODIS_SCALE,
        lambda im: im.select('NDVI').multiply(0.0001).rename('NDVI_MO'),
        fallback_days=8,
    )


def _modis_ndwi(date, geom):
    return _extract_first_fallback(
        'MODIS/061/MOD09GA_006',
        'NDWI_MO',
        date,
        geom,
        LEGACY_MODIS_SCALE,
        lambda im: im.normalizedDifference(['sur_refl_b02', 'sur_refl_b04']).rename('NDWI_MO'),
        fallback_days=8,
    )


def _modis_lst(date, geom, night=False):
    src_band = 'LST_Night_1km' if night else 'LST_Day_1km'
    out_band = 'LSTN_C' if night else 'LST_C'
    return _extract_first_fallback(
        'MODIS/061/MOD11A1',
        out_band,
        date,
        geom,
        LEGACY_MODIS_LST_SCALE,
        lambda im: im.select(src_band).multiply(0.02).subtract(273.15).rename(out_band),
        fallback_days=1,
    )


def _modis_et(date, geom):
    ee = _get_ee()
    for try_date in _fallback_dates(date, 8):
        try:
            end = (pd.to_datetime(try_date) + pd.Timedelta(days=8)).strftime('%Y-%m-%d')
            img = (ee.ImageCollection('MODIS/061/MOD16A2')
                     .filterBounds(geom)
                     .filterDate(try_date, end)
                     .first()
                     .select('ET')
                     .multiply(0.1)
                     .rename('ET'))
            val = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=500,
                maxPixels=1e13,
            ).get('ET').getInfo()
            if val is not None:
                return val
        except Exception:
            continue
    return None


def _smap_soil_moisture(date, geom):
    return _extract_first_fallback(
        'NASA_USDA/HSL/SMAP10KM_soil_moisture',
        'ssm',
        date,
        geom,
        10000,
        fallback_days=3,
    )


def _era5_hourly_daily_value(date, geom, band, out_name, postproc=None):
    value = _extract_first_fallback(
        'ECMWF/ERA5/HOURLY',
        out_name,
        date,
        geom,
        LEGACY_ERA5_SCALE,
        lambda im: im.select(band).rename(out_name),
        fallback_days=0,
    )
    if value is not None and postproc:
        return postproc(value)
    return value


def _built_up_ghsl(date, geom):
    ee = _get_ee()
    try:
        img = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020').select('built_surface')
        return img.reduceRegion(
            ee.Reducer.mean(), geom, scale=100, maxPixels=1e13
        ).get('built_surface').getInfo()
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
        'desc' : 'CHEAQI legacy products: MOD13Q1 EVI/NDVI, MOD09GA NDWI, MOD11A1 LST, MOD16A2 ET, SMAP soil moisture, fire.',
    },
    'era5_land': {
        'label': 'ERA5-Land Meteorology (daily aggregates)',
        'vars' : ['T2M', 'DEW', 'TP', 'SP', 'U10', 'V10', 'SSR',
                  'T2M_MAX', 'T2M_MIN', 'WIND_GUST', 'CLOUD_COVER', 'EVAP'],
        'desc' : 'CHEAQI legacy ERA5 hourly daily means: temperature, dew point, precipitation, pressure, wind, solar radiation.',
    },
    'era5_hourly': {
        'label': 'ERA5 Daily Atmosphere',
        'vars' : ['BLH', 'MSLP'],
        'desc' : 'Mean sea-level pressure from ERA5 Daily; boundary layer height as daily mean from ERA5 hourly source.',
    },
    's5p': {
        'label': 'Sentinel-5P / TROPOMI (7 km)',
        'vars' : ['NO2', 'AOD_S5P'],
        'desc' : 'NO₂ column and aerosol index from Sentinel-5P/TROPOMI. ±3-day window mean. PM2.5 labels come from input/SILAM data, not GEE.',
    },
    's5p_gases': {
        'label': 'Sentinel-5P Precursor & Pollutant Gases (7 km)',
        'vars' : ['SO2', 'CO', 'HCHO', 'O3'],
        'desc' : 'SO₂ (coal/industry), CO (combustion/biomass), HCHO (VOCs), O₃. ±3-day window mean.',
    },
    'aerosol': {
        'label': 'Quantitative Aerosol & Model PM2.5',
        'vars' : ['AOD_MAIAC', 'CAMS_PM25'],
        'desc' : 'MAIAC MODIS AOD 1 km (total aerosol, not just dust) and CAMS reanalysis surface PM2.5 prior.',
    },
    'population': {
        'label': 'Population & Built Environment (GHSL, 100 m)',
        'vars' : ['POP', 'BUILT_V', 'SMOD'],
        'desc' : 'GHSL 2020 population count, built-up volume, and settlement-degree (urban/rural). Emission intensity + personal-exposure weighting.',
    },
    'landcover': {
        'label': 'ESA WorldCover Land Cover (10 m)',
        'vars' : ['LandCover'],
        'desc' : 'ESA WorldCover v200 dominant land-cover class — emission source type.',
    },
    'impervious': {
        'label': 'Impervious Surface (Landsat 8 C2 L2, 30 m)',
        'vars' : ['NDII'],
        'desc' : 'Normalised Difference Impervious Index: (Red − TIR_norm) / (Red + TIR_norm). ±16-day window.',
    },
    'dynworld': {
        'label': 'Dynamic World Land Cover (10 m)',
        'vars' : ['DW_label', 'BuiltUp'],
        'desc' : 'Google DW dominant class and GHSL 2020 built-up surface.',
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
# Tropospheric S5P gases converted to surface µg/m³ via BLH (see add_gas_surface_ugm3)
GAS_UGM3_VARS = ['NO2_ugm3', 'HCHO_ugm3']
ROAD_VARS    = ['EM_m', 'EH_m', 'WRND_km_km2']  # from OSM shapefile
BULK_PROXY_VARS = {
    # Multi-band Sentinel-2 stack.
    'NDVI', 'NDBI', 'NDWI', 'MNDWI', 'SAVI', 'MSAVI', 'GCI', 'ARVI', 'EVI2',
    # Multi-band ERA5-Land daily aggregate stack.
    'T2M', 'DEW', 'TP', 'SP', 'U10', 'V10', 'SSR', 'Soil_Moist',
    # Static terrain.
    'Elevation', 'Slope',
    # Single-image groups that can still be sampled in date/chunk batches.
    'EVI', 'NDVI_MO', 'NDWI_MO', 'LST_C', 'LSTN_C', 'ET', 'FRP', 'FireMask',
    'BurnedArea', 'NO2', 'AOD_S5P', 'DW_label', 'BuiltUp', 'VIIRS_NTL',
    'T2M_MAX', 'T2M_MIN', 'WIND_GUST', 'CLOUD_COVER', 'EVAP',
    # New: S5P gases, quantitative aerosol, population/built, land cover.
    'SO2', 'CO', 'HCHO', 'O3', 'AOD_MAIAC', 'CAMS_PM25',
    'POP', 'BUILT_V', 'SMOD', 'LandCover',
}


def _s2_bulk_image(var_name, date, bounds_geom, s2_ndvi_mode='bimonthly'):
    ee = _get_ee()
    if s2_ndvi_mode == 'daily':
        start = pd.to_datetime(date).strftime('%Y-%m-%d')
        end = (pd.to_datetime(date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    else:
        start = (pd.to_datetime(date) - pd.DateOffset(days=S2_WINDOW_DAYS)).strftime('%Y-%m-%d')
        end = (pd.to_datetime(date) + pd.DateOffset(days=S2_WINDOW_DAYS + 1)).strftime('%Y-%m-%d')

    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(bounds_geom)
             .map(_s2_cloudmask))

    def _one(im):
        if var_name == 'NDVI':
            return im.normalizedDifference(['B8', 'B4']).rename('NDVI')
        if var_name == 'NDBI':
            return im.normalizedDifference(['B11', 'B8']).rename('NDBI')
        if var_name == 'NDWI':
            return im.normalizedDifference(['B3', 'B8']).rename('NDWI')
        if var_name == 'MNDWI':
            return im.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        return {
            'SAVI':  ('1.5*(NIR-RED)/(NIR+RED+0.5)', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
            'MSAVI': ('(2*NIR+1-sqrt((2*NIR+1)*(2*NIR+1)-8*(NIR-RED)))/2', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
            'GCI':   ('NIR/GREEN-1', {'NIR': im.select('B8'), 'GREEN': im.select('B3')}),
            'ARVI':  ('(NIR-(2*RED-BLUE))/(NIR+(2*RED-BLUE))', {'NIR': im.select('B8'), 'RED': im.select('B4'), 'BLUE': im.select('B2')}),
            'EVI2':  ('2.5*(NIR-RED)/(NIR+2.4*RED+1)', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
        }[var_name][0] and im.expression(*{
            'SAVI':  ('1.5*(NIR-RED)/(NIR+RED+0.5)', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
            'MSAVI': ('(2*NIR+1-sqrt((2*NIR+1)*(2*NIR+1)-8*(NIR-RED)))/2', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
            'GCI':   ('NIR/GREEN-1', {'NIR': im.select('B8'), 'GREEN': im.select('B3')}),
            'ARVI':  ('(NIR-(2*RED-BLUE))/(NIR+(2*RED-BLUE))', {'NIR': im.select('B8'), 'RED': im.select('B4'), 'BLUE': im.select('B2')}),
            'EVI2':  ('2.5*(NIR-RED)/(NIR+2.4*RED+1)', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
        }[var_name]).rename(var_name)

    img = _image_collection_with_fallback(col.map(_with_time(_one)), start, end).rename(var_name)
    return img, 20


def _proxy_image(var_name, date, bounds_geom, s2_ndvi_mode='bimonthly'):
    ee = _get_ee()
    sc = _build_scale_fns()
    date_dt = pd.to_datetime(date)
    if var_name in {'NDVI', 'NDBI', 'NDWI', 'MNDWI', 'SAVI', 'MSAVI', 'GCI', 'ARVI', 'EVI2'}:
        return _s2_bulk_image(var_name, date, bounds_geom, s2_ndvi_mode)
    if var_name in {'T2M', 'DEW', 'TP', 'SP', 'U10', 'V10', 'SSR', 'Soil_Moist'}:
        img = _collection_image_with_fallback(
            'ECMWF/ERA5_LAND/DAILY_AGGR',
            ee.Date(date), ee.Date(date).advance(1, 'day'), bounds_geom,
        )
        return sc[var_name](img), 11132
    if var_name in {'T2M_MAX', 'T2M_MIN', 'WIND_GUST', 'CLOUD_COVER', 'EVAP'}:
        img, _, scale = _era5_extra_multi_image([var_name], date, bounds_geom)
        return img.select(var_name), scale
    if var_name == 'MSLP':
        img = _collection_image_with_fallback(
            'ECMWF/ERA5/DAILY', ee.Date(date), ee.Date(date).advance(1, 'day'), bounds_geom
        )
        return sc[var_name](img), 27830
    if var_name == 'BLH':
        img = _collection_image_with_fallback(
            'ECMWF/ERA5/HOURLY', ee.Date(date), ee.Date(date).advance(1, 'day'), bounds_geom
        )
        return sc[var_name](img), 27830
    # ── Sentinel-5P / TROPOMI gases. ±3-day window mean fills QA-masked pixels
    #    and orbital swath gaps (the cause of blanks at scattered point grids). ──
    if var_name in {'NO2', 'AOD_S5P', 'SO2', 'CO', 'HCHO', 'O3'}:
        s5p_spec = {
            'NO2':     ('COPERNICUS/S5P/OFFL/L3_NO2',  'tropospheric_NO2_column_number_density'),
            'AOD_S5P': ('COPERNICUS/S5P/OFFL/L3_AER_AI', 'absorbing_aerosol_index'),
            'SO2':     ('COPERNICUS/S5P/OFFL/L3_SO2',  'SO2_column_number_density'),
            'CO':      ('COPERNICUS/S5P/OFFL/L3_CO',   'CO_column_number_density'),
            'HCHO':    ('COPERNICUS/S5P/OFFL/L3_HCHO', 'tropospheric_HCHO_column_number_density'),
            'O3':      ('COPERNICUS/S5P/OFFL/L3_O3',   'O3_column_number_density'),
        }
        col_id, band = s5p_spec[var_name]
        start = (date_dt - pd.DateOffset(days=3)).strftime('%Y-%m-%d')
        end = (date_dt + pd.DateOffset(days=4)).strftime('%Y-%m-%d')
        img = _collection_image_with_fallback(
            col_id, start, end, bounds_geom, select=band,
        ).rename(var_name)
        return img, 7000
    # ── Quantitative aerosol + model PM2.5 prior ───────────────────────────
    # AOD_MAIAC / CAMS are dense, always-populated collections: a direct window
    # mean is enough. The archive fallback would sort tens of thousands of granules
    # per query and blow GEE's compute memory / hang, so it is deliberately skipped.
    if var_name == 'AOD_MAIAC':
        start = (date_dt - pd.DateOffset(days=3)).strftime('%Y-%m-%d')
        end = (date_dt + pd.DateOffset(days=4)).strftime('%Y-%m-%d')
        img = (ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
                 .filterBounds(bounds_geom).filterDate(start, end)
                 .select('Optical_Depth_055').mean().multiply(0.001).rename('AOD_MAIAC'))
        return img, 1000
    if var_name == 'CAMS_PM25':
        start = (date_dt - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        end = (date_dt + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        img = (ee.ImageCollection('ECMWF/CAMS/NRT')
                 .filterBounds(bounds_geom).filterDate(start, end)
                 .select('particulate_matter_d_less_than_25_um_surface')
                 .mean().multiply(1e9).rename('CAMS_PM25'))  # kg/m3 -> ug/m3
        return img, 40000
    # ── Human activity / population / land use (annual, ~static per year) ──
    if var_name in {'POP', 'BUILT_V'}:
        ghsl = {'POP': ('JRC/GHSL/P2023A/GHS_POP', 'population_count'),
                'BUILT_V': ('JRC/GHSL/P2023A/GHS_BUILT_V', 'built_volume_total')}
        col_id, band = ghsl[var_name]
        img = (ee.ImageCollection(col_id)
                 .filter(ee.Filter.calendarRange(2020, 2020, 'year'))
                 .select(band).first().rename(var_name))
        return img, 100
    if var_name == 'SMOD':
        img = (ee.ImageCollection('JRC/GHSL/P2023A/GHS_SMOD_V2-0')
                 .filter(ee.Filter.calendarRange(2020, 2020, 'year'))
                 .select('smod_code').first().rename('SMOD'))
        return img, 1000
    if var_name == 'LandCover':
        img = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('LandCover')
        return img, 10
    if var_name == 'NDVI_MO':
        start = f'{date_dt.year}-{date_dt.month:02d}-01'
        end = (date_dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        col = (ee.ImageCollection('MODIS/061/MOD13A3')
                 .filterBounds(bounds_geom)
                 .map(_with_time(lambda im: im.select('NDVI').multiply(0.0001).rename('NDVI_MO'))))
        img = _image_collection_with_fallback(col, start, end)
        return img, 1000
    if var_name == 'VIIRS_NTL':
        start = f'{date_dt.year}-{date_dt.month:02d}-01'
        end = (date_dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        img = _collection_image_with_fallback(
            'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG', start, end, bounds_geom, select='avg_rad'
        ).rename('VIIRS_NTL')
        return img, 500
    if var_name == 'BurnedArea':
        start = f'{date_dt.year}-{date_dt.month:02d}-01'
        end = (date_dt + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        img = _collection_image_with_fallback(
            'MODIS/061/MCD64A1', start, end, bounds_geom, select='BurnDate'
        ).rename('BurnedArea')
        return img, 500
    if var_name == 'NDWI_MO':
        start = (date_dt - pd.DateOffset(days=MODIS_OPT_WINDOW_DAYS)).strftime('%Y-%m-%d')
        end = (date_dt + pd.DateOffset(days=MODIS_OPT_WINDOW_DAYS + 1)).strftime('%Y-%m-%d')
        col = (ee.ImageCollection('MODIS/061/MOD09A1')
                 .filterBounds(bounds_geom)
                 .map(_with_time(lambda im: im.normalizedDifference(['sur_refl_b04', 'sur_refl_b06']).rename('NDWI_MO'))))
        img = _image_collection_with_fallback(col, start, end)
        return img, 500
    if var_name in {'EVI', 'LST_C', 'LSTN_C', 'ET'}:
        specs = {
            'EVI': ('MODIS/061/MOD13A1', sc['EVI'], 500),
            'LST_C': ('MODIS/061/MOD11A1', sc['LST'], 1000),
            'LSTN_C': ('MODIS/061/MOD11A1', sc['LSTN'], 1000),
            'ET': ('MODIS/061/MOD16A2', sc['ET'], 500),
        }
        col_id, fn, scale = specs[var_name]
        start = (date_dt - pd.DateOffset(days=MODIS_OPT_WINDOW_DAYS)).strftime('%Y-%m-%d')
        end = (date_dt + pd.DateOffset(days=MODIS_OPT_WINDOW_DAYS + 1)).strftime('%Y-%m-%d')
        img = _collection_image_with_fallback(
            col_id, start, end, bounds_geom, fn=fn
        ).rename(var_name)
        return img, scale
    if var_name in {'FRP', 'FireMask'}:
        band = 'MaxFRP' if var_name == 'FRP' else 'FireMask'
        img = _collection_image_with_fallback(
            'MODIS/061/MOD14A1', ee.Date(date), ee.Date(date).advance(1, 'day'),
            bounds_geom, select=band,
        )
        if var_name == 'FRP':
            img = img.multiply(0.1)
        return img.rename(var_name), 1000
    if var_name == 'BuiltUp':
        start = (date_dt - pd.DateOffset(days=DW_WINDOW_DAYS)).strftime('%Y-%m-%d')
        end = (date_dt + pd.DateOffset(days=DW_WINDOW_DAYS + 1)).strftime('%Y-%m-%d')
        img = _collection_image_with_fallback(
            'GOOGLE/DYNAMICWORLD/V1', start, end, bounds_geom, select='built', reducer='median',
        ).rename('BuiltUp')
        return img, 10
    if var_name == 'DW_label':
        start = (date_dt - pd.DateOffset(days=DW_WINDOW_DAYS)).strftime('%Y-%m-%d')
        end = (date_dt + pd.DateOffset(days=DW_WINDOW_DAYS + 1)).strftime('%Y-%m-%d')
        col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(bounds_geom).select('label')
        img = _image_collection_with_fallback(col, start, end, reducer='median').rename('DW_label')
        return img, 10
    if var_name == 'Elevation':
        return ee.Image('USGS/SRTMGL1_003').select('elevation').rename('Elevation'), 30
    if var_name == 'Slope':
        return ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).rename('Slope'), 30
    raise KeyError(var_name)


def _s2_bulk_multi_image(var_names, date, bounds_geom, s2_ndvi_mode='bimonthly'):
    ee = _get_ee()
    if s2_ndvi_mode == 'daily':
        start = pd.to_datetime(date).strftime('%Y-%m-%d')
        end = (pd.to_datetime(date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    else:
        start = (pd.to_datetime(date) - pd.DateOffset(days=S2_WINDOW_DAYS)).strftime('%Y-%m-%d')
        end = (pd.to_datetime(date) + pd.DateOffset(days=S2_WINDOW_DAYS + 1)).strftime('%Y-%m-%d')

    def _one(im):
        im = _s2_cloudmask(im)
        bands = []
        for var_name in var_names:
            if var_name == 'NDVI':
                bands.append(im.normalizedDifference(['B8', 'B4']).rename('NDVI'))
            elif var_name == 'NDBI':
                bands.append(im.normalizedDifference(['B11', 'B8']).rename('NDBI'))
            elif var_name == 'NDWI':
                bands.append(im.normalizedDifference(['B3', 'B8']).rename('NDWI'))
            elif var_name == 'MNDWI':
                bands.append(im.normalizedDifference(['B3', 'B11']).rename('MNDWI'))
            else:
                exprs = {
                    'SAVI':  ('1.5*(NIR-RED)/(NIR+RED+0.5)', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
                    'MSAVI': ('(2*NIR+1-sqrt((2*NIR+1)*(2*NIR+1)-8*(NIR-RED)))/2', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
                    'GCI':   ('NIR/GREEN-1', {'NIR': im.select('B8'), 'GREEN': im.select('B3')}),
                    'ARVI':  ('(NIR-(2*RED-BLUE))/(NIR+(2*RED-BLUE))', {'NIR': im.select('B8'), 'RED': im.select('B4'), 'BLUE': im.select('B2')}),
                    'EVI2':  ('2.5*(NIR-RED)/(NIR+2.4*RED+1)', {'NIR': im.select('B8'), 'RED': im.select('B4')}),
                }
                expr, args = exprs[var_name]
                bands.append(im.expression(expr, args).rename(var_name))
        out = bands[0]
        for band in bands[1:]:
            out = out.addBands(band)
        return out

    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(bounds_geom)
             .map(_with_time(_one)))
    img = _image_collection_with_fallback(col, start, end).select(var_names)
    return img, var_names, 20


def _era5_land_multi_image(var_names, date, bounds_geom):
    ee = _get_ee()
    sc = _build_scale_fns()
    img = _collection_image_with_fallback(
        'ECMWF/ERA5_LAND/DAILY_AGGR',
        ee.Date(date), ee.Date(date).advance(1, 'day'), bounds_geom,
    )
    out = None
    for var_name in var_names:
        band = sc[var_name](img)
        out = band if out is None else out.addBands(band)
    return out.select(var_names), var_names, 11132


def _era5_extra_multi_image(var_names, date, bounds_geom):
    ee = _get_ee()
    daily = _collection_image_with_fallback(
        'ECMWF/ERA5_LAND/DAILY_AGGR',
        ee.Date(date), ee.Date(date).advance(1, 'day'), bounds_geom,
    )
    hourly_mean = None
    hourly_max = None
    hourly_min = None

    def _hourly(reducer):
        return _collection_image_with_fallback(
            'ECMWF/ERA5_LAND/HOURLY',
            ee.Date(date), ee.Date(date).advance(1, 'day'), bounds_geom,
            reducer=reducer,
        )

    def _atm_hourly(reducer):
        # ERA5 atmospheric (NOT ERA5-Land) — has cloud cover and wind gust.
        return _collection_image_with_fallback(
            'ECMWF/ERA5/HOURLY',
            ee.Date(date), ee.Date(date).advance(1, 'day'), bounds_geom,
            reducer=reducer,
        )

    out = None
    for var_name in var_names:
        if var_name == 'T2M_MAX':
            band = _select_first_available(daily, ['temperature_2m_max'], var_name, scale=1, offset=-273.15)
            if hourly_max is None:
                hourly_max = _hourly('max')
            hourly_band = _select_first_available(hourly_max, ['temperature_2m'], var_name, scale=1, offset=-273.15)
            band = band.unmask(hourly_band)
        elif var_name == 'T2M_MIN':
            band = _select_first_available(daily, ['temperature_2m_min'], var_name, scale=1, offset=-273.15)
            if hourly_min is None:
                hourly_min = _hourly('min')
            hourly_band = _select_first_available(hourly_min, ['temperature_2m'], var_name, scale=1, offset=-273.15)
            band = band.unmask(hourly_band)
        elif var_name == 'WIND_GUST':
            # ERA5 atmospheric daily-max gust (ERA5-Land has no gust variable).
            atm_max = _atm_hourly('max')
            band = _select_first_available(
                atm_max,
                ['instantaneous_10m_wind_gust', 'wind_gust_since_previous_post_processing_10m'],
                var_name,
            )
        elif var_name == 'CLOUD_COVER':
            # ERA5 atmospheric daily-mean total cloud cover (ERA5-Land has none).
            atm_mean = _atm_hourly('mean')
            band = _select_first_available(atm_mean, ['total_cloud_cover'], var_name)
        elif var_name == 'EVAP':
            band = _select_first_available(
                daily,
                ['total_evaporation_sum', 'evaporation_from_bare_soil_sum', 'evaporation_from_bare_soil'],
                var_name,
                scale=1000,
            )
        else:
            continue
        out = band if out is None else out.addBands(band)
    return out.select(var_names), var_names, 11132


def _terrain_multi_image(var_names):
    ee = _get_ee()
    elev = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('Elevation')
    slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).rename('Slope')
    img = elev.addBands(slope).select(var_names)
    return img, var_names, 30


def _bulk_image_groups(var_names, date, bounds_geom, s2_ndvi_mode='bimonthly'):
    ordered = list(var_names)
    groups = []

    def take(names):
        picked = [v for v in ordered if v in names]
        for v in picked:
            ordered.remove(v)
        return picked

    s2_vars = take({'NDVI', 'NDBI', 'NDWI', 'MNDWI', 'SAVI', 'MSAVI', 'GCI', 'ARVI', 'EVI2'})
    if s2_vars:
        groups.append(('Sentinel-2 indices', *_s2_bulk_multi_image(s2_vars, date, bounds_geom, s2_ndvi_mode)))

    era5_land_vars = take({'T2M', 'DEW', 'TP', 'SP', 'U10', 'V10', 'SSR', 'Soil_Moist'})
    if era5_land_vars:
        groups.append(('ERA5-Land', *_era5_land_multi_image(era5_land_vars, date, bounds_geom)))

    era5_extra_vars = take({'T2M_MAX', 'T2M_MIN', 'WIND_GUST', 'CLOUD_COVER', 'EVAP'})
    if era5_extra_vars:
        groups.append(('ERA5-Land extras', *_era5_extra_multi_image(era5_extra_vars, date, bounds_geom)))

    terrain_vars = take({'Elevation', 'Slope'})
    if terrain_vars:
        groups.append(('SRTM terrain', *_terrain_multi_image(terrain_vars)))

    for var_name in list(ordered):
        img, scale = _proxy_image(var_name, date, bounds_geom, s2_ndvi_mode)
        groups.append((var_name, img, [var_name], scale))
    return groups


def _extract_proxy_bulk(date_str, rows_df, var_names, col_lat, col_lon,
                        resolution_m=0, s2_ndvi_mode='bimonthly', log_fn=None,
                        geometry_mode='point'):
    ee = _get_ee()
    features = []
    use_cell = geometry_mode == 'cell' and resolution_m and resolution_m > 0
    for _, row in rows_df.iterrows():
        pt = ee.Geometry.Point([float(row[col_lon]), float(row[col_lat])])
        geom = pt.buffer(resolution_m / 2).bounds() if use_cell else pt
        features.append(ee.Feature(geom, {'_task_id': row['_task_id']}))

    fc = ee.FeatureCollection(features)
    bounds = fc.geometry()
    out = {task_id: {} for task_id in rows_df['_task_id'].astype(str)}
    # Keep (task_id, ee.Feature) pairs so a failed chunk can be split and retried
    # instead of silently blanking the whole group (the cause of the database gaps).
    pairs = list(zip(rows_df['_task_id'].astype(str).tolist(), features))

    for group_name, img, band_names, scale in _bulk_image_groups(var_names, date_str, bounds, s2_ndvi_mode):
        if log_fn:
            log_fn(f'Bulk {date_str}: extracting {group_name} ({len(band_names)} vars) for {len(rows_df)} cells')

        def _sample(batch):
            fc_local = ee.FeatureCollection([f for _, f in batch])
            if use_cell:
                # Cap the reduce scale at the cell size. At a dataset's native
                # coarse scale (ERA5 ~11 km, S5P ~7 km, CAMS ~40 km) a 100 m cell
                # contains no pixel and reduceRegions returns null for every cell.
                # Reducing the coarse image at the finer cell scale still samples
                # the overlapping pixel's value.
                reduce_scale = min(scale, resolution_m) if resolution_m else scale
                return img.reduceRegions(
                    collection=fc_local, reducer=ee.Reducer.mean(),
                    scale=reduce_scale, tileScale=16,
                ).getInfo()
            return img.sampleRegions(
                collection=fc_local, properties=['_task_id'],
                scale=scale, geometries=False, tileScale=16,
            ).getInfo()

        # Process the group; on error split the batch and retry (handles dense-grid
        # compute/timeout limits) down to a floor before giving up on those cells.
        stack = [pairs]
        while stack:
            batch = stack.pop()
            try:
                reduced = _sample(batch)
                # reduceRegions (cell mode) names its output 'mean' for a SINGLE-band
                # image but uses band names for multi-band; sampleRegions (point mode)
                # always uses band names. Fall back to 'mean' for single-band groups.
                single_band = len(band_names) == 1
                for feat in reduced.get('features', []):
                    props = feat.get('properties', {})
                    task_id = str(props.get('_task_id'))
                    for band_name in band_names:
                        val = props.get(band_name)
                        if val is None and single_band:
                            val = props.get('mean')
                        out.setdefault(task_id, {})[band_name] = val
            except Exception as exc:
                if len(batch) > 50:
                    mid = len(batch) // 2
                    stack.append(batch[:mid])
                    stack.append(batch[mid:])
                    if log_fn:
                        log_fn(f'  {group_name}: chunk of {len(batch)} failed '
                               f'({type(exc).__name__}); splitting and retrying')
                else:
                    if log_fn:
                        log_fn(f'  {group_name}: BLANKING {len(batch)} cells after retries '
                               f'({type(exc).__name__}: {exc})')
                    for task_id, _ in batch:
                        for band_name in band_names:
                            out.setdefault(task_id, {})[band_name] = None
    return out


def build_getters(var_groups=None, s2_ndvi_mode='bimonthly'):
    """
    Return the extraction getters dict, optionally filtered to selected groups.
    var_groups: list of group keys (e.g. ['s2', 'era5_land']) or None for all.
    """
    ndvi_getter = _s2_ndvi_bimonthly if s2_ndvi_mode == 'bimonthly' else \
        (lambda d, g: _nd_index(d, g, 'B8', 'B4', 'NDVI'))

    all_getters = {
        # ── Sentinel-2 ──────────────────────────────────────────────────────
        'NDVI'      : ndvi_getter,
        'NDBI'      : lambda d, g: _nd_index(d, g, 'B11', 'B8',  'NDBI'),
        'NDWI'      : lambda d, g: _nd_index(d, g, 'B3',  'B8',  'NDWI'),
        'MNDWI'     : lambda d, g: _nd_index(d, g, 'B3',  'B11', 'MNDWI'),
        'SAVI'      : lambda d, g: _s2_expr(d, g,
                          '1.5*(NIR-RED)/(NIR+RED+0.5)',
                          {'NIR': 'B8', 'RED': 'B4'}, 'SAVI'),
        'MSAVI'     : lambda d, g: _s2_expr(d, g,
                          '0.5*(2*NIR+1-sqrt((2*NIR+1)*(2*NIR+1)-8*(NIR-RED)))',
                          {'NIR': 'B8', 'RED': 'B4'}, 'MSAVI'),
        'GCI'       : lambda d, g: _s2_expr(d, g,
                          '(NIR/GREEN)-1', {'NIR': 'B8', 'GREEN': 'B3'}, 'GCI'),
        'ARVI'      : lambda d, g: _s2_expr(d, g,
                          '(NIR-(2*RED-BLUE))/(NIR+(2*RED-BLUE))',
                          {'NIR': 'B8', 'RED': 'B4', 'BLUE': 'B2'}, 'ARVI'),
        'EVI2'      : lambda d, g: _s2_expr(d, g,
                          '2.5*(NIR-RED)/(NIR+2.4*RED+1)',
                          {'NIR': 'B8', 'RED': 'B4'}, 'EVI2'),
        # ── MODIS ───────────────────────────────────────────────────────────
        'EVI'       : lambda d, g: _modis_evi(d, g),
        'NDVI_MO'   : lambda d, g: _modis_ndvi(d, g),
        'NDWI_MO'   : lambda d, g: _modis_ndwi(d, g),
        'LST_C'     : lambda d, g: _modis_lst(d, g, night=False),
        'LSTN_C'    : lambda d, g: _modis_lst(d, g, night=True),
        'ET'        : lambda d, g: _modis_et(d, g),
        'FRP'       : lambda d, g: _extract('MODIS/061/MOD14A1', 'MaxFRP',   d, g, 1000),
        'FireMask'  : lambda d, g: _extract('MODIS/061/MOD14A1', 'FireMask', d, g, 1000),
        'BurnedArea': lambda d, g: _burned_area(d, g),
        'BurnedArea_CCI': lambda d, g: _burned_area_cci(d, g),
        'Soil_Moist': lambda d, g: _smap_soil_moisture(d, g),
        # ── Dynamic World ────────────────────────────────────────────────────
        'DW_label'  : lambda d, g: _dw_label(d, g),
        'BuiltUp'   : lambda d, g: _built_up_ghsl(d, g),
        # ── SRTM terrain ─────────────────────────────────────────────────────
        'Elevation' : lambda d, g: _get_elevation(g),
        'Slope'     : lambda d, g: _get_slope(g),
        # ── ERA5-Land ────────────────────────────────────────────────────────
        'T2M'       : lambda d, g: _era5_hourly_daily_value(d, g, 'temperature_2m', 'T2M', lambda x: x - 273.15),
        'DEW'       : lambda d, g: _era5_hourly_daily_value(d, g, 'dewpoint_temperature_2m', 'DEW', lambda x: x - 273.15),
        'TP'        : lambda d, g: _era5_hourly_daily_value(d, g, 'total_precipitation', 'TP', lambda x: x * 1000),
        'SP'        : lambda d, g: _era5_hourly_daily_value(d, g, 'surface_pressure', 'SP'),
        'U10'       : lambda d, g: _era5_hourly_daily_value(d, g, 'u_component_of_wind_10m', 'U10'),
        'V10'       : lambda d, g: _era5_hourly_daily_value(d, g, 'v_component_of_wind_10m', 'V10'),
        'SSR'       : lambda d, g: _era5_hourly_daily_value(d, g, 'surface_solar_radiation_downwards', 'SSR', lambda x: x / 3600),
        'T2M_MAX'   : lambda d, g: _reduce_image_dict(_proxy_image('T2M_MAX', d, g)[0], g, 11132, ['T2M_MAX']).get('T2M_MAX'),
        'T2M_MIN'   : lambda d, g: _reduce_image_dict(_proxy_image('T2M_MIN', d, g)[0], g, 11132, ['T2M_MIN']).get('T2M_MIN'),
        'WIND_GUST' : lambda d, g: _reduce_image_dict(_proxy_image('WIND_GUST', d, g)[0], g, 11132, ['WIND_GUST']).get('WIND_GUST'),
        'CLOUD_COVER': lambda d, g: _reduce_image_dict(_proxy_image('CLOUD_COVER', d, g)[0], g, 11132, ['CLOUD_COVER']).get('CLOUD_COVER'),
        'EVAP'      : lambda d, g: _reduce_image_dict(_proxy_image('EVAP', d, g)[0], g, 11132, ['EVAP']).get('EVAP'),
        # ── ERA5 daily atmosphere ─────────────────────────────────────────────
        'BLH'       : lambda d, g: _era5_hourly_daily_value(d, g, 'boundary_layer_height', 'BLH'),
        'MSLP'      : lambda d, g: _era5_hourly_daily_value(d, g, 'mean_sea_level_pressure', 'MSLP'),
        # ── Sentinel-5P / TROPOMI ────────────────────────────────────────────
        'NO2'       : lambda d, g: _extract_first_fallback(
                          'COPERNICUS/S5P/OFFL/L3_NO2', 'NO2', d, g, 1000,
                          lambda im: im.select('tropospheric_NO2_column_number_density').rename('NO2'),
                          fallback_days=1),
        'AOD_S5P'   : lambda d, g: _extract_first_fallback(
                          'COPERNICUS/S5P/OFFL/L3_AER_AI', 'AOD_S5P', d, g, 1000,
                          lambda im: im.select('absorbing_aerosol_index').rename('AOD_S5P'),
                          fallback_days=1),
        # ── VIIRS NTL ────────────────────────────────────────────────────────
        'VIIRS_NTL' : lambda d, g: _viirs_ntl(d, g),
        # ── Impervious ───────────────────────────────────────────────────────
        'NDII'      : lambda d, g: _ndii(d, g),
        # ── Sentinel-5P precursor & pollutant gases ──────────────────────────
        'SO2'       : lambda d, g: _reduce_image_dict(_proxy_image('SO2', d, g)[0], g, 7000, ['SO2']).get('SO2'),
        'CO'        : lambda d, g: _reduce_image_dict(_proxy_image('CO', d, g)[0], g, 7000, ['CO']).get('CO'),
        'HCHO'      : lambda d, g: _reduce_image_dict(_proxy_image('HCHO', d, g)[0], g, 7000, ['HCHO']).get('HCHO'),
        'O3'        : lambda d, g: _reduce_image_dict(_proxy_image('O3', d, g)[0], g, 7000, ['O3']).get('O3'),
        # ── Quantitative aerosol & model PM2.5 ───────────────────────────────
        'AOD_MAIAC' : lambda d, g: _reduce_image_dict(_proxy_image('AOD_MAIAC', d, g)[0], g, 1000, ['AOD_MAIAC']).get('AOD_MAIAC'),
        'CAMS_PM25' : lambda d, g: _reduce_image_dict(_proxy_image('CAMS_PM25', d, g)[0], g, 40000, ['CAMS_PM25']).get('CAMS_PM25'),
        # ── Population / built environment / land cover ──────────────────────
        'POP'       : lambda d, g: _reduce_image_dict(_proxy_image('POP', d, g)[0], g, 100, ['POP']).get('POP'),
        'BUILT_V'   : lambda d, g: _reduce_image_dict(_proxy_image('BUILT_V', d, g)[0], g, 100, ['BUILT_V']).get('BUILT_V'),
        'SMOD'      : lambda d, g: _reduce_image_dict(_proxy_image('SMOD', d, g)[0], g, 1000, ['SMOD']).get('SMOD'),
        'LandCover' : lambda d, g: _reduce_image_dict(_proxy_image('LandCover', d, g)[0], g, 10, ['LandCover']).get('LandCover'),
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
        u10 = pd.to_numeric(df['U10'], errors='coerce')
        v10 = pd.to_numeric(df['V10'], errors='coerce')
        df['WS']   = np.hypot(u10, v10)
        df['WD10'] = (270 - np.degrees(np.arctan2(v10, u10))) % 360
    if 'T2M' in df.columns and 'DEW' in df.columns:
        t2m = pd.to_numeric(df['T2M'], errors='coerce')
        dew = pd.to_numeric(df['DEW'], errors='coerce')

        def sat_vp(T):
            return 6.1078 * 10 ** (7.5 * T / (237.3 + T))
        df['RH'] = (sat_vp(dew) / sat_vp(t2m) * 100).clip(0, 100)
    return df


# Molar masses (g/mol) for the S5P/TROPOMI trace gases.
GAS_MOLAR_MASS = {
    'NO2': 46.0055, 'SO2': 64.066, 'CO': 28.0101, 'HCHO': 30.026, 'O3': 47.997,
}
# Only the TROPOSPHERIC-column products can be turned into a surface concentration
# by dividing through the boundary-layer height. O3/CO/SO2 are TOTAL-column products
# (O3 is ~90% stratospheric, CO is well-mixed through the whole troposphere), so a
# BLH division is not physically meaningful for them — they stay as raw columns.
GAS_TROPOSPHERIC = {'NO2', 'HCHO'}


def add_gas_surface_ugm3(df: pd.DataFrame, gases=None) -> pd.DataFrame:
    """Approximate surface concentration (µg/m³) for tropospheric S5P gases.

    TROPOMI gives a vertical COLUMN number density (mol/m²); PM2.5 is a surface
    mass concentration (µg/m³). For species whose product is a TROPOSPHERIC column
    (NO2, HCHO), convert column → column mass → surface concentration assuming the
    gas is well-mixed through the boundary layer:

        C[µg/m³] ≈ N[mol/m²] · M[g/mol] · 1e6[µg/g] / BLH[m]

    Adds <gas>_ugm3 companion columns; the raw column densities are kept unchanged.
    Requires BLH (m); rows with missing/non-positive BLH yield NaN. AOD_S5P is a
    dimensionless index and is never converted. Pass `gases` to override the default
    tropospheric subset (e.g. include 'SO2'), accepting the physical caveats.
    """
    df = df.copy()
    if 'BLH' not in df.columns:
        return df
    gases = gases or GAS_TROPOSPHERIC
    blh = pd.to_numeric(df['BLH'], errors='coerce')
    blh = blh.where(blh > 0)  # guard against divide-by-zero / negative
    for gas in gases:
        if gas in df.columns and gas in GAS_MOLAR_MASS:
            col = pd.to_numeric(df[gas], errors='coerce')
            df[f'{gas}_ugm3'] = col * GAS_MOLAR_MASS[gas] * 1e6 / blh
    return df


def gap_fill(df: pd.DataFrame, col_id: str, date_col: str,
             var_cols: list, window: int = 7) -> pd.DataFrame:
    """Fill gaps only from same-cell observed values within a 7-day window.

    This preserves local variation: no rolling averages, same-date medians,
    global medians, or zero fills are used.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = (df.sort_values([col_id, date_col])
            .drop_duplicates(subset=[col_id, date_col], keep='last')
            .reset_index(drop=True))

    def _fill_group(g, col):
        g = g.sort_values(date_col).copy()
        values = pd.to_numeric(g[col], errors='coerce')
        dates = g[date_col]

        prev_values = values.ffill()
        prev_dates = dates.where(values.notna()).ffill()
        prev_age = (dates - prev_dates).dt.days
        fill_prev = values.isna() & prev_values.notna() & prev_age.between(0, window)
        values.loc[fill_prev] = prev_values.loc[fill_prev]

        next_values = values.bfill()
        next_dates = dates.where(values.notna()).bfill()
        next_age = (next_dates - dates).dt.days
        fill_next = values.isna() & next_values.notna() & next_age.between(0, window)
        values.loc[fill_next] = next_values.loc[fill_next]
        return values.reindex(g.index)

    for col in var_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df.groupby(col_id, group_keys=False).apply(
                lambda g, c=col: _fill_group(g, c),
                include_groups=False,
            ).sort_index()
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
        input_cols = list(grid_df.columns)

        missing = [c for c in [col_id, col_lat, col_lon] if c not in grid_df.columns]
        if missing:
            raise KeyError(
                f"Required column(s) {missing} not found. "
                f"Available columns: {list(grid_df.columns)}"
            )

        if col_date and col_date in grid_df.columns:
            grid_df[col_date] = pd.to_datetime(grid_df[col_date], errors='coerce')
            grid_df = grid_df.dropna(subset=[col_date])
            date_frequency = 'input date column'
            tasks_df = grid_df.copy()
        else:
            date_frequency = cfg.get('date_frequency', 'daily')
            date_range = _date_range_for_frequency(
                cfg['date_from'], cfg['date_to'], date_frequency
            )
            base = grid_df.copy()
            base['_j'] = 1
            dr_df = pd.DataFrame({'date': date_range, '_j': 1})
            tasks_df = base.merge(dr_df, on='_j').drop(columns='_j')
            col_date = 'date'

        date_limit = int(cfg.get('date_limit', 0) or 0)
        if date_limit > 0:
            task_dates = pd.to_datetime(tasks_df[col_date]).dt.normalize()
            keep_dates = sorted(task_dates.dropna().unique())[:date_limit]
            before_limit = len(tasks_df)
            tasks_df = tasks_df[task_dates.isin(keep_dates)].copy()
            self._log(
                f'Date limit: first {len(keep_dates)} unique dates '
                f'({before_limit:,} → {len(tasks_df):,} rows)'
            )

        tasks_df['_task_id'] = (
            tasks_df[col_id].astype(str) + '_' +
            pd.to_datetime(tasks_df[col_date]).dt.strftime('%Y%m%d')
        )
        before_dupes = len(tasks_df)
        tasks_df = tasks_df.drop_duplicates(subset=['_task_id'], keep='last').reset_index(drop=True)
        dropped_dupes = before_dupes - len(tasks_df)
        cell_count = len(grid_df[[col_id]].drop_duplicates())
        self._log(f'Grid: {cell_count} cells × {len(tasks_df):,} tasks')
        if dropped_dupes:
            self._log(f'Dropped {dropped_dupes:,} duplicate cell-date rows')
        self._log(f'Date schedule: {date_frequency}')

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
        s2_ndvi_mode = cfg.get('s2_ndvi_mode', 'bimonthly')
        getters = build_getters(cfg.get('var_groups'), s2_ndvi_mode=s2_ndvi_mode)
        selected_vars = set(getters)
        passthrough_cols = [c for c in input_cols if c in tasks_df.columns]
        passthrough_out_cols = []
        passthrough_name_map = {}
        for c in passthrough_cols:
            out_c = f'{c}_input' if c in selected_vars and c not in {col_id, col_lat, col_lon, col_date} else c
            passthrough_name_map[c] = out_c
            if out_c not in passthrough_out_cols:
                passthrough_out_cols.append(out_c)
        use_era5_batch = bool(cfg.get('use_era5_batch', False))
        era5_land_batch_vars = [v for v in ['T2M', 'DEW', 'TP', 'SP', 'U10', 'V10', 'SSR', 'Soil_Moist']
                                if use_era5_batch and v in selected_vars]
        era5_daily_batch_vars = [v for v in ['MSLP'] if use_era5_batch and v in selected_vars]
        era5_hourly_daily_mean_vars = [v for v in ['BLH'] if use_era5_batch and v in selected_vars]
        batch_vars = set(era5_land_batch_vars + era5_daily_batch_vars + era5_hourly_daily_mean_vars)
        self._log(
            f'Variables ({len(getters)}): {", ".join(getters.keys())}'
        )
        if s2_ndvi_mode == 'bimonthly' and 'NDVI' in selected_vars:
            self._log('Sentinel-2 NDVI mode: bi-monthly composites (1–15 / 16–month end) with cache')
        if era5_land_batch_vars:
            self._log(f'ERA5-Land batched request: {", ".join(era5_land_batch_vars)}')
        if era5_daily_batch_vars:
            self._log(f'ERA5 Daily batched request: {", ".join(era5_daily_batch_vars)}')
        if era5_hourly_daily_mean_vars:
            self._log(f'ERA5 daily mean from hourly source: {", ".join(era5_hourly_daily_mean_vars)}')
        self._log(
            f'Post-extraction: WS, WD10, RH (derived met)'
            + (f', EM_m / EH_m / WRND (road metrics)' if cfg.get('roads_shp') else '')
        )

        # 5 ── Extraction loop ─────────────────────────────────────────
        chk_write_lock = threading.Lock()
        cache_lock = threading.Lock()
        value_cache = {}
        bulk_proxy_vars = [v for v in getters if v in BULK_PROXY_VARS]
        remaining_nonbulk_vars = [v for v in getters if v not in BULK_PROXY_VARS]
        bulk_proxy_mode = bool(bulk_proxy_vars) and bool(cfg.get('use_bulk_proxy', True))
        bulk_chunk_size = int(cfg.get('bulk_chunk_size', 10000) or 10000)
        bulk_chunk_cache = bool(cfg.get('bulk_chunk_cache', True))
        bulk_cache_dir = os.path.join(cfg['output_dir'], 'bulk_chunk_cache')
        if bulk_proxy_mode and bulk_chunk_cache:
            os.makedirs(bulk_cache_dir, exist_ok=True)
        geometry_mode = cfg.get('geometry_mode', 'point')
        use_cell_geom = geometry_mode == 'cell'
        use_buffer_geom = geometry_mode == 'buffer'
        buffer_m = float(cfg.get('buffer_m', LEGACY_BUFFER_M) or LEGACY_BUFFER_M)
        if use_cell_geom and not cfg.get('grid_resolution_m', 0):
            geometry_mode = 'point'
            use_cell_geom = False
        self._log(
            f'Extraction geometry: '
            f'{"250 m buffered point" if use_buffer_geom else "cell area average" if use_cell_geom else "point sample"}'
        )

        def _cached(cache_key, fn):
            with cache_lock:
                if cache_key in value_cache:
                    return value_cache[cache_key]
            val = fn()
            with cache_lock:
                value_cache[cache_key] = val
            return val

        def _base_record(row):
            rec = {'_task_id': row['_task_id']}
            for c in passthrough_cols:
                rec[passthrough_name_map[c]] = row[c]
            rec['date'] = pd.to_datetime(row[col_date]).strftime('%Y-%m-%d')
            return rec

        def _row_geometry(row):
            resolution_m = cfg.get('grid_resolution_m', 0)
            pt = ee.Geometry.Point([float(row[col_lon]), float(row[col_lat])])
            if use_buffer_geom:
                geom = pt.buffer(buffer_m).bounds()
            elif use_cell_geom and resolution_m and resolution_m > 0:
                geom = pt.buffer(resolution_m / 2).bounds()
            else:
                geom = pt
            return geom

        def _extract_nonbulk_vars(rec, row, date_str, var_names):
            if not var_names:
                return rec
            geom = _row_geometry(row)
            var_workers = cfg.get('var_workers', 4)
            selected = [(k, getters[k]) for k in var_names if k in getters]

            def _extract_var(kf):
                k, f = kf
                try:
                    return k, f(date_str, geom)
                except Exception:
                    return k, None

            with ThreadPoolExecutor(max_workers=var_workers) as var_ex:
                for key, val in var_ex.map(_extract_var, selected):
                    rec[key] = val
            return rec

        def _write_checkpoint(records):
            if not records:
                return
            with chk_write_lock:
                pd.DataFrame(records).to_csv(
                    chk_path, mode='a',
                    header=not os.path.exists(chk_path),
                    index=False,
                )

        if bulk_proxy_mode:
            self._log(
                f'Bulk daily proxy mode: {", ".join(bulk_proxy_vars)} '
                f'({bulk_chunk_size} cells/request chunk)'
            )
            if bulk_chunk_cache:
                self._log(f'Bulk chunk cache: {bulk_cache_dir}')
            if remaining_nonbulk_vars:
                self._log(f'Non-bulk variables will be handled outside bulk mode: {", ".join(remaining_nonbulk_vars)}')
            new_records = []

            chunk_jobs = []
            for date_str, date_tasks in tasks_todo.groupby(
                pd.to_datetime(tasks_todo[col_date]).dt.strftime('%Y-%m-%d'),
                sort=True,
            ):
                n_date_tasks = len(date_tasks)
                for start_idx in range(0, n_date_tasks, bulk_chunk_size):
                    chunk_jobs.append((
                        date_str,
                        start_idx,
                        n_date_tasks,
                        date_tasks.iloc[start_idx:start_idx + bulk_chunk_size].copy(),
                    ))

            bulk_workers = int(cfg.get('bulk_workers') or min(cfg.get('max_workers', 24), 10))
            self._log(f'Bulk parallel workers: {bulk_workers} date/chunk jobs')

            def _run_bulk_chunk(job):
                date_str, start_idx, n_date_tasks, chunk = job
                chunk_ids = '|'.join(chunk['_task_id'].astype(str).tolist())
                chunk_key = hashlib.sha1(
                    f'{date_str}|{start_idx}|{geometry_mode}|{cfg.get("grid_resolution_m", 0)}|'
                    f'{",".join(bulk_proxy_vars)}|{",".join(remaining_nonbulk_vars)}|{chunk_ids}'.encode('utf-8')
                ).hexdigest()[:16]
                cache_path = os.path.join(bulk_cache_dir, f'{date_str}_{start_idx:08d}_{chunk_key}.csv')
                if bulk_chunk_cache and os.path.exists(cache_path):
                    return date_str, start_idx, n_date_tasks, pd.read_csv(cache_path).to_dict('records')

                values = _extract_proxy_bulk(
                    date_str, chunk, bulk_proxy_vars, col_lat, col_lon,
                    cfg.get('grid_resolution_m', 0), s2_ndvi_mode, self._log,
                    geometry_mode,
                )
                chunk_records = []
                for _, row in chunk.iterrows():
                    rec = _base_record(row)
                    rec.update(values.get(str(row['_task_id']), {}))
                    for var_name in bulk_proxy_vars:
                        rec.setdefault(var_name, None)
                    rec = _extract_nonbulk_vars(rec, row, date_str, remaining_nonbulk_vars)
                    chunk_records.append(rec)
                if bulk_chunk_cache:
                    pd.DataFrame(chunk_records).to_csv(cache_path, index=False)
                return date_str, start_idx, n_date_tasks, chunk_records

            with ThreadPoolExecutor(max_workers=bulk_workers) as bulk_ex:
                futures = [bulk_ex.submit(_run_bulk_chunk, job) for job in chunk_jobs]
                for fut in as_completed(futures):
                    with self._lock:
                        if self.state['status'] == 'stopped':
                            break
                    date_str, start_idx, n_date_tasks, chunk_records = fut.result()

                    _write_checkpoint(chunk_records)
                    new_records.extend(chunk_records)

                    with self._lock:
                        self.state['progress'] += len(chunk_records)
                        prog = self.state['progress']
                        self.state['pct'] = int(prog / n_total * 100) if n_total else 0
                    self._log(
                        f'[{prog}/{n_total}] {date_str}  '
                        f'{min(start_idx + len(chunk_records), n_date_tasks)}/{n_date_tasks} cells  '
                        f'({self.state["pct"]}%)'
                    )
        else:
            new_records = []

        def extract_one(i_row):
            i, row = i_row
            with self._lock:
                if self.state['status'] == 'stopped':
                    return None

            date_str = pd.to_datetime(row[col_date]).strftime('%Y-%m-%d')
            resolution_m = cfg.get('grid_resolution_m', 0)
            geom = _row_geometry(row)
            cell_key = (
                f'{row[col_id]}|{float(row[col_lat]):.8f}|'
                f'{float(row[col_lon]):.8f}|{resolution_m}|{geometry_mode}|{buffer_m}'
            )
            month_key = pd.to_datetime(date_str).strftime('%Y-%m')
            half_month_key = _half_month_bounds(date_str)[0]

            rec = _base_record(row)

            if era5_land_batch_vars:
                rec.update(_era5_land_batch(date_str, geom, era5_land_batch_vars))
            if era5_daily_batch_vars:
                rec.update(_era5_daily_batch(date_str, geom, era5_daily_batch_vars))
            if era5_hourly_daily_mean_vars:
                rec.update(_era5_hourly_batch(date_str, geom, era5_hourly_daily_mean_vars))

            if 'Elevation' in selected_vars:
                rec['Elevation'] = _cached(
                    ('Elevation', cell_key),
                    lambda: getters['Elevation'](date_str, geom),
                )
            if 'Slope' in selected_vars:
                rec['Slope'] = _cached(
                    ('Slope', cell_key),
                    lambda: getters['Slope'](date_str, geom),
                )
            if 'NDVI' in selected_vars and s2_ndvi_mode == 'bimonthly':
                rec['NDVI'] = _cached(
                    ('NDVI', cell_key, half_month_key),
                    lambda: getters['NDVI'](date_str, geom),
                )
            if 'NDVI_MO' in selected_vars:
                rec['NDVI_MO'] = _cached(
                    ('NDVI_MO', cell_key, month_key),
                    lambda: getters['NDVI_MO'](date_str, geom),
                )
            if 'VIIRS_NTL' in selected_vars:
                rec['VIIRS_NTL'] = _cached(
                    ('VIIRS_NTL', cell_key, month_key),
                    lambda: getters['VIIRS_NTL'](date_str, geom),
                )
            if 'BurnedArea' in selected_vars:
                rec['BurnedArea'] = _cached(
                    ('BurnedArea', cell_key, month_key),
                    lambda: getters['BurnedArea'](date_str, geom),
                )

            var_workers = cfg.get('var_workers', 6)
            remaining_getters = [(k, f) for k, f in getters.items()
                                 if k not in rec and k not in batch_vars]

            def _extract_var(kf):
                k, f = kf
                try:
                    return k, f(date_str, geom)
                except Exception:
                    return k, None

            with ThreadPoolExecutor(max_workers=var_workers) as var_ex:
                for key, val in var_ex.map(_extract_var, remaining_getters):
                    rec[key] = val

            # Write to checkpoint immediately
            _write_checkpoint([rec])

            with self._lock:
                self.state['progress'] += 1
                prog = self.state['progress']
                self.state['pct'] = int(prog / n_total * 100) if n_total else 0

            self._log(
                f'[{prog}/{n_total}] {row[col_id]}  {date_str}  '
                f'({self.state["pct"]}%)'
            )
            return rec

        if not bulk_proxy_mode:
            with ThreadPoolExecutor(max_workers=cfg.get('max_workers', 24)) as ex:
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

        meta = {'_task_id', 'date', *passthrough_out_cols}
        var_cols = [c for c in raw_df.columns if c not in meta]
        for col in var_cols:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

        self._log(f'Gap-filling {len(var_cols)} variable columns (same-cell ±7-day carry fill; no medians) …')
        missing_before = raw_df[var_cols].isna().sum() if var_cols else pd.Series(dtype='int64')
        final_df = gap_fill(raw_df, col_id, 'date', var_cols)
        missing_after = final_df[var_cols].isna().sum() if var_cols else pd.Series(dtype='int64')

        if var_cols:
            fill_summary = pd.DataFrame({
                'variable': var_cols,
                'missing_before': [int(missing_before.get(c, 0)) for c in var_cols],
                'missing_after': [int(missing_after.get(c, 0)) for c in var_cols],
                'filled_values': [int(missing_before.get(c, 0) - missing_after.get(c, 0)) for c in var_cols],
                'fill_method': ['same_cell_previous_then_next_within_7_days' for _ in var_cols],
            })
            fill_summary.to_csv(os.path.join(cfg['output_dir'], 'gap_fill_summary.csv'), index=False)
            self._log(
                f'Gap-fill summary written: '
                f'{os.path.join(cfg["output_dir"], "gap_fill_summary.csv")}'
            )

        self._log('Computing derived meteorology: WS, WD10, RH …')
        final_df = add_derived_met(final_df)

        gas_cols = [g for g in GAS_TROPOSPHERIC if g in final_df.columns]
        if gas_cols and 'BLH' in final_df.columns:
            self._log(f'Converting tropospheric S5P gases to µg/m³ via BLH: '
                      f'{", ".join(g + "_ugm3" for g in gas_cols)} …')
            final_df = add_gas_surface_ugm3(final_df)

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
        core = ['_task_id']
        for c in passthrough_out_cols:
            if c in final_df.columns and c not in core:
                core.append(c)
        if 'date' in final_df.columns and 'date' not in core:
            core.append('date')
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
