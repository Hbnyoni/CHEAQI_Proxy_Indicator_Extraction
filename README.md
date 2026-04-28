# GIPEX — Geospatial Indicators for Proxy Environmental eXposure

**CHEAQI-MNCH Research Tool · v2.1 · 2026**

GIPEX is a Dash web application that extracts 40+ environmental and geospatial proxy indicators from Google Earth Engine (GEE) for user-defined grid cells, with support for time-series analysis, spatial mapping, and automated report generation.

---

## Features

- **40+ variables** extracted via GEE: Sentinel-2, MODIS, ERA5-Land, ERA5-Hourly, Sentinel-5P, Dynamic World, SRTM terrain, VIIRS night-time lights, and the **Normalized Difference Impervious Index (NDII)** from Landsat 8 C2 L2
- **Dual-pool parallel extraction**: outer pool (tasks) × inner pool (variables per task) for maximum speed
- **Multi-format data ingestion**: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet, and URL upload
- **Pre-loaded OSM roads** SQLite database (2.8 M segments) for road-proximity metrics — no shapefile paths exposed in the UI
- **Interactive visualisations**: scatter map with OSM overlay, time-series, correlation heatmap, box plots
- **HTML report** with summary statistics, charts, and metadata — downloadable from the browser
- **Checkpoint/resume**: extraction saves progress to CSV so jobs can be interrupted and resumed

---

## Quick Start

```bash
# Install dependencies
pip install dash dash-bootstrap-components plotly pandas numpy geopandas \
            earthengine-api requests openpyxl xlrd pyarrow shapely

# Run
python app.py
# Open http://localhost:8087
```

---

## Indicators

| Group | Variables |
|---|---|
| Sentinel-2 | NDVI, NDBI, NDWI, MNDWI, SAVI, MSAVI, GCI, ARVI, EVI2 |
| MODIS | EVI, NDVI_MO, NDWI_MO, LST_C, LSTN_C, ET, FRP, FireMask, BurnedArea, Soil_Moist |
| ERA5-Land | T2M, DEW, TP, SP, U10, V10, SSR |
| ERA5 Hourly | BLH, MSLP |
| Sentinel-5P | NO2, AOD_S5P, SPM25 |
| Dynamic World | DW_label, BuiltUp |
| **Impervious** | **NDII** (Landsat 8 C2 L2, 30 m) |
| Terrain | Elevation, Slope |
| VIIRS NTL | VIIRS_NTL |
| Derived | WS, WD10, RH |
| Roads (OSM) | EM_m, EH_m, WRND_km_km2 |

### NDII — Normalized Difference Impervious Index

```
NDII = (Red − TIR_norm) / (Red + TIR_norm)
```

- **Red**: Landsat 8 SR_B4 (surface reflectance, scaled to [0, 1])
- **TIR_norm**: Landsat 8 ST_B10 (surface temperature, Kelvin), normalised to [0, 1] using a 250–350 K range
- Cloud-masked using QA_PIXEL (cloud + cloud shadow flags); ±16-day window for cloud-gap filling

---

## Grid CSV Format

```csv
cell_id,lat,lon,date_only
ZA_001,-26.1985,28.0464,2022-06-15
ZA_002,-26.2100,28.0600,2022-06-15
```

Columns `lat`, `lon`, `cell_id` are required. `date_only` is optional — if absent, a date range is used.

---

## Configuration

| Field | Description | Default |
|---|---|---|
| GEE Project | Earth Engine project ID | `ee-cheaqi` |
| Task workers (outer) | Parallel cell×date tasks | 50 |
| Var workers (inner) | Parallel variables per task | 6 |

---

## Project

Developed as part of the **CHEAQI-MNCH**  research programme.

**Developer**: CHEAQI-MNCH Research Team  
**Contact**: hbnyoni@gmail.com  /nyonih@staff.msu.ac.zw
**License**: MIT
