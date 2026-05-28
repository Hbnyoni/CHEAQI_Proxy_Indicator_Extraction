#!/usr/bin/env bash
# GeoGrid Indicators Dashboard — startup script
# Mirrors Spectra's start.sh pattern (Spectra: port 8086, GeoGrid: port 8087)
cd /home/bongani/grid_indicators_app
exec /opt/anaconda3/bin/python3 app.py
