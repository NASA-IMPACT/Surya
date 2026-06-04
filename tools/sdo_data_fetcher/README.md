# 🌞 SDO Data Fetcher

A lightweight Python tool for fetching live and historical Solar Dynamics Observatory (SDO) browse imagery for Surya-related experimentation and monitoring.

## Overview

While Surya primarily uses preprocessed SDO datasets, this tool provides a simple way to fetch current and recent-past AIA/HMI observations for:

- real-time solar activity monitoring
- flare and active-region review windows
- quick data exploration
- recent-event validation workflows
- prototype preprocessing and inference pipelines

## What's New

The downloader now supports redundant live providers plus Helioviewer-backed historical target-time downloads. You can fetch all available SDO wavelengths from a target date/time forward for several hours and review the results in a dependency-free local web UI.

### Provider fallback chain

By default, the fetchers try providers in this order:

1. `lmsal` — LMSAL Sun Today browse imagery
2. `jsoc` — Stanford JSOC latest HMI imagery
3. `nasa` — NASA SDO browse imagery
4. `helioviewer` — Helioviewer rendered imagery

This is available through both `sdo_fetcher_v2.py` and `sdo_data_fetcher.py` with `--provider auto`.

## Quick Start

```bash
cd tools/sdo_data_fetcher
pip install -r requirements.txt
```

### Basic usage

```bash
# Latest AIA 171 image using automatic fallback
python sdo_fetcher_v2.py --source AIA_171

# Force a specific provider
python sdo_fetcher_v2.py --source AIA_171 --provider lmsal

# Download latest HMI magnetogram from JSOC
python sdo_fetcher_v2.py --source HMI_Magnetogram --provider jsoc

# Download multiple channels
python sdo_fetcher_v2.py --multiple

# Download all available SDO sources from a UTC target time forward for 4 hours
python sdo_fetcher_v2.py --datetime "2026-02-06T12:30:00Z" --all --hours 4 --cadence 15

# Interpret a timezone-naive datetime as local time before converting to UTC
python sdo_fetcher_v2.py --datetime "2026-02-06T07:30:00" --timezone local --all --hours 3 --cadence 15

# Launch the retro local web UI
python sdo_web_ui.py
```

Open `http://127.0.0.1:8765` after launching the web UI.

### Original fetcher

```bash
python sdo_data_fetcher.py --source AIA_193 --provider auto
```

## Available providers

- `auto`
- `lmsal`
- `jsoc`
- `nasa`
- `helioviewer`

## Available data sources

### AIA channels

- `AIA_94`
- `AIA_131`
- `AIA_171`
- `AIA_193`
- `AIA_211`
- `AIA_304`
- `AIA_335`
- `AIA_1600`
- `AIA_1700`
- `AIA_4500`

### HMI channels

- `HMI_Continuum`
- `HMI_Magnetogram`

## Python example

```python
from sdo_fetcher_v2 import SDOFetcher

fetcher = SDOFetcher(output_dir="surya_inference_data")
metadata = fetcher.get_latest_image_direct(source="AIA_171", provider="auto")

if metadata:
    print(metadata["filepath"])
    print(metadata["provider_name"])
    print(metadata.get("observation_time"))

manifest = fetcher.download_time_series(
    sources=list(fetcher.SDO_SOURCES.keys()),
    start_time="2026-02-06T12:30:00Z",
    timezone_mode="utc",
    hours=4,
    cadence_minutes=15,
)
```

## Historical solar moment UI

Run the dependency-free local web app:

```bash
python sdo_web_ui.py
```

The Intel-blue retro console lets you choose:

- target date and time
- `UTC` or `Local timezone`
- forward-only duration in hours
- sampling cadence in minutes
- all wavelengths or selected wavelengths
- image width and format

Historical results are grouped by requested timestamp with links to each rendered image and JSON metadata. Historical fetching uses Helioviewer because the other providers in this tool are latest/browse feeds.

## Advanced examples

Run the menu-driven helper:

```bash
python sdo_advanced_examples.py
```

It includes:

- multi-wavelength comparison downloads
- active region monitoring
- prominence monitoring
- space weather quick checks
- continuous monitoring

## Output

Each download writes:

- an image file (`.jpg`, `.gif`, or `.png`, depending on provider)
- a `.json` metadata file containing provider, source, URL, and timing info
- for historical windows, a `manifest.json` summarizing requested samples, successful downloads, and failures

## Notes

- LMSAL provides daily AIA and HMI browse imagery.
- JSOC support is currently most useful for HMI live products.
- Helioviewer remains as an API fallback when browse-image hosts are unavailable.

## References

- NASA SDO: https://sdo.gsfc.nasa.gov/
- LMSAL Sun Today: https://suntoday.lmsal.com/suntoday/
- JSOC latest HMI: https://jsoc1.stanford.edu/hmi_latest.html
- Helioviewer: https://helioviewer.org/
