# 🌞 SDO Real-Time Data Fetcher

A lightweight Python tool for fetching live Solar Dynamics Observatory (SDO) browse imagery for Surya-related experimentation and monitoring.

## Overview

While Surya primarily uses preprocessed SDO datasets, this tool provides a simple way to fetch current AIA and HMI observations for:

- real-time solar activity monitoring
- quick data exploration
- recent-event validation workflows
- prototype preprocessing and inference pipelines

## What's New

The downloader now supports redundant live providers so it can continue working when a single upstream host is unavailable.

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
```

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
```

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

## Notes

- LMSAL provides daily AIA and HMI browse imagery.
- JSOC support is currently most useful for HMI live products.
- Helioviewer remains as an API fallback when browse-image hosts are unavailable.

## References

- NASA SDO: https://sdo.gsfc.nasa.gov/
- LMSAL Sun Today: https://suntoday.lmsal.com/suntoday/
- JSOC latest HMI: https://jsoc1.stanford.edu/hmi_latest.html
- Helioviewer: https://helioviewer.org/
