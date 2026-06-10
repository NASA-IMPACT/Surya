# 🌞 SDO Data Fetcher

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NASA SDO](https://img.shields.io/badge/NASA-SDO-red.svg)](https://sdo.gsfc.nasa.gov/)

A powerful Python application to fetch real-time and historical solar images from NASA's **Solar Dynamics Observatory (SDO)**. Get solar data in seconds with support for all AIA wavelengths and HMI instruments!

<p align="center">
  <img src="https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg" width="300" alt="SDO AIA 171">
  <img src="https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0304.jpg" width="300" alt="SDO AIA 304">
</p>

## ✨ Features

- 🔴 **Live Data** - Fetches the latest SDO observations with automatic provider fallback
- 🌈 **12 Sources** - All available SDO AIA channels plus HMI continuum and magnetogram
- 🕒 **Historical Target Times** - Fetch SDO imagery closest to a specific date/time through Helioviewer
- 🕹️ **Retro Web UI** - Dependency-free local Intel-blue console for reviewing flare windows
- 📊 **Auto Metadata** - Each image includes JSON metadata with observation details
- ⚡ **Redundant Sources** - Automatically falls back across LMSAL Sun Today, Stanford JSOC, NASA SDO, and Helioviewer
- 🎯 **CLI & Python API** - Use from command line or integrate into your code
- 📦 **Batch Downloads** - Get multiple wavelengths simultaneously
- 🔬 **Space Weather Ready** - Perfect for monitoring solar activity

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ep150de/sdo-data-fetcher.git
cd sdo-data-fetcher

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Get the latest AIA 171Å image (default)
python sdo_fetcher_v2.py

# Prefer high-resolution rendered output first, then fallback
python sdo_fetcher_v2.py --source AIA_171 --provider auto_highres --width 4096 --image-type png

# Force a specific provider
python sdo_fetcher_v2.py --source AIA_171 --provider lmsal

# Get a specific wavelength
python sdo_fetcher_v2.py --source AIA_304

# Download multiple wavelengths
python sdo_fetcher_v2.py --multiple

# Download all wavelengths from a target UTC time forward for 4 hours
python sdo_fetcher_v2.py --datetime "2026-02-06T12:30:00Z" --all --hours 4 --cadence 15

# Launch the local retro web UI
python sdo_web_ui.py

# List all available sources
python sdo_fetcher_v2.py --list
```

Then open `http://127.0.0.1:8765` to use the web console.

## 📡 Available Data Sources

| Source | Wavelength | Temperature | Best For |
|--------|------------|-------------|----------|
| **AIA_94** | 94 Å | ~6 MK | Hot flare plasma |
| **AIA_131** | 131 Å | ~10 MK | Flaring regions |
| **AIA_171** | 171 Å | ~0.6 MK | Quiet corona, coronal loops ⭐ |
| **AIA_193** | 193 Å | ~1.5 MK | Active regions |
| **AIA_211** | 211 Å | ~2 MK | Active regions |
| **AIA_304** | 304 Å | ~0.05 MK | Prominences, filaments |
| **AIA_335** | 335 Å | ~2.5 MK | Active regions |
| **AIA_1600** | 1600 Å | - | Upper photosphere |
| **AIA_1700** | 1700 Å | - | Temperature minimum |
| **AIA_4500** | 4500 Å | - | Visible light photosphere |
| **HMI_Continuum** | Visible | - | Solar surface |
| **HMI_Magnetogram** | - | - | Magnetic fields |

## 💡 Usage Examples

### Command Line

```bash
# Monitor solar activity
python sdo_fetcher_v2.py --source AIA_193

# Review a flare or active region from a specific UTC time
python sdo_fetcher_v2.py --datetime "2026-02-06T12:30:00Z" --source AIA_131 --hours 2 --cadence 10

# Interpret a timezone-naive time as your local timezone, then convert to UTC
python sdo_fetcher_v2.py --datetime "2026-02-06T07:30:00" --timezone local --all --hours 3 --cadence 15

# Space weather check
python sdo_advanced_examples.py  # Choose option 3

# Download full comparison set
python sdo_advanced_examples.py  # Choose option 1
```

### Python Code

```python
from sdo_fetcher_v2 import SDOFetcher

# Initialize fetcher
fetcher = SDOFetcher(output_dir="solar_images")

# Download latest image
metadata = fetcher.get_latest_image_direct(source="AIA_171")

if metadata:
    print(f"Image saved: {metadata['filepath']}")
  print(f"Provider: {metadata['provider_name']}")
  print(f"Observation time: {metadata['observation_time']}")

# Download multiple wavelengths
sources = ["AIA_171", "AIA_193", "AIA_304", "HMI_Magnetogram"]
results = fetcher.download_multiple(sources)

# Download all wavelengths forward from a target time
manifest = fetcher.download_time_series(
    sources=list(fetcher.SDO_SOURCES.keys()),
    start_time="2026-02-06T12:30:00Z",
    timezone_mode="utc",
    hours=4,
    cadence_minutes=15,
)
```

## 🕹️ Historical Solar Moment Web UI

Run the dependency-free local web app:

```bash
python sdo_web_ui.py
```

Open `http://127.0.0.1:8765` and enter:

- Target date and time
- Timezone mode: `UTC` or `Local timezone`
- Forward-only duration in hours
- Sampling cadence in minutes
- Image width and format
- All wavelengths or selected wavelengths

The UI downloads the closest available Helioviewer-rendered image for each selected SDO source at each sample time. Results are grouped by requested timestamp with links to each image and its JSON metadata.

Historical fetching uses Helioviewer because the other providers in this repository are latest/browse feeds rather than arbitrary-time APIs.

## 🔁 Redundant Live Data Providers

The fetchers now support a provider chain for current imagery:

- **LMSAL Sun Today** - Daily AIA and HMI browse images at `suntoday.lmsal.com`
- **Stanford JSOC** - Latest HMI browse products at `jsoc1.stanford.edu`
- **NASA SDO** - Latest public browse images at `sdo.gsfc.nasa.gov`
- **Helioviewer** - API-based rendered imagery fallback (and high-resolution first mode)

Use `--provider auto` for browse-first fallback, `--provider auto_highres` for high-resolution-first fallback, or pick one explicitly with `--provider lmsal`, `--provider jsoc`, `--provider nasa`, or `--provider helioviewer`.

For Helioviewer downloads (latest and historical), you can also request render settings with `--width` and `--image-type`.
If fallback lands on browse providers (LMSAL/JSOC/NASA), those settings are not applied and metadata marks `render_settings_applied: false` with `resolution_class: browse_fixed`.

## 🎓 Advanced Features

The `sdo_advanced_examples.py` script includes:

1. **Multi-wavelength comparison sets** - Download complementary wavelengths for analysis
2. **Active region monitoring** - Track solar flares and active regions
3. **Space weather quick check** - Rapid assessment tool
4. **Prominence monitoring** - Track eruptions and filaments
5. **Continuous monitoring** - Automated periodic downloads
6. **Monitoring daemon generator** - Create long-running monitoring scripts

```bash
python sdo_advanced_examples.py
```

## 📂 Output Structure

```
sdo_data/
├── SDO_AIA_171_20260206_123456.jpg    # Solar image
├── SDO_AIA_171_20260206_123456.json   # Metadata
├── SDO_AIA_304_20260206_123457.jpg
├── SDO_AIA_304_20260206_123457.json
└── historical_20260206_123000Z/
    ├── manifest.json
    └── 20260206_123000Z/
        ├── SDO_AIA_171_20260206_123000Z.png
        └── SDO_AIA_171_20260206_123000Z.json
```

Each JSON file contains:
- Source and wavelength information
- Exact observation timestamp
- Requested historical timestamp and closest actual observation timestamp
- Time delta between requested and actual observation
- Download metadata
- Direct image URL
- Render provenance fields (`requested_image_width`, `requested_image_type`, `render_settings_applied`, `resolution_class`)

## 🔬 About NASA's SDO

The **Solar Dynamics Observatory** is a NASA mission launched in February 2010 to study the Sun's atmosphere and magnetic activity. It provides:

- 🛰️ **24/7 observations** from geosynchronous orbit
- 📸 **4K images every 12 seconds** in 10 wavelengths
- 🧲 **Magnetic field measurements** of the Sun's surface
- ☀️ **Real-time space weather monitoring**
- 📊 **Over 20 million images captured** since launch

Learn more at [sdo.gsfc.nasa.gov](https://sdo.gsfc.nasa.gov/)

## 📖 Documentation

- **[SDO_GUIDE.md](SDO_GUIDE.md)** - Comprehensive guide with detailed examples
- **[QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)** - Quick command reference card
- **[NASA SDO Website](https://sdo.gsfc.nasa.gov/)** - Official mission website
- **[Helioviewer.org](https://helioviewer.org/)** - Interactive solar image viewer

## 🛠️ Requirements

- Python 3.7+
- `requests` library (installed via requirements.txt)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **NASA/SDO** and the AIA, EVE, and HMI science teams for providing open access to solar data
- **Helioviewer Project** for API access and tools
- All solar physics researchers and space weather forecasters

## 🔗 Useful Links

- [SDO Mission Overview](https://sdo.gsfc.nasa.gov/mission/)
- [Space Weather Prediction Center](https://www.swpc.noaa.gov/)
- [Solar Data Analysis Center](https://umbra.nascom.nasa.gov/)
- [Helioviewer API Docs](https://api.helioviewer.org/docs/)

---

**Made with ☀️ for solar physics research, education, and space weather monitoring**

*If you find this tool useful, please ⭐ star this repository!*
