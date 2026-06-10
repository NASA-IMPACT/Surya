# 🛠️ Surya Tools

Utility tools and scripts to support the Surya foundation model ecosystem.

## Available Tools

### 📡 SDO Data Fetcher

Lightweight tool for fetching real-time and historical Solar Dynamics Observatory observations.

**Location**: `tools/sdo_data_fetcher/`

**Purpose**: 
- Fetch latest SDO/AIA and HMI observations
- Fetch target-time SDO windows forward from a flare or active-region moment
- Download specific wavelengths for custom analysis
- Review all available wavelengths in a dependency-free local web UI
- Real-time solar activity monitoring
- Quick data exploration without full dataset downloads

**Quick Start**:
```bash
cd tools/sdo_data_fetcher
pip install -r requirements.txt
python sdo_fetcher_v2.py --list
python sdo_web_ui.py
```

See [sdo_data_fetcher/README.md](sdo_data_fetcher/README.md) for detailed documentation.

---

## Contributing New Tools

We welcome additional tools that support the Surya ecosystem! Consider contributing:
- Data preprocessing utilities
- Visualization tools
- Custom dataset generators
- Analysis scripts
- Integration helpers

Please follow the repository's contribution guidelines when adding new tools.
