# EVE-AIA Dataset Preparation and Loader for HelioFM

This repository provides utilities for preparing and loading EUV irradiance spectra from NASA's EVE instrument, aligned with input data from the HelioFM framework. The pipeline enables ML-ready formatting and timestamp-synchronized dataset loading using PyTorch.

---


## Setup

Ensure the following dependencies are installed:


### 🚀 Example Usage

For training run the below code after building the environment


```sh
cd downstream_examples/euv_spectra_prediction
bash download_data.sh
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
```

### Data and pretrained weights

- The dataset is hosted on Hugging Face: [nasa-ibm-ai4science/euv-spectra](https://huggingface.co/datasets/nasa-ibm-ai4science/euv-spectra/tree/main)
- The weights can be found at [model/nasa-ibm-ai4science/euv_spectra_surya](https://huggingface.co/nasa-ibm-ai4science/euv_spectra_surya/tree/main)

Preprocessing Details

    Zero handling: Zero values in EVE spectra are replaced by the wavelength-wise minimum (avoids -inf when taking log10)

    Log-scaling: Intensities are compressed using log10 for dynamic range reduction

    Normalization: Spectra are scaled globally using predefined min/max values (-9.00 to -1.96 in log10 space)

Output Format
Each training sample returns:

    ts: 3D temporal image data from HelioFM inputs (shape: (13, 4096, 4096))

    target: Normalized EVE spectrum vector (length 1343)

## EVE EUV Spectra Prediction 

This  contains code and model implementations for predicting the EVE spectra from spatiotemporal solar data. The dataset contains hundreds of thousands of temporally aligned AIA image cubes and EUV spectra, covering Solar Cycle 24 and parts of Solar Cycle 25, and including both quiet-Sun and active-region conditions. 

---

### 📊 Dataset Description

**Dataset can be found at [NASA-IMPACT HuggingFace Repository](https://huggingface.co/datasets/nasa-impact/surya-bench-euv-spectra)**

The dataset consists of three splits (70-15-15): train, val, and test, each containing:
- A timestamp
- A 1343-dimensional spectrum corresponding to EUV wavelengths ranging from approximately 6.5 to 33.3 nm, observed at a 1-minute cadence
- EUV measurements from both flare and quiet sun conditions
- Input shape: (1, 13, 4096, 4096)
- Output shape: (1, 1343)

