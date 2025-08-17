# Solarwind Dataset Preparation and Loader for Surya

This repository provides utilities for loading solar wind data, aligned with input data from the HelioFM framework. The pipeline enables ML-ready formatting and timestamp-synchronized dataset loading using PyTorch.

## Setup

Ensure the following dependencies are installed:


### 🚀 Example Usage

For training run the below code after building the environment

```sh
git clone git@github.com:NASA-IMPACT/release-downstream-surya.git
cd release-downstream-surya

uv sync

cd downstream_examples/solar_wind_forcasting
uv run torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
```

### 📊 Dataset Description

**Dataset can be found at [NASA-IMPACT HuggingFace Repository](https://huggingface.co/datasets/nasa-impact/Surya-bench-solarwind)**

The dataset it stored as `.csv` files. Each sample in the dataset corresponds to a tracked active region and is structured as follows:
- Input shape: (1, 13, 4096, 4096)
- Temporal coverage of the dataset is `2010-05-01` to `2023-12-31`
- 5 physical quantities: V, Bx(GSE), By(GSM), Bz(GSM), Number Density (N)
- Input timestamps: (120748,)
- cadence: Hourly
- Output shape: (1)
- Output prediction:  (single value per prediction)

