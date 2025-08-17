# EVE-AIA Dataset Preparation and Loader for HelioFM

This repository provides utilities for preparing and loading EUV irradiance spectra from NASA's EVE instrument, aligned with input data from the HelioFM framework. The pipeline enables ML-ready formatting and timestamp-synchronized dataset loading using PyTorch.

---


## Setup

Ensure the following dependencies are installed:


### 🚀 Example Usage

For training run the below code after building the environment


```sh
git clone git@github.com:NASA-IMPACT/release-downstream-surya.git
cd release-downstream-surya

uv run torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
```



Run the prepare_data.ipynb notebook to generate:

    X_train.pt: input tensor of shape (N, 13, 4096, 4096)

    Y_train.csv: corresponding target spectra of shape (N, 1343)

Both files will be saved in the current directory.

Inside the prepare_data.ipynb notebook, we use eve_dataloader to construct the PyTorch-ready dataset aligned with EVE spectra, use the EVEDSDataset class like so:

```bash
from eve_dataloader import EVEDSDataset
train_dataset = eve_dataloader.EVEDSDataset(
    #### All these lines are required by the parent HelioNetCDFDataset class
    index_path=config.data.train_data_path,
    time_delta_input_minutes=config.data.time_delta_input_minutes,
    time_delta_target_minutes=config.data.time_delta_target_minutes,
    n_input_timestamps=config.data.n_input_timestamps,
    rollout_steps=config.rollout_steps,
    channels=config.data.channels,
    drop_hmi_probablity=config.drop_hmi_probablity,
    num_mask_aia_channels=config.num_mask_aia_channels,
    use_latitude_in_learned_flow=config.use_latitude_in_learned_flow,
    scalers=scalers,
    phase="train",
    #### Put your donwnstream (DS) specific parameters below this line
    ds_eve_index_path= "../../hfmds/data/AIA_EVE_dataset_combined.nc",
    ds_time_column="train_time",
    ds_time_tolerance = "6m",
    ds_match_direction = "forward"    
)
```
To load validation or test data, just change:

    phase → "val" or "test"

    ds_time_column → "val_time" or "test_time"

You can also modify ds_time_tolerance to change the matching window (e.g., "1m", "10s", "15m").

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

