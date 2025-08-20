# Solare Flare Forecasting

The output includes binary labels based on both the **maximum flare class** and **cumulative flare intensity** within a given time window.


## Setup

Ensure the following dependencies are installed:

### 🚀 Example Usage

For training run the below code after building the environment


```sh
cd downstream_examples/solar_flare_forcasting
bash download_data.sh
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
```
### Data and pretrained weights

- The dataset is hosted on Hugging Face: [nasa-ibm-ai4science/surya-bench-flare-forecasting](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting/tree/main)
- The weights can be found at [model/nasa-ibm-ai4science/solar_flares_surya](https://huggingface.co/nasa-ibm-ai4science/euv_spectra_surya/tree/main)

### Dataset

The csv file should be in the format of

```csv
timestep,max_goes_class,cumulative_index,label_max,label_cum
2011-01-01 00:00:00,B8.3,0.0,0,0
2011-01-01 01:00:00,B8.3,0.0,0,0
```

And to use the dataset:
```python

from dataset import SolarFlareDataset

dataset = SolarFlareDataset(
    index_path=config["data"]["train_data_path"],
    time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
    time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
    n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
    rollout_steps=config["rollout_steps"],
    channels=config["data"]["channels"],
    drop_hmi_probability=config["drop_hmi_probability"],
    num_mask_aia_channels=config["num_mask_aia_channels"],
    use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
    scalers=scalers,
    phase="train",
    flare_index_path=config["data"]["flare_data_path"],
    pooling=config["data"]["pooling"],
    random_vert_flip=config["data"]["random_vert_flip"]
)

```
