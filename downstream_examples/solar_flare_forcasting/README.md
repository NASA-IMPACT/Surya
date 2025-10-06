# Solar Flare Forecasting

The output includes binary labels based on both the **maximum flare class** and **cumulative flare intensity** within a given time window.


## Setup

Ensure the following dependencies are installed:

### 🚀 Training

For training run the below code after building the environment

```sh
cd downstream_examples/solar_flare_forcasting
bash download_data.sh
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
```


## Inference

Run Solar Flare forecasting inference using either the interactive notebook or command-line scripts.
**Prerequisites**: Download all the data using the [download_data.sh](download_data.sh) script.

### Option A: Interactive Notebook (Recommended for beginners)

The [solar_flare_tutorial.ipynb](solar_flare_tutorial.ipynb) notebook provides step-by-step guidance with explanations of forecasting results.

### Option B: Command-Line Inference

**Basic GPU Inference**
```bash
python infer.py --checkpoint_path ./assets/solar_flare_weights.pth \
                --output_dir ./inference_results \
                --num_samples 3 \
                --device cuda 
```

**CPU Inference** (slower but no GPU required)
```bash
python infer.py --checkpoint_path ./assets/solar_flare_weights.pth \
                --output_dir ./inference_results \
                --num_samples 3 \
                --device cpu
```

**Advanced Usage**
```bash
# Custom configuration and more samples
python infer.py --config_path ./config.yaml \
                --checkpoint_path ./assets/solar_flare_weights.pth \
                --output_dir ./custom_results \
                --num_samples 10 \
                --data_type valid \
                --device cuda
```

### Parameters Reference
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config_path` | `./config.yaml` | Path to model configuration file |
| `--checkpoint_path` | `./assets/solar_flare_weights.pth` | Path to trained model weights |
| `--output_dir` | `./inference_results` | Directory for saving results |
| `--num_samples` | `3` | Number of samples to process and analyze |
| `--data_type` | `test` | Dataset split to use (`test` or `valid`) |
| `--device` | `cuda` | Computing device (`cuda` or `cpu`) |

#### Output
- **Forecasting Results**: Tabular output showing timestamps, predictions, and ground truth
- **Format**: Console output with formatted table
- **Metrics**: Binary predictions (0 = No Flare, 1 = Flare) vs actual outcomes
- **Timestamps**: Input observation time and target prediction time

### Example Output
```
================================================================================
Time Input           | Time Target          | Prediction      | Ground Truth
--------------------------------------------------------------------------------
2014-01-07T12:00     | 2014-01-07T18:00     | 1               | 1
2014-01-08T06:00     | 2014-01-08T12:00     | 0               | 0
2014-01-09T15:30     | 2014-01-09T21:30     | 1               | 0
================================================================================
```

## Dataset Information

### Input Data
- **Format**: SDO/AIA multi-channel solar images with flare event labels
- **Shape**: (13, 4096, 4096) - 13 channels including:
  - AIA channels: 94Å, 131Å, 171Å, 193Å, 211Å, 304Å, 335Å, 1600Å
  - HMI channels: Magnetogram, Bx, By, Bz, Velocity
- **Temporal coverage**: 2011-2014
- **Cadence**: Hourly intervals
- **Labels**: Binary flare occurrence (0/1) for future time windows

### Output Data
- **Format**: Binary classification labels
- **Classes**: No Flare (0) and Flare (1)
- **Prediction Window**: 1-24 hours ahead (configurable)


### Data and pretrained weights

- The dataset is hosted on Hugging Face: [nasa-ibm-ai4science/surya-bench-flare-forecasting](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting/tree/main)
- The weights can be found at [model/nasa-ibm-ai4science/solar_flares_surya](https://huggingface.co/nasa-ibm-ai4science/solar_flares_surya)

### Custom Dataset
To use your own flare labels:

1. Format your CSV file with the required columns: `timestep`, `max_goes_class`, `cumulative_index`, `label_max`, `label_cum`
2. Update the configuration file to point to your data
3. Modify `config.yaml` with your data paths and parameters

The csv file should be in the format as shown below

```csv
timestep,max_goes_class,cumulative_index,label_max,label_cum
2011-01-01 00:00:00,B8.3,0.0,0,0
2011-01-01 01:00:00,B8.3,0.0,0,0
```