# Solar Wind Speed Forecasting

The model predicts solar wind speed based on solar observations from the Solar Dynamics Observatory (SDO). This is a regression task that forecasts continuous wind speed values.

## Setup

Ensure the following dependencies are installed:

### 🚀 Training

For training run the below code after building the environment

```sh
cd downstream_examples/solar_wind_forcasting
bash download_data.sh
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
```

## Inference

Run Solar Wind speed forecasting inference using either the interactive notebook or command-line scripts.
**Prerequisites**: Complete setup and data download first.


```sh
# solar wind specific downloads
cd downstream_examples/solar_wind_forcasting
bash download_data.sh
```

### Option A: Interactive Notebook (Recommended for beginners)

The [solar_wind_tutorial.ipynb](solar_wind_tutorial.ipynb) notebook provides step-by-step guidance with explanations of forecasting results.

### Option B: Command-Line Inference

**Basic GPU Inference**
```bash
python infer.py --checkpoint_path ./assets/solar_wind_weights.pth \
                --output_dir ./inference_results \
                --num_samples 3 \
                --device cuda 
```

**CPU Inference** (slower but no GPU required)
```bash
python infer.py --checkpoint_path ./assets/solar_wind_weights.pth \
                --output_dir ./inference_results \
                --num_samples 3 \
                --device cpu
```

**Advanced Usage**
```bash
# Custom configuration and more samples
python infer.py --config_path ./config.yaml \
                --checkpoint_path ./assets/solar_wind_weights.pth \
                --output_dir ./custom_results \
                --num_samples 10 \
                --data_type valid \
                --device cuda
```

### Parameters Reference
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config_path` | `./config.yaml` | Path to model configuration file |
| `--checkpoint_path` | `./assets/solar_wind_weights.pth` | Path to trained model weights |
| `--output_dir` | `./inference_results` | Directory for saving results |
| `--num_samples` | `3` | Number of samples to process and analyze |
| `--data_type` | `test` | Dataset split to use (`test` or `valid`) |
| `--device` | `cuda` | Computing device (`cuda` or `cpu`) |

#### Output
- **Forecasting Results**: Tabular output showing timestamps, predicted wind speeds, and ground truth values
- **Format**: Console output with formatted table plus summary statistics
- **Metrics**: Continuous wind speed predictions (km/s) vs actual measurements
- **Statistics**: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R² score
- **Timestamps**: Input observation time and target prediction time

### Example Output
```
==========================================================================================
Time Input           | Time Target          | Prediction (km/s)    | Ground Truth (km/s) 
------------------------------------------------------------------------------------------
2011-01-07T01:00     | 2011-01-07T02:00     | 501.81               | 483.00              
==========================================================================================
```

## Dataset Information

### Input Data
- **Format**: SDO/AIA multi-channel solar images with solar wind speed labels
- **Shape**: (13, 4096, 4096) - 13 channels including:
  - AIA channels: 94Å, 131Å, 171Å, 193Å, 211Å, 304Å, 335Å, 1600Å
  - HMI channels: Magnetogram, Bx, By, Bz, Velocity
- **Temporal coverage**: 2010-2023
- **Cadence**: Hourly intervals
- **Labels**: Continuous solar wind speed values (km/s)

### Output Data
- **Format**: Continuous regression values
- **Units**: Solar wind speed in kilometers per second (km/s)
- **Prediction Window**: 4 days ahead (96 hours)
- **Typical Range**: 250-800 km/s (can vary during solar storms)

### Data and pretrained weights

- The dataset is hosted on Hugging Face: [nasa-ibm-ai4science/Surya-bench-solarwind](https://huggingface.co/datasets/nasa-ibm-ai4science/Surya-bench-solarwind/tree/main)
- The weights can be found at [nasa-ibm-ai4science/solar_wind_surya](https://huggingface.co/nasa-ibm-ai4science/solar_wind_surya/tree/main)

### Custom Dataset
To use your own solar wind data:

1. Format your CSV file with the required columns: `Epoch`, `V` (wind speed)
2. Update the configuration file to point to your data
3. Modify `config.yaml` with your data paths and parameters

The csv file should be in the format as shown below

```csv
Epoch,V,Bx,By,Bz,N
2010-05-01 00:00:00,425.6,1.2,-0.8,2.3,5.7
2010-05-01 01:00:00,432.1,1.5,-0.6,2.1,6.2
```

### 📊 Dataset Description

**Dataset can be found at [NASA-IMPACT HuggingFace Repository](https://huggingface.co/datasets/nasa-impact/Surya-bench-solarwind)**

The dataset is stored as `.csv` files. Each sample in the dataset corresponds to a tracked active region and is structured as follows:
- Input shape: (1, 13, 4096, 4096)
- Temporal coverage of the dataset is `2010-05-01` to `2023-12-31`
- 5 physical quantities: V (wind speed), Bx(GSE), By(GSM), Bz(GSM), Number Density (N)
- Input timestamps: (120748,)
- Cadence: Hourly
- Output shape: (1)
- Output prediction: Continuous wind speed value (km/s)

