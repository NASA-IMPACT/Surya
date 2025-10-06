# EUV Spectra Prediction

The model predicts extreme ultraviolet (EUV) irradiance spectra from NASA's EVE instrument based on solar observations from the Solar Dynamics Observatory (SDO). This is a regression task that forecasts 1343-dimensional spectral vectors corresponding to EUV wavelengths from approximately 6.5 to 33.3 nm.

## Setup

Ensure the following dependencies are installed:

### 🚀 Training

For training run the below code after building the environment

```sh
cd downstream_examples/euv_spectra_prediction
bash download_data.sh
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py
```

## Inference

Run EUV spectra prediction inference using either the interactive notebook or command-line scripts.
**Prerequisites**: Complete setup and data download first.

```sh
# EUV specific downloads (inference data)
cd euv_spectra_prediction
bash download_data.sh
```

### Option A: Interactive Notebook (Recommended for beginners)

The [euv_spectra_tutorial.ipynb](euv_spectra_tutorial.ipynb) notebook provides step-by-step guidance with explanations of spectral prediction results.

### Option B: Command-Line Inference

**Basic GPU Inference**
```bash
python infer.py --checkpoint_path ./assets/euv_spectra_weights.pth \
                --output_dir ./inference_results \
                --num_samples 3 \
                --device cuda 
```

**CPU Inference** (slower but no GPU required)
```bash
python infer.py --checkpoint_path ./assets/euv_spectra_weights.pth \
                --output_dir ./inference_results \
                --num_samples 3 \
                --device cpu
```

**Advanced Usage**
```bash
# Custom configuration and more samples
python infer.py --config_path ./config.yaml \
                --checkpoint_path ./assets/euv_spectra_weights.pth \
                --output_dir ./custom_results \
                --num_samples 5 \
                --data_type valid \
                --device cuda
```

### Parameters Reference
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config_path` | `./config.yaml` | Path to model configuration file |
| `--checkpoint_path` | `./assets/euv_spectra_weights.pth` | Path to trained model weights |
| `--output_dir` | `./inference_results` | Directory for saving results |
| `--num_samples` | `3` | Number of samples to process and analyze |
| `--data_type` | `test` | Dataset split to use (`test`, `valid`, or `train`) |
| `--device` | `cuda` | Computing device (`cuda` or `cpu`) |

#### Output
- **Spectral Predictions**: Tabular output showing timestamps, spectrum dimensions, and value ranges
- **Format**: Console output with formatted table plus summary statistics and visualizations
- **Metrics**: 1343-dimensional spectral vectors vs actual measurements
- **Statistics**: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), R² score, and spectral correlations
- **Visualizations**: Spectral plots comparing predictions vs ground truth saved as PNG files

### Example Output
```
======================================================================================================================================================
Sample 1
Time Input           | Time Target          | Spectrum Shape       | Pred Range                  | GT Range                   
------------------------------------------------------------------------------------------------------------------------------------------------------
2011-01-07T07:12     | 2011-01-07T08:12     | 1343                 | [0.3542, 0.9644]            | [0.2851, 0.9661]           
======================================================================================================================================================

======================================================================================================================================================
Sample 2
Time Input           | Time Target          | Spectrum Shape       | Pred Range                  | GT Range                   
------------------------------------------------------------------------------------------------------------------------------------------------------
2011-01-07T07:00     | 2011-01-07T08:00     | 1343                 | [0.3392, 0.9637]            | [0.3725, 0.9662]           
======================================================================================================================================================

======================================================================================================================================================
Sample 3
Time Input           | Time Target          | Spectrum Shape       | Pred Range                  | GT Range                   
------------------------------------------------------------------------------------------------------------------------------------------------------
2011-01-07T06:48     | 2011-01-07T07:48     | 1343                 | [0.3350, 0.9633]            | [0.3986, 0.9658]           
======================================================================================================================================================

====================================================================================================
SUMMARY STATISTICS
====================================================================================================
Metric                              | Value               
----------------------------------------------------------------------------------------------------
Mean Absolute Error                 | 0.005164            
Root Mean Square Error              | 0.008369            
R² Score                            | 0.985208            
Average Spectral Correlation        | 0.994224            
Number of Samples                   | 3                   
Spectrum Dimension                  | 1343                
====================================================================================================

📊 Visualization saved to: ./inference_results/euv_spectra_predictions.png
```

## Dataset Information

### Input Data
- **Format**: SDO/AIA multi-channel solar images with EUV spectra labels
- **Shape**: (13, 4096, 4096) - 13 channels including:
  - AIA channels: 94Å, 131Å, 171Å, 193Å, 211Å, 304Å, 335Å, 1600Å
  - HMI channels: Magnetogram, Bx, By, Bz, Velocity
- **Temporal coverage**: Solar Cycle 24 and parts of Solar Cycle 25
- **Cadence**: 1-minute intervals
- **Labels**: 1343-dimensional EUV spectra vectors

### Output Data
- **Format**: Continuous spectral regression values
- **Dimensions**: 1343 wavelength bins from ~6.5 to 33.3 nm
- **Preprocessing**: Log-scaled and normalized intensities
- **Range**: Normalized values between 0 and 1
- **Physical meaning**: EUV irradiance intensity at specific wavelengths

### Data and pretrained weights

- The dataset is hosted on Hugging Face: [nasa-ibm-ai4science/euv-spectra](https://huggingface.co/datasets/nasa-ibm-ai4science/euv-spectra/tree/main)
- The weights can be found at [nasa-ibm-ai4science/euv_spectra_surya](https://huggingface.co/nasa-ibm-ai4science/euv_spectra_surya/tree/main)

### Preprocessing Details

- **Zero handling**: Zero values in EVE spectra are replaced by the wavelength-wise minimum (avoids -inf when taking log10)
- **Log-scaling**: Intensities are compressed using log10 for dynamic range reduction
- **Normalization**: Spectra are scaled globally using predefined min/max values (-9.00 to -1.96 in log10 space)

### Output Format
Each training sample returns:
- **ts**: 3D temporal image data from HelioFM inputs (shape: (13, 4096, 4096))
- **target**: Normalized EVE spectrum vector (length 1343)

### Custom Dataset
To use your own EUV spectra data:

1. Format your NetCDF file with the required structure: timestamps and spectra arrays
2. Update the configuration file to point to your data
3. Modify `config.yaml` with your data paths and parameters

The NetCDF file should contain:
- Time variables: `train_time`, `val_time`, `test_time`
- Spectra variables: `train_spectra`, `val_spectra`, `test_spectra`
- Each spectrum should be a 1343-dimensional array

### 📊 Dataset Description

**Dataset can be found at [NASA-IMPACT HuggingFace Repository](https://huggingface.co/datasets/nasa-impact/surya-bench-euv-spectra)**

The dataset consists of three splits (70-15-15): train, val, and test, each containing:
- A timestamp
- A 1343-dimensional spectrum corresponding to EUV wavelengths ranging from approximately 6.5 to 33.3 nm, observed at a 1-minute cadence
- EUV measurements from both flare and quiet sun conditions
- Input shape: (1, 13, 4096, 4096)
- Output shape: (1, 1343)

