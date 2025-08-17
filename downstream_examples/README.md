# Finetuning tasks

This repository contains downstream finetuning code related to six key tasks that are structured as prediction or segmentation. These tasks are related to various aspects of solar and heliospheric events ecosystem and are listed below:
1. **[Active Region Segmentation](ar_segmentation/)**
2. **[Solar EUV Spectra Modeling](euv_spectra_prediction/)**
3. **[Solar Flare Forecasting](solar_flare_forcasting/)**
4. **[Solar Wind Forecasting](solar_wind_forcasting/)**



### Download SDO Data

Downloads and generates csv files. The data_path is defined in the main function

1. Downloads the data from huggingface repo -> `data_path/tars/{validate,test}_splitted_tars/`
2. combines the extracted tar parts -> `data_path/tars/{validate,test}_splitted_tars/{validate,test}.tar`
3. Extracts data from the tar file -> `data_path/{validate,test}/*.nc`
4. CSV files (copies the same csv to downstream_tasks) -> `{downstream_tasks}/assets/sdo_{validate, test}.csv`

## Run the python file to download data

```bash
uv run python download_data.py
```
