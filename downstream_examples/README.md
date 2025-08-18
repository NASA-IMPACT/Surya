# Finetuning tasks

This repository contains downstream finetuning code related to six key tasks that are structured as prediction or segmentation. These tasks are related to various aspects of solar and heliospheric events ecosystem and are listed below:
1. **[Active Region Segmentation](ar_segmentation/)**
2. **[Solar EUV Spectra Modeling](euv_spectra_prediction/)**
3. **[Solar Flare Forecasting](solar_flare_forcasting/)**
4. **[Solar Wind Forecasting](solar_wind_forcasting/)**



## Download SDO Data

The data is located at [nasa-ibm-ai4science/SDO_training](https://huggingface.co/datasets/nasa-ibm-ai4science/SDO_training)

Downloads and generates csv files. The `data_path` is defined in the main function

1. Downloads the data from huggingface repo -> `data_path/tars/{validate,test}_splitted_tars/`
2. combines the extracted tar parts -> `data_path/tars/{validate,test}_splitted_tars/{validate,test}.tar`
3. Extracts data from the tar file -> `data_path/{validate,test}/*.nc`
4. CSV files (copies the same csv to downstream_tasks) -> `{downstream_tasks}/assets/sdo_{validate, test}.csv`


### Run the python file to download data

```bash
python download_data.py
```

The structure of the downloaded file should give the following structure:

```
downstream_examples
├── ar_segmentation
│   ├── assets
│   │   ├── sdo_test.csv
│   │   └── sdo_validate.csv
│   ├── ...
│   └── ...
├── common_data
│   ├── tars
│   │   ├── test_splitted_tars
│   │   │   ├── test.tar
│   │   │   ├── test.tar.part_aa
│   │   │   ├── test.tar.part_aa
│   │   │   ├── ...
│   │   │   ├── ...
│   │   │   └── test.tar.part_aq
│   │   └── validate_splitted_tars
│   │       ├── validate.tar
│   │       ├── validate.tar.part_aa
│   │       ├── validate.tar.part_aa
│   │       ├── ...
│   │       ├── ...
│   │       └── validate.tar.part_aq
│   ├── test
│   │   ├── 20110103_1200.nc
│   │   ├── ...
│   │   ├── ...
│   │   └── 20110105_2348.nc
│   └── validate
│       ├── 20110106_0000.nc
│       ├── ...
│       ├── ...
│       └── 20110108_1148.nc
├── download_data.py
├── euv_spectra_prediction
│   ├── assets
│   │   ├── sdo_test.csv
│   │   └── sdo_validate.csv
│   ├── ...
│   └── ...
├── README.md
├── solar_flare_forcasting
│   ├── assets
│   │   ├── sdo_test.csv
│   │   └── sdo_validate.csv
│   ├── ...
│   └── ...
└── solar_wind_forcasting
    ├── assets
    │   ├── sdo_test.csv
    │   └── sdo_validate.csv
    ├── ...
    └── ...
```

The created `sdo_test.csv` and `sdo_validate.csv` are created with the following format

```
,path,timestep,present
0,/full/path/to/downstream_examples/common_data/test/20110101_0000.nc,2011-01-01 00:00:00,0
1,/full/path/to/downstream_examples/common_data/test/20110101_0012.nc,2011-01-01 00:12:00,0
2,/full/path/to/downstream_examples/common_data/test/20110101_0024.nc,2011-01-01 00:24:00,1
3,/full/path/to/downstream_examples/common_data/test/20110101_0036.nc,2011-01-01 00:36:00,1
```

- `path`: path to the nc files
- `timestamp`: timetamp of the data
- `present`: 1 represent we have the nc file.


Note: 
1. In the `sdo_{test,validate}.csv`, an entry for every 12 minute cadence are created from the start date and month to end date and month. The `present` column represent if we have the nc file. So, if we have 297 files and two month of range we will have 6962 entries, with 297 entries having `1` and the rest with `0`. These are used by the dataloaders to determine the files to use.
2. These csv are created for all the downstreams present in `downstream_tasks` variable in the `download_data.py` file.
3. All the tar part files must be downloaded to create a complete tar file. If any of the data is missing, we cannot create a tar file and extract the `nc` files. So be sure to have enough data space for the 3 steps:
    a. part tars
    b. complete (combined) tars
    c. Extracted nc files
4. The intermediate files are not deleted automatically. This is left for the user to cleanup.
