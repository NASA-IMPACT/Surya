from surya.datasets.helio import HelioNetCDFDataset
import pandas as pd
import xarray as xr
import numpy as np
import torch


class EVEDSDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset to show an example of how to create a
    dataset for donwstream applications. It includes both the necessary parameters
    to initialize the parent class, as well as those of the child

    HelioFM Parameters
    ------------------
    index_path : str
        Path to HelioFM index
    time_delta_input_minutes : list[int]
        Input delta times to define the input stack in minutes from the present
    time_delta_target_minutes : int
        Target delta time to define the output stack on rollout in minutes from the present
    n_input_timestamps : int
        Number of input timestamps
    rollout_steps : int
        Number of rollout steps
    scalers : optional
        scalers used to perform input data normalization, by default None
    num_mask_aia_channels : int, optional
        Number of aia channels to mask during training, by default 0
    drop_hmi_probablity : int, optional
        Probability of removing hmi during training, by default 0
    use_latitude_in_learned_flow : bool, optional
        Switch to provide heliographic latitude for each datapoint, by default False
    channels : list[str] | None, optional
        Input channels to use, by default None
    phase : str, optional
        Descriptor of the phase used for this database, by default "train"

    Downstream (DS) Parameters
    --------------------------
    ds_eve_index_path : str, optional
        DS index.  In this example a eve dataset, by default None
    ds_time_column : str, optional
        Name of the column to use as datestamp to compare with HelioFM's index, by default None
    ds_time_tolerance : str, optional
        How much time difference is tolerated when finding matches between HelioFM and the DS, by default None
    ds_match_direction : str, optional
        Direction used to find matches using pd.merge_asof possible values are "forward", "backward",
        or "nearest".  For causal relationships is better to use "forward", by default "forward"

    Raises
    ------
    ValueError
        Error is raised if there is not overlap between the HelioFM and DS indices
        given a tolerance
    """

    def __init__(
        self,
        #### All these lines are required by the parent HelioNetCDFDataset class
        index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probablity=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        #### Put your donwnstream (DS) specific parameters below this line
        ds_eve_index_path: str = None,
        ds_time_column: str = None,  # choose "train_time", "val_time" or "test_time", the spectra will be chose accordingly.
        ds_time_tolerance: str = None,
        ds_match_direction: str = "forward",
        sdo_data_root_path: str = None,        
    ):

        ## Initialize parent class
        super().__init__(
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            sdo_data_root_path=sdo_data_root_path,
        )

        # Load ds index and find intersection with HelioFM index
        with xr.open_dataset(ds_eve_index_path, engine="h5netcdf", cache=False, lock=False) as ds:

            if ds_time_column.startswith("train"):
                spectra_key = "train_spectra"
            elif ds_time_column.startswith("val"):
                spectra_key = "val_spectra"
            elif ds_time_column.startswith("test"):
                spectra_key = "test_spectra"
            else:
                raise ValueError(
                    f"Could not determine spectra field from ds_time_column='{ds_time_column}'"
                )

            # Extract timestamps and spectra from NetCDF
            # Reason: Each row has a timestamp and a 1343-point spectrum
            timestamps = pd.to_datetime(ds[ds_time_column].values)
            spectra_array = ds[spectra_key].values  # shape: (N_samples, 1343)

        # Replace zeros with wavelength-wise minimums BEFORE log10
        # Reason: Avoid -inf or outliers like -9 distorting the normalization
        spectra_array[spectra_array == 0] = np.nan
        min_per_wavelength = np.nanmin(spectra_array, axis=0)
        nan_mask = np.isnan(spectra_array)
        spectra_array[nan_mask] = np.take(min_per_wavelength, np.where(nan_mask)[1])

        # Apply log10 safely
        # Reason: Helps compress dynamic range (intensities span 10^-9 to 10^-1)
        spectra_log = np.log10(spectra_array)

        # Global min-max normalization
        # Reason: Required to keep relative shape of spectrum unchanged
        global_min = -9.00  # np.min(spectra_log)
        global_max = -1.96  # np.max(spectra_log)
        spectra_norm = (spectra_log - global_min) / (global_max - global_min)

        # Store as DataFrame with list of normalized spectra per timestamp
        # Reason: Needed for merge_asof and PyTorch-style indexing
        self.ds_index = pd.DataFrame(
            {
                "ds_index": timestamps,
                "normalized_spectrum": list(spectra_norm),  # each row is a 1343-long list
            }
        )
        self.ds_index.sort_values("ds_index", inplace=True)

        # Create HelioFM valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame({"valid_indices": self.valid_indices}).sort_values(
            "valid_indices"
        )
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(["ds_index", "index_delta"])
        self.df_valid_indices.drop_duplicates(subset="ds_index", keep="first", inplace=True)
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between HelioFM and DS indices")

        # Override valid indices variables to reflect matches between HelioFM and DS
        self.valid_indices = [pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                # HelioFM keys--------------------------------
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                forecast (torch.Tensor):          C, L, H, W
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
                # HelioFM keys--------------------------------
                eve_intensity_target (torch.Tensor):    1
                ds_time (torch.Tensor):                 1
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        # This lines assembles the dictionary that HelioFM's dataset returns (defined above)
        base_dictionary, metadata = super().__getitem__(idx=idx)

        # We now add the eve intensity label
        base_dictionary["target"] = torch.tensor(
            self.df_valid_indices.iloc[idx]["normalized_spectrum"]
        )
        base_dictionary["label"] = torch.tensor(
            self.df_valid_indices.iloc[idx]["normalized_spectrum"]
        )
        # base_dictionary["ds_time"] = self.df_valid_indices.index[idx]

        return base_dictionary, metadata
