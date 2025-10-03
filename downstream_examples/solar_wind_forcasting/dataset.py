from surya.datasets.helio import HelioNetCDFDataset
import pandas as pd
import numpy as np
import torch
from surya.utils.distributed import print0

def logsign(x):
    return torch.sign(x) * torch.log10(torch.abs(x) + 1)


def invlogsin(x):
    return torch.sign(x) * (torch.pow(torch.abs(x)) - 1)


class WindSpeedDSDataset(HelioNetCDFDataset):
    """
    A solar wind class, inheriting from HelioNetCDFDataset.

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
    drop_hmi_probability : int, optional
        Probability of removing hmi during training, by default 0
    use_latitude_in_learned_flow : bool, optional
        Switch to provide heliographic latitude for each datapoint, by default False
    channels : list[str] | None, optional
        Input channels to use, by default None
    phase : str, optional
        Descriptor of the phase used for this database, by default "train"

    Downstream (DS) Parameters
    --------------------------
    ds_solar_wind_path : str, optional
        DS index. A path to csv file containing the DS timestamps and the corresponding solar wind speed.
    ds_time_column : str, optional
        Name of the column to use as datestamp to compare with HelioFM's index, by default None
    ds_time_delta_in_out: str, optional
        Time delta between the AIA timestamp and the target solar wind speed. Given as a string.
        For example, 1h, means 1 hour, 4D means 4 days, etc.
    ds_time_tolerance : str, optional
        How much time difference is tolerated when finding matches between HelioFM and the DS, by default None
    ds_match_direction : str, optional
        Direction used to find matches using pd.merge_asof possible values are "forward", "backward",
        or "nearest".  For causal relationships is better to use "forward", by default "forward"
    ds_normalize: bool, optional
        Flag to enable or disable normalization. Enabling normalization will cause the data to be transformed using
        a signum-log function, defined as: sign(x)*log_10(abs(x)+1). By default the values are not normalized
    ds_scaler: list, optional
        A two element sequence that provides the mean and stddev of the wind dataset. If we use only 1 variable, this would be a
        list of two floats. Otherwise, it would be a list of two vectors. Only used with the ds_normalize flag. Default values are
        set for the wind speed pre-computed by hand.

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
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        #### Put your donwnstream (DS) specific parameters below this line
        ds_solar_wind_path: str = None,
        ds_time_column: str = None,
        ds_time_delta_in_out: str = None,
        ds_time_tolerance: str = None,
        ds_match_direction: str = "forward",
        ds_normalize=True,
        ds_scaler=[2.61, 0.09],
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
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            sdo_data_root_path=sdo_data_root_path,
        )

        # Load ds index and find intersection with HelioFM index
        self.ds_index = pd.read_csv(ds_solar_wind_path)
        self.ds_time_delta_in_out = np.timedelta64(ds_time_delta_in_out[0], ds_time_delta_in_out[1])
        self.ds_index["ds_index"] = pd.to_datetime(self.ds_index[ds_time_column]).values.astype(
            "datetime64[ns]"
        )
        self.ds_index.sort_values("ds_index", inplace=True)
        self.un_norm_ptp = np.ptp(self.ds_index["V"])
        self.un_norm_min = np.min(self.ds_index["V"])
        print0("Timedelta", self.ds_time_delta_in_out)
        # Get the matched solar wind speed time index. We will match index[FM] with index[DS]-dT
        self.ds_index["sw_match_index"] = self.ds_index["ds_index"] - self.ds_time_delta_in_out

        # Implement normalization.  This is going to be DS application specific, no two will look the same
        self.ds_index["V"] = (self.ds_index["V"] - self.un_norm_min) / self.un_norm_ptp
        # Create HelioFM valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame({"valid_indices": self.valid_indices}).sort_values(
            "valid_indices"
        )

        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="sw_match_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["sw_match_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(["sw_match_index", "index_delta"])
        self.df_valid_indices.drop_duplicates(subset="sw_match_index", keep="first", inplace=True)
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance), :
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between HelioFM and DS indices")

        # Override valid indices variables to reflect matches between HelioFM and DS
        self.valid_indices = [pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)
        self.ds_normalize = ds_normalize
        if self.ds_normalize:
            self.ds_scaler = ds_scaler
            if type(self.ds_scaler[0]) == "float" or type(self.ds_scaler[0]) == "int":
                self.ds_scaler = [torch.tensor(self.ds_scaler[0]), torch.tensor(self.ds_scaler[1])]
            elif torch.is_tensor(self.ds_scaler[0]):
                pass
            elif type(self.ds_scaler[0]) == np.ndarray:
                self.ds_scaler = [
                    torch.from_numpy(self.ds_scaler[0]),
                    torch.from_numpy(self.ds_scaler[1]),
                ]
            else:
                print0(
                    "Scalers are not a list of torch tensors, float, int or np.ndarray. What are you feeding in?"
                )
        else:
            self.ds_scaler = [torch.tensor(1), torch.tensor(1)]

    
    def un_norm_output(self,x):
        return x * self.un_norm_ptp + self.un_norm_min

    def __len__(self):
        return self.adjusted_length

    def __scale__(self, x):
        if self.ds_normalize:
            return (logsign(x) - self.ds_scaler[0]) / self.ds_scaler[1]
        else:
            return x

    def __unscale__(self, x):
        if self.ds_normalize:
            return invlogsin(x * self.ds_scaler[1] + self.ds_scaler[0])
        else:
            return x

    def __getitem__(self, idx: int, unscale=False) -> dict:
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
                V (torch.Tensor):             1
                ds_time (torch.Tensor):       1
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        # This lines assembles the dictionary that HelioFM's dataset returns (defined above)
        base_dictionary, metadata = super().__getitem__(idx=idx)

        # We now add the wind speed label
        target = torch.tensor(self.df_valid_indices.iloc[idx]["V"])
        # if self.ds_normalize and not unscale:
        #     target = self.__scale__(target)
        base_dictionary["target"] = torch.tensor(
            [
                target,
            ]
        )
        base_dictionary["ds_time"] = self.df_valid_indices.index[idx].timestamp()

        return base_dictionary, metadata
