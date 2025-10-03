import sys
import os
import re
import glob
import h5py
import numpy as np
import pandas as pd
import torch

# Append base path.  May need to be modified if the folder structure changes
from surya.datasets.helio import HelioNetCDFDataset


class ArDSDataset(HelioNetCDFDataset):
    def __init__(
        self,
        #### All these lines are required by the parent HelioNetCDFDataset class
        sdo_data_root_path: str,
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
        ds_ar_index_paths: list = None,
    ):
        super().__init__(
            sdo_data_root_path=sdo_data_root_path,
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
        )

        # Load ds index and find intersection with HelioFM index
        self.create_logger()
        self.ar_index = pd.DataFrame()
        all_data = [pd.read_csv(file) for file in ds_ar_index_paths]
        self.ar_index = pd.concat(all_data, ignore_index=True)
        self.ar_index = self.ar_index.loc[self.ar_index["present"] == 1, :]

        self.ar_index["timestamp"] = pd.to_datetime(self.ar_index["timestamp"]).values.astype(
            "datetime64[ns]"
        )
        self.ar_index.sort_values("timestamp", inplace=True)

        # Create HelioFM valid indices and find closest match to DS index
        self.ar_valid_indices = pd.DataFrame({"valid_indices": self.valid_indices}).sort_values(
            "valid_indices"
        )

        self.ar_valid_indices = pd.merge(
            self.ar_index,
            self.ar_valid_indices,
            how="inner",
            left_on="timestamp",
            right_on="valid_indices",
        )

        # Override valid indices variables to reflect matches between HelioFM and DS
        self.valid_indices = [pd.Timestamp(date) for date in self.ar_valid_indices["valid_indices"]]
        self.adjusted_length = len(self.valid_indices)
        self.ar_valid_indices.set_index("valid_indices", inplace=True)

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
                flare_intensity_target
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        # This lines assembles the dictionary that HelioFM's dataset returns (defined above)
        # base_dictionary, metadata = super().__getitem__(idx=idx)
        base_dictionary = {}
        timestep = self.valid_indices[idx]
        base_dictionary["ts"] = self.transform_data(
            self.load_nc_data(self.index.loc[timestep, "path"], timestep, self.channels)
        )
        base_dictionary["ts"] = np.stack([base_dictionary["ts"]], axis=1)
        file_path = self.ar_valid_indices.iloc[idx]["file_path"]

        file_path = os.path.join("./assets/surya-bench-ar-segmentation", file_path)

        try:
            with h5py.File(file_path, "r") as f:
                mask = torch.from_numpy(f["union_with_intersect"][...])

        except Exception as e:
            print(f"Error loading mask from {file_path}: {e}")
            raise e

        base_dictionary["forecast"] = mask / 255.0
        base_dictionary["time_delta_input"] = torch.tensor([0])
        # base_dictionary["forecast"] = mask["union_with_intersect"][:, :]

        timestep = np.datetime64(timestep)
        metadata = {
            "timestamps_input": timestep,
            "timestamps_targets": timestep,
        }

        return base_dictionary, metadata
