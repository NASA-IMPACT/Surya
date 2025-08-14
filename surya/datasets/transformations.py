import abc
from logging import info
from typing import Tuple

import numpy as np
import torch
import xarray as xr


class Transformation(object):
    @abc.abstractmethod
    def fit(self, data: xr.DataArray):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, data: xr.DataArray):
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform(self, data: xr.DataArray):
        raise NotImplementedError()

    @abc.abstractmethod
    def fit_transform(self, data: xr.DataArray):
        return self.fit(data).transform(data)

    @abc.abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def from_dict(info: dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()


class MinMaxScaler(Transformation):
    """_summary_
    Minmax scaling on the entire data
    """

    def __init__(self, new_min=1, new_max=2):
        self._is_fitted = False
        self.new_min = new_min
        self.new_max = new_max
        self._min = None
        self._max = None

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, data: xr.DataArray):
        if not self.is_fitted:
            self._max = data.max().values
            self._min = data.min().values
            self._is_fitted = True
        else:
            info("Already fitted, skipping function.")
        return self

    def _transform(self, data: xr.DataArray):
        return (
            ((data - self.min) / (self.max - self.min)) * (self.new_max - self.new_min)
        ) + self.new_min

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        assert self.min is not None and self.max is not None, "You must run fit first."

        data = xr.apply_ufunc(self._transform, data, dask="forbidden")

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

    def to_dict(self) -> dict:
        out_dict = {
            "base": self.__module__,
            "class": self.__class__.__name__,
            "new_min": str(self.new_min),
            "new_max": str(self.new_max),
            "min": str(self.min),
            "max": str(self.max),
            "is_fitted": self.is_fitted,
        }
        return out_dict

    @staticmethod
    def from_dict(info: dict):
        # with open(yaml_path, 'r') as file:
        #     data = yaml.load(file, Loader=yaml.SafeLoader)
        out = MinMaxScaler(
            new_min=np.float32(info["new_min"]), new_max=np.float32(info["new_max"])
        )
        out._min = np.float32(info["min"])
        out._max = np.float32(info["max"])
        out._is_fitted = info["is_fitted"]
        return out

    def reset(self):
        self.__init__(self.new_min, self.new_max)

    def __str__(self):
        return (
            f"min: {self.min}, "
            f"max: {self.max}, "
            f"new_max: {self.new_max}, "
            f"new_min: {self.new_min}"
        )


class StandardScaler(Transformation):
    """_summary_
    Standard scaling on the entire data
    """

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self._is_fitted = False
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
        self._sl_scale_factor = None

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def sl_scale_factor(self) -> float:
        return self._sl_scale_factor

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, data):
        if not self.is_fitted:
            self._mean = data.mean().values
            self._std = data.std().values
            self._min = data.min().values
            self._max = data.max().values
            self._is_fitted = True
        else:
            info("Already fitted, skipping function.")

        return self

    def _transform(self, data: xr.DataArray):
        return (data - self.mean) / (self.std + self.epsilon)

    def _signum_log_transform(self, data: xr.DataArray):
        data = data * self.sl_scale_factor
        return np.sign(data) * np.log1p(np.abs(data))

    def signum_log_transform(self, data: xr.DataArray):
        assert self.mean is not None and self.std is not None, "You must run fit first."

        data = xr.apply_ufunc(self._signum_log_transform, data, dask="forbidden")
        data = xr.apply_ufunc(self._transform, data, dask="forbidden")
        return data

    def transform(self, data: xr.DataArray):
        assert self.mean is not None and self.std is not None, "You must run fit first."

        data = xr.apply_ufunc(self._transform, data, dask="forbidden")
        return data

    def fit_transform(self, data: xr.DataArray):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            return data * (
                torch.Tensor([self.std]).to(data.device)
                + torch.Tensor([self.epsilon]).to(data.device)
            ) + torch.Tensor([self.mean]).to(data.device)
        else:
            return data * (self.std + self.epsilon) + self.mean

    def inverse_signum_log_transform(self, data):
        if isinstance(data, torch.Tensor):
            return (
                torch.sign(data)
                * torch.expm1(torch.abs(data))
                / torch.Tensor([self.sl_scale_factor]).to(data.device)
            )
        else:
            return np.sign(data) * np.expm1(np.abs(data)) / self.sl_scale_factor

    def to_dict(self) -> dict:
        return {
            "base": self.__module__,
            "class": self.__class__.__name__,
            "epsilon": str(self.epsilon),
            "mean": str(self.mean),
            "std": str(self.std),
            "is_fitted": self.is_fitted,
            "min": str(self.min),
            "max": str(self.max),
            "sl_scale_factor": str(self.sl_scale_factor),
        }

    @staticmethod
    def from_dict(info: dict):
        out = StandardScaler(epsilon=np.float32(info["epsilon"]))
        out._mean = np.float32(info["mean"])
        out._std = np.float32(info["std"])
        out._is_fitted = info["is_fitted"]
        out._min = np.float32(info["min"])
        out._max = np.float32(info["max"])
        out._sl_scale_factor = np.float32(info["sl_scale_factor"])
        return out

    def reset(self):
        self.__init__(self.epsilon)

    def __str__(self):
        return f"mean: {self.mean}, " f"std: {self.std}, " f"epsilon: {self.epsilon}"


class MaskUnits2D:
    """
    Transformation that takes a tuple of numpy tensors and returns a sequence of mask units. These are generally in the form `channel, dim_0, dim_1, dim_2, ...`. The returned data is largely of shape `mask unit sequence, channel, lat, lon`. Masked patches are not returned.
    The return values contain sets of indices. The indices indicate which mask units where dropped (masked) or not. The 1D indexing here simply relies on flattening the 2D space of mask units. The class methods `reconstruct` and `reconstruct_batch` show how to re-assemble the entire sequence.
    """

    def __init__(
        self,
        n_lat_mu: int,
        n_lon_mu: int,
        padding,
        seed=None,
        mask_ratio_vals: float = 0.5,
        mask_ratio_tars: float = 0.0,
        n_lats: int = 361,
        n_lons: int = 576,
    ):
        self.n_lat_mu = n_lat_mu
        self.n_lon_mu = n_lon_mu
        self.mask_ratio_vals = mask_ratio_vals
        self.mask_ratio_tars = mask_ratio_tars
        self.padding = padding
        self.n_lats = n_lats + padding[0][0] + padding[0][1]
        self.n_lons = n_lons + padding[1][0] + padding[1][1]

        if self.n_lats % n_lat_mu != 0:
            raise ValueError(
                f"Padded latitudes {self.n_lats} are not an integer multiple of the mask unit size {n_lat_mu}."
            )
        if self.n_lons % n_lon_mu != 0:
            raise ValueError(
                f"Padded longitudes {self.n_lons} are not an integer multiple of the mask unit size {n_lon_mu}."
            )

        self.mask_shape = (self.n_lats // self.n_lat_mu, self.n_lons // self.n_lon_mu)

        self.rng = np.random.default_rng(seed=seed)

    def n_units_masked(self, mask_type="vals"):
        if mask_type == "vals":
            return int(self.mask_ratio_vals * np.prod(self.mask_shape))
        elif mask_type == "tars":
            return int(self.mask_ratio_tars * np.prod(self.mask_shape))
        else:
            raise ValueError(
                f"`{mask_type}` not an allowed value for `mask_type`. Use `vals` or `tars`."
            )

    @staticmethod
    def reconstruct(
        idx_masked: torch.Tensor,
        idx_unmasked: torch.Tensor,
        data_masked: torch.Tensor,
        data_unmasked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstructs a tensor along the mask unit dimension. Non-batched version.

        Args:
            idx_masked: Tensor of shape `mask unit sequence`.
            idx_unmasked: Tensor of shape `mask unit sequence`.
            data_masked: Tensor of shape `mask unit sequence, ...`. Should have same size along mask unit sequence dimension as idx_masked. Dimensions beyond the first two, marked here as ... will typically be `local_sequence, channel` or `channel, lat, lon`. These dimensions should agree with data_unmasked.
            data_unmasked: Tensor of shape `mask unit sequence, ...`. Should have same size along mask unit sequence dimension as idx_unmasked. Dimensions beyond the first two, marked here as ... will typically be `local_sequence, channel` or `channel, lat, lon`. These dimensions should agree with data_masked.
        Returns:
            Tensor of same shape as inputs data_masked and data_unmasked. I.e. `mask unit sequence, ...`.
        """
        idx_total = torch.argsort(torch.cat([idx_masked, idx_unmasked], dim=0), dim=0)
        idx_total = idx_total.reshape(
            *idx_total.shape,
            *[1 for _ in range(len(idx_total.shape), len(data_unmasked.shape))],
        )
        idx_total = idx_total.expand(*idx_total.shape[:1], *data_unmasked.shape[1:])
        data = torch.cat([data_masked, data_unmasked], dim=0)
        data = torch.gather(data, dim=0, index=idx_total)
        return data

    @staticmethod
    def reconstruct_batch(
        idx_masked: torch.Tensor,
        idx_unmasked: torch.Tensor,
        data_masked: torch.Tensor,
        data_unmasked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstructs a tensor along the mask unit dimension. Batched version.

        Args:
            idx_masked: Tensor of shape `batch, mask unit sequence`.
            idx_unmasked: Tensor of shape `batch, mask unit sequence`.
            data_masked: Tensor of shape `batch, mask unit sequence, ...`. Should have same size along mask unit sequence dimension as idx_masked. Dimensions beyond the first two, marked here as ... will typically be `local_sequence, channel` or `channel, lat, lon`. These dimensions should agree with data_unmasked.
            data_unmasked: Tensor of shape `batch, mask unit sequence, ...`. Should have same size along mask unit sequence dimension as idx_unmasked. Dimensions beyond the first two, marked here as ... will typically be `local_sequence, channel` or `channel, lat, lon`. These dimensions should agree with data_masked.
        Returns:
            Tensor of same shape as inputs data_masked and data_unmasked. I.e. `batch, mask unit sequence, ...`.
        """
        idx_total = torch.argsort(torch.cat([idx_masked, idx_unmasked], dim=1), dim=1)
        idx_total = idx_total.reshape(
            *idx_total.shape,
            *[1 for _ in range(len(idx_total.shape), len(data_unmasked.shape))],
        )
        idx_total = idx_total.expand(*idx_total.shape[:2], *data_unmasked.shape[2:])
        data = torch.cat([data_masked, data_unmasked], dim=1)
        data = torch.gather(data, dim=1, index=idx_total)
        return data

    def __call__(self, data: Tuple[np.array]) -> Tuple[torch.Tensor]:
        """
        Args:
            data: Tuple of numpy tensors. These are interpreted as `(sur_static, ulv_static, sur_vals, ulv_vals, sur_tars, ulv_tars)`.
        Returns:
            Tuple of torch tensors. If the target is unmasked (`mask_ratio_tars` is zero), the tuple contains
            `(static, indices_masked_vals, indices_unmaked_vals, vals, tars)`. When targets are masked as well, we are dealing with
            `(static, indices_masked_vals, indices_unmaked_vals, vals, indices_masked_tars, indices_unmasked_tars, tars)`.
            Their shapes are as follows:
                static: mask unit sequence, channel, lat, lon
                indices_masked_vals: mask unit sequence
                indices_unmaked_vals: mask unit sequence
                vals: mask unit sequence, channel, lat, lon
                tars: mask unit sequence, channel, lat, lon
        """
        sur_static, ulv_static, sur_vals, ulv_vals, sur_tars, ulv_tars = data

        sur_vals, ulv_vals = np.squeeze(sur_vals, axis=1), np.squeeze(ulv_vals, axis=1)
        sur_tars, ulv_tars = np.squeeze(sur_tars, axis=1), np.squeeze(ulv_tars, axis=1)

        vals = np.concatenate(
            [
                sur_vals,
                ulv_vals.reshape(
                    ulv_vals.shape[0] * ulv_vals.shape[1], *ulv_vals.shape[-2:]
                ),
            ],
            axis=0,
        )
        tars = np.concatenate(
            [
                sur_tars,
                ulv_tars.reshape(
                    ulv_tars.shape[0] * ulv_tars.shape[1], *ulv_tars.shape[-2:]
                ),
            ],
            axis=0,
        )

        padding = ((0, 0), *self.padding)
        static = np.pad(sur_static, padding)
        vals = np.pad(vals, padding)
        tars = np.pad(tars, padding)

        static = static.reshape(
            static.shape[0],
            static.shape[-2] // self.n_lat_mu,
            self.n_lat_mu,
            static.shape[-1] // self.n_lon_mu,
            self.n_lon_mu,
        ).transpose(1, 3, 0, 2, 4)
        vals = vals.reshape(
            vals.shape[0],
            vals.shape[-2] // self.n_lat_mu,
            self.n_lat_mu,
            vals.shape[-1] // self.n_lon_mu,
            self.n_lon_mu,
        ).transpose(1, 3, 0, 2, 4)
        tars = tars.reshape(
            tars.shape[0],
            tars.shape[-2] // self.n_lat_mu,
            self.n_lat_mu,
            tars.shape[-1] // self.n_lon_mu,
            self.n_lon_mu,
        ).transpose(1, 3, 0, 2, 4)

        maskable_indices = np.arange(np.prod(self.mask_shape))
        maskable_indices = self.rng.permutation(maskable_indices)
        indices_masked_vals = maskable_indices[: self.n_units_masked()]
        indices_unmasked_vals = maskable_indices[self.n_units_masked() :]

        vals = vals.reshape(-1, *vals.shape[2:])[indices_unmasked_vals, :, :, :]

        if self.mask_ratio_tars > 0.0:
            maskable_indices = np.arange(np.prod(self.mask_shape))
            maskable_indices = self.rng.permutation(maskable_indices)
            indices_masked_tars = maskable_indices[: self.n_units_masked("tars")]
            indices_unmasked_tars = maskable_indices[self.n_units_masked("tars") :]

            tars = tars.reshape(-1, *tars.shape[2:])[indices_unmasked_tars, :, :, :]

            return_value = (
                torch.from_numpy(static).flatten(0, 1),
                torch.from_numpy(indices_masked_vals),
                torch.from_numpy(indices_unmasked_vals),
                torch.from_numpy(vals),
                torch.from_numpy(indices_masked_tars),
                torch.from_numpy(indices_unmasked_tars),
                torch.from_numpy(tars),
            )
            return return_value
        else:
            return_value = (
                torch.from_numpy(static).flatten(0, 1),
                torch.from_numpy(indices_masked_vals),
                torch.from_numpy(indices_unmasked_vals),
                torch.from_numpy(vals),
                torch.from_numpy(tars).flatten(0, 1),
            )
            return return_value
