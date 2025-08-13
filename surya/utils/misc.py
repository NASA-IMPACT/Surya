import numbers
from logging import Logger
from time import time

import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import DataLoader


def view_as_windows(arr_in: np.ndarray, window_shape, step=1) -> np.ndarray:
    """Rolling window view of the input n-dimensional array.
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).

    Ref: https://github.com/scikit-image/scikit-image/blob/5e74a4a3a5149a8a14566b81a32bb15499aa3857/skimage/util/shape.py#L97-L247
    Parameters
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (
        (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
    ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def class_from_name(module_name: str, class_name: str) -> object:
    # load the module, will raise ImportError if module cannot be loaded
    m = __import__(module_name, globals(), locals(), [class_name])
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


@torch.no_grad()
def throughput(data_loader: DataLoader, model: torch.nn.Module, logger: Logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info("throughput averaged with 30 times")
        tic1 = time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
