import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import checkpoint as dist_checkpoint
from torch.distributed import fsdp

import functools
import itertools

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from typing import Any, Dict, Optional

from .schemas import TrainState


def init_dist(device: str, rank: int, world_size: int):
    torch.distributed.init_process_group(
        device,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=60),
    )


def init_ddp(use_gpu: bool):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if use_gpu:
        assert (
            torch.cuda.is_available()
        ), "GPU requested but none was found in the system."

    if use_gpu:
        init_dist("nccl", rank, world_size)
        torch.cuda.set_device(local_rank)
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        cudnn.benchmark = True
    else:
        init_dist("gloo", rank, world_size)
    return local_rank, rank


def set_global_seed(rank):
    random.seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


# def save_model_singular(model, *args, **kwargs):
#     """Stream all model parameters to rank 0 on the CPU, then pass all
#     other given arguments to `torch.save` to save the model, but only on
#     the root process.
#     """
#     save_policy = fsdp.FullStateDictConfig(
#         offload_to_cpu=True, rank0_only=True)
#     with fsdp.FullyShardedDataParallel.state_dict_type(
#             model,
#             fsdp.StateDictType.FULL_STATE_DICT,
#             save_policy,
#     ):
#         cpu_state = model.state_dict()
#     # We do *not* want to write to the same location with multiple
#     # processes at the same time.
#     if is_root_process():
#         torch.save(cpu_state, *args, **kwargs)


def save_model(model, save_dir):
    """Obtain sharded model parameters from the GPU, then save the model
    as a distributed checkpoint to the given directory. Saving a
    distributed checkpoint means that the checkpoint will be split into
    individual files, one for each process.
    """
    state_dict_config = fsdp.ShardedStateDictConfig(offload_to_cpu=False)
    with fsdp.FullyShardedDataParallel.state_dict_type(
        model,
        fsdp.StateDictType.SHARDED_STATE_DICT,
        state_dict_config,
    ):
        cp_state_dict = {"model": model.state_dict()}
    dist_checkpoint.save_state_dict(
        cp_state_dict,
        dist_checkpoint.FileSystemWriter(save_dir),
    )


def load_model(model, load_dir):
    """Set the given model's state dictionary in-place from the given
    distributed checkpoint directory.
    """
    state_dict_config = fsdp.ShardedStateDictConfig(offload_to_cpu=False)
    with fsdp.FullyShardedDataParallel.state_dict_type(
        model,
        fsdp.StateDictType.SHARDED_STATE_DICT,
        state_dict_config,
    ):
        cp_state_dict = {"model": model.state_dict()}
    dist_checkpoint.load_state_dict(
        cp_state_dict,
        dist_checkpoint.FileSystemReader(load_dir),
    )
    model.load_state_dict(cp_state_dict["model"])


@functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


# The reason we define this is that `torch.distributed` does not
# implement it; for the global rank, there's
# `torch.distributed.get_rank()`.
@functools.lru_cache(maxsize=None)
def get_local_rank():
    """Return the local rank of this process."""
    return int(os.getenv("LOCAL_RANK"))


def print0(*args, **kwargs):
    """Print something only on the root process."""
    if (not dist.is_initialized()) or is_root_process():
        print(*args, **kwargs)


def save_model_singular(model, save_path, parallelism, *args, **kwargs):
    """Stream all model parameters to rank 0 on the CPU, then pass all
    other given arguments to `torch.save` to save the model, but only on
    the root process.
    """

    match parallelism:
        case "fsdp":
            save_policy = fsdp.FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with fsdp.FullyShardedDataParallel.state_dict_type(
                model,
                fsdp.StateDictType.FULL_STATE_DICT,
                save_policy,
            ):
                cpu_state = model.state_dict()
            # We do *not* want to write to the same location with multiple
            # processes at the same time.
            if is_main_process():
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(obj=cpu_state, f=save_path, *args, **kwargs)

        case "ddp":
            if is_main_process():
                torch.save(obj=model.module.state_dict(), f=save_path, *args, **kwargs)
            dist.barrier()
        case _:
            raise ValueError(
                f'`parallelism` should be one of "ddp" and "fsdp". Got {parallelism}.'
            )


def save_optim_singular(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    parallelism: str = "fsdp",
):
    match parallelism:
        case "fsdp":
            optim_state_dict_config = fsdp.FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )

            with fsdp.FullyShardedDataParallel.state_dict_type(
                model,
                fsdp.StateDictType.FULL_STATE_DICT,
                optim_state_dict_config=optim_state_dict_config,
            ):
                optim_state_dict = fsdp.FullyShardedDataParallel.optim_state_dict(
                    model, optimizer
                )

                if is_main_process():
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    checkpoint = {
                        "optimizer_state_dict": optim_state_dict,
                    }
                    torch.save(checkpoint, f=save_path)
        case "ddp":
            if is_main_process():
                optim_state_dict = optimizer.state_dict()
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(obj=optim_state_dict, f=save_path)
            dist.barrier()
        case _:
            raise ValueError(
                f'`parallelism` should be one of "ddp" and "fsdp". Got {parallelism}.'
            )


def collect_optim_singular(
    model: nn.Module, optimizer: torch.optim.Optimizer, parallelism: str = "fsdp"
) -> dict:
    optim_state_dict = {}
    match parallelism:
        case "fsdp":
            optim_state_dict_config = fsdp.FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )

            with fsdp.FullyShardedDataParallel.state_dict_type(
                model,
                fsdp.StateDictType.FULL_STATE_DICT,
                optim_state_dict_config=optim_state_dict_config,
            ):
                optim_state_dict = fsdp.FullyShardedDataParallel.optim_state_dict(
                    model, optimizer
                )

        case "ddp":
            if is_main_process():
                optim_state_dict = optimizer.state_dict()
            dist.barrier()
        case _:
            raise ValueError(
                f'`parallelism` should be one of "ddp" and "fsdp". Got {parallelism}.'
            )

    return optim_state_dict


def save_state_singular(states: TrainState, save_path, *args, **kwargs):
    """Stream all model parameters to rank 0 on the CPU, then pass all
    other given arguments to `torch.save` to save paramters, but only on
    the root process.
    """
    if is_main_process():
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(obj=states, f=save_path, *args, **kwargs)
    dist.barrier()


class StatefulDistributedSampler(DistributedSampler):
    _YIELDED = "yielded"

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.yielded = 0
        self.next_yielded = None

    def __iter__(self):
        self.yielded = 0
        if self.next_yielded is not None:
            self.yielded = self.next_yielded
            self.next_yielded = None
        it = super().__iter__()
        for idx in itertools.islice(it, self.yielded, None):
            self.yielded += 1
            yield idx

    def state_dict(self) -> Dict[str, Any]:
        return {self._YIELDED: self.yielded}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self._YIELDED not in state_dict:
            raise ValueError("Invalid state_dict")
        if state_dict[self._YIELDED] < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")
        self.next_yielded = state_dict[self._YIELDED]
