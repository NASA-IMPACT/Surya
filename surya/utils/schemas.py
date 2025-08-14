from typing import TypedDict, Dict, Any
import torch


class TrainState(TypedDict):
    dataloader: torch.utils.data.DataLoader
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    sampler: Any  # Changed from torch.utils.data.sampler to Any
    profiler: bool
    epoch: int
    iteration: int
    loss: float
    wandb_state: int
