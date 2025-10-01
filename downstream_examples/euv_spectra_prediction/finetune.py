import argparse
import sys
import os

import numpy as np
import torch
import torch.distributed as dist
import wandb

# Now try imports
from dataset import EVEDSDataset
from torch.amp import GradScaler, autocast
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
from surya.utils import distributed
import yaml

from surya.utils.data import build_scalers
from surya.utils.distributed import (
    StatefulDistributedSampler,
    init_ddp,
    print0,
    save_model_singular,
    set_global_seed,
)

from models import (
    HelioSpectformer1D,
)

from surya.utils.log import log


def custom_collate_fn(batch):
    """
    Custom collate function for handling batches of data and metadata in a PyTorch DataLoader.

    This function separately processes the data and metadata from the input batch.

    - The `data_batch` is collated using PyTorch's `default_collate`. If collation fails due to incompatible data types,
    the batch is returned as-is.

    - The `metadata_batch` is assumed to be a dictionary, where each key corresponds to a list of values across the batch.
    Each key is collated using `default_collate`. If collation fails for a particular key, the original list of values
    is retained.

    Example usage for accessing collated metadata:
        - `collated_metadata['timestamps_input'][batch_idx][input_time]`
        - `collated_metadata['timestamps_input'][batch_idx][rollout_step]`

    Args:
        batch (list of tuples): Each tuple contains (data, metadata), where:
            - `data` is a tensor or other data structure used for training.
            - `metadata` is a dictionary containing additional information.

    Returns:
        tuple: (collated_data, collated_metadata)
            - `collated_data`: The processed batch of data.
            - `collated_metadata`: The processed batch of metadata.
    """

    # Unpack batch into separate lists of data and metadata
    data_batch, metadata_batch = zip(*batch)

    # Attempt to collate the data batch using PyTorch's default collate function
    try:
        collated_data = torch.utils.data.default_collate(data_batch)
    except TypeError:
        # If default_collate fails (e.g., due to incompatible types), return the data batch as-is
        collated_data = data_batch

    # Handle metadata collation
    if isinstance(metadata_batch[0], dict):
        collated_metadata = {}
        for key in metadata_batch[0].keys():
            values = [d[key] for d in metadata_batch]
            try:
                # Attempt to collate values under the current key
                collated_metadata[key] = torch.utils.data.default_collate(values)
            except TypeError:
                # If collation fails, keep the values as a list
                collated_metadata[key] = values
    else:
        # If metadata is not a dictionary, try to collate it as a whole
        try:
            collated_metadata = torch.utils.data.default_collate(metadata_batch)
        except TypeError:
            # If collation fails, return metadata as-is
            collated_metadata = metadata_batch

    return collated_data, collated_metadata


def evaluate_model(dataloader, epoch, model, device, run, criterion=torch.nn.MSELoss()):
    model.eval()

    # Initialize accumulators
    # Accumulators (tensors so they can be reduced across ranks)
    abs_err_sum = torch.tensor(0.0, device=device)
    sq_err_sum = torch.tensor(0.0, device=device)
    targ_sum = torch.tensor(0.0, device=device)
    targ_sq_sum = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0.0, device=device)
    total, correct = 0, 0
    running_loss, num_batches = 0.0, 0
    # Inference loop
    with torch.no_grad():
        for i, (batch, metadata) in enumerate(dataloader):
            curr_batch = {k: v.to(device) for k, v in batch.items()}
            if config["iters_per_epoch_valid"] == i:
                break

            with autocast(device_type="cuda", dtype=config["dtype"]):
                outputs = model(curr_batch)
                target = curr_batch["label"]
                loss = criterion(outputs, target)

            reduced_loss = loss.detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= dist.get_world_size()

            running_loss += loss.item()
            num_batches += 1

            if i % config["wandb_log_train_after"] == 0 and distributed.is_main_process():
                print(f"Epoch: {epoch}, batch: {i}, loss: {reduced_loss.item()}")
                # print(f"Batch {i}, Loss: {reduced_loss.item()}")
                log(run, {"val_loss": reduced_loss.item()})

            diff = outputs - target
            abs_err_sum += torch.abs(diff).sum()
            sq_err_sum += (diff**2).sum()
            targ_sum += target.sum()
            targ_sq_sum += (target**2).sum()
            total_n += torch.tensor(target.numel(), device=device)

    # Aggregate metrics across all ranks
    for t in [abs_err_sum, sq_err_sum, targ_sum, targ_sq_sum, total_n]:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    mae = abs_err_sum.item() / total_n.item()
    mse = sq_err_sum.item() / total_n.item()
    rmse = mse**0.5

    # R² calculation: 1 - SSE/SST
    var_y = (targ_sq_sum.item() - (targ_sum.item() ** 2) / total_n.item()) / total_n.item()
    r2 = float("nan") if var_y == 0 else 1.0 - (mse / var_y)

    # Compute final metrics
    avg_loss = running_loss / max(num_batches, 1)

    # Print and log
    if distributed.is_main_process():
        print(
            f"Validation — MAE: {mae:.4f}  RMSE: {rmse:.4f}  R2: {r2:.4f}  "
            f"Avg Loss: {avg_loss:.4f}  Samples: {int(total_n.item())}"
        )
        log(
            run,
            {
                "valid/mae": mae,
                "valid/rmse": rmse,
                "valid/r2": r2,
                "valid/loss": avg_loss,
                "valid/total": int(total_n.item()),
            },
        )

    return mae, rmse, r2, avg_loss


def wrap_all_checkpoints(model):
    for name, module in model.named_children():
        if (
            isinstance(module, torch.nn.Sequential)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.Conv2d)
        ):
            setattr(
                model,
                name,
                checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            )


def get_model(config, wandb_logger) -> torch.nn.Module:
    """
    Function to initialize and return the model based on the configuration.

    Args:
        config (ExperimentConfig): Configuration object containing model parameters.
        wandb_logger (Any): Weights & Biases logger for model visualization.
        logger (logging.Logger): Standard Python logger for informational messages.

    Returns:
        Module: Initialized PyTorch model.
    """

    if torch.distributed.is_initialized() and distributed.is_main_process():
        print("Creating the model.")

    match config["model"]["model_type"]:
        case "spectformer":
            print0("Initializing HelioSpectformer1D.")
            model = HelioSpectformer1D(
                img_size=config["model"]["img_size"],
                patch_size=config["model"]["patch_size"],
                in_chans=config["model"]["in_channels"],
                embed_dim=config["model"]["embed_dim"],
                time_embedding=config["model"]["time_embedding"],
                depth=config["model"]["depth"],
                num_heads=config["model"]["num_heads"],
                mlp_ratio=config["model"]["mlp_ratio"],
                drop_rate=config["model"]["drop_rate"],
                dtype=config["dtype"],
                window_size=config["model"]["window_size"],
                dp_rank=config["model"]["dp_rank"],
                learned_flow=config["model"]["learned_flow"],
                use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
                init_weights=config["model"]["init_weights"],
                checkpoint_layers=config["model"]["checkpoint_layers"],
                n_spectral_blocks=config["model"]["spectral_blocks"],
                rpe=config["model"]["rpe"],
                ensemble=config["model"]["ensemble"],
                finetune=config["model"]["finetune"],
                nglo=config["model"]["nglo"],
                # Put finetuning additions below this line
                dropout=config["model"]["dropout"],
                num_penultimate_transformer_layers=0,
                num_penultimate_heads=0,
                num_outputs=1343,
                config=config,
            )
        case "resnet18":
            print0("Initializing ResNet18Classifier.")
            model = ResNet18Classifier(
                in_channels=config["model"]["in_channels"],
                time_steps=config["model"]["time_embedding"]["time_dim"],
                num_classes=1343,
            )
        case "resnet34":
            print0("Initializing ResNet34Classifier.")
            model = ResNet34Classifier(
                in_channels=config["model"]["in_channels"],
                time_steps=config["model"]["time_embedding"]["time_dim"],
                num_classes=1343,
            )
        case "resnet50":
            print0("Initializing ResNet50Classifier.")
            model = ResNet50Classifier(
                in_channels=config["model"]["in_channels"],
                time_steps=config["model"]["time_embedding"]["time_dim"],
                num_classes=1343,
            )
        case "alexnet":
            print0("Initializing AlexNetClassifier.")
            model = AlexNetClassifier(
                in_channels=config["model"]["in_channels"],
                time_steps=config["model"]["time_embedding"]["time_dim"],
                num_classes=1343,
            )
        case "mobilenet":
            print0("Initializing MobileNetClassifier.")
            model = MobileNetClassifier(
                in_channels=config["model"]["in_channels"],
                time_steps=config["model"]["time_embedding"]["time_dim"],
                num_classes=1343,
            )
        case _:
            model_type = config["model"]["model_type"]
            raise ValueError(f"Unknown model type {model_type}.")

    if torch.cuda.is_available():
        print0("GPU is available")
        device = torch.cuda.current_device()
    else:
        raise Exception("Training pipeline is not configured to run on CPU.")

    pretrained_path = config["pretrained_path"]

    if config["model"]["model_type"] == "spectformer":
        if (pretrained_path is not None) and os.path.exists(pretrained_path):
            print0(f"Loading pretrained model from {pretrained_path}.")
            model_state = model.state_dict()
            checkpoint_state = torch.load(pretrained_path, weights_only=True)

            filtered_checkpoint_state = {
                k: v
                for k, v in checkpoint_state.items()
                if k in model_state and v.shape == model_state[k].shape
            }

            # 2. Load the filtered weights
            model_state.update(filtered_checkpoint_state)
            model.load_state_dict(model_state, strict=True)

        else:
            raise ValueError(f"No checkpoint or pretrained model found at {pretrained_path}.")

    if config["model"]["freeze_backbone"]:
        for name, param in model.named_parameters():
            if "embedding" in name or "backbone" in name:
                param.requires_grad = False
        parameters_with_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                parameters_with_grads.append(name)
        print0(
            f"{len(parameters_with_grads)} parameters require gradients: {', '.join(parameters_with_grads)}."
        )

    if torch.distributed.is_initialized() and distributed.is_main_process():
        active = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        total = sum(p.numel() for p in model.parameters()) / 1e6
        print0(f"MODEL: {active:.2f} M ACTIVE / {total:.2f} M TOTAL PARAMETERS.")

    return model


def get_dataloaders(config, scalers):

    train_dataset = EVEDSDataset(
        #### All these lines are required by the parent HelioNetCDFDataset class
        index_path=config["data"]["train_data_path"],
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probablity=config["drop_hmi_probablity"],
        num_mask_aia_channels=config["num_mask_aia_channels"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="train",
        #### Put your donwnstream (DS) specific parameters below this line
        ds_eve_index_path=config["data"]["train_solar_data_path"],
        ds_time_column="train_time",
        ds_time_tolerance="6m",
        ds_match_direction="forward",
    )
    print0(f"Total dataset size: {len(train_dataset)}")

    valid_dataset = EVEDSDataset(
        #### All these lines are required by the parent HelioNetCDFDataset class
        index_path=config["data"]["train_data_path"],
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probablity=config["drop_hmi_probablity"],
        num_mask_aia_channels=config["num_mask_aia_channels"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="train",
        #### Put your donwnstream (DS) specific parameters below this line
        ds_eve_index_path=config["data"]["train_solar_data_path"],
        ds_time_column="val_time",
        ds_time_tolerance="6m",
        ds_match_direction="forward",
    )
    print0(f"Total dataset size: {len(valid_dataset)}")
    # print0(f"Total dataset size: {len(dataset)}")
    dl_kwargs = dict(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_data_workers"],
        prefetch_factor=config["data"]["prefetch_factor"],
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=StatefulDistributedSampler(train_dataset, drop_last=True),
        **dl_kwargs,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        sampler=StatefulDistributedSampler(valid_dataset, drop_last=True),
        **dl_kwargs,
    )

    return train_loader, valid_loader


def main(config, use_gpu: bool, use_wandb: bool, profile: bool):

    run = None
    local_rank, rank = init_ddp(use_gpu)
    print(f"RANK: {rank}; LOCAL_RANK: {local_rank}.")
    scalers = build_scalers(info=config["data"]["scalers"])
    os.makedirs(config["path_experiment"], exist_ok=True)

    if use_wandb and distributed.is_main_process():
        # https://docs.wandb.ai/guides/track/log/distributed-training

        job_id = os.getenv("PBS_JOBID")
        print(f"Job ID: {job_id}")
        print(f"local_rank: {local_rank}, rank: {rank}: WANDB")

        run = wandb.init(
            project=config["wandb_project"],
            entity="nasa-impact",
            name=f'[JOB: {job_id}] EVE {config["job_id"]}',
            config=config,
            mode="offline",
        )
        wandb.save(args.config_path)

    torch.distributed.barrier()

    train_loader, valid_loader = get_dataloaders(config, scalers)
    model = get_model(config, run)
    model.to(rank)

    if len(config["model"]["checkpoint_layers"]) > 0:
        print0("Using checkpointing.")
        wrap_all_checkpoints(model)

    total_params = sum(p.numel() for p in model.parameters())
    print0(f"Total number of parameters: {total_params:,}")

    model = DistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()],
        find_unused_parameters=False,
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning_rate"])
    device = local_rank

    scaler = GradScaler()

    print(f"Starting training for {config['optimizer']['max_epochs']} epochs.")
    for epoch in range(config["optimizer"]["max_epochs"]):
        print(f"Epoch {epoch} of {config['optimizer']['max_epochs']}")
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        running_batch = torch.tensor(0, device=device)

        for i, (batch, metadata) in enumerate(train_loader):

            if config["iters_per_epoch_train"] == i:
                break

            curr_batch = {k: v.to(local_rank) for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=config["dtype"]):
                outputs = model(curr_batch)
                target = curr_batch["label"]
                loss = criterion(outputs, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Reduce loss across all processes
            reduced_loss = loss.detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= dist.get_world_size()

            running_loss += reduced_loss
            running_batch += 1

            # Print/log only from rank 0
            if i % config["wandb_log_train_after"] == 0 and distributed.is_main_process():
                print(f"Epoch: {epoch}, batch: {i}, loss: {reduced_loss.item()}")
                # print(f"Batch {i}, Loss: {reduced_loss.item()}")
                log(run, {"train_loss": reduced_loss.item()})

            if (i + 1) % config["save_wt_after_iter"] == 0:
                print0(f"Reached save_wt_after_iter ({config['save_wt_after_iter']}).")
                fp = os.path.join(config["path_experiment"], "checkpoint.pth")
                distributed.save_model_singular(model, fp, parallelism=config["parallelism"])

        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_batch, op=dist.ReduceOp.SUM)

        if distributed.is_main_process():
            log(run, {"epoch_loss": running_loss.item() / running_batch.item()})

        fp = os.path.join(config["path_experiment"], f"epoch_{epoch}.pth")
        save_model_singular(model, fp, parallelism=config["parallelism"])
        print0(f"Epoch {epoch}: Model saved at {fp}")

        evaluate_model(valid_loader, epoch, model, rank, run, criterion)


if __name__ == "__main__":

    set_global_seed(0)

    parser = argparse.ArgumentParser("Solar Wind Downstream baseline Training")
    parser.add_argument(
        "--config_path",
        default="./config.yaml",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument("--gpu", default=True, action="store_true", help="Run on GPU CUDA.")
    parser.add_argument("--wandb", default=False, action="store_true", help="Log into WanDB.")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))

    if config["dtype"] == "float16":
        config["dtype"] = torch.float16
    elif config["dtype"] == "bfloat16":
        config["dtype"] = torch.bfloat16
    elif config["dtype"] == "float32":
        config["dtype"] = torch.float32
    else:
        raise NotImplementedError("Please choose from [float16,bfloat16,float32]")

    if not args.gpu:
        raise ValueError(
            "Training scripts are not configured for CPU use. Please set the `--gpu` flag."
        )

    main(config=config, use_gpu=args.gpu, use_wandb=args.wandb, profile=args.profile)
    torch.distributed.destroy_process_group()
