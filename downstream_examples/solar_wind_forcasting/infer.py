import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml
from tqdm import tqdm
from huggingface_hub import snapshot_download
import torch.nn.functional as F
# Import from surya
from surya.utils.data import build_scalers
from surya.utils.distributed import set_global_seed

from dataset import WindSpeedDSDataset
from finetune import custom_collate_fn, get_model, apply_peft_lora


def load_model(config, checkpoint_path, device):
    """
    Load the trained model from checkpoint
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model
    model = get_model(config, wandb_logger=None)
    
    # Apply LoRA if needed
    if config["model"]["use_lora"]: 
        model = apply_peft_lora(model, config) 
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint
    
    # Remove 'module.' prefix if present (from DistributedDataParallel)
    if any(key.startswith('module.') for key in model_state.keys()):
        model_state = {key.replace('module.', ''): value for key, value in model_state.items()}
    
    # Load state dict
    try:
        model.load_state_dict(model_state, strict=True)
    except Exception as e:
        print(f"Failed to load with strict=True: {e}")
        raise e
    
    model.to(device)
    model.eval()
    
    return model


def get_dataloader(config, scalers, data_type="test", num_samples=3):
    """
    Create dataloader for inference
    """
    dataset = WindSpeedDSDataset(
        #### All these lines are required by the parent HelioNetCDFDataset class
        sdo_data_root_path=config["data"]["sdo_data_root_path"],
        index_path=config["data"]["train_data_path"],
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=config["drop_hmi_probability"],
        num_mask_aia_channels=config["num_mask_aia_channels"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="train",
        #### Put your downstream (DS) specific parameters below this line
        ds_solar_wind_path=config["data"]["solarwind_train_index"],
        ds_time_column=config["data"]["ds_time_column"],
        ds_time_delta_in_out=config["data"]["ds_time_delta_in_out"],
        ds_time_tolerance=config["data"]["ds_time_tolerance"],
        ds_match_direction=config["data"]["ds_match_direction"],
        ds_normalize=config["data"]["ds_normalize"],
        ds_scaler=config["data"]["ds_scaler"],
    )

    assert len(dataset) > 0, "No data found"
    
    random_ids = (
        torch.randperm(len(dataset) - 1)[: num_samples-1] + 1
    )

    dataloader = DataLoader(
        dataset=Subset(dataset, [0] + random_ids.tolist()),
        batch_size=1,
        num_workers=config["data"]["num_data_workers"],
        prefetch_factor=None,
        pin_memory=True,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    return dataset, dataloader


def run_inference(config, checkpoint_path, output_dir, device, data_type="test", num_samples=3, device_type="cuda"):
    """
    Run inference on the dataset
    """
    
    # Build scalers
    scalers = build_scalers(info=config["data"]["scalers"])
    
    # Load model
    model = load_model(config, checkpoint_path, device)
    
    # Get dataloader
    dataset, dataloader = get_dataloader(config, scalers, data_type, num_samples)
    
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Run inference
    infer_single_sample(
        model=model,
        local_rank=device,
        validation_loader=dataloader,
        gpu=True,
        dtype=config["dtype"],
        save_path=os.path.join(output_dir, 'test.png'),
        device_type=device_type,
        dataset=dataset
    )


def infer_single_sample(
    model: torch.nn.Module,
    local_rank: int,
    validation_loader: DataLoader,
    gpu: bool,
    dtype: torch.dtype,
    save_path: str = "test.png",
    device_type="cuda",
    dataset: WindSpeedDSDataset = None
):
    model.eval()
    predictions = []
    ground_truths = []
    timestamps_input_list = []
    timestamps_targets_list = []
    
    with torch.no_grad():
        print(f"Running inference on {len(validation_loader.dataset)} samples")
        for i, (batch, metadata) in enumerate(validation_loader):
            print(f"Running inference on {i}th sample")
            if gpu:
                batch = {k: v.to(local_rank) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device_type, dtype=dtype):

                ground_truth = batch["target"].item()
                timestamps_input = metadata["timestamps_input"]
                timestamps_targets = metadata["timestamps_targets"]
                
                # Convert numpy datetime64 to readable string format
                timestamps_input_str = np.datetime_as_string(timestamps_input, unit='m')[0][0]
                timestamps_targets_str = np.datetime_as_string(timestamps_targets, unit='m')[0][0]
                
                # Get model prediction (continuous value for wind speed)
                wind_speed_prediction = model(batch).item()

                # Un-normalize the prediction and ground truth
                wind_speed_prediction = dataset.un_norm_output(wind_speed_prediction)
                ground_truth = dataset.un_norm_output(ground_truth)
                
                # Store results for summary
                predictions.append(wind_speed_prediction)
                ground_truths.append(ground_truth)
                timestamps_input_list.append(timestamps_input_str)
                timestamps_targets_list.append(timestamps_targets_str)

                # Display results in table format
                print("\n" + "="*90)
                print(f"{'Time Input':<20} | {'Time Target':<20} | {'Prediction (km/s)':<20} | {'Ground Truth (km/s)':<20}")
                print("-"*90)
                print(f"{timestamps_input_str:<20} | {timestamps_targets_str:<20} | {wind_speed_prediction:<20.2f} | {ground_truth:<20.2f}")
                print("="*90)

    # Calculate and display summary statistics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths)**2))
    
    # R² calculation
    ss_res = np.sum((ground_truths - predictions)**2)
    ss_tot = np.sum((ground_truths - np.mean(ground_truths))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    
    print("\n" + "="*90)
    print("SUMMARY STATISTICS")
    print("="*90)
    print(f"{'Metric':<25} | {'Value':<20}")
    print("-"*90)
    print(f"{'Mean Absolute Error':<25} | {mae:<20.4f}")
    print(f"{'Root Mean Square Error':<25} | {rmse:<20.4f}")
    print(f"{'R² Score':<25} | {r2:<20.4f}")
    print(f"{'Number of Samples':<25} | {len(predictions):<20}")
    print("="*90)


def main():
    parser = argparse.ArgumentParser("Solar Wind Speed Inference")
    parser.add_argument(
        "--config_path",
        default="./config_infer.yaml",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="./assets/solar_wind_weights.pth",
        type=str,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default="./inference_results",
        type=str,
        help="Directory to save inference results.",
    )
    parser.add_argument(
        "--data_type",
        default="test",
        choices=["test", "valid"],
        help="Type of data to run inference on.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--num_samples",
        default=3,
        type=int,
        help="Number of samples to visualize.",
    )
    args = parser.parse_args()
    
    # Set global seed for reproducibility
    set_global_seed(42)
    
    # Load config
    config = yaml.safe_load(open(args.config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))
    
    # Set dtype
    if config["dtype"] == "float16":
        config["dtype"] = torch.float16
    elif config["dtype"] == "bfloat16":
        config["dtype"] = torch.bfloat16
    elif config["dtype"] == "float32":
        config["dtype"] = torch.float32
    else:
        raise NotImplementedError("Please choose from [float16,bfloat16,float32]")
    
    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Run inference
    run_inference(
        config=config,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        device=device,
        data_type=args.data_type,
        num_samples=args.num_samples,
        device_type=args.device
    )


if __name__ == "__main__":

    snapshot_download(
        repo_id="nasa-ibm-ai4science/solar_wind_surya",
        local_dir="./assets",
        allow_patterns='*.pth',
        token=None,
    )

    main()
