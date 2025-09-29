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
import matplotlib.pyplot as plt

# Import from surya
from surya.utils.data import build_scalers
from surya.utils.distributed import set_global_seed

from dataset import EVEDSDataset
from finetune import custom_collate_fn, get_model


def load_model(config, checkpoint_path, device):
    """
    Load the trained model from checkpoint
    """
    
    # Initialize model
    model = get_model(config, wandb_logger=None)
    
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
        print(f"Loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load with strict=True: {e}")
        # Try to load with strict=False for partial matches
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    model.to(device)
    model.eval()
    
    return model


def get_dataloader(config, scalers, data_type="test", num_samples=3):

    dataset = EVEDSDataset(
        #### All these lines are required by the parent HelioNetCDFDataset class
        sdo_data_root_path=config["data"]["sdo_data_root_path"],
        index_path=config["data"]["infer_data_path"],
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
        ds_eve_index_path=config["data"]["infer_solar_data_path"],
        ds_time_column="val_time",
        ds_time_tolerance="6m",
        ds_match_direction="forward",
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

    return dataloader


def run_inference(config, checkpoint_path, output_dir, device, data_type="test", num_samples=3, device_type="cuda"):
    """
    Run inference on the dataset
    """
    
    # Build scalers
    scalers = build_scalers(info=config["data"]["scalers"])
    
    # Load model
    model = load_model(config, checkpoint_path, device)
    
    # Get dataloader
    dataloader = get_dataloader(config, scalers, data_type, num_samples)
    
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    infer_samples(
        model=model,
        local_rank=device,
        validation_loader=dataloader,
        gpu=True,
        dtype=config["dtype"],
        save_path=os.path.join(output_dir, 'euv_spectra_predictions.png'),
        device_type=device_type
    )


def infer_samples(
    model: torch.nn.Module,
    local_rank: int,
    validation_loader: DataLoader,
    gpu: bool,
    dtype: torch.dtype,
    save_path: str = "euv_spectra_predictions.png",
    device_type="cuda"
):
    model.eval()
    predictions = []
    ground_truths = []
    timestamps_input_list = []
    timestamps_targets_list = []
    
    # Define wavelength range for EUV spectra (approximate)
    wavelengths = np.linspace(6.5, 33.3, 1343)  # From ~6.5 to 33.3 nm
    
    with torch.no_grad():
        for i, (batch, metadata) in enumerate(validation_loader):

            if gpu:
                batch = {k: v.to(local_rank) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                ground_truth = batch["label"].cpu().numpy()  # Shape: (1, 1343)
                timestamps_input = metadata["timestamps_input"]
                timestamps_targets = metadata["timestamps_targets"]
                
                # Convert numpy datetime64 to readable string format
                timestamps_input_str = np.datetime_as_string(timestamps_input, unit='m')[0][0]
                timestamps_targets_str = np.datetime_as_string(timestamps_targets, unit='m')[0][0]
                
                # Get model prediction (1343-dimensional spectrum)
                spectrum_prediction = model(batch).cpu().numpy()  # Shape: (1, 1343)
                
                # Store results for summary
                predictions.append(spectrum_prediction[0])  # Remove batch dimension
                ground_truths.append(ground_truth[0])  # Remove batch dimension
                timestamps_input_list.append(timestamps_input_str)
                timestamps_targets_list.append(timestamps_targets_str)

                # Display results in table format
                print("\n" + "="*150)
                print(f"Sample {i+1}")
                print(f"{'Time Input':<20} | {'Time Target':<20} | {'Spectrum Shape':<20} | {'Pred Range':<27} | {'GT Range':<27}")
                print("-"*150)
                pred_min, pred_max = spectrum_prediction[0].min(), spectrum_prediction[0].max()
                gt_min, gt_max = ground_truth[0].min(), ground_truth[0].max()
                print(f"{timestamps_input_str:<20} | {timestamps_targets_str:<20} | {spectrum_prediction.shape[1]:<20} | [{pred_min:.4f}, {pred_max:.4f}]{'':<11} | [{gt_min:.4f}, {gt_max:.4f}]{'':<11}")
                print("="*150)

    # Calculate and display summary statistics
    predictions = np.array(predictions)  # Shape: (num_samples, 1343)
    ground_truths = np.array(ground_truths)  # Shape: (num_samples, 1343)
    
    # Calculate element-wise metrics
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths)**2))
    
    # R² calculation
    ss_res = np.sum((ground_truths - predictions)**2)
    ss_tot = np.sum((ground_truths - np.mean(ground_truths))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    
    # Calculate spectral correlation (average correlation per sample)
    correlations = []
    for i in range(len(predictions)):
        corr = np.corrcoef(predictions[i], ground_truths[i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    avg_correlation = np.mean(correlations) if correlations else 0.0
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"{'Metric':<35} | {'Value':<20}")
    print("-"*100)
    print(f"{'Mean Absolute Error':<35} | {mae:<20.6f}")
    print(f"{'Root Mean Square Error':<35} | {rmse:<20.6f}")
    print(f"{'R² Score':<35} | {r2:<20.6f}")
    print(f"{'Average Spectral Correlation':<35} | {avg_correlation:<20.6f}")
    print(f"{'Number of Samples':<35} | {len(predictions):<20}")
    print(f"{'Spectrum Dimension':<35} | {predictions.shape[1]:<20}")
    print("="*100)

    # Create visualization
    create_spectrum_plots(predictions, ground_truths, wavelengths, timestamps_input_list, save_path)
    
    return predictions, ground_truths, timestamps_input_list


def create_spectrum_plots(predictions, ground_truths, wavelengths, timestamps, save_path):
    """
    Create visualization plots for EUV spectra predictions
    """
    num_samples = len(predictions)
    
    # Create subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot predicted and ground truth spectra
        ax.plot(wavelengths, ground_truths[i], 'b-', label='Ground Truth', alpha=0.7, linewidth=2)
        ax.plot(wavelengths, predictions[i], 'r--', label='Prediction', alpha=0.7, linewidth=2)
        
        # Calculate correlation for this sample
        corr = np.corrcoef(predictions[i], ground_truths[i])[0, 1]
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Sample {i+1} - {timestamps[i]}\nCorrelation: {corr:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-limits
        y_min = min(ground_truths[i].min(), predictions[i].min()) - 0.05
        y_max = max(ground_truths[i].max(), predictions[i].max()) + 0.05
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser("EUV Spectra Prediction Inference")
    parser.add_argument(
        "--config_path",
        default="./config_infer.yaml",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="./assets/euv_spectra_weights.pth",
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
        choices=["test", "valid", "train"],
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
        repo_id="nasa-ibm-ai4science/euv_spectra_surya",
        local_dir="./assets",
        allow_patterns='*.pth',
        token=None,
    )

    main()
