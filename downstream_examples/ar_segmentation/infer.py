import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sunpy.visualization.colormaps as sunpy_cm
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Import from the same directory
from models import HelioSpectformer2D, UNet
from peft import LoraConfig, get_peft_model

# Import from surya
from surya.utils.data import build_scalers
from surya.utils.distributed import set_global_seed
from surya.datasets.helio import HelioNetCDFDataset
from finetune import custom_collate_fn


def get_model(config) -> torch.nn.Module:
    """
    Initialize and return the model based on the configuration.
    """
    print("Creating the model.")

    if config["model"]["model_type"] == "spectformer_lora":
        print("Initializing spectformer with LoRA.")
        model = HelioSpectformer2D(
            img_size=config["model"]["img_size"],
            patch_size=config["model"]["patch_size"],
            in_chans=config["model"]["in_channels"],
            embed_dim=config["model"]["embed_dim"],
            time_embedding=config["model"]["time_embedding"],
            depth=config["model"]["depth"],
            n_spectral_blocks=config["model"]["spectral_blocks"],
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
            rpe=config["model"]["rpe"],
            finetune=config["model"]["finetune"],
            config=config,
        )
    elif config["model"]["model_type"] == "unet":
        print("Initializing UNet.")
        model = UNet(
            in_chans=config["model"]["in_channels"],
            embed_dim=config["model"]["unet_embed_dim"],
            out_chans=1,
            n_blocks=config["model"]["unet_blocks"],
        )
    else:
        raise ValueError(f"Unknown model type {config['model']['model_type']}.")

    return model


def apply_peft_lora(model: torch.nn.Module, config) -> torch.nn.Module:
    """
    Apply PEFT LoRA to the model
    """
    if not "lora_config" in config["model"]:
        print("No LoRA configuration found. Using default LoRA settings.")
        lora_config = {
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            "lora_dropout": 0.1,
            "bias": "none",
        }
    else:
        lora_config = config["model"]["lora_config"]

    print(f"Applying PEFT LoRA with configuration: {lora_config}")

    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        ),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        bias=lora_config.get("bias", "none"),
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    return model


def load_model(config, checkpoint_path, device):
    """
    Load the trained model from checkpoint
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model
    model = get_model(config)
    
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
        print("Model loaded successfully with strict=True")
    except Exception as e:
        print(f"Failed to load with strict=True: {e}")
        raise e
    
    model.to(device)
    model.eval()
    
    return model


def get_dataloader(config, scalers, data_type="test",num_samples=3):
    """
    Create dataloader for inference
    """

    index_path = config["data"]["valid_data_path"]

    dataset = HelioNetCDFDataset(
        sdo_data_root_path=config["data"]["sdo_data_root_path"],
        index_path=index_path,
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=len(config["data"]["time_delta_input_minutes"]),
        rollout_steps=1,
        channels=config["data"]["channels"],
        scalers=scalers,
        phase="valid",
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


def save_predictions(predictions, targets, output_dir, sample_idx):
    """
    Save prediction visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy and apply sigmoid
    pred = torch.sigmoid(predictions).cpu().numpy()
    target = targets.cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot prediction
    axes[0].imshow(pred.squeeze(), cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Prediction')
    axes[0].axis('off')
    
    # Plot target
    axes[1].imshow(target.squeeze(), cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot difference
    diff = np.abs(pred.squeeze() - target.squeeze())
    axes[2].imshow(diff, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{sample_idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def run_inference(config, checkpoint_path, output_dir, device, data_type="test",num_samples=3,device_type="cuda"):
    """
    Run inference on the dataset
    """
    print(f"Starting inference with checkpoint: {checkpoint_path}")
    
    # Build scalers
    scalers = build_scalers(info=config["data"]["scalers"])
    
    # Load model
    model = load_model(config, checkpoint_path, device)
    
    # Get dataloader
    dataloader = get_dataloader(config, scalers, data_type,num_samples )
    
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Run inference
    infer_single_sample(
            model=model,
            local_rank=device,
            validation_loader=dataloader,
            gpu=True,
            scalers=scalers,
            channels=config["data"]["channels"],
            dtype=config["dtype"],
            save_path=os.path.join(output_dir,'test.png'),
            time_dim=config["model"]["time_embedding"]["time_dim"],
            device_type=device_type
        )
    
    print(f"Inference complete. Results saved to: {output_dir}")

def infer_single_sample(
    model: torch.nn.Module,
    local_rank: int,
    validation_loader: DataLoader,
    gpu: bool,
    scalers,
    channels,
    dtype: torch.dtype,
    save_path: str = "test.png",
    time_dim=1,
    device_type="cuda"
):
    model.eval()
    with torch.no_grad():
        for i, (batch, metadata) in enumerate(validation_loader):
            print(f"Running inference on batch {i}.")
            if gpu:
                batch = {k: v.to(local_rank) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device_type, dtype=dtype):

                forecast_hat = model(batch)
                if forecast_hat.ndim == 5:
                    forecast_hat = forecast_hat[:, 0]

                def assort_all_channels(tensor):
                    return torch.cat(
                        [
                            (
                                tensor[channels.index(channel), :, :][None, :, :]
                                if channel in channels
                                else torch.empty_like(tensor[0, :, :][None, :, :])
                            )
                            for channel in channels
                        ],
                        dim=0,
                    )

                save_dict = {}
                for input_idx in range(time_dim):
                    save_dict[f"input_{input_idx}"] = assort_all_channels(
                        batch["ts"][0, :, input_idx, :, :].squeeze().cpu()
                    )

                forecast_hat = forecast_hat.expand(1,13,-1,-1)
                forecast_hat = torch.sigmoid(forecast_hat)
                
                save_dict |= {
                    "prediction": assort_all_channels(
                        forecast_hat[0, ...].squeeze().cpu()
                    ),
                }

                iter_save_path = save_path.replace(".png", f"_{i}.png")
                os.makedirs(os.path.dirname(iter_save_path), exist_ok=True)

                # Save images as numpy
                # for key in save_dict.keys():
                #     np.save(
                #         iter_save_path.replace(".png", f"_{key}.npy"),
                #         save_dict[key].float().numpy(),
                #     )

                metadata_str = format_metadata(metadata)

                plot_sun_sdo_cmap2(
                    list(save_dict.values()),
                    scaler=scalers,
                    save_path=iter_save_path,
                    dpi=100,
                    scaler_keys=channels[:-5],
                    title_txt=metadata_str,
                )


def format_metadata(metadata, target_timestamp=0):
    """
    Convert a dictionary of datetime64[ns] arrays into a formatted string
    where timestamps are shown up to the minute.

    Args:
        metadata (dict): Dictionary where values are lists of numpy datetime64 arrays.

    Returns:
        str: Formatted string in "key: value1 value2 ..., key2: value3 value4 ..." format.
    """
    # Step 1: Convert datetime64 arrays to formatted strings
    formatted_metadata = {}
    for key, value in metadata.items():
        formatted_values = []
        for arr in value:
            formatted_values.extend(
                np.datetime_as_string(arr, unit="m")
            )  # Keep only YYYY-MM-DDTHH:MM
        formatted_metadata[key] = formatted_values

    # Step 2: Convert the formatted dictionary to a string representation
    formatted_strings = []
    formatted_strings.append(
        f"timestamps_input: {' '.join(formatted_metadata['timestamps_input'])}"
    )
    formatted_strings.append(
        f"timestamps_targets: {formatted_metadata['timestamps_targets'][target_timestamp]}"
    )
    # Step 3: Join all key-value pairs into a single string

    return ", ".join(formatted_strings)

def inverse_transform_single_channel(data, mean, std, sl_scale_factor, epsilon):
    """
    Implements inverse signum log transform.

    Args:
        data: Numpy array of shape C, H, W
        means: Numpy array of shape C. Mean per channel.
        stds: Numpy array of shape C. Standard deviation per channel.
        sl_scale_factors: Numpy array of shape C. Signum-log scale factors.
        epsilons: Numpy array of shape C. Constant to avoid zero division.

    Returns:
        Numpy array of shape C, H, W.
    """
    data = data * (std + epsilon) + mean

    data = np.sign(data) * np.expm1(np.abs(data))

    data = data / sl_scale_factor

    return data

def tensor_to_numpy(tensor, all_channels, scaler=None, channel_idx=None):
    """
    Convert a tensor to a numpy array and apply unnormalization if scaler is provided.

    Args:
        tensor (torch.Tensor): Tensor to convert
        scaler (dict): Dictionary containing 'mean', 'std', and 'epsilon' for unnormalization
        channel_idx (int): Index of the channel (used to select the scaler)

    Returns:
        numpy.ndarray: Converted numpy array
    """

    tensor = tensor.float().numpy()

    if scaler:
        band = all_channels[channel_idx]

        mean = scaler[band].mean
        std = scaler[band].std
        epsilon = scaler[band].epsilon
        sl_scale_factor = scaler[band].sl_scale_factor

        tensor = inverse_transform_single_channel(tensor, mean, std, sl_scale_factor, epsilon)
        tensor = np.sign(tensor) * np.log1p(np.abs(tensor))


    return tensor


def process_tensors(tensor_list, all_channels, scaler, n_channels):
    np_list = []
    for c, tensor in enumerate(tensor_list):
        
        if c != 0:
            scaler = None
        
        np = [
            tensor_to_numpy(tensor[i], all_channels, scaler=scaler, channel_idx=i)
            for i in range(n_channels)
        ]
        # np = [_inverse_signum_log_transform(tensor[i]) / scaler[i].sl_scale_factor for i in range(n_channels)]
        np_list.append(np)
    return np_list


def plot_sun_sdo_cmap2(
    images_tensor,
    save_path,
    scaler,
    dpi=100,
    scaler_keys=["0094", "0131", "0171", "0193", "0211", "0304", "0335", "hmi"],
    title_txt="AIA/HMI Plots",
):
    """
    Plot and save a grid of images with different colormaps for each channel.

    Args:
        images_tensor (list of torch.tensor): List of 3D tensors, each of shape [8, 4096, 4096],
                                            representing the image data for different tensors.
                                            Each tensor corresponds to a different set of images.
        save_path (str): Path to save the combined image. The file format is inferred from the file extension.
        scaler (dict): Dictionary containing scaling factors for the bands.
        dpi (int): Resolution of the saved image.

    Description:
        This function arranges and plots the images from the provided list of tensors (`images_tensor`).
        Each tensor in the list is expected to contain 8 channels. The function uses predefined colormaps
        from the `sunpy` library to visualize each channel of each tensor. The colormaps used are:
        'sdoaia94', 'sdoaia131', 'sdoaia171', 'sdoaia193', 'sdoaia211', 'sdoaia304', 'sdoaia335',
        and 'hmimag'.

        The channels for each tensor are displayed in rows, with each row corresponding to one tensor and
        each column representing a different channel. A single horizontal colorbar is added for each column
        (channel). The final combined image is saved to the path specified by `save_path`.
    """

    # Process the tensors (e.g., scaling or normalization)
    color_channel_mapping = {
        "aia94": sunpy_cm.cmlist["sdoaia94"],
        "aia131": sunpy_cm.cmlist["sdoaia131"],
        "aia171": sunpy_cm.cmlist["sdoaia171"],
        "aia193": sunpy_cm.cmlist["sdoaia193"],
        "aia211": sunpy_cm.cmlist["sdoaia211"],
        "aia304": sunpy_cm.cmlist["sdoaia304"],
        "aia335": sunpy_cm.cmlist["sdoaia335"],
        "aia1600": sunpy_cm.cmlist["sdoaia1600"],
        "hmi_m": sunpy_cm.cmlist["hmimag"],
        "hmi_bx": sunpy_cm.cmlist["hmimag"],
        "hmi_by": sunpy_cm.cmlist["hmimag"],
        "hmi_bz": sunpy_cm.cmlist["hmimag"],
        "hmi_v": plt.get_cmap("bwr"),  # Special case for 'hmi_v'
    }

    n_channels = len(scaler_keys)
    images_tensor = process_tensors(
        images_tensor, scaler_keys, scaler, n_channels=n_channels
    )

    # Using the colour_maps as references
    num_rows = len(images_tensor)
    band_name = list(scaler.keys())

    # Create figure with subplots, using GridSpec for more control
    fig = plt.figure(figsize=(5 * n_channels, 5 * num_rows), dpi=dpi)
    gs = GridSpec(num_rows, n_channels, figure=fig, wspace=0, hspace=0.005)

    fig.suptitle(title_txt, fontsize=24, fontweight="bold")
    # Loop through each channel and each tensor to plot
    for i, channel in enumerate(scaler_keys):
        vmin = scaler[channel].min
        vmax = scaler[channel].max
        for n in range(num_rows):
            ax = fig.add_subplot(gs[n, i])  # Select the subplot using GridSpec
            
            if n == 0:  # Set title for each column on the top row
                try:
                    if "hmi" in channel:
                        vmin = -vmax  # for hmi we need to have center at 0.

                    im = ax.imshow(
                        images_tensor[n][i],
                        cmap=color_channel_mapping[channel],
                        vmin=vmin,
                        vmax=vmax,
                    )
                except TypeError as e:
                    sizes = [t.shape for t in images_tensor]
                    print(f"Sizes: {sizes}.")
                    raise e

                ax.set_title(f"Band {band_name[i]}", fontsize=18)
            
            else:
                im = ax.imshow(
                        images_tensor[n][i],
                        cmap=plt.get_cmap("gray"),
                        vmin=0,
                        vmax=1,
                    )
            ax.axis("off")
        # Add a single horizontal colorbar directly below each column
        cbar_ax = fig.add_axes(
            [
                ax.get_position().x0,
                ax.get_position().y0 - 0.02,
                ax.get_position().width,
                0.01,
            ]
        )
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal", fraction=0.046)
        # cbar.ax.xaxis.set_tick_params(rotation=-90)

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved image at: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser("AR Segmentation Inference")
    parser.add_argument(
        "--config_path",
        default="./config_infer.yaml",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="./assets/ar_segmentation_weights.pth",
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
        repo_id="nasa-ibm-ai4science/ar_segmentation_surya",
        local_dir="./assets",
        allow_patterns='*.pth',
        token=None,
    )

    main()