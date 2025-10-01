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

from dataset import SolarFlareDataset
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
    
    # print(model)
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


def get_dataloader(config, scalers, data_type="test",num_samples=3):
    """
    Create dataloader for inference
    """

    dataset = SolarFlareDataset(
        #### All these lines are required by the parent HelioNetCDFDataset class
        sdo_data_root_path=config["data"]["sdo_data_root_path"],
        index_path=config["data"]["valid_data_path"],
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=config["model"]["time_embedding"]["time_dim"],
        rollout_steps=config["rollout_steps"],
        channels=config["data"]["channels"],
        drop_hmi_probability=config["drop_hmi_probability"],
        num_mask_aia_channels=config["num_mask_aia_channels"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="valid",
        #### Put your donwnstream (DS) specific parameters below this line
        flare_index_path=config["data"]["flare_data_path"],
        pooling=config["data"]["pooling"],
        random_vert_flip=False,
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


def run_inference(config, checkpoint_path, output_dir, device, data_type="test",num_samples=3,device_type="cuda"):
    """
    Run inference on the dataset
    """
    
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
            dtype=config["dtype"],
            save_path=os.path.join(output_dir,'test.png'),
            device_type=device_type
        )


def infer_single_sample(
    model: torch.nn.Module,
    local_rank: int,
    validation_loader: DataLoader,
    gpu: bool,
    dtype: torch.dtype,
    save_path: str = "test.png",
    device_type="cuda"
):
    model.eval()
    with torch.no_grad():
        for i, (batch, metadata) in enumerate(validation_loader):

            if gpu:
                batch = {k: v.to(local_rank) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device_type, dtype=dtype):

                ground_truth = batch["label"].item()
                timestamps_input = metadata["timestamps_input"]
                timestamps_targets = metadata["timestamps_targets"]
                
                # Convert numpy datetime64 to readable string format
                timestamps_input_str = np.datetime_as_string(timestamps_input, unit='m')[0][0]
                timestamps_targets_str = np.datetime_as_string(timestamps_targets, unit='m')[0][0]
                
                forecast_hat = int(F.sigmoid(model(batch)).item() > 0.5)
                

                # Display results in table format
                print("\n" + "="*80)
                print(f"{'Time Input':<20} | {'Time Target':<20} | {'Prediction':<15} | {'Ground Truth':<12}")
                print("-"*80)
                print(f"{timestamps_input_str:<20} | {timestamps_targets_str:<20} | {forecast_hat:<15} | {ground_truth:<12}")
                print("="*80)



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
        default="./assets/solar_flare_weights.pth",
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
        repo_id="nasa-ibm-ai4science/solar_flares_surya",
        local_dir="./assets",
        allow_patterns='*.pth',
        token=None,
    )

    main()