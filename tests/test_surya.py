import os
from huggingface_hub import snapshot_download
import pytest
import yaml
import logging
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import sunpy.visualization.colormaps as sunpy_cm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from surya.datasets.helio import HelioNetCDFDataset, inverse_transform_single_channel
from surya.models.helio_spectformer import HelioSpectFormer
from surya.utils.data import build_scalers, custom_collate_fn

REFERENCE_LOSS = np.array(
    [
        0.24890851974487305,
        0.31166040152311325,
        0.32497961074113846,
        0.32198894023895264,
        0.3090871721506119,
        0.3190929591655731,
        0.36053016781806946,
        0.33693042397499084,
    ]
)
SDO_CHANNELS = [
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "aia1600",
    "hmi_m",
    "hmi_bx",
    "hmi_by",
    "hmi_bz",
    "hmi_v",
]

logger = logging.getLogger(__name__)


@dataclass
class SDOImage:
    channel: str
    data: np.ndarray
    timestamp: str
    type: str


@pytest.fixture(scope="module", autouse=True)
def download_data():
    snapshot_download(
        repo_id="nasa-ibm-ai4science/Surya-1.0",
        local_dir="data/Surya-1.0",
        allow_patterns=["config.yaml", "scalers.yaml", "surya.366m.v1.pt"],
        token=None,
    )
    snapshot_download(
        repo_id="nasa-ibm-ai4science/Surya-1.0_validation_data",
        repo_type="dataset",
        local_dir="data/Surya-1.0_validation_data",
        allow_patterns="20140107_1[5-9]??.nc",
        token=None,
    )


@pytest.fixture
def config() -> dict:
    with open("data/Surya-1.0/config.yaml") as fp:
        config = yaml.safe_load(fp)

    return config


@pytest.fixture
def model(config) -> HelioSpectFormer:
    model = HelioSpectFormer(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=len(config["data"]["sdo_channels"]),
        embed_dim=config["model"]["embed_dim"],
        time_embedding={
            "type": "linear",
            "time_dim": len(config["data"]["time_delta_input_minutes"]),
        },
        depth=config["model"]["depth"],
        n_spectral_blocks=config["model"]["n_spectral_blocks"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        drop_rate=config["model"]["drop_rate"],
        dtype=torch.bfloat16,
        window_size=config["model"]["window_size"],
        dp_rank=config["model"]["dp_rank"],
        learned_flow=config["model"]["learned_flow"],
        use_latitude_in_learned_flow=config["model"]["learned_flow"],
        init_weights=False,
        checkpoint_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        rpe=config["model"]["rpe"],
        ensemble=config["model"]["ensemble"],
        finetune=config["model"]["finetune"],
    )
    logger.info("Initialized the model.")

    return model


@pytest.fixture
def scalers() -> dict:
    scalers_info = yaml.safe_load(open("data/Surya-1.0/scalers.yaml", "r"))
    scalers = build_scalers(info=scalers_info)
    logger.info("Built the scalers.")
    return scalers


@pytest.fixture
def dataset(config, scalers) -> HelioNetCDFDataset:
    dataset = HelioNetCDFDataset(
        index_path="tests/test_surya_index.csv",
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=len(config["data"]["time_delta_input_minutes"]),
        rollout_steps=1,
        channels=config["data"]["sdo_channels"],
        drop_hmi_probability=config["data"]["drop_hmi_probability"],
        num_mask_aia_channels=config["data"]["num_mask_aia_channels"],
        use_latitude_in_learned_flow=config["data"]["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="valid",
        pooling=config["data"]["pooling"],
        random_vert_flip=config["data"]["random_vert_flip"],
    )
    logger.info(f"Initialized the dataset. {len(dataset)} samples.")

    return dataset


@pytest.fixture
def dataloader(dataset) -> DataLoader:
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2,
        prefetch_factor=None,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    return dataloader


def batch_step(
    model: HelioSpectFormer,
    batch_data: dict,
    batch_metadata: dict,
    device: int | str,
    rollout: int = 0,
) -> tuple[float, list[SDOImage]]:
    """
    Perform a single batch step for the given model, batch data, metadata, and device.

    Args:
        model: The PyTorch model to use for prediction.
        batch_data: A dictionary containing input and target data for the batch.
        batch_metadata: A dictionary containing metadata for the batch, including timestamps.
        device: The device to use for computation ('cpu', 'cuda' or device number).
        rollout: The number of steps to forecast ahead. Defaults to 0.

    Returns:
        tuple[float, list[SDOImage]]: A tuple containing the average loss for the batch and a list of SDOImage objects for input and output data.
    """

    loss = 0.0
    n_samples_x_steps = 0
    data_returned = []
    forecast_hat = None  # Initialize forecast_hat
    for t_idx in range(2):
        timestamp = np.datetime_as_string(
            batch_metadata["timestamps_input"][0][t_idx], unit="s"
        )
        data_returned.append(
            SDOImage(
                channel="aia94",
                data=batch_data["ts"][0, SDO_CHANNELS.index("aia94"), t_idx].numpy(),
                timestamp=timestamp,
                type="input",
            )
        )

    for step in range(0, rollout + 1):
        if step == 0:
            curr_batch = {
                key: batch_data[key].to(device) for key in ["ts", "time_delta_input"]
            }
        else:
            # Use the previous forecast_hat from the previous iteration
            if forecast_hat is not None:
                curr_batch["ts"] = torch.cat(
                    (curr_batch["ts"][:, :, 1:, ...], forecast_hat[:, :, None, ...]),
                    dim=2,
                )

        forecast_hat = model(curr_batch)
        curr_target = batch_data["forecast"][:, :, step, ...].to(device)
        curr_batch_loss = F.mse_loss(forecast_hat, curr_target)
        loss += curr_batch_loss.item()
        n_samples_x_steps += curr_batch["ts"].shape[0]

        timestamp = np.datetime_as_string(
            batch_metadata["timestamps_targets"][0][step], unit="s"
        )
        data_returned.append(
            SDOImage(
                channel="aia94",
                data=forecast_hat.to(dtype=torch.float32)
                .cpu()[0, SDO_CHANNELS.index("aia94")]
                .numpy(),
                timestamp=timestamp,
                type="output",
            )
        )

    loss = loss / n_samples_x_steps

    return loss, data_returned


def test_surya_20140107(model: HelioSpectFormer, dataloader: DataLoader, caplog):
    """
    End to end test for the Surya foundation model.

    Args:
        model: An instance of the HelioSpectFormer model
        dataloader: A PyTorch DataLoader wrapping the HelioNetCDF dataset class.
        caplog: A pytest fixture for capturing logging output.

    Returns:
        None
    """
    os.makedirs("logs/data", exist_ok=True)
    caplog.set_level(logging.INFO)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        logger.info(f"GPU detected. Running the test on device {device}.")
    else:
        device = "cpu"
        logger.warning(f"No GPU detected. Running the test on CPU.")

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Surya FM: {n_parameters:.2f} M total parameters.")
    path_weights = "data/Surya-1.0/surya.366m.v1.pt"
    weights = torch.load(
        path_weights, map_location=torch.device(device), weights_only=True
    )
    model.load_state_dict(weights, strict=True)
    logger.info("Loaded weights.")

    logger.info("Starting inference run.")
    loss = []
    plot_data = []
    for batch_idx, (batch_data, batch_metadata) in enumerate(dataloader):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                batch_loss, data_returned = batch_step(
                    model, batch_data, batch_metadata, device, rollout=1
                )
                loss.append(batch_loss)
                plot_data.append(data_returned)

    ref_loss_mean = np.mean(REFERENCE_LOSS)
    loss = np.array(loss)
    loss_mean = np.mean(loss)
    loss_delta = loss_mean - ref_loss_mean
    logger.info(
        f"Completed validation run. Local loss {loss_mean:.5f}. "
        f"Reference loss {ref_loss_mean:.5f}. Deviation {loss_delta}."
    )

    assert np.abs(loss_delta) < 1.0e-4

    logger.info("Preparing visualization")
    plot_data = sorted(plot_data, key=lambda data_returned: data_returned[0].timestamp)
    means, stds, epsilons, sl_scale_factors = dataloader.dataset.transformation_inputs()
    c_idx = SDO_CHANNELS.index("aia94")
    vmin = float("-inf")
    vmax = float("inf")
    for j in range(8):
        for sdo_image in plot_data[j]:
            sdo_image.data = inverse_transform_single_channel(
                sdo_image.data,
                mean=means[c_idx],
                std=stds[c_idx],
                epsilon=epsilons[c_idx],
                sl_scale_factor=sl_scale_factors[c_idx],
            )
            vmin = max(vmin, sdo_image.data.min())
            vmax = min(vmax, np.quantile(sdo_image.data, 0.99))

    plt_kwargs = {"vmin": vmin, "vmax": vmax, "cmap": sunpy_cm.cmlist["sdoaia94"], "origin" : "lower"}
    fig, ax = plt.subplots(8, 4, figsize=(16, 32))
    for j in range(8):
        for i in range(4):
            ax[j, i].axis("off")
            ax[j, i].imshow(plot_data[j][i].data, **plt_kwargs)
            ax[j, i].set_title(
                f"{plot_data[j][i].type.capitalize()} - {plot_data[j][i].timestamp}"
            )

    fig.suptitle("Surya - AIA94", y=0.9)
    plt.savefig("surya_model_validation.png", dpi=120, bbox_inches="tight")
    logger.info("Saved visualization at surya_model_validation.png.")
    assert True
