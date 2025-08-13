import torch
import torch.nn as nn
import torch.nn.functional as F


class HelioFlowModel(nn.Module):
    def __init__(self, img_size=(4096, 4096), use_latitude_in_learned_flow=False):
        super().__init__()

        self.use_latitude_in_learned_flow = use_latitude_in_learned_flow

        u = torch.linspace(-1, 1, img_size[0])
        v = torch.linspace(-1, 1, img_size[1])
        u, v = torch.meshgrid(u, v, indexing="xy")
        self.register_buffer(
            "grid", torch.stack((u, v), dim=2).view(1, *img_size, 2)
        )  # B, H, W, 2

        # Higher modes can be used for explicit feature engineering for flow features.
        if self.use_latitude_in_learned_flow:
            higher_modes = [u, v, torch.ones_like(u)]
        else:
            higher_modes = [
                u,
                v,
            ]
        self.register_buffer(
            "higher_modes", torch.stack(higher_modes, dim=2).view(1, *img_size, -1)
        )

        self.flow_generator = nn.Sequential(
            nn.Linear(self.higher_modes.shape[3], 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, batch):
        """
        Args:
            batch: Dictionary containing keys `ts` and
            `forecast_latitude` (optionally).
                ts (torch.Tensor):                B, C, T, H, W
                forecast_latitude (torch.Tensor): B, L
            B - Batch size, C - Channels, T - Input times, H - Image height,
            W - Image width, L - Lead time.
        """

        x = batch["ts"]
        B, C, T, H, W = x.shape
        if T == 1:
            x = x[:, :, -1, :, :]
        else:
            # Taking the average of the last two time stamps
            x = (x[:, :, -1, :, :] + x[:, :, -2, :, :]) / 2

        # Flow fields have the shape B, H_out, W_out, 2
        if self.use_latitude_in_learned_flow:
            broadcast_lat = batch["forecast_latitude"] / 7
            broadcast_lat = torch.concatenate(
                [
                    torch.ones_like(broadcast_lat),
                    torch.ones_like(broadcast_lat),
                    broadcast_lat,
                ],
                1,
            )[:, None, None, :]
            higher_modes = self.higher_modes * broadcast_lat
            flow_field = self.grid + self.flow_generator(higher_modes)
        else:
            flow_field = self.grid + self.flow_generator(self.higher_modes)
        flow_field = flow_field.expand(B, H, W, 2)

        y_hat = F.grid_sample(
            x,
            flow_field,
            mode="bilinear",
            padding_mode="border",  # Possible values: zeros, border, or reflection.
            align_corners=False,
        )

        return y_hat
