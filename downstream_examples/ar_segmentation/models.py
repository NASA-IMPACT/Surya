import torch
from torch import nn


from surya.models.helio_spectformer import HelioSpectFormer

from surya.models.embedding import (
    LinearDecoder,
    PerceiverDecoder,
)
import torch.nn.functional as F
from typing import Callable
from functools import partial


class ChannelAdapter(nn.Module):
    def __init__(self, model, num_data_chans, time_dim, out_chans: int=13,):
        super().__init__()
        self.num_data_chans = num_data_chans
        self.out_chans = out_chans
        self.time_dim = time_dim
        
        self.adapter = nn.Conv3d(self.num_data_chans, self.out_chans, kernel_size=1, padding=0)
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch['ts'] = self.adapter(batch['ts'])
        x = self.model(batch)
        return x

class HelioSpectformer1D(HelioSpectFormer):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        time_embedding: dict,
        depth: int,
        n_spectral_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        window_size: int,
        dp_rank: int,
        learned_flow: bool = False,
        use_latitude_in_learned_flow: bool = False,
        init_weights: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        checkpoint_layers: list[int] | None = None,
        rpe: bool = False,
        finetune: bool = False,
        # Put finetuning additions below this line
        nglo: int = 0,
        dropout: float = 0.1,
        num_outputs: int = 1,
        num_penultimate_transformer_layers: int = 1,
        num_penultimate_heads: int = 8,
        config=None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            dtype=dtype,
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            finetune=finetune,
        )

        self.pooling_strategies = [
            config["model"]["global_average_pooling"],
            config["model"]["global_max_pooling"],
            config["model"]["attention_pooling"],
            config["model"]["transformer_pooling"],
        ]

        assert (
            sum(self.pooling_strategies) == 1
        ), "No or multiple pooling strategy selected. Aborting."

        self.global_average_pooling = False
        self.global_max_pooling = False
        self.attention_pooling = False
        self.transformer_pooling = False

        if config["model"]["dropout"] is not None:
            self.dropout_layer = nn.Dropout(config["model"]["dropout"])
            self.dropout = True

        if config["model"]["global_average_pooling"]:
            self.global_average_pooling = True

        elif config["model"]["global_max_pooling"]:
            self.global_max_pooling = True

        elif config["model"]["attention_pooling"]:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_penultimate_heads, dropout=dropout
            )
            self.attention_pooling = True

        elif config["model"]["transformer_pooling"]:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # (batch, 1, 1, token_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_penultimate_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
            )
            self.downstream_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_penultimate_transformer_layers
            )
            self.transformer_pooling = True

        else:
            raise Exception("No valid pooling strategy selected.")

        if config["model"]["penultimate_linear_layer"]:
            self.linear = nn.Linear(embed_dim, embed_dim)
            self.penultimate_linear_layer = True

        self.unembed = nn.Linear(embed_dim, num_outputs)

    def forward(self, batch):

        tokens = super().forward(batch=batch)

        if self.dropout is not None:
            tokens = self.dropout_layer(tokens)

        if self.penultimate_linear_layer:
            tokens = self.linear(tokens)

        # Global average pooling
        if self.global_average_pooling:
            agg_tokens = torch.mean(tokens, dim=1)  # (B, L, D) -> (B, D)

        # Global max pooling
        if self.global_max_pooling:
            agg_tokens, _ = torch.max(tokens, dim=1)  # (B, L, D) -> (B, D)

        # Global attention pooling
        if self.attention_pooling:
            tokens = tokens.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)
            tokens = self.attention(query=tokens, key=tokens, value=tokens)  # (L, B, D)
            agg_tokens = tokens.sum(dim=0)  # (B, D)

        if self.transformer_pooling:
            batch_size = tokens.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  #  (B, 1, D)
            tokens = torch.cat((cls_tokens, tokens), dim=1)  #  (B, L+1, D)
            tokens = tokens.permute(1, 0, 2)  # (B, L+1, D) -> (L+1, B, D)
            tokens = self.downstream_transformer(tokens)  # (L+1, B, D)
            agg_tokens = tokens[0, :, :]  # (B, D)

        if self.dropout is not None:
            out = self.dropout_layer(agg_tokens)

        out = self.unembed(out)

        return out


class HelioSpectformer2D(HelioSpectFormer):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        time_embedding: dict,
        depth: int,
        n_spectral_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        window_size: int,
        dp_rank: int,
        learned_flow: bool = False,
        use_latitude_in_learned_flow: bool = False,
        init_weights: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        checkpoint_layers: list[int] | None = None,
        rpe: bool = False,
        finetune: bool = False,
        # Put finetuning additions below this line
        config=None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            dtype=dtype,
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            finetune=finetune,
        )

        match config["model"]["ft_unembedding_type"]:
            case "linear":
                self.unembed = LinearDecoder(
                    patch_size=patch_size,
                    out_chans=config["model"]["ft_out_chans"],
                    embed_dim=embed_dim,
                )
            case "perceiver":
                self.unembed = PerceiverDecoder(
                    embed_dim=embed_dim,
                    patch_size=patch_size,
                    out_chans=config["model"]["ft_out_chans"],
                )
            case _:
                raise NotImplementedError(
                    f'Embedding {time_embedding["type"]} has not been implemented.'
                )

        # self.sigmoid = nn.Sigmoid()

    def forward(self, batch):

        tokens = super().forward(batch=batch)

        # Unembed the tokens
        # BE L D -> BE C H W
        forecast_hat = self.unembed(tokens)
        # forecast_hat = self.sigmoid(forecast_hat)

        return forecast_hat


class UNet(nn.Module):
    """U-Net [1]: encoder–decoder with skip connections for precise segmentation.

    Core idea: downsampling path captures context; upsampling path restores
    resolution while fusing encoder features via concatenation skip connections.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        out_chans: int,
        n_blocks: int = 5,
        activation: Callable = F.gelu,
    ):
        super().__init__()

        self.encoder = UNetEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            n_down=n_blocks,
            activation=activation,
        )
        self.decoder = UNetDecoder(
            embed_dim=embed_dim,
            n_up=n_blocks,
            concat_activations=True,
            activation=activation,
        )
        self.out = nn.Conv2d(embed_dim, out_chans, kernel_size=1, padding=0)

    def forward(self, batch: dict[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing key `ts` which defines tensor
            of shape (B, C_in, T, H, W) OR (B, C_in, H, W). Here, C_in
            equals in_chans. In the latter case, the time dimension
            T is being discarded.
        Returns:
            Tensor of shape (B, C_out, H, W).
        """
        x = batch["ts"]

        if x.ndim == 5:
            x = x[:, :, -1, :, :]

        h = self.encoder(x)
        x = self.decoder(h)
        x = self.out(x)

        return x


class DoubleConv(nn.Module):
    """Two Conv2d layers with nonlinearity, as in standard U-Net blocks."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        padding: int = 1,
        activation: Callable = F.gelu,
    ):
        super().__init__()

        self.activation = activation
        self.conv_a = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding)
        self.conv_b = nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        r_value = self.activation(self.conv_b(self.activation(self.conv_a(x))))

        return r_value


class UNetEncoder(nn.Module):
    """U-Net encoder producing a pyramid of feature maps at decreasing scales."""

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        n_down: int,
        activation: Callable = F.gelu,
        pooling: Callable = partial(F.max_pool2d, kernel_size=2),
    ):
        super().__init__()

        self.pooling = pooling

        blocks = [DoubleConv(in_chans, embed_dim, kernel_size=7, padding=3, activation=activation)]
        for i in range(n_down):
            blocks.append(
                DoubleConv(
                    2**i * embed_dim,
                    2 ** (i + 1) * embed_dim,
                    kernel_size=3,
                    padding=1,
                    activation=activation,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, C_in, H_in, W_in)
        Returns:
            List of tensors of shape (B, C, H, W).
            These will have varying values for C, H, W. E.g. (for n_down=2, embed_dim=4):
            [
                torch.Size([1, 4, 16, 16]),
                torch.Size([1, 8, 8, 8]),
                torch.Size([1, 16, 4, 4]),
            ]
        """

        x = self.blocks[0](x)

        intermediates = [x]
        for b in self.blocks[1:]:
            x_p = self.pooling(x)
            x = b(x_p)
            intermediates.append(x)

        return intermediates


class UNetDecoder(nn.Module):
    """U-Net decoder that upsamples and fuses encoder features via skip connections."""

    def __init__(
        self,
        embed_dim: int,
        n_up: int,
        concat_activations: bool,
        activation: Callable = F.gelu,
        unpool_factor: int = 2,
    ):
        """
        Args:
            concat_activations: Whether to copy and concatenate the UNet activations.
                For a standard UNet, this is true. Setting this to false corresponds
                more to autoencoder behavior.
        """
        super().__init__()

        self.concat_activations = concat_activations
        concat_factor = (
            2 if concat_activations else 1
        )  # If we concat, we need to double the size of the input ...

        blocks = []
        for i in range(n_up):
            blocks.append(
                nn.ModuleDict(
                    {
                        "up": nn.Sequential(
                            DoubleConv(
                                2 ** (n_up - i) * embed_dim,
                                unpool_factor**2 * 2 ** (n_up - i - 1) * embed_dim,
                                kernel_size=1,
                                padding=0,
                                activation=activation,
                            ),
                            nn.PixelShuffle(unpool_factor),
                        ),
                        "conv": DoubleConv(
                            concat_factor * 2 ** (n_up - i - 1) * embed_dim,
                            2 ** (n_up - i - 1) * embed_dim,
                            kernel_size=3,
                            padding=1,
                            activation=activation,
                        ),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: List of tensors. Has to have same length as n_up. Should simply be the
                output of UNetEncoder. If not, here is an example of shape
                for n_up=2, embed_dim=4:
                [
                    torch.Size([1, 4, 16, 16]),
                    torch.Size([1, 8, 8, 8]),
                    torch.Size([1, 16, 4, 4]),
                ]
        """
        intermediates = x
        x = intermediates.pop()

        for h, b in zip(reversed(intermediates), self.blocks):
            x = b["up"](x)
            if self.concat_activations:
                x = torch.cat((x, h), dim=1)
            x = b["conv"](x)

        return x
