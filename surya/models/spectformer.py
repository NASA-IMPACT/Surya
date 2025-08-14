import math
import logging
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.layers import DropPath, trunc_normal_
import torch.fft

from .transformer_ls import AttentionLS

_logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape  # torch.Size([1, 262144, 1024])
        if spatial_size is None:
            a = b = int(math.sqrt(N))  # a=b=512
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)  # torch.Size([1, 512, 512, 1024])

        # FROM HERE USED TO BE AUTOCAST to float32
        dtype = x.dtype
        x = x.to(torch.float32)
        x = torch.fft.rfft2(
            x, dim=(1, 2), norm="ortho"
        )  # torch.Size([1, 512, 257, 1024])
        weight = torch.view_as_complex(
            self.complex_weight.to(torch.float32)
        )  # torch.Size([512, 257, 1024])
        x = x * weight
        x = torch.fft.irfft2(
            x, s=(a, b), dim=(1, 2), norm="ortho"
        )  # torch.Size([1, 512, 512, 1024])
        x = x.to(dtype)

        x = x.reshape(B, N, C)  # torch.Size([1, 262144, 1024])
        # UP TO HERE USED TO BE AUTOCAST to float32

        return x


class BlockSpectralGating(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        h=14,
        w=8,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, *args):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class BlockAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads: int = 8,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        w=2,
        dp_rank=2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rpe=False,
        adaLN=False,
        nglo=0,
    ):
        """
        num_heads: Attention heads. 4 for tiny, 8 for small and 12 for base
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.attn = AttentionLS(
            dim=dim,
            num_heads=num_heads,
            w=w,
            dp_rank=dp_rank,
            nglo=nglo,
            rpe=rpe,
        )

        if adaLN:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(dim, dim, bias=True),
                act_layer(),
                nn.Linear(dim, 6 * dim, bias=True),
            )
        else:
            self.adaLN_modulation = None

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.adaLN_modulation is not None:
            (
                shift_mha,
                scale_mha,
                gate_mha,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(c).chunk(6, dim=2)
        else:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = 6 * (1.0,)

        x = x + gate_mha * self.drop_path(
            self.attn(
                self.norm1(x) * scale_mha + shift_mha,
            )
        )
        x = x + gate_mlp * self.drop_path(
            self.mlp(self.norm2(x) * scale_mlp + shift_mlp)
        )

        return x


class SpectFormer(nn.Module):
    def __init__(
        self,
        grid_size: int = 224 // 16,
        embed_dim=768,
        depth=12,
        n_spectral_blocks=4,
        num_heads: int = 8,
        mlp_ratio=4.0,
        uniform_drop=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        window_size=2,
        dp_rank=2,
        norm_layer=nn.LayerNorm,
        checkpoint_layers: list[int] | None = None,
        rpe=False,
        ensemble: int | None = None,
        nglo: int = 0,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            n_spectral_blocks (int): number of spectral gating blocks
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            uniform_drop (bool): true for uniform, false for linearly increasing drop path probability.
            drop_rate (float): dropout rate
            drop_path_rate (float): drop path (stochastic depth) rate
            window_size: window size for long/short attention
            dp_rank: dp rank for long/short attention
            norm_layer: (nn.Module): normalization layer for attention blocks
            checkpoint_layers: indicate which layers to use for checkpointing
            rpe: Use relative position encoding in Long-Short attention blocks.
            ensemble: Integer indicating ensemble size or None for deterministic model.
            nglo: Number of (additional) global tokens.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_spectral_blocks = n_spectral_blocks
        self._checkpoint_layers = checkpoint_layers or []
        self.ensemble = ensemble
        self.nglo = nglo

        h = grid_size
        w = h // 2 + 1

        if uniform_drop:
            _logger.info(f"Using uniform droppath with expect rate {drop_path_rate}.")
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            _logger.info(
                f"Using linear droppath with expect rate {drop_path_rate * 0.5}."
            )
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks_spectral_gating = nn.ModuleList()
        self.blocks_attention = nn.ModuleList()
        for i in range(depth):
            if i < n_spectral_blocks:
                layer = BlockSpectralGating(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    h=h,
                    w=w,
                )
                self.blocks_spectral_gating.append(layer)
            else:
                layer = BlockAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    w=window_size,
                    dp_rank=dp_rank,
                    rpe=rpe,
                    adaLN=True if ensemble is not None else False,
                    nglo=nglo,
                )
                self.blocks_attention.append(layer)

        self.apply(self._init_weights)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor of shape B, N, C for deterministic of BxE, N, C for ensemble forecast.
        Returns:
            Tensor of same shape as input.
        """
        if self.ensemble:
            BE, N, C = tokens.shape
            noise = torch.randn(
                size=(BE, N, C), dtype=tokens.dtype, device=tokens.device
            )
        else:
            noise = None

        for i, blk in enumerate(
            chain(self.blocks_spectral_gating, self.blocks_attention)
        ):
            if i in self._checkpoint_layers:
                tokens = checkpoint(blk, tokens, noise, use_reentrant=False)
            else:
                tokens = blk(tokens, noise)

        return tokens

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
