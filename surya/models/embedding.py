"""
Perceiver code is based on Aurora: https://github.com/microsoft/aurora/blob/main/aurora/model/perceiver.py

Some conventions for notation:
B - Batch
T - Time
H - Height (pixel space)
W - Width (pixel space)
HT - Height (token space)
WT - Width (token space)
ST - Sequence (token space)
C - Input channels
D - Model (embedding) dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_


class PatchEmbed3D(nn.Module):
    """Timeseries Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, time_dim=2
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.time_dim = time_dim

        self.proj = nn.Conv2d(
            in_chans * time_dim,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, T, H, W)
        Returns:
            Tensor of shape (B, ST, D)
        """
        B, C, T, H, W = x.shape
        x = self.proj(x.flatten(1, 2))  # (B, C, T, H, W) -> (B, D, HT, WT)
        x = rearrange(x, "B D HT WT -> B (HT WT) D")  # (B, N, D)
        return x


class LinearEmbedding(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        time_dim=2,
        embed_dim=768,
        drop_rate=0.0,
    ):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_dim=time_dim,
        )

        self._generate_position_encoding(img_size, patch_size, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def _generate_position_encoding(self, img_size, patch_size, embed_dim):
        """
        Generates a positional encoding signal for the model. The generated
        positional encoding signal is stored as a buffer (`self.fourier_signal`).

        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of each patch in the image.
            embed_dim (int): The embedding dimension of the model.

        Returns:
            None.
        """
        # Generate signal of shape (C, H, W)
        x = torch.linspace(0.0, 1.0, img_size // patch_size)
        y = torch.linspace(0.0, 1.0, img_size // patch_size)
        x, y = torch.meshgrid(x, y, indexing="xy")
        fourier_signal = []

        frequencies = torch.linspace(1, (img_size // patch_size) / 2.0, embed_dim // 4)

        for f in frequencies:
            fourier_signal.extend(
                [
                    torch.cos(2.0 * torch.pi * f * x),
                    torch.sin(2.0 * torch.pi * f * x),
                    torch.cos(2.0 * torch.pi * f * y),
                    torch.sin(2.0 * torch.pi * f * y),
                ]
            )
        fourier_signal = torch.stack(fourier_signal, dim=2)
        fourier_signal = rearrange(fourier_signal, "h w c -> 1 (h w) c")
        self.register_buffer("pos_embed", fourier_signal)

    def forward(self, x, dt):
        """
        Args:
            x: Tensor of shape (B, C, T, H, W).
            dt: Tensor of shape (B, T). However it is not used.
        Returns:
            Tensor of shape (B, ST, D)
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        return x


class LinearDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        out_chans: int,
        embed_dim: int,
    ):
        """
        Args:
            patch_size: patch size
            in_chans: number of iput channels
            embed_dim: embedding dimension
        """
        super().__init__()

        self.unembed = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=(patch_size**2) * out_chans,
                kernel_size=1,
            ),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, D). For ensembles, we have implicitly B = (B E).
        Returns:
            Tensor of shape (B C H W).
            Here
            - C equals num_queries
            - H == W == sqrt(L) x patch_size
        """
        # Reshape the tokens to 2d token space: (B, C, H_token, W_token)
        _, L, _ = x.shape
        H_token = W_token = int(L**0.5)
        x = rearrange(x, "B (H W) D -> B D H W", H=H_token, W=W_token)

        # Unembed the tokens. Convolution + pixel shuffle.
        x = self.unembed(x)

        return x


class MLP(nn.Module):
    """A simple one-hidden-layer MLP."""

    def __init__(self, dim: int, hidden_features: int, dropout: float = 0.0) -> None:
        """Initialise.

        Args:
            dim (int): Input dimensionality.
            hidden_features (int): Width of the hidden layer.
            dropout (float, optional): Drop-out rate. Defaults to no drop-out.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""
        return self.net(x)


class PerceiverAttention(nn.Module):
    """Cross attention module from the Perceiver architecture."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
    ) -> None:
        """Initialise.

        Args:
            latent_dim (int): Dimensionality of the latent features given as input.
            context_dim (int): Dimensionality of the context features also given as input.
            head_dim (int): Attention head dimensionality.
            num_heads (int): Number of heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.to_q = nn.Linear(latent_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, latent_dim, bias=False)

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Run the cross-attention module.

        Args:
            latents (:class:`torch.Tensor`): Latent features of shape `(B, L1, Latent_D)`
                where typically `L1 < L2` and `Latent_D <= Context_D`. `Latent_D` is equal to
                `self.latent_dim`.
            x (:class:`torch.Tensor`): Context features of shape `(B, L2, Context_D)`.

        Returns:
            :class:`torch.Tensor`: Latent values of shape `(B, L1, Latent_D)`.
        """
        h = self.num_heads

        q = self.to_q(latents)  # (B, L1, D2) to (B, L1, D)
        k, v = self.to_kv(x).chunk(2, dim=-1)  # (B, L2, D1) to twice (B, L2, D)
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L1 D -> B L1 (H D)")  # (B, L1, D)
        return self.to_out(out)  # (B, L1, Latent_D)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler module from the Flamingo paper."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        residual_latent: bool = True,
        ln_eps: float = 1e-5,
    ) -> None:
        """Initialise.

        Args:
            latent_dim (int): Dimensionality of the latent features given as input.
            context_dim (int): Dimensionality of the context features also given as input.
            depth (int, optional): Number of attention layers.
            head_dim (int, optional): Attention head dimensionality. Defaults to `64`.
            num_heads (int, optional): Number of heads. Defaults to `16`
            mlp_ratio (float, optional): Rimensionality of the hidden layer divided by that of the
                input for all MLPs. Defaults to `4.0`.
            drop (float, optional): Drop-out rate. Defaults to no drop-out.
            residual_latent (bool, optional): Use residual attention w.r.t. the latent features.
                Defaults to `True`.
            ln_eps (float, optional): Epsilon in the layer normalisation layers. Defaults to
                `1e-5`.
        """
        super().__init__()

        self.residual_latent = residual_latent
        self.layers = nn.ModuleList([])
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            latent_dim=latent_dim,
                            context_dim=context_dim,
                            head_dim=head_dim,
                            num_heads=num_heads,
                        ),
                        MLP(
                            dim=latent_dim, hidden_features=mlp_hidden_dim, dropout=drop
                        ),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                    ]
                )
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Run the module.

        Args:
            latents (:class:`torch.Tensor`): Latent features of shape `(B, L1, D1)`.
            x (:class:`torch.Tensor`): Context features of shape `(B, L2, D1)`.

        Returns:
            torch.Tensor: Latent features of shape `(B, L1, D1)`.
        """
        for attn, ff, ln1, ln2 in self.layers:
            # We use post-res-norm like in Swin v2 and most Transformer architectures these days.
            # This empirically works better than the pre-norm used in the original Perceiver.
            attn_out = ln1(attn(latents, x))
            # HuggingFace suggests using non-residual attention in Perceiver might work better when
            # the semantics of the query and the output are different:
            #
            #   https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/perceiver/modeling_perceiver.py#L398
            #
            latents = attn_out + latents if self.residual_latent else attn_out
            latents = ln2(ff(latents)) + latents
        return latents


class PerceiverChannelEmbedding(nn.Module):
    def __init__(
        self,
        in_chans: int,
        img_size: int,
        patch_size: int,
        time_dim: int,
        num_queries: int,
        embed_dim: int,
        drop_rate: float,
    ):
        super().__init__()

        if embed_dim % 2 != 0:
            raise ValueError(
                f"Temporal embeddings require `embed_dim` to be even. Currently we have {embed_dim}."
            )

        self.num_patches = (img_size // patch_size) ** 2
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels=in_chans * time_dim,
            out_channels=in_chans * embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            groups=in_chans,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.num_patches))
        trunc_normal_(self.pos_embed, std=0.02)

        self.latent_queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        trunc_normal_(self.latent_queries, std=0.02)

        self.perceiver = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=1,
            head_dim=embed_dim // 16,
            num_heads=16,
            mlp_ratio=4.0,
            drop=0.0,
            residual_latent=False,
            ln_eps=1e-5,
        )

        self.latent_aggregation = nn.Linear(num_queries * embed_dim, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x, dt):
        """
        Args:
            x: Tensor of shape (B, C, T, H, W)
            dt: Tensor of shape (B, T) identifying time deltas.
        Returns:
            Tensor of shape (B, ST, D)
        """
        B, C, T, H, W = x.shape
        x = rearrange(x, "B C T H W -> B (C T) H W")
        x = self.proj(x)  # B (C T) H W -> B (C D) HT WT
        x = x.flatten(2, 3)  # B (C D) ST
        ST = x.shape[2]
        assert ST == self.num_patches
        x = rearrange(x, "B (C D) ST -> (B C) D ST", B=B, ST=ST, C=C, D=self.embed_dim)
        x = x + self.pos_embed
        x = rearrange(x, "(B C) D ST -> (B ST) C D", B=B, ST=ST, C=C, D=self.embed_dim)

        # ((B ST) NQ D), ((B ST) C D) -> ((B ST) NQ D)
        x = self.perceiver(self.latent_queries.expand(B * ST, -1, -1), x)
        x = rearrange(
            x,
            "(B ST) NQ D -> B ST (NQ D)",
            B=B,
            ST=self.num_patches,
            NQ=self.num_queries,
            D=self.embed_dim,
        )
        x = self.latent_aggregation(x)  # B ST (NQ D) -> B ST D'

        assert x.shape[1] == self.num_patches
        assert x.shape[2] == self.embed_dim

        x = self.pos_drop(x)

        return x


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        out_chans: int,
    ):
        """
        Args:
            embed_dim: embedding dimension
            patch_size: patch size
            out_chans: number of output channels. This determines the number of latent queries.
            drop_rate: dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_chans = out_chans

        self.latent_queries = nn.Parameter(torch.zeros(1, out_chans, embed_dim))
        trunc_normal_(self.latent_queries, std=0.02)

        self.perceiver = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=1,
            head_dim=embed_dim // 16,
            num_heads=16,
            mlp_ratio=4.0,
            drop=0.0,
            residual_latent=False,
            ln_eps=1e-5,
        )
        self.proj = nn.Conv2d(
            in_channels=out_chans * embed_dim,
            out_channels=out_chans * patch_size**2,
            kernel_size=1,
            padding=0,
            groups=out_chans,
        )
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, D) For ensembles, we have implicitly B = (B E).
        Returns:
            Tensor of shape (B C H W).
            Here
            - C equals out_chans
            - H == W == sqrt(L) x patch_size
        """
        B, L, D = x.shape
        H_token = W_token = int(L**0.5)

        x = rearrange(x, "B L D -> (B L) 1 D")
        # (B L) 1 D -> (B L) C D
        x = self.perceiver(self.latent_queries.expand(B * L, -1, -1), x)
        x = rearrange(x, "(B H W) C D -> B (C D) H W", H=H_token, W=W_token)
        # B (C D) H_token W_token -> B (C patch_size patch_size) H_token W_token
        x = self.proj(x)
        # B (C patch_size patch_size) H_token W_token -> B C H W
        x = self.pixel_shuffle(x)

        return x
