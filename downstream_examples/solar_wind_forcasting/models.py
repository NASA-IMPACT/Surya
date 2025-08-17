import torch
from torch import nn
import torchvision.models as models
from itertools import chain

from torch.utils.checkpoint import checkpoint

from surya.models.helio_spectformer import HelioSpectFormer

from surya.models.embedding import (
    LinearDecoder,
    PerceiverDecoder,
)


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
        checkpoint_layers: list[int] | None = None,
        rpe: bool = False,
        ensemble: int | None = None,
        finetune: bool = False,
        nglo: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        # Put finetuning additions below this line
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
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            ensemble=ensemble,
            finetune=finetune,
            dtype=dtype,
            nglo=nglo,
        )

        self.pooling_strategies = [
            config["model"]["global_average_pooling"],
            config["model"]["global_max_pooling"],
            config["model"]["attention_pooling"],
            config["model"]["transformer_pooling"],
            config["model"]["global_class_token"],
        ]

        assert (
            sum(self.pooling_strategies) == 1
        ), "No or multiple pooling strategy selected. Aborting."

        self.global_average_pooling = False
        self.global_max_pooling = False
        self.attention_pooling = False
        self.transformer_pooling = False
        self.penultimate_linear_layer = False
        self.global_class_token = False

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

        elif config["model"]["global_class_token"]:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.global_class_token = True

        else:
            raise Exception("No valid pooling strategy selected.")

        if config["model"]["penultimate_linear_layer"]:
            self.linear = nn.Linear(embed_dim, embed_dim)
            self.penultimate_linear_layer = True

        self.unembed = nn.Linear(embed_dim, num_outputs)

    def _forward_cls_token(self, batch):
        x = batch["ts"]
        dt = batch["time_delta_input"]
        B, C, T, H, W = x.shape

        if self.learned_flow:
            y_hat_flow = self.learned_flow_model(batch)  # B, C, H, W
            if any([param.requires_grad for param in self.learned_flow_model.parameters()]):
                return y_hat_flow
            else:
                x = torch.concat((x, y_hat_flow.unsqueeze(2)), dim=2)  # B, C, T+1, H, W
                if self.time_embedding["type"] == "perceiver":
                    dt = torch.cat((dt, batch["lead_time_delta"].reshape(-1, 1)), dim=1)

        # embed the data
        tokens = self.embedding(x, dt)

        if self.ensemble:
            raise NotImplementedError(
                "Use of CLS token has not been implemented with ensemble modifications."
            )
        else:
            noise = None

        # pass the time series through the encoder
        for i, blk in enumerate(
            chain(self.backbone.blocks_spectral_gating, self.backbone.blocks_attention)
        ):
            if i == self.backbone.n_spectral_blocks:
                tokens = torch.cat(
                    (
                        self.cls_token.expand(B, 1, self.embed_dim),
                        tokens,
                    ),
                    dim=1,
                )
            if i in self.backbone._checkpoint_layers:
                tokens = checkpoint(blk, tokens, noise, use_reentrant=False)
            else:
                tokens = blk(tokens, noise)
        tokens = tokens[:, [0], :]

        return tokens

    def forward(self, batch):
        # batch['ts'] = batch['ts'].unsqueeze(2)
        if self.global_class_token:
            tokens = self._forward_cls_token(batch)
        else:
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

        if self.global_class_token:
            agg_tokens = torch.squeeze(tokens, dim=1)

        if self.dropout is not None:
            out = self.dropout_layer(agg_tokens)

        out = self.unembed(agg_tokens)
        # out = torch.sigmoid(out)
        # out = out.squeeze(dim=1)

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

    def forward(self, batch):

        tokens = super().forward(batch=batch)

        # Unembed the tokens
        # BE L D -> BE C H W
        forecast_hat = self.unembed(tokens)

        return forecast_hat


class ResNet18Classifier(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(ResNet18Classifier, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.resnet.conv1 = nn.Conv2d(
            merged_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        # x['ts'] = ['ts'].unsqueeze(2)
        x = x["ts"]
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through ResNet
        features = self.resnet(x_merged)  # B, 512, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 512

        # Classification
        features = self.dropout(features)
        output = self.classifier(features)
        # output = torch.sigmoid(output)

        return output.squeeze(-1)


class ResNet34Classifier(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(ResNet34Classifier, self).__init__()
        # Load pretrained ResNet34
        self.resnet = models.resnet34(pretrained=True)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.resnet.conv1 = nn.Conv2d(
            merged_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        # x['ts'] = x['ts'].unsqueeze(2)
        x = x["ts"]
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through ResNet
        features = self.resnet(x_merged)  # B, 512, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 512

        # Classification
        features = self.dropout(features)
        output = self.classifier(features)
        # output = torch.sigmoid(output)

        return output.squeeze(-1)


class ResNet50Classifier(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(ResNet50Classifier, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.resnet.conv1 = nn.Conv2d(
            merged_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        # x['ts'] = x['ts'].unsqueeze(2)
        x = x["ts"]
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through ResNet
        features = self.resnet(x_merged)  # B, 2048, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 2048

        # Classification
        features = self.dropout(features)
        output = self.classifier(features)
        # output = torch.sigmoid(output)

        return output.squeeze(-1)


class AlexNetClassifier(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(AlexNetClassifier, self).__init__()
        # Load pretrained AlexNet
        self.alexnet = models.alexnet(pretrained=True)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.alexnet.features[0] = nn.Conv2d(
            merged_channels, 64, kernel_size=11, stride=4, padding=2
        )

        # Remove the final classification layer
        self.alexnet = nn.Sequential(*list(self.alexnet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        # Use adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(9216, num_classes)  # 6*6*256 = 9216

    def forward(self, x):
        # Input: B, C, T, H, W
        # x['ts'] = x['ts'].unsqueeze(2)
        x = x["ts"]
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through AlexNet features
        features = self.alexnet(x_merged)  # B, 256, H', W'

        # Adaptive pooling to ensure consistent size
        features = self.adaptive_pool(features)  # B, 256, 6, 6

        # Flatten
        features = features.view(B, -1)  # B, 9216

        # Classification
        features = self.dropout(features)
        output = self.classifier(features)
        # output = torch.sigmoid(output)

        return output.squeeze(-1)


class MobileNetClassifier(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(MobileNetClassifier, self).__init__()
        # Load pretrained MobileNet
        self.mobilenet = models.mobilenet_v2(pretrained=True)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.mobilenet.features[0][0] = nn.Conv2d(
            merged_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Remove the final classification layer
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        # x['ts'] = x['ts'].unsqueeze(2)
        x = x["ts"]
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through MobileNet
        features = self.mobilenet(x_merged)  # B, 1280, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 1280

        # Classification
        features = self.dropout(features)
        output = self.classifier(features)
        # output = torch.sigmoid(output)

        return output.squeeze(-1)
