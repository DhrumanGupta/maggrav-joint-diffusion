import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm_groups(channels: int, max_groups: int = 32) -> int:
    """Find largest divisor of channels up to max_groups."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResConvBlock3D(nn.Module):
    """Conv3D + GroupNorm + SiLU with residual connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = _group_norm_groups(out_ch)

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)

        # Shortcut for channel mismatch
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.silu(out + identity)


class ResConvStack3D(nn.Module):
    """Stack of residual conv blocks."""

    def __init__(self, in_ch: int, out_ch: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        layers = [ResConvBlock3D(in_ch, out_ch)]
        for _ in range(num_layers - 1):
            layers.append(ResConvBlock3D(out_ch, out_ch))
        self.stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class DownBlockRes3D(nn.Module):
    """Residual conv stack + stride-2 downsample."""

    def __init__(self, in_ch: int, out_ch: int, num_layers: int):
        super().__init__()
        self.block = ResConvStack3D(in_ch, out_ch, num_layers)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.down(x)


class UpBlockTrilinear3D(nn.Module):
    """Trilinear interpolation to target size + residual conv stack."""

    def __init__(self, in_ch: int, out_ch: int, target_size: int, num_layers: int):
        super().__init__()
        self.target_size = target_size
        self.block = ResConvStack3D(in_ch, out_ch, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(self.target_size, self.target_size, self.target_size),
            mode="trilinear",
            align_corners=False,
        )
        return self.block(x)


class NonLocalBlock3D(nn.Module):
    """Self-attention lite: 1x1 Q/K/V projections with learnable residual weight."""

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        inner_ch = max(channels // reduction, 1)
        self.query = nn.Conv3d(channels, inner_ch, kernel_size=1)
        self.key = nn.Conv3d(channels, inner_ch, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = inner_ch**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W

        q = self.query(x).view(B, -1, N)  # B, C', N
        k = self.key(x).view(B, -1, N)  # B, C', N
        v = self.value(x).view(B, C, N)  # B, C, N

        # Attention: softmax(Q^T K / sqrt(d)) @ V^T
        attn = torch.softmax(
            torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1
        )  # B, N, N
        out = torch.bmm(v, attn.transpose(1, 2))  # B, C, N
        out = out.view(B, C, D, H, W)

        return x + self.gamma * out


@dataclass
class VAE3DAttentionConfig:
    in_channels: int = 2
    base_channels: int = 32  # Conservative at 200³
    latent_channels: int = 96
    bottleneck_channels: int = 192  # Capped at 25³
    latent_size: int = 40  # Output latent spatial size
    blocks_per_stage: int = 3
    num_attention_blocks: int = 2


class VAE3DAttention(nn.Module):
    """
    VAE with attention at bottleneck.

    Encoder: 200 → 100 → 50 → 25 (attention) → 40 (μ/logvar)
    Decoder: 40 → 50 → 100 → 200

    Channel progression: 32 → 64 → 128 → 192 (capped)
    """

    def __init__(self, cfg: VAE3DAttentionConfig):
        super().__init__()
        self.cfg = cfg
        base = cfg.base_channels

        # Channel progression: 32 → 64 → 128 → 192
        ch_200 = base  # 32
        ch_100 = base * 2  # 64
        ch_50 = base * 4  # 128
        ch_25 = cfg.bottleneck_channels  # 192 (capped)

        # ========== ENCODER ==========
        # Initial conv at 200³
        self.enc_in = ResConvStack3D(cfg.in_channels, ch_200, cfg.blocks_per_stage)

        # Downsample stages
        self.enc_down1 = DownBlockRes3D(ch_200, ch_100, cfg.blocks_per_stage)  # 200→100
        self.enc_down2 = DownBlockRes3D(ch_100, ch_50, cfg.blocks_per_stage)  # 100→50
        self.enc_down3 = DownBlockRes3D(ch_50, ch_25, cfg.blocks_per_stage)  # 50→25

        # Mid processing at 25³ with attention
        self.enc_mid = ResConvStack3D(ch_25, ch_25, cfg.blocks_per_stage)
        self.attention_blocks = nn.ModuleList(
            [NonLocalBlock3D(ch_25) for _ in range(cfg.num_attention_blocks)]
        )

        # 25 → 40 upsample before latent head
        self.pre_latent_up = nn.Sequential(
            nn.Upsample(size=(cfg.latent_size,) * 3, mode="trilinear", align_corners=False),
            ResConvStack3D(ch_25, ch_25, 1),  # Light refinement
        )

        # Latent projections at 40³
        self.to_latent = nn.Conv3d(ch_25, cfg.latent_channels, kernel_size=1)
        self.to_mu = nn.Conv3d(cfg.latent_channels, cfg.latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv3d(cfg.latent_channels, cfg.latent_channels, kernel_size=1)

        # ========== DECODER ==========
        self.from_latent = nn.Conv3d(cfg.latent_channels, ch_25, kernel_size=1)

        # Upsample stages: 40 → 50 → 100 → 200
        self.dec_up1 = UpBlockTrilinear3D(ch_25, ch_50, 50, cfg.blocks_per_stage)
        self.dec_up2 = UpBlockTrilinear3D(ch_50, ch_100, 100, cfg.blocks_per_stage)
        self.dec_up3 = UpBlockTrilinear3D(ch_100, ch_200, 200, cfg.blocks_per_stage)

        # Output
        self.dec_mid = ResConvStack3D(ch_200, ch_200, cfg.blocks_per_stage)
        self.out_conv = nn.Conv3d(ch_200, cfg.in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 200³ input
        h = self.enc_in(x)

        # 200 → 100 → 50 → 25
        h = self.enc_down1(h)
        h = self.enc_down2(h)
        h = self.enc_down3(h)

        # Mid + attention at 25³
        h = self.enc_mid(h)
        for attn in self.attention_blocks:
            h = attn(h)

        # 25 → 40
        h = self.pre_latent_up(h)

        # Project to latent
        h = self.to_latent(h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # From latent (40³)
        h = self.from_latent(z)

        # 40 → 50 → 100 → 200
        h = self.dec_up1(h)
        h = self.dec_up2(h)
        h = self.dec_up3(h)

        # Output
        h = self.dec_mid(h)
        return self.out_conv(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
