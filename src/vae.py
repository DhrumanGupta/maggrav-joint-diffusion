import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_downsample_factor(downsample_factor: int) -> int:
    if downsample_factor not in (4, 8, 16):
        raise ValueError("downsample_factor must be 4, 8, or 16.")
    return int(math.log2(downsample_factor))


def _group_norm_groups(channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = _group_norm_groups(out_ch)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class ConvStack3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        layers = [ConvBlock3D(in_ch, out_ch)]
        for _ in range(num_layers - 1):
            layers.append(ConvBlock3D(out_ch, out_ch))
        self.stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int):
        super().__init__()
        self.block = ConvStack3D(in_ch, out_ch, num_layers)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.down(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int):
        super().__init__()
        self.block = ConvStack3D(in_ch, out_ch, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.block(x)


@dataclass
class VAE3DConfig:
    in_channels: int = 2
    base_channels: int = 48
    latent_channels: int = 96
    downsample_factor: int = 8
    blocks_per_stage: int = 3


class VAE3D(nn.Module):
    def __init__(self, cfg: VAE3DConfig):
        super().__init__()
        self.cfg = cfg
        num_down = _validate_downsample_factor(cfg.downsample_factor)

        self.enc_in = ConvStack3D(
            cfg.in_channels, cfg.base_channels, cfg.blocks_per_stage
        )

        enc_blocks = []
        ch = cfg.base_channels
        for i in range(num_down):
            out_ch = cfg.base_channels * (2 ** (i + 1))
            enc_blocks.append(DownBlock3D(ch, out_ch, cfg.blocks_per_stage))
            ch = out_ch
        self.enc_blocks = nn.ModuleList(enc_blocks)

        self.enc_mid = ConvStack3D(ch, ch, cfg.blocks_per_stage)
        self.to_latent = nn.Conv3d(ch, cfg.latent_channels, kernel_size=1)
        self.to_mu = nn.Conv3d(cfg.latent_channels, cfg.latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv3d(
            cfg.latent_channels, cfg.latent_channels, kernel_size=1
        )

        self.from_latent = nn.Conv3d(cfg.latent_channels, ch, kernel_size=1)

        dec_blocks = []
        for i in reversed(range(num_down)):
            out_ch = cfg.base_channels * (2**i)
            dec_blocks.append(UpBlock3D(ch, out_ch, cfg.blocks_per_stage))
            ch = out_ch
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.dec_mid = ConvStack3D(ch, ch, cfg.blocks_per_stage)
        self.out_conv = nn.Conv3d(ch, cfg.in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_in(x)
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_mid(h)
        h = self.to_latent(h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        for block in self.dec_blocks:
            h = block(h)
        h = self.dec_mid(h)
        return self.out_conv(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
