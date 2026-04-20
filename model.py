import torch

from torch.nn import Module, ModuleList, LeakyReLU, Sigmoid, BatchNorm2d, PixelShuffle
from torch.nn import Conv2d

from constants import PRELIMINARY_NETWORK_DEPTH, REFINEMENT_NETWORK_DEPTH
from utilities import reverseTransmissionMap, applyMapBasedAttention

IN_CHANNEL_NUM = 9
OUT_CHANNEL_NUM = 3

# ── Residual double-conv block ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

# ── Single-stream encoder (returns features + skips) ──────────────────────
class StreamEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc1 = ResBlock(in_ch, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        self.enc4 = ResBlock(256, 512)
        self.down = nn.Conv2d(1, 1, 4, stride=2, padding=1)  # channel-wise strided conv placeholder
    def _down(self, x):
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        x = F.avg_pool2d(x, 2)      # simple spatial halving per channel
        return x.view(B, C, H // 2, W // 2)
    def forward(self, x):
        s1 = self.enc1(x)           # (B, 64,  H,   W)
        s2 = self.enc2(self._down(s1))  # (B, 128, H/2, W/2)
        s3 = self.enc3(self._down(s2))  # (B, 256, H/4, W/4)
        s4 = self.enc4(self._down(s3))  # (B, 512, H/8, W/8)
        return s4, [s1, s2, s3]     # bottleneck + skips

# ── Channel attention (SE block) ──────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).view(x.shape[0], x.shape[1], 1, 1)

# ── Decoder ───────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # up3: 512 → 256, then concat lf_skip[2](256) + hf_skip[2](256) = 256+256+256 = 768
        self.up3  = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dec3 = ResBlock(256 + 256 * 2, 256)   # 768 → 256

        # up2: 256 → 128, then concat lf_skip[1](128) + hf_skip[1](128) = 128+128+128 = 384
        self.up2  = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = ResBlock(128 + 128 * 2, 128)   # 384 → 128

        # up1: 128 → 64, then concat lf_skip[0](64) + hf_skip[0](64) = 64+64+64 = 192
        self.up1  = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = ResBlock(64 + 64 * 2, 64)      # 192 → 64

        self.out  = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Sigmoid())

    def forward(self, bottleneck, lf_skips, hf_skips):
        # lf_skips / hf_skips = [s1(64), s2(128), s3(256)]
        x = self.up3(bottleneck)
        x = torch.cat([x, lf_skips[2], hf_skips[2]], dim=1)  # 256+256+256 = 768
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, lf_skips[1], hf_skips[1]], dim=1)  # 128+128+128 = 384
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, lf_skips[0], hf_skips[0]], dim=1)  # 64+64+64 = 192
        x = self.dec1(x)

        return self.out(x)

# ── Physics-guided bottleneck ─────────────────────────────────────────────
class PhysicsBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_proj = nn.Conv2d(1, 512, 1)          # project transmission map into channel space
        self.ca     = ChannelAttention(512)
        self.res    = ResBlock(512, 512)
    def forward(self, x, transmission_map):
        t_feat = self.t_proj(F.interpolate(transmission_map, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = x + t_feat
        x = self.ca(x)
        return self.res(x)

# ── Top-level network (replaces ImageEnhancementNetwork) ─────────────────
class ImageEnhancementNetwork(nn.Module):
    def __init__(self, **kwargs):   # **kwargs for drop-in compatibility
        super().__init__()
        self.lf_encoder = StreamEncoder(IN_CHANNEL_NUM)
        self.hf_encoder = StreamEncoder(IN_CHANNEL_NUM)
        self.bottleneck = PhysicsBottleneck()
        self.decoder    = Decoder()
    def forward(self, lf, hf):
        # Physics prior from raw RGB of LF stream
        t_map, _ = DCPTransmission(lf[:, :3, :, :])
        lf_bot, lf_skips = self.lf_encoder(lf)
        hf_bot, hf_skips = self.hf_encoder(hf)
        bottleneck = self.bottleneck(lf_bot + hf_bot, t_map)
        output = self.decoder(bottleneck, lf_skips, hf_skips)
        # Return same 3-tuple as original for CompositeLoss compatibility
        lf_out = output   # simplified: use output as proxy for sub-outputs
        hf_out = output
        return output, lf_out, hf_out


if __name__ == "__main__":
    lf = torch.randn(1, 9, 256, 256)
    hf = torch.randn(1, 9, 256, 256)
    model = ImageEnhancementNetwork()
    y = model(lf, hf)
    print(y.shape)