from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsample by 2x using strided convolution, then double conv block."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv3d(in_ch, in_ch, kernel_size=2, stride=2, bias=False)
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # Upsample to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PhaseUNet3D(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()
        C = base_channels

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, C)       # → C
        self.enc2 = DownBlock(C, C * 2)                # → 2C
        self.enc3 = DownBlock(C * 2, C * 4)            # → 4C

        # Bottleneck
        self.bottleneck = DownBlock(C * 4, C * 8)      # → 8C

        # Decoder (with skip connections)
        self.dec3 = UpBlock(C * 8, C * 4, C * 4)       # 8C + 4C → 4C
        self.dec2 = UpBlock(C * 4, C * 2, C * 2)       # 4C + 2C → 2C
        self.dec1 = UpBlock(C * 2, C, C)                # 2C + C  → C

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv3d(C, C // 2, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(C // 2, out_channels, kernel_size=1),
            nn.Tanh(),  # Phase ∈ [-1, 1] (multiply by π for radians)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encoder
        s1 = self.enc1(x)             # [B, C,  N,   N,   N]
        s2 = self.enc2(s1)            # [B, 2C, N/2, N/2, N/2]
        s3 = self.enc3(s2)            # [B, 4C, N/4, N/4, N/4]

        # Bottleneck
        b = self.bottleneck(s3)       # [B, 8C, N/8, N/8, N/8]

        # Decoder with skip connections
        d3 = self.dec3(b, s3)         # [B, 4C, N/4, N/4, N/4]
        d2 = self.dec2(d3, s2)        # [B, 2C, N/2, N/2, N/2]
        d1 = self.dec1(d2, s1)        # [B, C,  N,   N,   N]

        # Output
        return self.out_conv(d1)      # [B, 1,  N,   N,   N]


# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

class BCDIPhaseLoss(nn.Module):


    def __init__(self,
                 alpha_amp: float = None, beta_phase: float = None,
                 gamma_diff: float = None,
                 alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.01):
        super().__init__()
        # The GUI's descriptive names take precedence; fall back to the
        # Greek-letter ones (kept for backward compatibility with old scripts)
        self.alpha = alpha_amp if alpha_amp is not None else alpha
        self.beta  = beta_phase if beta_phase is not None else beta
        self.gamma = gamma_diff if gamma_diff is not None else gamma

    def forward(
        self,
        phase_pred: torch.Tensor,
        phase_true: torch.Tensor,
        support: torch.Tensor,
        amplitude: torch.Tensor = None,
        diff_amp: torch.Tensor = None,
    ) -> dict:

        # Accept either keyword for the amplitude tensor
        if amplitude is None and diff_amp is not None:
            amplitude = diff_amp
        # ── 1. Masked MSE loss ────────────────────────────────────────────
        n_support = support.sum().clamp(min=1)
        mse = ((phase_pred - phase_true) ** 2 * support).sum() / n_support

        result = {'mse': mse}
        total = self.alpha * mse

        # ── 2. FFT consistency loss ───────────────────────────────────────
        if amplitude is not None and self.beta > 0:
            # Construct complex object: support * exp(i * phase_pred * π)
            phase_rad = phase_pred * torch.pi
            obj = support * torch.exp(1j * phase_rad.squeeze(1))

            # Forward FFT
            obj_fft = torch.fft.fftshift(
                torch.fft.fftn(torch.fft.ifftshift(obj, dim=(-3, -2, -1)),
                               dim=(-3, -2, -1)),
                dim=(-3, -2, -1)
            )

            # The amplitude of the FFT should match the measured amplitude
            pred_amp = torch.abs(obj_fft)
            # Normalize both to compare shapes rather than absolute values
            pred_amp_n = pred_amp / (pred_amp.max() + 1e-8)
            true_amp_n = amplitude.squeeze(1) / (amplitude.squeeze(1).max() + 1e-8)

            fft_loss = F.mse_loss(pred_amp_n, true_amp_n)
            result['fft_consistency'] = fft_loss
            total = total + self.beta * fft_loss

        # ── 3. Gradient smoothness ────────────────────────────────────────
        if self.gamma > 0:
            # Finite differences along each axis (inside support)
            dx = (phase_pred[:, :, 1:, :, :] - phase_pred[:, :, :-1, :, :]) ** 2
            dy = (phase_pred[:, :, :, 1:, :] - phase_pred[:, :, :, :-1, :]) ** 2
            dz = (phase_pred[:, :, :, :, 1:] - phase_pred[:, :, :, :, :-1]) ** 2

            # Mask with support (intersection of adjacent voxels)
            sx = support[:, :, 1:, :, :] * support[:, :, :-1, :, :]
            sy = support[:, :, :, 1:, :] * support[:, :, :, :-1, :]
            sz = support[:, :, :, :, 1:] * support[:, :, :, :, :-1]

            smooth = (
                (dx * sx).sum() / sx.sum().clamp(min=1) +
                (dy * sy).sum() / sy.sum().clamp(min=1) +
                (dz * sz).sum() / sz.sum().clamp(min=1)
            ) / 3.0

            result['smoothness'] = smooth
            total = total + self.gamma * smooth

        result['total'] = total
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, grid_size: int = 64):
    """Print model architecture summary."""
    n_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"PhaseUNet3D Summary")
    print(f"{'='*50}")
    print(f"  Parameters:     {n_params:,}")
    print(f"  Grid size:      {grid_size}³")

    # Estimate memory
    param_mb = n_params * 4 / 1e6  # float32
    # Activation memory (rough: ~4x input per level, 4 levels)
    act_mb = 4 * grid_size**3 * 4 * (1 + 2 + 4 + 8) / 1e6
    print(f"  Param memory:   {param_mb:.1f} MB")
    print(f"  Act. memory:    ~{act_mb:.0f} MB (per sample)")
    print(f"  Recommended:    batch_size=8 for ≥8GB GPU")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    # Test model creation and forward pass
    model = PhaseUNet3D(in_channels=1, base_channels=32)
    model_summary(model, grid_size=64)

    # Test forward pass
    x = torch.randn(2, 1, 64, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}  range [{x.min():.2f}, {x.max():.2f}]")
    print(f"Output: {y.shape}  range [{y.min():.2f}, {y.max():.2f}]")

    # Test loss
    loss_fn = BCDIPhaseLoss(alpha=1.0, beta=0.1, gamma=0.01)
    phase_true = torch.randn(2, 1, 64, 64, 64).clamp(-1, 1)
    support = (torch.randn(2, 1, 64, 64, 64) > 0).float()
    amplitude = torch.rand(2, 1, 64, 64, 64)

    losses = loss_fn(y, phase_true, support, amplitude)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
