"""
nn_autophase_model.py — Physics-aware dual-decoder network for BCDI.

Merges AutoPhaseNN (Yao et al., npj Comp Mat 2022) architecture with your
existing U-Net backbone. Key differences from nn_phase_model.py:

    1. TWO decoder heads — one for amplitude, one for phase
       (previously: one head for phase only)

    2. Physics-enforcing activations:
       - amplitude: sigmoid → [0, 1]
       - phase: tanh × π → [-π, π]

    3. Zero-padding support layer:
       The amplitude is padded to N/2 in each dimension, enforcing the
       oversampling requirement for phase retrieval (Miao 2000).
       This means the reconstructed object is automatically confined
       to half the grid in each direction.

    4. Shape support constraint:
       During the forward model, amplitude is thresholded at 10% max
       to create a dynamic support mask.

Trained with the physics-aware unsupervised loss in nn_autophase_train.py:
no ground-truth phase or amplitude needed, only diffraction intensity.

Reference:
    Yao et al., "AutoPhaseNN: unsupervised physics-aware deep learning of 3D
    nanoscale Bragg coherent diffraction imaging", npj Comp. Mat. 8, 124 (2022).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Building blocks (compatible with your nn_phase_model.py conventions)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock3D(nn.Module):
    """Two 3×3×3 convolutions with InstanceNorm + LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsample 2× via strided conv, then ConvBlock3D."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv3d(in_ch, in_ch, 2, stride=2, bias=False)
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """Upsample + skip concatenation + ConvBlock3D."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder heads (one per output: amplitude and phase)
# ═══════════════════════════════════════════════════════════════════════════════

class DecoderBranch(nn.Module):
    """
    One decoder branch: takes the bottleneck + skip connections,
    produces a single-channel output with custom final activation.
    """
    def __init__(self, base_channels: int, activation: str):
        super().__init__()
        C = base_channels
        self.dec3 = UpBlock(C * 8, C * 4, C * 4)
        self.dec2 = UpBlock(C * 4, C * 2, C * 2)
        self.dec1 = UpBlock(C * 2, C, C)

        # Final head
        self.head = nn.Sequential(
            nn.Conv3d(C, C // 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(C // 2, 1, 1),
        )

        if activation == 'sigmoid':
            self.final = nn.Sigmoid()          # for amplitude ∈ [0, 1]
        elif activation == 'tanh':
            self.final = nn.Tanh()             # for phase ∈ [-1, 1] (multiply by π)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, bottleneck, s1, s2, s3):
        d3 = self.dec3(bottleneck, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.final(self.head(d1))


# ═══════════════════════════════════════════════════════════════════════════════
# Main dual-output network
# ═══════════════════════════════════════════════════════════════════════════════

class AutoPhaseNet3D(nn.Module):
    """
    Dual-decoder 3D U-Net predicting both amplitude and phase.

    Input:
        diffraction magnitude (log-normalized), shape [B, 1, N, N, N]

    Outputs:
        amplitude, shape [B, 1, N, N, N]  ∈ [0, 1]
        phase, shape     [B, 1, N, N, N]  ∈ [-1, 1] (multiply by π for radians)

    Architecture (base_channels=32, N=64):

        Input → ConvBlock(32) → DownBlock(64) → DownBlock(128) → Bottleneck(256)
                     ↓                ↓              ↓
                     s1               s2             s3
                     ↓                ↓              ↓
        ┌──── Amp decoder: UpBlocks(128,64,32) → 1×1 conv → sigmoid ──→ amplitude
        │
        └──── Phase decoder: UpBlocks(128,64,32) → 1×1 conv → tanh ───→ phase

    Parameters: ~2.3M at base_channels=32

    Zero-padding support:
        If enforce_oversampling=True, the amplitude is zeroed outside a central
        N/2 × N/2 × N/2 region. This implements the Miao oversampling condition
        as a hard constraint, matching AutoPhaseNN's zero-padding layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        enforce_oversampling: bool = True,
    ):
        super().__init__()
        C = base_channels
        self.enforce_oversampling = enforce_oversampling

        # Shared encoder
        self.enc1 = ConvBlock3D(in_channels, C)
        self.enc2 = DownBlock(C, C * 2)
        self.enc3 = DownBlock(C * 2, C * 4)
        self.bottleneck = DownBlock(C * 4, C * 8)

        # Two independent decoder branches
        self.amp_decoder = DecoderBranch(C, activation='sigmoid')
        self.phase_decoder = DecoderBranch(C, activation='tanh')

    def _zero_pad_mask(self, shape, device):
        """
        Create a central-N/2 support mask (zero-padding layer, AutoPhaseNN §3.1).

        This enforces the oversampling constraint: the real-space object must
        fit inside half the grid in each dimension.
        """
        B, _, N1, N2, N3 = shape
        mask = torch.zeros((1, 1, N1, N2, N3), device=device)
        q1, q2, q3 = N1 // 4, N2 // 4, N3 // 4
        mask[:, :, q1:q1 + N1 // 2, q2:q2 + N2 // 2, q3:q3 + N3 // 2] = 1.0
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : tensor [B, 1, N, N, N]
            Log-normalized diffraction magnitude.

        Returns
        -------
        amplitude : tensor [B, 1, N, N, N]  ∈ [0, 1]
        phase     : tensor [B, 1, N, N, N]  ∈ [-1, 1]  (multiply by π for radians)
        """
        # Shared encoder
        s1 = self.enc1(x)                  # [B, C, N, N, N]
        s2 = self.enc2(s1)                 # [B, 2C, N/2, N/2, N/2]
        s3 = self.enc3(s2)                 # [B, 4C, N/4, N/4, N/4]
        b = self.bottleneck(s3)            # [B, 8C, N/8, N/8, N/8]

        # Dual decoders
        amplitude = self.amp_decoder(b, s1, s2, s3)
        phase = self.phase_decoder(b, s1, s2, s3)

        # Enforce oversampling via zero-padding the amplitude
        if self.enforce_oversampling:
            mask = self._zero_pad_mask(amplitude.shape, amplitude.device)
            amplitude = amplitude * mask

        return amplitude, phase


# ═══════════════════════════════════════════════════════════════════════════════
# Physics forward model (used during training)
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsForwardModel(nn.Module):
    """
    X-ray scattering forward model (AutoPhaseNN §Methods).

    Given predicted (amplitude, phase), forms the complex object, applies a
    dynamic shape support derived from the amplitude itself, and returns the
    predicted diffraction magnitude.

    This module has NO trainable parameters — it purely implements physics:
        ρ(r) = A(r) · exp(i · φ(r)) · S(r)
        F(q) = FFT[ ρ(r) ]
        |F(q)| = |FFT[ ρ(r) ]|

    where S(r) is a shape support computed by thresholding A(r) at `threshold`.
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        amplitude : [B, 1, N, N, N]   ∈ [0, 1]
        phase     : [B, 1, N, N, N]   ∈ [-1, 1]

        Returns
        -------
        pred_diff_mag : [B, 1, N, N, N]   predicted diffraction magnitude
        shape_support : [B, 1, N, N, N]   dynamic support (for regularization)
        """
        # 1. Dynamic shape support by thresholding amplitude
        # Use a soft threshold (sigmoid) so it stays differentiable
        amp_max = amplitude.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
        amp_norm = amplitude / amp_max
        # Soft support: sigmoid((A - threshold) * steepness)
        shape_support = torch.sigmoid((amp_norm - self.threshold) * 25.0)

        # 2. Form complex object:  A(r) · S(r) · e^{iφ(r)}
        phase_rad = phase * torch.pi   # [-π, π]
        real = amplitude * shape_support * torch.cos(phase_rad)
        imag = amplitude * shape_support * torch.sin(phase_rad)
        obj = torch.complex(real.squeeze(1), imag.squeeze(1))

        # 3. Centered FFT
        obj_shifted = torch.fft.ifftshift(obj, dim=(-3, -2, -1))
        F_q = torch.fft.fftn(obj_shifted, dim=(-3, -2, -1))
        F_q = torch.fft.fftshift(F_q, dim=(-3, -2, -1))

        # 4. Magnitude
        pred_diff_mag = torch.abs(F_q).unsqueeze(1)

        return pred_diff_mag, shape_support


# ═══════════════════════════════════════════════════════════════════════════════
# Unsupervised loss (AutoPhaseNN's MAE on sqrt(intensity))
# ═══════════════════════════════════════════════════════════════════════════════

class UnsupervisedBCDILoss(nn.Module):
    """
    Physics-aware unsupervised loss (Eq. 1 in AutoPhaseNN paper):

        Loss = Σ |√I_estimated − √I_measured|  /  N³

    That is: mean absolute error on the diffraction AMPLITUDE (sqrt of
    intensity). Using sqrt compresses the high dynamic range (the center of
    the Bragg peak can be 10⁶× brighter than the fringes) and gives the
    fringes comparable weight to the peak.

    Optional regularizers:
        - support_smoothness : penalty on gradients of the shape support
          (discourages fragmented reconstructions)
        - tv_phase : total-variation on phase inside the support
          (encourages smooth strain fields)
    """

    def __init__(
        self,
        support_smoothness: float = 0.0,
        tv_phase: float = 0.0,
    ):
        super().__init__()
        self.support_smoothness = support_smoothness
        self.tv_phase = tv_phase

    def forward(
        self,
        pred_diff_mag: torch.Tensor,
        measured_diff_mag: torch.Tensor,
        amplitude: torch.Tensor = None,
        phase: torch.Tensor = None,
        shape_support: torch.Tensor = None,
    ) -> dict:
        """
        Parameters
        ----------
        pred_diff_mag     : [B, 1, N, N, N]  predicted |F(q)|
        measured_diff_mag : [B, 1, N, N, N]  measured  |F(q)|
        amplitude, phase, shape_support : optional, for regularizers

        Returns
        -------
        dict with 'total', 'mae_diff', and optional regularizer values.
        """
        # Normalize to match scales (AutoPhaseNN normalizes to max=1)
        pred_norm = pred_diff_mag / (pred_diff_mag.amax(dim=(2,3,4), keepdim=True) + 1e-8)
        meas_norm = measured_diff_mag / (measured_diff_mag.amax(dim=(2,3,4), keepdim=True) + 1e-8)

        # Main loss: MAE on normalized magnitudes
        mae = torch.abs(pred_norm - meas_norm).mean()
        total = mae
        result = {'mae_diff': mae}

        # Optional regularizers
        if self.support_smoothness > 0 and shape_support is not None:
            dx = (shape_support[:, :, 1:] - shape_support[:, :, :-1]).abs().mean()
            dy = (shape_support[:, :, :, 1:] - shape_support[:, :, :, :-1]).abs().mean()
            dz = (shape_support[:, :, :, :, 1:] - shape_support[:, :, :, :, :-1]).abs().mean()
            sm = (dx + dy + dz) / 3.0
            total = total + self.support_smoothness * sm
            result['support_smoothness'] = sm

        if self.tv_phase > 0 and phase is not None and shape_support is not None:
            # TV on phase, weighted by shape support (gradients only matter inside)
            w = (shape_support[:, :, :-1] * shape_support[:, :, 1:]).detach()
            dx = ((phase[:, :, 1:] - phase[:, :, :-1]) ** 2 * w).mean()
            w = (shape_support[:, :, :, :-1] * shape_support[:, :, :, 1:]).detach()
            dy = ((phase[:, :, :, 1:] - phase[:, :, :, :-1]) ** 2 * w).mean()
            w = (shape_support[:, :, :, :, :-1] * shape_support[:, :, :, :, 1:]).detach()
            dz = ((phase[:, :, :, :, 1:] - phase[:, :, :, :, :-1]) ** 2 * w).mean()
            tv = (dx + dy + dz) / 3.0
            total = total + self.tv_phase * tv
            result['tv_phase'] = tv

        result['total'] = total
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: AutoPhaseNet3D, grid_size: int = 64):
    n = count_parameters(model)
    print(f"\n{'='*52}")
    print(f"  AutoPhaseNet3D (dual-decoder)")
    print(f"{'='*52}")
    print(f"  Parameters:       {n:,}")
    print(f"  Grid size:        {grid_size}³")
    print(f"  Enforce ovsmp:    {model.enforce_oversampling}")
    print(f"  Outputs:          amplitude ∈ [0,1], phase ∈ [-1,1]×π")
    print(f"{'='*52}\n")


if __name__ == '__main__':
    # Smoke test
    model = AutoPhaseNet3D(base_channels=32)
    model_summary(model, 64)

    x = torch.randn(2, 1, 64, 64, 64)
    with torch.no_grad():
        amp, ph = model(x)
        forward = PhysicsForwardModel(threshold=0.1)
        pred_diff, support = forward(amp, ph)

    print(f"Input:             {x.shape}")
    print(f"Predicted amp:     {amp.shape}  range [{amp.min():.3f}, {amp.max():.3f}]")
    print(f"Predicted phase:   {ph.shape}  range [{ph.min():.3f}, {ph.max():.3f}]")
    print(f"Forward diff:      {pred_diff.shape}")
    print(f"Shape support:     {support.shape}  mean={support.mean():.3f}")

    # Loss test
    loss_fn = UnsupervisedBCDILoss(support_smoothness=0.01, tv_phase=0.01)
    measured = torch.rand(2, 1, 64, 64, 64) * 1000
    losses = loss_fn(pred_diff, measured, amp, ph, support)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.5f}")
