"""
nn_dataset.py — PyTorch Dataset for BCDI phase retrieval training.

Loads the .npz files created by nn_data_generator.py and prepares them
for training the U-Net phase predictor.

Input to the network:
    - log10(amplitude + 1), normalized to [0, 1]  → shape [1, N, N, N]

Target:
    - phase field (ground truth)                   → shape [1, N, N, N]
    - support mask                                 → shape [1, N, N, N]

Augmentations (applied randomly during training):
    - Random 90° rotations along each axis
    - Random flips along each axis
    - Gaussian noise on the input amplitude

These augmentations exploit the fact that BCDI diffraction patterns have
symmetry under rotation/flip (Friedel's law in centrosymmetric cases),
and the physics is invariant to these transformations.

Usage:
    from cdi_st.nn_dataset import BCDIDataset
    ds = BCDIDataset('./training_data', augment=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
"""

from __future__ import annotations
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional, Tuple


class BCDIDataset(Dataset):
    """
    PyTorch Dataset for BCDI (amplitude → phase) training pairs.

    Each sample contains:
        input:  log-scaled amplitude  [1, N, N, N]  float32
        target: ground-truth phase    [1, N, N, N]  float32
        support: binary mask          [1, N, N, N]  float32
    """

    def __init__(
        self,
        data_dir: str,
        augment: bool = False,
        normalize_phase: bool = True,
        max_samples: int = None,
    ):
        """
        Parameters
        ----------
        data_dir : str
            Directory containing sample_XXXXX.npz files.
        augment : bool
            Apply random augmentations (rotations, flips, noise).
        normalize_phase : bool
            Normalize phase to [-1, 1] range (divide by π).
        max_samples : int or None
            Limit number of samples loaded (for debugging).
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.normalize_phase = normalize_phase

        # Find all .npz files
        self.files = sorted(self.data_dir.glob("sample_*.npz"))
        # Filter out metadata JSONs
        self.files = [f for f in self.files if f.suffix == '.npz']

        if max_samples is not None:
            self.files = self.files[:max_samples]

        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        print(f"BCDIDataset: {len(self.files)} samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns dict with keys: 'input', 'target_phase', 'support', 'metadata_path'
        """
        fpath = self.files[idx]

        # Load arrays
        data = np.load(fpath)
        amplitude = data['amplitude'].astype(np.float32)    # [N, N, N]
        phase = data['phase_true'].astype(np.float32)        # [N, N, N]
        support = data['support'].astype(np.float32)         # [N, N, N]

        # ── Preprocessing ─────────────────────────────────────────────────

        # Log-scale the amplitude and normalize to [0, 1]
        log_amp = np.log10(amplitude + 1.0)
        max_val = log_amp.max()
        if max_val > 0:
            log_amp = log_amp / max_val
        # Store the normalization factor for reconstruction
        amp_scale = max_val

        # Normalize phase: divide by π to get [-1, 1] range
        if self.normalize_phase:
            phase = phase / np.pi

        # ── Augmentation ──────────────────────────────────────────────────

        if self.augment:
            log_amp, phase, support = self._augment(log_amp, phase, support)

        # ── Convert to tensors ────────────────────────────────────────────

        # Add channel dimension: [N, N, N] → [1, N, N, N]
        input_tensor = torch.from_numpy(log_amp[np.newaxis]).float()
        phase_tensor = torch.from_numpy(phase[np.newaxis]).float()
        support_tensor = torch.from_numpy(support[np.newaxis]).float()

        return {
            'input': input_tensor,
            'target_phase': phase_tensor,
            'support': support_tensor,
            'amp_scale': amp_scale,
            'file': str(fpath.name),
        }

    def _augment(
        self,
        log_amp: np.ndarray,
        phase: np.ndarray,
        support: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply physics-respecting augmentations.

        Rotations and flips are valid because:
          - The diffraction pattern of a rotated/flipped crystal
            is the rotated/flipped diffraction pattern
          - Phase changes sign under inversion (Friedel's law)
        """
        rng = np.random.default_rng()

        # Random 90° rotations (0, 1, 2, or 3 times) for each axis pair
        for axes in [(0, 1), (0, 2), (1, 2)]:
            k = rng.integers(0, 4)
            if k > 0:
                log_amp = np.rot90(log_amp, k, axes)
                phase = np.rot90(phase, k, axes)
                support = np.rot90(support, k, axes)

        # Random flips along each axis
        for axis in range(3):
            if rng.random() < 0.5:
                log_amp = np.flip(log_amp, axis)
                phase = -np.flip(phase, axis)  # Phase inverts under flip!
                support = np.flip(support, axis)

        # Small additive noise on input (simulates measurement noise)
        if rng.random() < 0.3:
            noise_level = rng.uniform(0.005, 0.02)
            log_amp = log_amp + rng.normal(0, noise_level, log_amp.shape).astype(np.float32)
            log_amp = np.clip(log_amp, 0, 1)

        # Ensure contiguous arrays after rot/flip
        return (
            np.ascontiguousarray(log_amp),
            np.ascontiguousarray(phase),
            np.ascontiguousarray(support),
        )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    val_fraction: float = 0.15,
    test_fraction: float = 0.05,
    num_workers: int = 4,
    max_samples: int = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from a directory of .npz files.

    Parameters
    ----------
    data_dir : str
        Directory with sample_XXXXX.npz files.
    batch_size : int
        Batch size for training.
    val_fraction : float
        Fraction of data for validation.
    test_fraction : float
        Fraction of data for testing.
    num_workers : int
        Number of dataloader workers.
    max_samples : int or None
        Limit dataset size.
    seed : int
        Random seed for splitting.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    # Full dataset (no augmentation for splitting)
    full_ds = BCDIDataset(data_dir, augment=False, max_samples=max_samples)

    n = len(full_ds)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    # Wrap train split with augmentation
    train_ds.dataset = BCDIDataset(data_dir, augment=True, max_samples=max_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick test
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './training_data'
    ds = BCDIDataset(data_dir, augment=True)
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    print(f"Input shape:   {sample['input'].shape}")
    print(f"Phase shape:   {sample['target_phase'].shape}")
    print(f"Support shape: {sample['support'].shape}")
    print(f"Input range:   [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
    print(f"Phase range:   [{sample['target_phase'].min():.3f}, {sample['target_phase'].max():.3f}]")
    print(f"Support sum:   {sample['support'].sum():.0f} voxels")
