"""
nn_autophase_train.py — Unsupervised training for AutoPhaseNet3D.

Trains the dual-decoder network WITHOUT requiring ground-truth phase or
support. Only the measured diffraction magnitude is used.

Training loop:
    1. Input diffraction magnitude → encoder
    2. Predict amplitude and phase
    3. Physics forward model:  ρ(r) = A·S·e^{iφ} → FFT → |F|_pred
    4. Loss = MAE(|F|_pred, |F|_measured)
    5. Backprop through the physics model AND the network

This is a direct implementation of the AutoPhaseNN training procedure
(Yao et al., npj Comp Mat 2022), adapted to work with your existing
dataset structure.

Usage:
    # From scratch on simulated data (no phase_true needed):
    python nn_autophase_train.py --data_dir ./training_data --epochs 60

    # Fine-tune on experimental data (loaded from .npz or .h5):
    python nn_autophase_train.py \\
        --data_dir ./experimental_data \\
        --resume checkpoints_autophase/best_model.pt \\
        --lr 1e-5 --epochs 30
"""

from __future__ import annotations
import os, sys, time, json, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from .nn_autophase_model import (
    AutoPhaseNet3D, PhysicsForwardModel, UnsupervisedBCDILoss,
    count_parameters
)


# ═══════════════════════════════════════════════════════════════════════════════
# Unsupervised dataset: only diffraction magnitude is needed
# ═══════════════════════════════════════════════════════════════════════════════

class UnsupervisedBCDIDataset(Dataset):
    """
    Dataset that ONLY loads the measured diffraction magnitude.

    Works with:
        - .npz files from nn_data_generator.py (simulated)
        - .npz files exported from .h5 via nn_experimental_loader.py
        - raw .h5 files from ID01 beamline (auto-detected)

    Returns a dict with:
        'input'   : log-normalized diffraction magnitude [1, N, N, N]
        'measured': linear-scale diffraction magnitude   [1, N, N, N]
        'amp_scale': normalization factor for reconstruction
    """

    def __init__(
        self,
        data_dir: str,
        augment: bool = False,
        max_samples: int = None,
        grid_size: int = 64,
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.grid_size = grid_size

        # Collect .npz and .h5 files
        self.files = []
        self.files += sorted(self.data_dir.glob("sample_*.npz"))
        self.files += sorted(self.data_dir.glob("*.h5"))
        if max_samples is not None:
            self.files = self.files[:max_samples]

        if len(self.files) == 0:
            raise ValueError(f"No .npz or .h5 files found in {data_dir}")

        print(f"UnsupervisedBCDIDataset: {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def _load_diffraction(self, fpath: Path) -> np.ndarray:
        """Load diffraction from either .npz or .h5."""
        if fpath.suffix == '.npz':
            data = np.load(fpath)
            if 'diffraction' in data:
                return data['diffraction'].astype(np.float32)
            elif 'diffraction_volume' in data:
                return data['diffraction_volume'].astype(np.float32)
            elif 'amplitude' in data:
                # amplitude was stored; square it to get intensity-like
                return (data['amplitude'].astype(np.float32)) ** 2
            else:
                raise KeyError(f"No diffraction data in {fpath}")

        elif fpath.suffix == '.h5':
            # Lazy import so h5py isn't required unless .h5 files are used
            from cdi_st.nn_experimental_loader import load_h5_diffraction
            return load_h5_diffraction(fpath, target_size=self.grid_size)

        else:
            raise ValueError(f"Unsupported file type: {fpath}")

    def __getitem__(self, idx: int) -> dict:
        fpath = self.files[idx]
        diffraction = self._load_diffraction(fpath)

        # Ensure float32 and non-negative
        diffraction = np.maximum(diffraction, 0).astype(np.float32)

        # Magnitude = sqrt(intensity)
        magnitude = np.sqrt(diffraction)

        # Log-normalized input to the network (0..1)
        log_mag = np.log10(magnitude + 1.0)
        amp_scale = log_mag.max()
        if amp_scale > 0:
            log_mag = log_mag / amp_scale

        # Augmentations (valid for BCDI: rotations preserve physics)
        if self.augment:
            rng = np.random.default_rng()
            for axes in [(0, 1), (0, 2), (1, 2)]:
                k = rng.integers(0, 4)
                if k > 0:
                    magnitude = np.rot90(magnitude, k, axes)
                    log_mag = np.rot90(log_mag, k, axes)
            for axis in range(3):
                if rng.random() < 0.5:
                    magnitude = np.flip(magnitude, axis)
                    log_mag = np.flip(log_mag, axis)
            magnitude = np.ascontiguousarray(magnitude)
            log_mag = np.ascontiguousarray(log_mag)

        return {
            'input':    torch.from_numpy(log_mag[np.newaxis]).float(),
            'measured': torch.from_numpy(magnitude[np.newaxis]).float(),
            'amp_scale': float(amp_scale),
            'file':     str(fpath.name),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Training / validation loops
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, physics, loss_fn, loader, optimizer, scaler, device):
    model.train()
    running = {}
    n = 0
    for batch_idx, batch in enumerate(loader):
        inp = batch['input'].to(device)
        measured = batch['measured'].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=(device.type == 'cuda')):
            amp, phase = model(inp)
            pred_diff, support = physics(amp, phase)
            losses = loss_fn(pred_diff, measured, amp, phase, support)
            loss = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            running[k] = running.get(k, 0) + v.item()
        n += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}  loss={running['total']/n:.5f}")

    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, physics, loss_fn, loader, device):
    model.eval()
    running = {}
    n = 0
    for batch in loader:
        inp = batch['input'].to(device)
        measured = batch['measured'].to(device)
        with autocast('cuda', enabled=(device.type == 'cuda')):
            amp, phase = model(inp)
            pred_diff, support = physics(amp, phase)
            losses = loss_fn(pred_diff, measured, amp, phase, support)
        for k, v in losses.items():
            running[k] = running.get(k, 0) + v.item()
        n += 1
    return {k: v / max(n, 1) for k, v in running.items()}


def save_ckpt(model, optimizer, scheduler, scaler, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
    }, path)


def load_ckpt(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('val_loss', float('inf'))


# ═══════════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    data_dir: str,
    output_dir: str = './checkpoints_autophase',
    epochs: int = 60,
    batch_size: int = 8,
    lr: float = 1e-3,
    base_channels: int = 32,
    support_smoothness: float = 0.005,
    tv_phase: float = 0.005,
    threshold: float = 0.1,
    enforce_oversampling: bool = True,
    patience: int = 20,
    resume: str = None,
    num_workers: int = 2,
    max_samples: int = None,
    val_fraction: float = 0.15,
    seed: int = 42,
    grid_size: int = 64,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*62}")
    print(f"  AutoPhaseNet3D Unsupervised Training")
    print(f"{'='*62}")
    print(f"  Device:         {device}")
    if device.type == 'cuda':
        print(f"  GPU:            {torch.cuda.get_device_name()}")
    print(f"  Data dir:       {data_dir}")
    print(f"  Output:         {out.resolve()}")
    print(f"  Epochs:         {epochs}   Batch size: {batch_size}   LR: {lr}")
    print(f"  Base channels:  {base_channels}  (oversampling={enforce_oversampling})")
    print(f"  Regularizers:   support_smoothness={support_smoothness} tv_phase={tv_phase}")

    # Dataset
    full_ds = UnsupervisedBCDIDataset(data_dir, augment=False,
                                        max_samples=max_samples,
                                        grid_size=grid_size)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_ds.dataset = UnsupervisedBCDIDataset(data_dir, augment=True,
                                                 max_samples=max_samples,
                                                 grid_size=grid_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print(f"  Train: {n_train}   Val: {n_val}")

    # Model + physics + loss
    model = AutoPhaseNet3D(base_channels=base_channels,
                            enforce_oversampling=enforce_oversampling).to(device)
    physics = PhysicsForwardModel(threshold=threshold).to(device)
    loss_fn = UnsupervisedBCDILoss(support_smoothness=support_smoothness,
                                     tv_phase=tv_phase)
    n_params = count_parameters(model)
    print(f"  Parameters:     {n_params:,}")
    print(f"{'='*62}\n")

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_epoch = 0
    best_val = float('inf')
    if resume and os.path.exists(resume):
        start_epoch, best_val = load_ckpt(resume, model, optimizer, None, scaler)
        print(f"  Resumed from {resume} (epoch {start_epoch}, val {best_val:.5f})\n")

    # CSV log
    log_path = out / 'training_log.csv'
    log_file = open(log_path, 'a' if (log_path.exists() and resume) else 'w', newline='')
    log_writer = csv.writer(log_file)
    if log_path.stat().st_size == 0:
        log_writer.writerow(['epoch', 'lr', 'train_total', 'train_mae',
                              'val_total', 'val_mae', 'time_sec'])

    # Training loop
    no_improve = 0
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}  lr={lr_now:.2e}")

        train_losses = train_one_epoch(model, physics, loss_fn,
                                         train_loader, optimizer, scaler, device)
        val_losses = validate(model, physics, loss_fn, val_loader, device)
        scheduler.step(val_losses['total'])

        elapsed = time.time() - t0
        print(f"  train: total={train_losses['total']:.5f}  mae={train_losses['mae_diff']:.5f}")
        print(f"  val:   total={val_losses['total']:.5f}  mae={val_losses['mae_diff']:.5f}  ({elapsed:.1f}s)")

        log_writer.writerow([
            epoch + 1, lr_now,
            train_losses['total'], train_losses['mae_diff'],
            val_losses['total'], val_losses['mae_diff'],
            elapsed,
        ])
        log_file.flush()

        # Best-model checkpointing
        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            save_ckpt(model, optimizer, scheduler, scaler,
                     epoch + 1, best_val, out / 'best_model.pt')
            print(f"  ★ New best (val={best_val:.5f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")

        if (epoch + 1) % 10 == 0:
            save_ckpt(model, optimizer, scheduler, scaler,
                     epoch + 1, val_losses['total'],
                     out / f'ckpt_epoch{epoch+1}.pt')

        if no_improve >= patience:
            print(f"\n  Early stopping.")
            break
        print()

    log_file.close()

    # Save config
    config = {
        'data_dir': str(data_dir),
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val,
        'base_channels': base_channels,
        'enforce_oversampling': enforce_oversampling,
        'support_smoothness': support_smoothness,
        'tv_phase': tv_phase,
        'threshold': threshold,
        'grid_size': grid_size,
        'n_params': n_params,
    }
    with open(out / 'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*62}")
    print(f"  Training complete. Best val: {best_val:.5f}")
    print(f"  Saved: {out / 'best_model.pt'}")
    print(f"{'='*62}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoPhaseNet unsupervised training')
    parser.add_argument('--data_dir', type=str, default='./training_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_autophase')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--support_smoothness', type=float, default=0.005)
    parser.add_argument('--tv_phase', type=float, default=0.005)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--no_oversampling', action='store_true')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grid_size', type=int, default=64)
    args = parser.parse_args()

    kwargs = vars(args)
    kwargs['enforce_oversampling'] = not kwargs.pop('no_oversampling')
    train(**kwargs)
