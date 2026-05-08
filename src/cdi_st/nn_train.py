"""
nn_train.py — Training loop for the BCDI phase prediction U-Net.

Handles:
    - Training with mixed-precision (AMP) for faster GPU utilization
    - Cosine annealing learning rate schedule
    - Early stopping based on validation loss
    - Checkpoint saving (best model + periodic)
    - Logging training curves to CSV for plotting
    - Gradient clipping to prevent explosion

Usage:
    python nn_train.py --data_dir ./training_data --epochs 50 --batch_size 8

    # Resume from checkpoint:
    python nn_train.py --data_dir ./training_data --resume checkpoints/best_model.pt

    # Smaller model for limited GPU:
    python nn_train.py --data_dir ./training_data --base_channels 16 --batch_size 4
"""

from __future__ import annotations
import os, sys, time, json, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from pathlib import Path

from .nn_dataset import BCDIDataset, create_dataloaders
from .nn_phase_model import PhaseUNet3D, BCDIPhaseLoss, count_parameters


def train_one_epoch(
    model, loader, loss_fn, optimizer, scaler, device, epoch, max_grad_norm=1.0,
):
    """Train for one epoch. Returns dict of mean losses."""
    model.train()
    running = {'total': 0, 'mse': 0, 'fft_consistency': 0, 'smoothness': 0}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        inp = batch['input'].to(device)
        target = batch['target_phase'].to(device)
        support = batch['support'].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with autocast('cuda', enabled=(device.type == 'cuda')):
            pred = model(inp)
            losses = loss_fn(pred, target, support, inp)
            loss = losses['total']

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate
        for k in running:
            if k in losses:
                running[k] += losses[k].item()
        n_batches += 1

        # Progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            avg_loss = running['total'] / n_batches
            print(f"    batch {batch_idx+1}/{len(loader)}  loss={avg_loss:.5f}")

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Validate. Returns dict of mean losses."""
    model.eval()
    running = {'total': 0, 'mse': 0, 'fft_consistency': 0, 'smoothness': 0}
    n_batches = 0

    for batch in loader:
        inp = batch['input'].to(device)
        target = batch['target_phase'].to(device)
        support = batch['support'].to(device)

        with autocast('cuda', enabled=(device.type == 'cuda')):
            pred = model(inp)
            losses = loss_fn(pred, target, support, inp)

        for k in running:
            if k in losses:
                running[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, path):
    """Save training state for resuming."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training state."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('val_loss', float('inf'))


def train(
    data_dir: str,
    output_dir: str = './checkpoints',
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    base_channels: int = 32,
    loss_alpha: float = 1.0,
    loss_beta: float = 0.1,
    loss_gamma: float = 0.01,
    patience: int = 15,
    resume: str = None,
    num_workers: int = 4,
    max_samples: int = None,
    seed: int = 42,
):
    """
    Full training loop.

    Parameters
    ----------
    data_dir : str
        Directory with .npz training samples.
    output_dir : str
        Directory for checkpoints and logs.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size (reduce if OOM).
    lr : float
        Initial learning rate.
    base_channels : int
        U-Net base channels (32=~1.9M params, 16=~0.5M params).
    loss_alpha, loss_beta, loss_gamma : float
        Loss component weights.
    patience : int
        Early stopping patience (epochs without improvement).
    resume : str or None
        Path to checkpoint to resume from.
    num_workers : int
        DataLoader workers.
    max_samples : int or None
        Limit dataset size (for debugging).
    seed : int
        Random seed.
    """
    # Setup
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"BCDI Phase U-Net Training")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    if device.type == 'cuda':
        print(f"  GPU:          {torch.cuda.get_device_name()}")
        print(f"  GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Data:         {data_dir}")
    print(f"  Output:       {out.resolve()}")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  LR:           {lr}")
    print(f"  Base ch:      {base_channels}")
    print(f"  Loss weights: α={loss_alpha} β={loss_beta} γ={loss_gamma}")
    print(f"  Patience:     {patience}")

    # Data
    train_loader, val_loader, _ = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers,
        max_samples=max_samples, seed=seed,
    )

    # Model
    model = PhaseUNet3D(in_channels=1, base_channels=base_channels).to(device)
    n_params = count_parameters(model)
    print(f"  Parameters:   {n_params:,}")

    # Loss, optimizer, scheduler
    loss_fn = BCDIPhaseLoss(alpha=loss_alpha, beta=loss_beta, gamma=loss_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if resume and os.path.exists(resume):
        start_epoch, best_val_loss = load_checkpoint(
            resume, model, optimizer, scheduler, scaler
        )
        print(f"  Resumed from epoch {start_epoch}, best val_loss={best_val_loss:.5f}")

    print(f"{'='*60}\n")

    # Logging
    log_path = out / 'training_log.csv'
    log_exists = log_path.exists() and resume
    log_file = open(log_path, 'a' if log_exists else 'w', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            'epoch', 'lr', 'train_loss', 'train_mse', 'train_fft', 'train_smooth',
            'val_loss', 'val_mse', 'val_fft', 'val_smooth', 'time_sec'
        ])

    # Training loop
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}  lr={lr_now:.2e}")

        # Train
        train_losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch
        )

        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t0

        # Log
        print(
            f"  train: loss={train_losses['total']:.5f} "
            f"mse={train_losses['mse']:.5f} "
            f"fft={train_losses.get('fft_consistency', 0):.5f}"
        )
        print(
            f"  val:   loss={val_losses['total']:.5f} "
            f"mse={val_losses['mse']:.5f} "
            f"fft={val_losses.get('fft_consistency', 0):.5f}  "
            f"({elapsed:.1f}s)"
        )

        log_writer.writerow([
            epoch + 1, lr_now,
            train_losses['total'], train_losses['mse'],
            train_losses.get('fft_consistency', 0), train_losses.get('smoothness', 0),
            val_losses['total'], val_losses['mse'],
            val_losses.get('fft_consistency', 0), val_losses.get('smoothness', 0),
            elapsed,
        ])
        log_file.flush()

        # Checkpointing
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch + 1, best_val_loss, out / 'best_model.pt'
            )
            print(f"  ★ New best model saved (val_loss={best_val_loss:.5f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch + 1, val_losses['total'], out / f'checkpoint_epoch{epoch+1}.pt'
            )

        # Early stopping
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

        print()

    log_file.close()

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        epoch + 1, val_losses['total'], out / 'final_model.pt'
    )

    # Save training config
    config = {
        'data_dir': data_dir,
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'base_channels': base_channels,
        'loss_alpha': loss_alpha,
        'loss_beta': loss_beta,
        'loss_gamma': loss_gamma,
        'lr': lr,
        'batch_size': batch_size,
        'n_params': n_params,
    }
    with open(out / 'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.5f}")
    print(f"  Model saved:   {out / 'best_model.pt'}")
    print(f"  Log:           {log_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BCDI phase U-Net')
    parser.add_argument('--data_dir', type=str, default='./training_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--loss_alpha', type=float, default=1.0)
    parser.add_argument('--loss_beta', type=float, default=0.1)
    parser.add_argument('--loss_gamma', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(**vars(args))
