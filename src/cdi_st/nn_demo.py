"""
nn_demo.py — End-to-end demonstration of the NN-accelerated BCDI pipeline.

This script ties everything together:
    1. Generate a small training dataset (50 samples for demo)
    2. Train the U-Net for a few epochs
    3. Run phase retrieval in all 3 modes
    4. Compare results and plot

For a real workflow, you would:
    - Generate 2000+ samples:  python nn_data_generator.py --num_samples 2000
    - Train for 50+ epochs:    python nn_train.py --epochs 50
    - Reconstruct:             python nn_phase_retrieval.py --input data.npz --mode hybrid

Usage:
    python nn_demo.py                          # Full demo
    python nn_demo.py --skip_training          # Skip to reconstruction (needs trained model)
    python nn_demo.py --num_samples 200        # More training data
    python nn_demo.py --material Au            # Focus on gold nanocrystals
"""

from __future__ import annotations
import os, sys, argparse, time
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Generate training data
# ═══════════════════════════════════════════════════════════════════════════════

def step_generate(num_samples=50, grid_size=64, material=None, output_dir='./demo_data'):
    """Generate training samples using bcdi_core."""
    from cdi_st.nn_data_generator import generate_dataset

    print("\n" + "█" * 60)
    print("  STEP 1: Generate training data")
    print("█" * 60)

    generate_dataset(
        output_dir=output_dir,
        num_samples=num_samples,
        grid_size=grid_size,
        seed=42,
        fixed_material=material,
        add_noise=True,
        resume=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Train the U-Net
# ═══════════════════════════════════════════════════════════════════════════════

def step_train(data_dir='./demo_data', output_dir='./demo_checkpoints',
               epochs=10, batch_size=4):
    """Train the phase prediction U-Net."""
    from nn_train import train

    print("\n" + "█" * 60)
    print("  STEP 2: Train phase prediction U-Net")
    print("█" * 60)

    train(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        base_channels=32,
        loss_alpha=1.0,
        loss_beta=0.1,
        loss_gamma=0.01,
        patience=10,
        num_workers=0,  # 0 for demo (avoids multiprocessing issues)
        seed=42,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Reconstruct and compare
# ═══════════════════════════════════════════════════════════════════════════════

def step_reconstruct(test_sample_path, model_path='./demo_checkpoints/best_model.pt'):
    """Run phase retrieval in all 3 modes and compare."""
    from cdi_st.nn_phase_retrieval import HybridPhaseRetriever

    print("\n" + "█" * 60)
    print("  STEP 3: Phase retrieval comparison")
    print("█" * 60)

    # Load test sample
    data = np.load(test_sample_path)
    diffraction = data['diffraction']
    phase_true = data['phase_true']
    support = data['support']

    print(f"\n  Test sample: {test_sample_path}")
    print(f"  Volume shape: {diffraction.shape}")
    print(f"  Max intensity: {diffraction.max():.2e}")

    # Create retriever
    retriever = HybridPhaseRetriever(
        model_path=model_path,
        base_channels=32,
    )

    # Compare all methods
    results = retriever.compare_methods(diffraction, support)

    # ── Compute quality metrics against ground truth ──────────────────────
    print(f"\n{'='*60}")
    print(f"  Quality comparison (against ground truth)")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Phase RMSE':>12} {'Phase corr':>12} {'R-factor':>10} {'Time':>8}")
    print(f"{'-'*60}")

    for name, result in results.items():
        # Phase RMSE inside support
        mask = support > 0.5
        if mask.sum() > 0:
            # Handle phase wrapping: compare on unit circle
            diff = result.phase[mask] - phase_true[mask]
            # Wrap to [-π, π]
            diff = np.angle(np.exp(1j * diff))
            rmse = np.sqrt(np.mean(diff ** 2))

            # Phase correlation
            corr = np.corrcoef(
                result.phase[mask].ravel(),
                phase_true[mask].ravel()
            )[0, 1]
        else:
            rmse = float('nan')
            corr = float('nan')

        print(
            f"{name:<15} {rmse:>12.4f} {corr:>12.4f} "
            f"{result.error_metric[-1]:>10.4f} {result.elapsed_seconds:>7.2f}s"
        )

    print(f"{'='*60}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Plot results
# ═══════════════════════════════════════════════════════════════════════════════

def step_plot(results, test_sample_path, output_dir='./demo_results'):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    data = np.load(test_sample_path)
    phase_true = data['phase_true']
    support = data['support']

    N = phase_true.shape[0]
    c = N // 2

    # ── Figure 1: Phase slices comparison ─────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Phase retrieval comparison (central slices)', fontsize=14)

    titles = ['Ground truth', 'NN-only', 'Hybrid', 'Classical']
    phases = [
        phase_true,
        results['nn_only'].phase,
        results['hybrid'].phase,
        results['classical'].phase,
    ]

    for col, (title, phase) in enumerate(zip(titles, phases)):
        # XY slice
        ax = axes[0, col]
        im = ax.imshow(
            phase[:, :, c] * support[:, :, c],
            cmap='twilight', vmin=-np.pi, vmax=np.pi,
            origin='lower', aspect='equal',
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('qx-qy' if col == 0 else '')
        ax.tick_params(labelsize=7)

        # XZ slice
        ax = axes[1, col]
        ax.imshow(
            phase[:, c, :] * support[:, c, :],
            cmap='twilight', vmin=-np.pi, vmax=np.pi,
            origin='lower', aspect='equal',
        )
        ax.set_ylabel('qx-qz' if col == 0 else '')
        ax.tick_params(labelsize=7)

    fig.colorbar(im, ax=axes, label='Phase (rad)', shrink=0.6)
    plt.tight_layout()
    plt.savefig(out / 'phase_comparison.png', dpi=150)
    print(f"  Saved: {out / 'phase_comparison.png'}")

    # ── Figure 2: Convergence curves ──────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_title('Convergence: R-factor vs iteration')

    colors = {'nn_only': '#e85d24', 'hybrid': '#3fb950', 'classical': '#4f98a3'}
    for name, result in results.items():
        if len(result.error_metric) > 1:
            ax.semilogy(result.error_metric, label=f'{name} ({result.elapsed_seconds:.1f}s)',
                       color=colors.get(name, '#888'), linewidth=2)
        else:
            ax.axhline(result.error_metric[0], linestyle='--',
                      label=f'{name} (R={result.error_metric[0]:.3f})',
                      color=colors.get(name, '#888'), linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('R-factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'convergence.png', dpi=150)
    print(f"  Saved: {out / 'convergence.png'}")

    # ── Figure 3: Amplitude (electron density) comparison ─────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Reconstructed electron density (central slice)', fontsize=14)

    amps = [
        np.abs(support),
        results['nn_only'].amplitude,
        results['hybrid'].amplitude,
        results['classical'].amplitude,
    ]
    titles = ['Support (truth)', 'NN-only |ρ|', 'Hybrid |ρ|', 'Classical |ρ|']

    for ax, amp, title in zip(axes, amps, titles):
        ax.imshow(amp[:, :, c], cmap='hot', origin='lower', aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out / 'amplitude_comparison.png', dpi=150)
    print(f"  Saved: {out / 'amplitude_comparison.png'}")

    plt.close('all')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='BCDI NN Phase Retrieval Demo')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Training samples to generate')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid size N')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs')
    parser.add_argument('--material', type=str, default=None,
                        help='Lock to specific material')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip data generation and training')
    parser.add_argument('--data_dir', type=str, default='./demo_data')
    parser.add_argument('--checkpoint_dir', type=str, default='./demo_checkpoints')
    parser.add_argument('--results_dir', type=str, default='./demo_results')
    args = parser.parse_args()

    t_total = time.time()

    print("\n" + "═" * 60)
    print("  BCDI NN-Accelerated Phase Retrieval — Full Demo")
    print("═" * 60)
    print(f"  Samples:    {args.num_samples}")
    print(f"  Grid:       {args.grid_size}³")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Material:   {args.material or 'random'}")
    print("═" * 60)

    if not args.skip_training:
        # Step 1: Generate data
        step_generate(
            num_samples=args.num_samples,
            grid_size=args.grid_size,
            material=args.material,
            output_dir=args.data_dir,
        )

        # Step 2: Train
        step_train(
            data_dir=args.data_dir,
            output_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=4,
        )

    # Find a test sample
    data_dir = Path(args.data_dir)
    test_files = sorted(data_dir.glob("sample_*.npz"))
    if not test_files:
        print("ERROR: No test samples found!")
        return

    # Use the last sample as test (not seen during training with default split)
    test_sample = str(test_files[-1])

    # Step 3: Reconstruct and compare
    model_path = Path(args.checkpoint_dir) / 'best_model.pt'
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("  Run without --skip_training first.")
        return

    results = step_reconstruct(test_sample, str(model_path))

    # Step 4: Plot
    step_plot(results, test_sample, args.results_dir)

    elapsed = time.time() - t_total
    print(f"\n{'═'*60}")
    print(f"  Demo complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
