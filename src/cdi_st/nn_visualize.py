"""
nn_visualize.py — Visualize BCDI phase retrieval reconstruction results.

Fixed version with:
    - Correct real-space vs reciprocal-space comparison
    - Phase unwrapping / global offset removal
    - Diffraction-space comparison (the scientifically meaningful one)
    - Better interpretation of convergence curves
    - Helper to find samples with actual strain for meaningful testing

Usage:
    %run nn_visualize.py --input reconstruction.npz
    %run nn_visualize.py --input reconstruction.npz --ground_truth training_data/sample_00000.npz
    %run nn_visualize.py --find_strained                  # List samples WITH strain/dislocation
    %run nn_visualize.py --input reconstruction.npz --save_dir ./figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_reconstruction(path: str) -> dict:
    """Load reconstruction.npz and reconstruct complex object."""
    data = np.load(path)
    out = dict(data)
    if "object_real" in data and "object_imag" in data:
        out["object_3d"] = data["object_real"] + 1j * data["object_imag"]
    elif "object_3d" not in data:
        out["object_3d"] = data["amplitude"] * np.exp(1j * data["phase"])

    print(f"Loaded reconstruction: {path}")
    print(f"  Shape:          {out['phase'].shape}")
    print(f"  Phase range:    [{out['phase'].min():.3f}, {out['phase'].max():.3f}] rad")
    print(f"  Amplitude max:  {out['amplitude'].max():.4f}")
    if "error_metric" in out and len(out["error_metric"]) > 0:
        print(f"  R-factor:       {out['error_metric'][-1]:.5f}")
        print(f"  Iterations:     {len(out['error_metric'])}")
    return out


def load_ground_truth(path: str) -> dict:
    """Load original sample .npz for comparison."""
    data = np.load(path)
    d = dict(data)
    print(f"\nLoaded ground truth: {path}")
    print(
        f"  Phase range:  [{d['phase_true'].min():.4f}, {d['phase_true'].max():.4f}] rad"
    )
    print(f"  Support vox:  {int(d['support'].sum()):,}")

    meta_path = Path(path).parent / (Path(path).stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        d["_meta"] = meta
        print(f"  Material:     {meta.get('material','?')}")
        print(f"  Shape:        {meta.get('shape','?')}")
        print(f"  Strain type:  {meta.get('strain_type','?')}")
        print(f"  Dislocation:  {meta.get('dislocation_type','?')}")
        print(f"  hkl:          {meta.get('hkl','?')}")
    return d


def find_strained_samples(data_dir: str, limit: int = 20):
    """List samples that have actual strain or dislocations."""
    data_dir = Path(data_dir)
    metas = sorted(data_dir.glob("sample_*_meta.json"))
    if not metas:
        print(f"No metadata files found in {data_dir}")
        return

    print(f"\n{'='*76}")
    print("  Samples with non-trivial phase (strain or dislocation)")
    print(f"{'='*76}")
    print(
        f"  {'ID':<6} {'Material':<12} {'Shape':<10} {'Strain':<18} {'Disloc':<8} {'hkl'}"
    )
    print(f"  {'-'*70}")

    found = 0
    for m in metas:
        with open(m) as f:
            meta = json.load(f)
        strain = meta.get("strain_type", "none")
        disloc = meta.get("dislocation_type", "none")
        if strain != "none" or disloc != "none":
            sid = meta["sample_id"]
            print(
                f"  {sid:<6d} {meta['material']:<12} {meta['shape']:<10} "
                f"{strain:<18} {disloc:<8} {meta.get('hkl','?')}"
            )
            found += 1
            if found >= limit:
                break

    print(f"{'='*76}")
    print(f"  Found {found} samples. Run one of these for a meaningful test:")
    print("    python nn_phase_retrieval.py \\")
    print(f"      --input {data_dir}/sample_XXXXX.npz --mode hybrid \\")
    print("      --output reconstruction_strained.npz")
    print(f"{'='*76}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing: remove trivial ambiguities
# ═══════════════════════════════════════════════════════════════════════════════


def remove_global_phase(phase: np.ndarray, support: np.ndarray) -> np.ndarray:
    """
    Remove global phase offset (trivial phase-retrieval ambiguity).

    The diffraction pattern |F(q)|² is invariant under ρ(r) → ρ(r)·e^{iφ₀},
    so phase is only recoverable up to a constant. Subtract the circular mean
    over the support so comparisons are meaningful.
    """
    mask = support > 0.5
    if mask.sum() == 0:
        return phase
    mean_phase = np.angle(np.mean(np.exp(1j * phase[mask])))
    unwrapped = phase - mean_phase
    return np.angle(np.exp(1j * unwrapped))  # Re-wrap to [-π, π]


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Real-space reconstruction overview
# ═══════════════════════════════════════════════════════════════════════════════


def fig_overview(recon: dict, save_path: str = None):
    support = recon.get("support", np.ones_like(recon["phase"]))
    phase_clean = remove_global_phase(recon["phase"], support)
    N = recon["phase"].shape[0]
    c = N // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Reconstructed Crystal (Real Space — Central Slice)", fontsize=13, y=1.02
    )

    im0 = axes[0].imshow(
        recon["amplitude"][:, :, c], cmap="hot", origin="lower", aspect="equal"
    )
    axes[0].set_title("Electron density  |ρ(r)|", fontsize=11)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    phase_slice = phase_clean[:, :, c] * (support[:, :, c] > 0.5)
    im1 = axes[1].imshow(
        phase_slice,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
        aspect="equal",
    )
    axes[1].set_title("Phase  φ(r)  [global offset removed]", fontsize=11)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    cb1 = plt.colorbar(
        im1, ax=axes[1], shrink=0.8, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    )
    cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

    im2 = axes[2].imshow(
        support[:, :, c], cmap="Blues", origin="lower", aspect="equal", vmin=0, vmax=1
    )
    axes[2].set_title("Support mask", fontsize=11)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Diffraction-space comparison (the RIGHT one)
# ═══════════════════════════════════════════════════════════════════════════════


def fig_diffraction_comparison(recon: dict, truth: dict, save_path: str = None):
    """
    FFT(reconstructed_object) vs measured amplitude.

    This is apples-to-apples: both are reciprocal-space amplitudes.
    """
    obj = recon["object_3d"]
    F_recon = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
    amp_recon_q = np.abs(F_recon).astype(np.float32)
    amp_true_q = truth["amplitude"].astype(np.float32)

    amp_recon_q = amp_recon_q / max(amp_recon_q.max(), 1e-12)
    amp_true_q = amp_true_q / max(amp_true_q.max(), 1e-12)

    N = amp_recon_q.shape[0]
    c = N // 2

    def log_n(x):
        return np.log10(x + 1e-5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Diffraction Pattern: Measured vs Reconstructed (reciprocal space)",
        fontsize=12,
        y=0.99,
    )

    slices = [
        ("qx-qy", lambda a: a[:, :, c]),
        ("qx-qz", lambda a: a[:, c, :]),
        ("qy-qz", lambda a: a[c, :, :]),
    ]

    for col, (name, getter) in enumerate(slices):
        t_sl = log_n(getter(amp_true_q))
        r_sl = log_n(getter(amp_recon_q))
        vmin = min(t_sl.min(), r_sl.min())
        vmax = max(t_sl.max(), r_sl.max())

        im0 = axes[0, col].imshow(
            t_sl, cmap="jet", origin="lower", vmin=vmin, vmax=vmax, aspect="equal"
        )
        axes[0, col].set_title(f"Measured  ({name})", fontsize=10)
        axes[0, col].tick_params(labelsize=7)

        im1 = axes[1, col].imshow(
            r_sl, cmap="jet", origin="lower", vmin=vmin, vmax=vmax, aspect="equal"
        )
        axes[1, col].set_title(f"Reconstructed FFT  ({name})", fontsize=10)
        axes[1, col].tick_params(labelsize=7)

    r = np.sum(np.abs(amp_true_q - amp_recon_q)) / np.sum(amp_true_q)
    fig.text(
        0.5,
        0.02,
        f"Global reciprocal-space R-factor = {r:.4f}",
        ha="center",
        fontsize=11,
        color="#1f6feb",
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Phase comparison with offset removed
# ═══════════════════════════════════════════════════════════════════════════════


def fig_phase_comparison(recon: dict, truth: dict, save_path: str = None):
    support = recon.get("support", truth["support"]).astype(np.float32)

    phase_recon = remove_global_phase(recon["phase"], support)
    phase_true = remove_global_phase(truth["phase_true"], truth["support"])

    ps = phase_recon * (support > 0.5)
    pt = phase_true * (truth["support"] > 0.5)

    perr = np.angle(np.exp(1j * (phase_recon - phase_true)))
    perr = perr * (support > 0.5) * (truth["support"] > 0.5)

    common = (support > 0.5) & (truth["support"] > 0.5)
    if common.sum() > 10:
        rmse = np.sqrt(np.mean(perr[common] ** 2))
        pr_vals = phase_recon[common]
        pt_vals = phase_true[common]
        if pt_vals.std() < 1e-6:
            corr_str = "N/A (ground truth phase is constant)"
        else:
            corr = np.corrcoef(pr_vals, pt_vals)[0, 1]
            corr_str = f"{corr:.3f}"
    else:
        rmse = float("nan")
        corr_str = "N/A"

    N = phase_recon.shape[0]
    c = N // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Phase Comparison  |  RMSE = {rmse:.3f} rad  |  correlation = {corr_str}",
        fontsize=12,
    )

    kw = dict(cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower", aspect="equal")
    titles = [
        "Ground truth φ (offset removed)",
        "Reconstructed φ (offset removed)",
        "Error map  φ_recon − φ_true",
    ]
    data = [pt[:, :, c], ps[:, :, c], perr[:, :, c]]

    for ax, dd, tt in zip(axes, data, titles):
        im = ax.imshow(dd, **kw)
        ax.set_title(tt, fontsize=10)
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_ticks([-np.pi, 0, np.pi])
        cb.set_ticklabels(["-π", "0", "π"])

    meta = truth.get("_meta", {})
    if (
        meta.get("strain_type", "none") == "none"
        and meta.get("dislocation_type", "none") == "none"
    ):
        fig.text(
            0.5,
            0.01,
            "⚠ This sample has no strain and no dislocation — ground truth phase is constant. "
            "Try a strained sample for a meaningful test.",
            ha="center",
            fontsize=10,
            color="#f0883e",
            style="italic",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Convergence with annotations
# ═══════════════════════════════════════════════════════════════════════════════


def fig_convergence(recon: dict, save_path: str = None):
    if "error_metric" not in recon or len(recon["error_metric"]) == 0:
        return
    errors = np.asarray(recon["error_metric"])
    n = len(errors)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogy(errors, color="#3fb950", linewidth=2)

    raar_end = int(n * 0.83)  # 100 RAAR + 20 ER default split
    ax.axvspan(0, raar_end, alpha=0.08, color="#f0883e", label="RAAR phase")
    ax.axvspan(raar_end, n, alpha=0.15, color="#4f98a3", label="ER polish")
    ax.axhline(
        errors[-1],
        color="#f0883e",
        linestyle="--",
        alpha=0.7,
        label=f"Final R = {errors[-1]:.4f}",
    )

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("R-factor", fontsize=11)
    ax.set_title("Phase retrieval convergence", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend()

    if errors.max() > 1.0:
        ax.annotate(
            "RAAR lets intensity leak outside\nsupport — normal with β=0.87",
            xy=(raar_end * 0.7, errors.max() * 0.7),
            xytext=(raar_end * 0.25, errors.max() * 0.25),
            fontsize=9,
            color="#888",
            arrowprops=dict(arrowstyle="->", color="#888", alpha=0.5),
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Visualize BCDI reconstruction")
    parser.add_argument("--input", type=str, default="reconstruction.npz")
    parser.add_argument("--ground_truth", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--find_strained",
        action="store_true",
        help="List training samples with non-trivial phase",
    )
    parser.add_argument("--data_dir", type=str, default="./training_data")
    args = parser.parse_args()

    if args.find_strained:
        find_strained_samples(args.data_dir)
        return

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        sp = lambda n: str(Path(args.save_dir) / n)
    else:
        sp = lambda _: None

    print("\n" + "=" * 60)
    print("  BCDI Reconstruction Visualizer")
    print("=" * 60)

    recon = load_reconstruction(args.input)
    truth = load_ground_truth(args.ground_truth) if args.ground_truth else None

    print("\n[1/4] Reconstruction overview...")
    fig_overview(recon, sp("1_overview.png"))

    print("[2/4] Convergence...")
    fig_convergence(recon, sp("2_convergence.png"))

    if truth is not None:
        print("[3/4] Diffraction-space comparison (reciprocal)...")
        fig_diffraction_comparison(recon, truth, sp("3_diffraction.png"))

        print("[4/4] Phase comparison (real space)...")
        fig_phase_comparison(recon, truth, sp("4_phase.png"))
    else:
        print("[3/4] (skipped — no --ground_truth provided)")

    print("\nDone.")


if __name__ == "__main__":
    main()
