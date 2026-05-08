"""
nn_data_generator.py — Generate training data for NN-accelerated BCDI phase retrieval.

Uses bcdi_core.py to create thousands of (support, phase, diffraction_pattern) triplets
by randomizing:
  - Material (from MATERIAL_PRESETS)
  - Crystal shape (cube, sphere, cylinder, hexagonal)
  - Supercell size (controls particle size → fringe spacing)
  - Strain type and magnitude
  - Dislocation parameters
  - Reflection (hkl)

Each sample is saved as a compressed .npz file containing:
  - 'amplitude': sqrt(diffraction_intensity), shape [N, N, N]
  - 'phase_true': the ground-truth phase field, shape [N, N, N]
  - 'support': binary support mask, shape [N, N, N]
  - 'diffraction': the full diffraction volume, shape [N, N, N]
  - 'metadata': dict with generation parameters

Usage:
    python nn_data_generator.py --output_dir ./training_data --num_samples 2000 --grid_size 64
"""

from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
from pathlib import Path
from typing import Optional

# ── Import from your existing bcdi_core ──────────────────────────────────────
from .bcdi_core import (
    MATERIAL_PRESETS, BCDIConfig, CrystalBuilder, ReflectionCalculator,
    BCDISimulator, DislocationConfig, add_experimental_noise,
    default_shape_for_material, compatible_shapes
)


# ═══════════════════════════════════════════════════════════════════════════════
# Random parameter samplers
# ═══════════════════════════════════════════════════════════════════════════════

def random_material(rng: np.random.Generator) -> str:
    """Pick a random material from presets."""
    names = list(MATERIAL_PRESETS.keys())
    return rng.choice(names)


def random_shape(material_name: str, rng: np.random.Generator) -> str:
    """Pick a compatible random shape for the material."""
    shapes = compatible_shapes(material_name)
    return rng.choice(shapes)


def random_supercell(rng: np.random.Generator, min_n: int = 10, max_n: int = 30) -> tuple:
    """Random supercell multipliers. Keeps aspect ratio reasonable."""
    nx = rng.integers(min_n, max_n + 1)
    ny = rng.integers(min_n, max_n + 1)
    nz = rng.integers(min_n, max_n + 1)
    return (int(nx), int(ny), int(nz))


def random_strain(rng: np.random.Generator) -> tuple:
    """Random strain type and magnitude."""
    strain_types = ['none', 'none', 'radial_gradient', 'edge_dislocation', 'random']
    # Weight 'none' more heavily for balanced training
    stype = rng.choice(strain_types)
    if stype == 'none':
        return stype, 0.0
    mag = float(10 ** rng.uniform(-5, -3))  # 1e-5 to 1e-3
    return stype, mag


def random_dislocation(rng: np.random.Generator, lattice_a: float) -> Optional[DislocationConfig]:
    """Randomly decide whether to add a dislocation, and configure it."""
    if rng.random() < 0.6:  # 60% no dislocation
        return None
    dtype = rng.choice(['edge', 'screw', 'mixed'])
    px = float(rng.uniform(0.3, 0.7))
    py = float(rng.uniform(0.3, 0.7))
    line_dir = rng.choice(['X', 'Y', 'Z'])
    b = float(lattice_a * rng.uniform(0.8, 1.2))
    nu = float(rng.uniform(0.2, 0.4))
    return DislocationConfig(
        dtype=dtype, pos_frac=(px, py), line_dir=line_dir,
        b_angstrom=b, nu=nu
    )


def random_reflection(reflections_df, rng: np.random.Generator):
    """Pick a reflection, biased toward high-intensity ones."""
    if len(reflections_df) == 0:
        return None
    # Prefer BCDI-flagged reflections
    bcdi = reflections_df[reflections_df['BCDI_flag']]
    if len(bcdi) > 0 and rng.random() < 0.8:
        row = bcdi.iloc[rng.integers(0, len(bcdi))]
    else:
        row = reflections_df.iloc[rng.integers(0, len(reflections_df))]
    return row['hkl']


# ═══════════════════════════════════════════════════════════════════════════════
# Core generation function
# ═══════════════════════════════════════════════════════════════════════════════

def generate_single_sample(
    sample_id: int,
    grid_size: int = 64,
    rng: np.random.Generator = None,
    add_noise: bool = True,
    fixed_material: str = None,
    randomize_dislocation: bool = True,
    randomize_strain: bool = True,
) -> dict:
    """
    Generate one (support, phase, diffraction) triplet.

    Parameters
    ----------
    sample_id : int
        Unique identifier for this sample.
    grid_size : int
        Detector grid size N (diffraction volume is NxNxN).
    rng : numpy random generator
        For reproducibility.
    add_noise : bool
        Whether to add experimental noise to some samples.
    fixed_material : str or None
        If set, only use this material (useful for material-specific training).
    randomize_dislocation : bool
        If True, occasionally add a random line dislocation to the sample.
        If False, all samples are dislocation-free (clean crystals).
    randomize_strain : bool
        If True, occasionally apply random strain (radial, edge, random).
        If False, all samples are strain-free (uniform unit cells).

    Returns
    -------
    dict with keys: amplitude, phase_true, support, diffraction, metadata
    """
    if rng is None:
        rng = np.random.default_rng(sample_id)

    # ── 1. Random configuration ───────────────────────────────────────────
    material = fixed_material if fixed_material else random_material(rng)
    cfg = BCDIConfig(material)

    shape = random_shape(material, rng)
    cfg.PARTICLE_SHAPE = shape
    cfg.SUPERCELL_MULT = random_supercell(rng)
    # DETECTOR_N_PIXELS is a derived @property = max(NX, NY) — set the
    # underlying settable attributes instead.
    cfg.DETECTOR_NX = grid_size
    cfg.DETECTOR_NY = grid_size

    strain_type, strain_mag = random_strain(rng)
    if not randomize_strain:
        strain_type, strain_mag = 'none', 0.0
    cfg.STRAIN_TYPE = strain_type
    cfg.STRAIN_MAGNITUDE = strain_mag

    if randomize_dislocation:
        disloc = random_dislocation(rng, cfg.LATTICE_A)
    else:
        disloc = None
    cfg.DISLOCATION = disloc

    # Randomize oversampling slightly
    cfg.TARGET_OVERSAMPLING = float(rng.uniform(3.0, 7.0))

    # ── 2. Build crystal ──────────────────────────────────────────────────
    builder = CrystalBuilder(cfg)
    builder.build()
    builder.apply_shape_filter()
    if cfg.DISLOCATION:
        builder.apply_dislocation_displacement()

    # ── 3. Calculate reflections and pick one ──────────────────────────────
    rc = ReflectionCalculator(cfg, builder)
    df = rc.calculate()
    if len(df) == 0:
        return None  # Skip if no valid reflections

    hkl = random_reflection(df, rng)
    if hkl is None:
        return None
    refl = rc.select_reflection(hkl)

    # ── 4. Run forward simulation ─────────────────────────────────────────
    sim = BCDISimulator(cfg, refl, builder)
    sim.simulate()

    if sim.diff_volume is None:
        return None

    # ── 5. Extract the ground-truth phase and support ─────────────────────
    N = grid_size
    tA = cfg.particle_size_angstrom
    OS = cfg.TARGET_OVERSAMPLING

    # Reconstruct the support mask (same logic as simulator)
    support = sim._sup(N, OS).astype(np.float32)

    # Reconstruct the phase field (same logic as simulator)
    phase_true = sim._displacement(N, tA, OS)

    # The diffraction pattern
    diffraction = sim.diff_volume.copy()

    # Optionally add noise
    noisy = False
    if add_noise and rng.random() < 0.3:
        noise_opts = {}
        if rng.random() < 0.7:
            noise_opts['poisson'] = True
        if rng.random() < 0.3:
            noise_opts['readout_noise'] = float(rng.uniform(1, 10))
        if rng.random() < 0.2:
            noise_opts['air_scatter'] = float(rng.uniform(10, 200))
        if noise_opts:
            diffraction = add_experimental_noise(diffraction, **noise_opts,
                                                  seed=sample_id)
            noisy = True

    # Amplitude = sqrt(intensity)
    amplitude = np.sqrt(np.maximum(diffraction, 0)).astype(np.float32)

    # ── 6. Build metadata ─────────────────────────────────────────────────
    metadata = {
        'sample_id': sample_id,
        'material': material,
        'shape': shape,
        'supercell': list(cfg.SUPERCELL_MULT),
        'grid_size': grid_size,
        'strain_type': strain_type,
        'strain_magnitude': float(strain_mag),
        'has_dislocation': disloc is not None,
        'dislocation_type': disloc.dtype if disloc else 'none',
        'hkl': list(hkl),
        'oversampling': float(cfg.TARGET_OVERSAMPLING),
        'noisy': noisy,
        'particle_size_nm': cfg.particle_size_nm.tolist(),
    }

    return {
        'amplitude': amplitude,
        'phase_true': phase_true,
        'support': support,
        'diffraction': diffraction.astype(np.float32),
        'metadata': metadata,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Batch generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    output_dir: str,
    num_samples: int = 2000,
    grid_size: int = 64,
    seed: int = 42,
    fixed_material: str = None,
    add_noise: bool = True,
    resume: bool = True,
    randomize_dislocation: bool = True,
    randomize_strain: bool = True,
):
    """
    Generate a full training dataset.

    Creates output_dir/sample_XXXXX.npz files and a manifest.json.

    Parameters
    ----------
    output_dir : str
        Directory to save .npz files.
    num_samples : int
        Total number of samples to generate.
    grid_size : int
        NxNxN grid for each diffraction volume.
    seed : int
        Base random seed for reproducibility.
    fixed_material : str or None
        Lock to one material for focused training.
    add_noise : bool
        Whether to add noise to some samples.
    resume : bool
        Skip already-generated samples.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    generated = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    print(f"{'='*60}")
    print(f"BCDI Training Data Generator")
    print(f"{'='*60}")
    print(f"  Output:     {out.resolve()}")
    print(f"  Samples:    {num_samples}")
    print(f"  Grid size:  {grid_size}³")
    print(f"  Material:   {fixed_material or 'random (all presets)'}")
    print(f"  Noise:      {'yes (30% of samples)' if add_noise else 'no'}")
    print(f"  Seed:       {seed}")
    print(f"{'='*60}\n")

    for i in range(num_samples):
        fname = f"sample_{i:05d}.npz"
        fpath = out / fname

        # Resume support
        if resume and fpath.exists():
            skipped += 1
            continue

        rng = np.random.default_rng(seed + i)

        try:
            result = generate_single_sample(
                sample_id=i,
                grid_size=grid_size,
                rng=rng,
                add_noise=add_noise,
                fixed_material=fixed_material,
                randomize_dislocation=randomize_dislocation,
                randomize_strain=randomize_strain,
            )

            if result is None:
                failed += 1
                continue

            # Save compressed
            # Real-space voxel pitch (nm) — needed by reconstruction GUI
            # voxel pitch = (object size × oversampling) / grid_size
            psize_arr = np.asarray(cfg.particle_size_nm, dtype=np.float32)
            voxel_size_nm = (psize_arr * float(cfg.TARGET_OVERSAMPLING)
                              / float(grid_size))
            np.savez_compressed(
                fpath,
                amplitude=result['amplitude'],
                phase_true=result['phase_true'],
                support=result['support'],
                diffraction=result['diffraction'],
                voxel_size_nm=voxel_size_nm,
                particle_size_nm=psize_arr,
            )

            # Save metadata separately for fast loading
            meta_path = out / f"sample_{i:05d}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(result['metadata'], f, indent=2)

            manifest.append(result['metadata'])
            generated += 1

            # Progress
            elapsed = time.time() - t0
            rate = generated / max(elapsed, 1)
            eta = (num_samples - i - 1) / max(rate, 0.01)

            if (generated % 10 == 0) or generated == 1:
                m = result['metadata']
                print(
                    f"  [{generated:4d}/{num_samples}] "
                    f"{m['material']:12s} {m['shape']:10s} "
                    f"hkl={m['hkl']}  strain={m['strain_type']:16s} "
                    f"disl={'Y' if m['has_dislocation'] else 'N'}  "
                    f"noise={'Y' if m['noisy'] else 'N'}  "
                    f"({rate:.1f} samp/s, ETA {eta/60:.0f}min)"
                )

        except Exception as e:
            failed += 1
            if failed < 20:
                print(f"  [FAIL] Sample {i}: {e}")

    # Save manifest
    manifest_path = out / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            'num_samples': generated,
            'grid_size': grid_size,
            'seed': seed,
            'samples': manifest
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Generated: {generated}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Manifest:  {manifest_path}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate BCDI training data for NN phase retrieval'
    )
    parser.add_argument('--output_dir', type=str, default='./training_data',
                        help='Output directory for .npz files')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of samples to generate')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid size N (volume is NxNxN)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--material', type=str, default=None,
                        help='Lock to a specific material (e.g., "Au")')
    parser.add_argument('--no_noise', action='store_true',
                        help='Disable noise augmentation')
    parser.add_argument('--no_resume', action='store_true',
                        help='Regenerate all samples from scratch')
    parser.add_argument('--no_dislocations', action='store_true',
                        help='Disable random dislocations (all samples are clean)')
    parser.add_argument('--no_strain', action='store_true',
                        help='Disable random strain (all samples are unstrained)')
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        grid_size=args.grid_size,
        seed=args.seed,
        fixed_material=args.material,
        add_noise=not args.no_noise,
        resume=not args.no_resume,
        randomize_dislocation=not args.no_dislocations,
        randomize_strain=not args.no_strain,
    )
