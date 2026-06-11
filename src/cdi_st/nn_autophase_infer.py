"""
nn_autophase_infer.py — Inference and hybrid refinement for AutoPhaseNet3D.

Runs the trained unsupervised network on new data (simulated or experimental),
optionally followed by iterative refinement (ER + RAAR) from your existing
nn_phase_retrieval module.

Three modes:
    1. 'nn_only'  — just the network forward pass (~100 ms, like AutoPhaseNN Fig 3b)
    2. 'refined'  — NN prediction + short refinement (AutoPhaseNN + your RAAR)
    3. 'compare'  — run all modes and compare quality metrics

Works with:
    - .npz files from your data generator (any sample)
    - .npz files converted from experiment via nn_experimental_loader.py
    - .h5 files directly (ID01 / 34-ID-C format, auto-loaded)

Usage:
    # On a simulated test sample:
    python nn_autophase_infer.py \\
        --input training_data/sample_00123.npz \\
        --model checkpoints_autophase/best_model.pt \\
        --output recon_autophase.npz \\
        --mode refined

    # On an experimental .h5 file:
    python nn_autophase_infer.py \\
        --input scan_042.h5 \\
        --model checkpoints_autophase/best_model.pt \\
        --output recon_exp.npz \\
        --mode refined

    # Compare NN-only vs refined:
    python nn_autophase_infer.py --input data.npz --model best.pt --mode compare
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from .nn_autophase_model import AutoPhaseNet3D


# ═══════════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AutoPhaseResult:
    object_3d: np.ndarray          # complex, [N,N,N]
    amplitude: np.ndarray          # |ρ|, [N,N,N]
    phase: np.ndarray              # φ ∈ [-π, π], [N,N,N]
    support: np.ndarray            # dynamic support from amplitude threshold
    error_metric: List[float]      # R-factor (or χ² per AutoPhaseNN)
    method: str
    elapsed_seconds: float


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (supports both .npz and .h5)
# ═══════════════════════════════════════════════════════════════════════════════

def load_input(path: str, target_size: int = 64,
                apply_gap_mask: bool = True) -> tuple:
    """
    Load a diffraction volume from .npz or .h5.

    Returns (diffraction, truth_dict_or_none, voxel_size_nm_or_none).
    `truth_dict` contains phase_true/support if available (simulated data).
    `voxel_size_nm` is a 3-vector (per axis) if available, else None.

    If `apply_gap_mask` and the data has NO ground truth (i.e. it's
    experimental rather than simulated), runs the detector-gap heuristic
    on the loaded volume. This catches multi-chip detector gaps that
    survived an earlier .npz conversion.
    """
    p = Path(path)

    if p.suffix == '.npz':
        data = np.load(p)
        if 'diffraction' in data:
            diff = data['diffraction'].astype(np.float32)
        elif 'diffraction_volume' in data:
            diff = data['diffraction_volume'].astype(np.float32)
        elif 'amplitude' in data:
            diff = (data['amplitude'].astype(np.float32)) ** 2
        else:
            raise KeyError(f"No diffraction data in {path}. Keys found: {list(data.keys())}")

        truth = None
        if 'phase_true' in data and 'support' in data:
            truth = {
                'phase_true': data['phase_true'],
                'support': data['support'],
            }

        # If this looks like experimental data (no ground truth), check
        # for detector gaps and fill them. Simulated data typically has
        # no zero-strips so this no-ops.
        if apply_gap_mask and truth is None:
            try:
                from cdi_st.nn_experimental_loader import mask_detector_gaps
                before_zero_frac = float((diff <= 0).mean())
                diff = mask_detector_gaps(diff, detector='auto')
                after_zero_frac = float((diff <= 0).mean())
                if before_zero_frac - after_zero_frac > 0.001:
                    print(f"[load_input] Filled detector gaps "
                          f"({100*(before_zero_frac - after_zero_frac):.1f}% of voxels)")
            except Exception as e:
                print(f"[load_input] Gap masking skipped: {e}")

        # Voxel pitch in nm if recorded
        voxel_nm = None
        if 'voxel_size_nm' in data:
            voxel_nm = np.asarray(data['voxel_size_nm'], dtype=np.float32)
        return diff, truth, voxel_nm

    elif p.suffix == '.h5':
        from cdi_st.nn_experimental_loader import load_h5_diffraction
        diff = load_h5_diffraction(path, target_size=target_size)
        return diff, None, None

    else:
        raise ValueError(f"Unsupported format: {p.suffix}")


# ═══════════════════════════════════════════════════════════════════════════════
# NN-only inference
# ═══════════════════════════════════════════════════════════════════════════════

def nn_only_infer(
    diffraction: np.ndarray,
    model_path: str,
    base_channels: int = 32,
    device: str = None,
    enforce_oversampling: bool = True,
    support_threshold: float = 0.05,
    support_method: str = 'percentile',
) -> AutoPhaseResult:
    """
    Run the trained AutoPhaseNet3D in pure prediction mode (no refinement).

    Improvements vs naive amplitude-threshold support:
    - 'percentile' support method: keeps the top X% of voxels by amplitude,
      which preserves SHAPE (cube stays cube, hexagon stays hexagon) instead
      of forcing roundness through threshold-based truncation.
    - Crystal-aware target support fraction estimated from autocorrelation
      of the measured diffraction (gives realistic crystal volume).

    support_method:
        'percentile' — keep top N voxels (N estimated from autocorrelation)
                       Best for general use, preserves shape best.
        'threshold'  — keep voxels above support_threshold * amp.max()
                       Original behavior, can clip cubes/hexagons.
    """
    t0 = time.time()
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load model
    model = AutoPhaseNet3D(base_channels=base_channels,
                            enforce_oversampling=enforce_oversampling).to(device)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Preprocess input (matches training preprocessing)
    diff = np.maximum(diffraction, 0).astype(np.float32)
    magnitude = np.sqrt(diff)
    log_mag = np.log10(magnitude + 1.0)
    norm_scale = log_mag.max()
    if norm_scale > 0:
        log_mag = log_mag / norm_scale

    # ── Detect grid-size mismatch with trained model ────────────────────
    # The dual-decoder net has explicit zero-pad for central N/2; if the input
    # grid is much larger or smaller than the training grid, the network's
    # learned features are spatially miscalibrated and produces garbage with
    # an empty center (the dreaded "cross artifact").
    N_input = diffraction.shape[0]
    expected_N = ckpt.get('grid_size', None)
    if expected_N is not None and expected_N != N_input:
        # Real mismatch — useful warning
        print(f"[nn_only] WARNING: grid size mismatch! Model trained at {expected_N}\u00b3, "
              f"input is {N_input}\u00b3. The GUI auto-resamples to fix this — "
              f"if you're calling nn_only_infer directly, downsample first.")
    # If expected_N is None (old checkpoint without metadata), do NOT print
    # anything. The model is fully convolutional and runs fine at any size;
    # the print was alarming users unnecessarily.

    x = torch.from_numpy(log_mag[None, None]).float().to(device)

    # Forward pass
    with torch.no_grad():
        amp_pred, phase_pred = model(x)

    amp = amp_pred[0, 0].cpu().numpy().astype(np.float32)
    phase = phase_pred[0, 0].cpu().numpy().astype(np.float32) * np.pi  # → radians

    # ── Twin-image suppression ───────────────────────────────────────────
    # When AutoPhaseNet outputs near-constant phase (which is common for
    # unfamiliar shapes), the result is essentially IFFT(|F|), which contains
    # both the object AND its centro-symmetric twin (Fienup 1982). The 3D
    # view then shows two parallel slabs separated by a gap — and the central
    # slices land IN the gap, looking empty.
    #
    # Standard BCDI fix: identify the object hemisphere by total amplitude and
    # zero out the other half. Then center the surviving object.
    try:
        from scipy.ndimage import center_of_mass, shift as nd_shift, label
        N = amp.shape[0]
        amp_max = max(amp.max(), 1e-12)

        # First check if there are multiple disconnected blobs above threshold
        # — if so, this is twin-image and we keep only the brightest.
        binary = (amp > 0.20 * amp_max).astype(np.int32)
        if binary.sum() > 5:
            labels, n_blobs = label(binary)
            if n_blobs > 1:
                # Multiple blobs detected — likely twin image
                blob_amplitudes = []
                for blob_id in range(1, n_blobs + 1):
                    mask = (labels == blob_id)
                    blob_amplitudes.append((mask.sum() * float(amp[mask].mean()), blob_id))
                blob_amplitudes.sort(reverse=True)
                # Keep only the top blob
                kept_id = blob_amplitudes[0][1]
                kept_mask = (labels == kept_id).astype(np.float32)
                # Soft mask: erode then dilate to keep some neighborhood
                from scipy.ndimage import binary_dilation
                kept_mask = binary_dilation(kept_mask > 0.5, iterations=2).astype(np.float32)
                amp = amp * kept_mask
                # Phase outside kept region: zero (will get filtered by support later)
                phase = phase * kept_mask
                print(f"[nn_only] Twin-image suppression: kept 1 of {n_blobs} blobs "
                      f"(removed {n_blobs - 1} centro-symmetric or noise blobs)")

        # ── Now shift the surviving object to grid center ────────────────
        sup_for_com = (amp > 0.10 * max(amp.max(), 1e-12)).astype(np.float32)
        if sup_for_com.sum() > 5:
            com = np.array(center_of_mass(sup_for_com))
            target = np.array(amp.shape) / 2.0
            shift_vec = target - com
            if np.linalg.norm(shift_vec) > 1.0:
                amp = nd_shift(amp, shift_vec, order=1, mode='constant', cval=0)
                cplx = np.exp(1j * phase) * (amp > 0).astype(np.float32)
                cplx_real = nd_shift(cplx.real, shift_vec, order=1,
                                      mode='constant', cval=0)
                cplx_imag = nd_shift(cplx.imag, shift_vec, order=1,
                                      mode='constant', cval=0)
                phase = np.angle(cplx_real + 1j * cplx_imag)
    except ImportError:
        pass

    # ── Build support ────────────────────────────────────────────────────
    n_total = amp.size

    if support_method == 'percentile':
        # Estimate the target support size from the autocorrelation of measured
        # diffraction. Use multiple AC thresholds and pick the one giving the
        # most plausible object size.
        auto_corr = np.abs(np.fft.fftshift(np.fft.ifftn(
            np.fft.ifftshift(magnitude ** 2))))
        ac_max = max(auto_corr.max(), 1e-12)

        # Find AC volume at multiple thresholds and average — robust estimate
        ac_sizes = []
        for thr in [0.10, 0.20, 0.30]:
            n_ac = int((auto_corr > thr * ac_max).sum())
            # Object volume = AC volume / 8 (3D autocorrelation is 2× linearly = 8× volume)
            obj_estimate = n_ac // 8
            if obj_estimate > 50:
                ac_sizes.append(obj_estimate)
        if ac_sizes:
            target_n = int(np.median(ac_sizes))
        else:
            target_n = max(int(n_total * 0.02), 100)  # 2% fallback
        target_n = min(target_n, int(n_total * 0.15))  # cap at 15% of grid

        # Keep top target_n voxels of the NN amplitude
        amp_sorted = np.sort(amp.ravel())[::-1]
        threshold_value = amp_sorted[min(target_n, len(amp_sorted) - 1)]
        support_raw = (amp >= threshold_value)
    else:
        # Classic threshold method (legacy)
        amp_max = max(amp.max(), 1e-12)
        support_raw = (amp > support_threshold * amp_max)
        # Cap at 15% (was 25% — too loose)
        if support_raw.sum() > n_total * 0.15:
            n_keep = int(n_total * 0.15)
            thr = np.sort(amp.ravel())[::-1][n_keep]
            support_raw = amp >= thr

    # Morphological cleanup
    try:
        from scipy.ndimage import binary_closing, binary_dilation, binary_opening
        # Closing fills small gaps inside (good for cubes)
        support = binary_closing(support_raw, iterations=2)
        # Opening removes isolated noise voxels
        support = binary_opening(support, iterations=1)
        # If opening killed everything, fall back to original
        if support.sum() < 20:
            support = binary_closing(support_raw, iterations=1)
        support = support.astype(np.float32)
    except ImportError:
        support = support_raw.astype(np.float32)

    # Build complex object at NN-predicted scale (amp ∈ [0,1])
    obj_unscaled = amp * support * np.exp(1j * phase)
    F_unscaled = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj_unscaled)))
    F_max = float(np.abs(F_unscaled).max())

    # Scale object so its FFT amplitude matches measured magnitude max
    if F_max > 1e-12 and magnitude.max() > 0:
        rescale = magnitude.max() / F_max
        obj = obj_unscaled * rescale
        amp = amp * rescale
    else:
        obj = obj_unscaled

    # Compute BOTH absolute R-factor (for backward compat) and normalized
    # diffraction R-factor (the meaningful AutoPhaseNN-style metric)
    F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
    pred_mag = np.abs(F)
    r_abs = np.sum(np.abs(pred_mag - magnitude)) / max(np.sum(magnitude), 1e-12)
    # Normalized R (max=1 on both sides, what AutoPhaseNN trained on)
    pred_n = pred_mag / max(pred_mag.max(), 1e-12)
    meas_n = magnitude / max(magnitude.max(), 1e-12)
    r_norm = float(np.sum(np.abs(pred_n - meas_n)) / max(np.sum(meas_n), 1e-12))

    # Use normalized R as the primary metric (more meaningful)
    return AutoPhaseResult(
        object_3d=obj, amplitude=amp, phase=phase, support=support,
        error_metric=[r_norm], method='nn_only',
        elapsed_seconds=time.time() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid refinement (NN + RAAR + ER from your existing module)
# ═══════════════════════════════════════════════════════════════════════════════

def _center_object(obj_3d, amp, phase, support, voxel_threshold=0.10):
    """
    Shift the reconstruction so its center-of-mass lands on the grid center.

    Phase retrieval algorithms have a translation ambiguity: any spatial
    shift of the object produces the same diffraction magnitude. Without
    explicit centering, the object can end up anywhere in the volume —
    making the central XY/XZ/YZ slices show empty regions.

    Returns the shifted (object_3d, amplitude, phase, support).
    """
    from scipy.ndimage import center_of_mass, shift as nd_shift
    import numpy as _np

    sup_mask = (support > 0.5)
    if sup_mask.sum() < 10:
        return obj_3d, amp, phase, support

    # Use amplitude-weighted COM inside the support (more robust than
    # binary COM when the amplitude has a defined peak).
    weight = amp * sup_mask
    if weight.sum() < 1e-12:
        return obj_3d, amp, phase, support
    com = _np.array(center_of_mass(weight))
    target = _np.array(amp.shape) / 2.0
    shift_vec = target - com

    # Skip if already close to center (less than 1 voxel)
    if _np.linalg.norm(shift_vec) < 1.0:
        return obj_3d, amp, phase, support

    # Shift complex object using its real and imaginary parts
    obj_real = nd_shift(obj_3d.real, shift_vec, order=1, mode='constant', cval=0)
    obj_imag = nd_shift(obj_3d.imag, shift_vec, order=1, mode='constant', cval=0)
    obj_new = obj_real + 1j * obj_imag

    amp_new = nd_shift(amp, shift_vec, order=1, mode='constant', cval=0)
    sup_new = (nd_shift(support.astype(_np.float32), shift_vec,
                          order=1, mode='constant', cval=0) > 0.5).astype(_np.float32)
    phase_new = _np.angle(obj_new)

    print(f"  [center] Shifted by ({shift_vec[0]:.1f}, {shift_vec[1]:.1f}, "
          f"{shift_vec[2]:.1f}) voxels to center COM")

    return obj_new, amp_new, phase_new, sup_new


def refined_infer(
    diffraction: np.ndarray,
    model_path: str,
    base_channels: int = 32,
    n_raar: int = 50,
    n_er: int = 20,
    n_hio: int = 50,
    beta: float = 0.9,
    device: str = None,
    enforce_oversampling: bool = True,
    support_threshold: float = 0.05,
    use_shrink_wrap: bool = True,
    max_support_fraction: float = 0.20,
    algorithm: str = 'hybrid',
) -> AutoPhaseResult:
    """
    AutoPhaseNet prediction + iterative refinement (HIO + RAAR + ER).

    Why this works when "RAAR + ER alone" fails on a partial NN solution:

    1. The NN gives a partial reconstruction (R~0.6) that's stuck in a local
       minimum. RAAR alone can't always escape.

    2. HIO (Fienup 1982) is more aggressive: it uses negative feedback outside
       the support, which is much better at escaping local minima than RAAR.
       Standard practice in BCDI is HIO → RAAR → ER, not RAAR alone.

    3. We preserve the NN's amplitude prediction (not just phase) as the
       starting amplitude. This was being thrown away in the previous version.

    4. We track the best iterate by R-factor and return that one, since
       HIO/RAAR oscillate.

    Schedule (matches Argonne 34-ID-C standard pipeline):
        n_hio iterations of HIO (default 50)   — escape local minima
        n_raar iterations of RAAR (default 50) — smooth refinement
        n_er  iterations of ER (default 20)    — final polish
    """
    t0 = time.time()

    # Step 1: NN prediction
    nn_result = nn_only_infer(
        diffraction, model_path, base_channels, device,
        enforce_oversampling, support_threshold,
    )

    amplitude = np.sqrt(np.maximum(diffraction, 0)).astype(np.float32)
    n_total = nn_result.support.size
    nn_sup_count = int((nn_result.support > 0.5).sum())

    # Step 2: Build SAFE initial support
    nn_support_bool = nn_result.support > 0.5

    if nn_sup_count == 0:
        print("[refined] NN support empty → autocorrelation seed")
        auto_corr = np.abs(np.fft.fftshift(np.fft.ifftn(
            np.fft.ifftshift(amplitude ** 2))))
        auto_thr = 0.10 * auto_corr.max()
        initial_support = (auto_corr > auto_thr)

    elif nn_sup_count > n_total * max_support_fraction:
        print(f"[refined] NN support too large ({nn_sup_count/n_total:.1%}) → "
              "autocorrelation + central N/2 box")
        auto_corr = np.abs(np.fft.fftshift(np.fft.ifftn(
            np.fft.ifftshift(amplitude ** 2))))
        flat_sorted = np.sort(auto_corr.ravel())[::-1]
        k = int(n_total * max_support_fraction)
        auto_thr = flat_sorted[k] if k < len(flat_sorted) else 0
        initial_support = auto_corr > auto_thr
        N = initial_support.shape[0]
        q = N // 4
        center_box = np.zeros_like(initial_support, dtype=bool)
        center_box[q:N-q, q:N-q, q:N-q] = True
        initial_support = initial_support & center_box

    else:
        initial_support = nn_support_bool

    # Final cap
    sup_count = int(initial_support.sum())
    if sup_count > n_total * max_support_fraction:
        n_keep = int(n_total * max_support_fraction)
        scores = nn_result.amplitude.copy()
        scores[~initial_support] = 0
        thr = np.sort(scores.ravel())[::-1][n_keep]
        initial_support = scores >= thr

    # Slight dilation for edges (helps with cubes/rectangles)
    try:
        from scipy.ndimage import binary_dilation, binary_closing
        initial_support = binary_closing(initial_support, iterations=1)
        initial_support = binary_dilation(initial_support, iterations=1)
    except ImportError:
        pass

    initial_support = initial_support.astype(np.float32)
    final_sup_count = int((initial_support > 0.5).sum())
    print(f"[refined] Initial support: {final_sup_count:,} voxels "
          f"({final_sup_count/n_total:.1%} of volume)")

    # Step 3: Run HIO + RAAR + ER (PRESERVING NN AMPLITUDE)
    from cdi_st.nn_phase_retrieval import run_phase_retrieval
    refined = run_phase_retrieval(
        measured_amplitude=amplitude,
        initial_phase=nn_result.phase.astype(np.float32),
        initial_amplitude=nn_result.amplitude.astype(np.float32),  # <- key fix
        support=initial_support,
        n_hio=n_hio,
        n_raar=n_raar,
        n_er=n_er,
        beta=beta,
        algorithm=algorithm,
        use_shrink_wrap=use_shrink_wrap,
        shrink_wrap_interval=10,
        shrink_wrap_threshold=0.10,
        shrink_wrap_sigma=2.0,
        max_support_fraction=max_support_fraction,
    )

    # Sanity check
    final_sup = int((refined.support > 0.5).sum())
    if final_sup > n_total * 0.8:
        print(f"[refined] WARNING: final support is {final_sup/n_total:.1%} of "
              "volume — reconstruction may be degenerate")

    # Recompute refined R-factor on NORMALIZED magnitudes (same metric as nn_only)
    # This makes the fallback comparison fair.
    F_refined = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(refined.object_3d)))
    pred_mag_r = np.abs(F_refined)
    pred_n_r = pred_mag_r / max(pred_mag_r.max(), 1e-12)
    meas_n = amplitude / max(amplitude.max(), 1e-12)
    refined_r_norm = float(np.sum(np.abs(pred_n_r - meas_n)) / max(np.sum(meas_n), 1e-12))

    nn_r_norm = nn_result.error_metric[-1]  # already normalized

    print(f"[refined] NN-only R={nn_r_norm:.4f}   Refined R={refined_r_norm:.4f}")

    # Fallback: keep NN result if refinement made things worse
    if refined_r_norm > nn_r_norm * 1.05:  # only 5% worse triggers fallback
        print(f"[refined] Refinement worse than NN-only — falling back to NN result")
        return AutoPhaseResult(
            object_3d=nn_result.object_3d,
            amplitude=nn_result.amplitude,
            phase=nn_result.phase,
            support=nn_result.support,
            error_metric=[nn_r_norm],
            method='refined\u2192nn_only_fallback',
            elapsed_seconds=time.time() - t0,
        )

    # Replace last R in error history with the normalized R for consistency
    refined_errors = list(refined.error_metric)
    if refined_errors:
        refined_errors[-1] = refined_r_norm

    # NOTE: We do NOT shift the reconstruction here. Phase retrieval has a
    # fundamental translation ambiguity, and shifting via real/imag
    # interpolation degrades the result (the support boundary gets eroded
    # by the linear interpolation + 0.5 threshold). Centering for display
    # is done in the GUI viewer at draw time without modifying the data.

    return AutoPhaseResult(
        object_3d=refined.object_3d,
        amplitude=refined.amplitude,
        phase=refined.phase,
        support=refined.support,
        error_metric=[nn_r_norm] + refined_errors,
        method='refined',
        elapsed_seconds=time.time() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble inference: AutoPhaseNet + supervised PhaseUNet3D, run in parallel
# ═══════════════════════════════════════════════════════════════════════════════

def supervised_nn_infer(
    diffraction: np.ndarray,
    model_path: str,
    base_channels: int = 32,
    device: str = None,
) -> AutoPhaseResult:
    """
    Inference using the SUPERVISED PhaseUNet3D model (predicts phase only,
    using ground-truth phase labels during training).

    This is complementary to AutoPhaseNet3D (unsupervised, dual-decoder).
    The supervised model often has better phase prediction inside the
    support but doesn't predict amplitude — we derive amplitude from
    autocorrelation seed.
    """
    t0 = time.time()
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    from cdi_st.nn_phase_model import PhaseUNet3D

    model = PhaseUNet3D(in_channels=1, base_channels=base_channels).to(device)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Same preprocessing
    diff = np.maximum(diffraction, 0).astype(np.float32)
    magnitude = np.sqrt(diff)
    log_mag = np.log10(magnitude + 1.0)
    norm_scale = log_mag.max()
    if norm_scale > 0:
        log_mag = log_mag / norm_scale

    x = torch.from_numpy(log_mag[None, None]).float().to(device)

    with torch.no_grad():
        phase_pred = model(x)  # [1,1,N,N,N] in [-1,1]

    phase = phase_pred[0, 0].cpu().numpy().astype(np.float32) * np.pi

    # Build amplitude from autocorrelation (this model doesn't predict amp)
    auto_corr = np.abs(np.fft.fftshift(np.fft.ifftn(
        np.fft.ifftshift(magnitude ** 2))))
    n_total = magnitude.size
    n_target = max(int(n_total * 0.05), 200)  # ~5% of grid as crystal
    flat_sorted = np.sort(auto_corr.ravel())[::-1]
    thr = flat_sorted[min(n_target, len(flat_sorted) - 1)]
    support = (auto_corr >= thr).astype(np.float32)

    try:
        from scipy.ndimage import binary_closing
        support = binary_closing(support > 0.5, iterations=2).astype(np.float32)
    except ImportError:
        pass

    # Amplitude seed: use sqrt(autocorrelation) inside support — this is a
    # real-valued density estimate consistent with the diffraction magnitude.
    # (Better than constant-1 inside support: avoids hard top-hat edges.)
    amp_seed = np.sqrt(np.maximum(auto_corr, 0)).astype(np.float32)
    amp_seed = amp_seed * support
    # Normalize so max=1 (we'll rescale below)
    amp_max = max(amp_seed.max(), 1e-12)
    amp = amp_seed / amp_max

    obj_unscaled = amp * np.exp(1j * phase) * support
    F_unscaled = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj_unscaled)))
    F_max = float(np.abs(F_unscaled).max())
    if F_max > 1e-12:
        rescale = magnitude.max() / F_max
        obj = obj_unscaled * rescale
        amp = np.abs(obj)
    else:
        obj = obj_unscaled

    F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
    pred_n = np.abs(F) / max(np.abs(F).max(), 1e-12)
    meas_n = magnitude / max(magnitude.max(), 1e-12)
    r_norm = float(np.sum(np.abs(pred_n - meas_n)) / max(np.sum(meas_n), 1e-12))

    return AutoPhaseResult(
        object_3d=obj, amplitude=amp, phase=phase, support=support,
        error_metric=[r_norm], method='supervised_nn',
        elapsed_seconds=time.time() - t0,
    )


def ensemble_infer(
    diffraction: np.ndarray,
    autophase_model: str,
    supervised_model: str = None,
    base_channels_autophase: int = 32,
    base_channels_supervised: int = 32,
    refine: bool = True,
    n_hio: int = 50,
    n_raar: int = 50,
    n_er: int = 20,
    weight_strategy: str = 'auto',
) -> AutoPhaseResult:
    """
    Ensemble two complementary networks:
        - AutoPhaseNet3D (unsupervised, predicts amplitude AND phase)
        - PhaseUNet3D    (supervised, predicts phase only)

    Then optionally refine with HIO/RAAR/ER.

    Strategy:
        1. Run both networks independently
        2. Combine: average phases (weighted), use AutoPhaseNet's amplitude
           (since supervised model doesn't predict amplitude well)
        3. Build initial support from union of both models' supports
        4. Optionally feed combined object to RAAR refinement

    Why this helps: the two models have different biases. AutoPhaseNet sees
    only the diffraction so it learns physics; supervised model sees ground
    truth so it learns shape priors. Their errors are largely independent,
    so averaging reduces variance.

    weight_strategy:
        'auto'   — weight by inverse R-factor of each model
        'equal'  — 50/50
        'autophase' — only use AutoPhaseNet (fallback if supervised fails)
        'supervised' — only use supervised
    """
    t0 = time.time()

    # Run both models
    print("[ensemble] Running AutoPhaseNet (unsupervised)...")
    auto_result = nn_only_infer(diffraction, autophase_model,
                                  base_channels=base_channels_autophase)
    auto_r = auto_result.error_metric[-1]
    print(f"  AutoPhaseNet R = {auto_r:.4f}")

    if supervised_model is None or weight_strategy == 'autophase':
        print("[ensemble] Skipping supervised model")
        if refine:
            return refined_infer(
                diffraction, autophase_model,
                base_channels=base_channels_autophase,
                n_hio=n_hio, n_raar=n_raar, n_er=n_er,
            )
        return auto_result

    print("[ensemble] Running supervised PhaseUNet3D...")
    try:
        sup_result = supervised_nn_infer(
            diffraction, supervised_model,
            base_channels=base_channels_supervised,
        )
        sup_r = sup_result.error_metric[-1]
        print(f"  Supervised R = {sup_r:.4f}")
    except Exception as e:
        print(f"[ensemble] Supervised model failed: {e}")
        print("[ensemble] Falling back to AutoPhaseNet only")
        if refine:
            return refined_infer(
                diffraction, autophase_model,
                base_channels=base_channels_autophase,
                n_hio=n_hio, n_raar=n_raar, n_er=n_er,
            )
        return auto_result

    if weight_strategy == 'supervised':
        if refine:
            # Use supervised result as init for refinement
            return _refine_from_result(
                sup_result, diffraction,
                n_hio=n_hio, n_raar=n_raar, n_er=n_er,
            )
        return sup_result

    # ── Combine both ────────────────────────────────────────────────────
    # Strategy: AutoPhaseNet provides shape (amplitude + support). Supervised
    # provides phase. Weighted-average phases inside the AutoPhaseNet support.

    # 1. Use AutoPhaseNet's support (it's what actually constrains the object)
    support_combined = (auto_result.support > 0.5).astype(np.float32)

    # 2. Use AutoPhaseNet's amplitude (it predicts amplitude; supervised doesn't)
    amp_combined = auto_result.amplitude.copy()

    # 3. Weighted circular mean of the two phases inside the support
    if weight_strategy == 'auto':
        w_auto = 1.0 / max(auto_r, 1e-3)
        w_sup = 1.0 / max(sup_r, 1e-3)
        total = w_auto + w_sup
        w_auto /= total
        w_sup /= total
    else:
        w_auto = 0.5
        w_sup = 0.5

    print(f"[ensemble] Phase weights: AutoPhase={w_auto:.2f}, Supervised={w_sup:.2f}")

    # Circular mean of phases (only meaningful inside the AutoPhaseNet support;
    # supervised's phase outside its own support is meaningless)
    z_combined = (w_auto * np.exp(1j * auto_result.phase) +
                  w_sup * np.exp(1j * sup_result.phase))
    phase_combined = np.angle(z_combined).astype(np.float32)
    # Outside support, set phase to AutoPhase value (won't matter — amp=0 there)
    out_mask = support_combined < 0.5
    phase_combined[out_mask] = auto_result.phase[out_mask]

    # 4. Build combined complex object
    # CRITICAL: amplitude already has shape info — don't multiply by anything else
    obj = amp_combined * np.exp(1j * phase_combined)

    # 5. Rescale so |F(obj)|.max() matches measured magnitude (consistency)
    F_unscaled = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
    F_max = float(np.abs(F_unscaled).max())
    measured_mag = np.sqrt(np.maximum(diffraction, 0)).astype(np.float32)
    if F_max > 1e-12 and measured_mag.max() > 0:
        rescale_factor = float(measured_mag.max()) / F_max
        obj = obj * rescale_factor
        amp_combined = np.abs(obj).astype(np.float32)

    # 6. Compute combined R-factor (normalized)
    F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
    pred_n = np.abs(F) / max(np.abs(F).max(), 1e-12)
    meas_n = measured_mag / max(measured_mag.max(), 1e-12)
    r_combined = float(np.sum(np.abs(pred_n - meas_n)) / max(np.sum(meas_n), 1e-12))
    print(f"[ensemble] Combined R = {r_combined:.4f}")

    # 7. Defensive: if combined R is much WORSE than the better of the two
    # individual models, keep that better model instead of the combination.
    best_solo_r = min(auto_r, sup_r)
    if r_combined > best_solo_r * 1.10:
        if auto_r <= sup_r:
            print(f"[ensemble] Combined ({r_combined:.4f}) worse than AutoPhase "
                  f"({auto_r:.4f}) — keeping AutoPhaseNet result")
            best_result = auto_result
        else:
            print(f"[ensemble] Combined ({r_combined:.4f}) worse than supervised "
                  f"({sup_r:.4f}) — keeping supervised result")
            best_result = sup_result
        if refine:
            return _refine_from_result(
                best_result, diffraction,
                n_hio=n_hio, n_raar=n_raar, n_er=n_er,
            )
        return best_result

    combined_result = AutoPhaseResult(
        object_3d=obj,
        amplitude=amp_combined,
        phase=phase_combined,
        support=support_combined,
        error_metric=[auto_r, sup_r, r_combined],
        method='ensemble',
        elapsed_seconds=time.time() - t0,
    )

    if not refine:
        return combined_result

    # Refine the combined result
    return _refine_from_result(
        combined_result, diffraction,
        n_hio=n_hio, n_raar=n_raar, n_er=n_er,
    )


def _refine_from_result(seed_result, diffraction, n_hio=50, n_raar=50, n_er=20):
    """Helper: refine an existing AutoPhaseResult with HIO/RAAR/ER."""
    from cdi_st.nn_phase_retrieval import run_phase_retrieval
    amplitude = np.sqrt(np.maximum(diffraction, 0)).astype(np.float32)

    refined = run_phase_retrieval(
        measured_amplitude=amplitude,
        initial_phase=seed_result.phase.astype(np.float32),
        initial_amplitude=seed_result.amplitude.astype(np.float32),
        support=seed_result.support.astype(np.float32),
        n_hio=n_hio,
        n_raar=n_raar,
        n_er=n_er,
        algorithm='hybrid',
        use_shrink_wrap=True,
        shrink_wrap_interval=10,
        shrink_wrap_threshold=0.10,
        shrink_wrap_sigma=2.0,
        max_support_fraction=0.20,
    )

    # Normalized R
    F = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(refined.object_3d)))
    pred_n = np.abs(F) / max(np.abs(F).max(), 1e-12)
    meas_n = amplitude / max(amplitude.max(), 1e-12)
    r_norm = float(np.sum(np.abs(pred_n - meas_n)) / max(np.sum(meas_n), 1e-12))

    seed_r = seed_result.error_metric[-1]
    if r_norm > seed_r * 1.05:
        print(f"[refine] Refinement worse than seed ({r_norm:.4f} vs {seed_r:.4f}), "
              "keeping seed")
        return seed_result

    # NOTE: no shift — see refined_infer for rationale. Centering is done
    # only at display time in the GUI, not on the underlying data.

    return AutoPhaseResult(
        object_3d=refined.object_3d,
        amplitude=refined.amplitude,
        phase=refined.phase,
        support=refined.support,
        error_metric=seed_result.error_metric + [r_norm],
        method='ensemble_refined' if seed_result.method == 'ensemble' else 'refined',
        elapsed_seconds=seed_result.elapsed_seconds,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison mode
# ═══════════════════════════════════════════════════════════════════════════════

def compare_methods(
    diffraction: np.ndarray,
    model_path: str,
    truth: dict = None,
    base_channels: int = 32,
) -> dict:
    """Run nn_only and refined; report timings and quality."""
    results = {}

    print("\n── [1/2] AutoPhaseNet — NN-only prediction ──")
    results['nn_only'] = nn_only_infer(diffraction, model_path, base_channels)
    r = results['nn_only']
    print(f"  Time: {r.elapsed_seconds:.3f}s   R-factor: {r.error_metric[-1]:.4f}")

    print("\n── [2/2] AutoPhaseNet + RAAR/ER refinement ──")
    results['refined'] = refined_infer(diffraction, model_path, base_channels,
                                         n_raar=50, n_er=20)
    r = results['refined']
    print(f"  Time: {r.elapsed_seconds:.3f}s   R-factor: {r.error_metric[-1]:.4f}")

    # Quality vs ground truth if available
    if truth is not None:
        print(f"\n{'='*56}")
        print(f"  Quality vs ground truth")
        print(f"{'='*56}")
        print(f"  {'Method':<15} {'R-factor':>10} {'Phase RMSE':>12} {'Time':>10}")
        print(f"  {'-'*52}")
        for name, result in results.items():
            phase_true = truth['phase_true']
            sup = truth['support'] > 0.5
            if sup.sum() > 10:
                # Remove global phase offset for fair comparison
                mean_r = np.angle(np.mean(np.exp(1j * result.phase[sup])))
                mean_t = np.angle(np.mean(np.exp(1j * phase_true[sup])))
                perr = result.phase[sup] - mean_r - (phase_true[sup] - mean_t)
                perr = np.angle(np.exp(1j * perr))
                rmse = np.sqrt(np.mean(perr ** 2))
            else:
                rmse = float('nan')
            print(f"  {name:<15} {result.error_metric[-1]:>10.4f} "
                  f"{rmse:>12.3f} {result.elapsed_seconds:>9.2f}s")
        print(f"{'='*56}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Save utility
# ═══════════════════════════════════════════════════════════════════════════════

def save_result(result: AutoPhaseResult, path: str):
    """Save reconstruction to .npz compatible with nn_visualize.py."""
    np.savez_compressed(
        path,
        object_real=np.real(result.object_3d).astype(np.float32),
        object_imag=np.imag(result.object_3d).astype(np.float32),
        amplitude=result.amplitude.astype(np.float32),
        phase=result.phase.astype(np.float32),
        support=result.support.astype(np.float32),
        error_metric=np.asarray(result.error_metric, dtype=np.float32),
    )
    print(f"  Saved: {path}  ({result.method}, {result.elapsed_seconds:.2f}s)")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoPhaseNet inference')
    parser.add_argument('--input', required=True,
                        help='.npz (simulated or preprocessed) or .h5 (raw)')
    parser.add_argument('--model', required=True,
                        help='Trained AutoPhaseNet checkpoint')
    parser.add_argument('--output', default='autophase_reconstruction.npz')
    parser.add_argument('--mode', choices=['nn_only', 'refined', 'compare'],
                        default='refined')
    parser.add_argument('--n_raar', type=int, default=50)
    parser.add_argument('--n_er', type=int, default=20)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--support_threshold', type=float, default=0.1)
    parser.add_argument('--no_oversampling', action='store_true')
    parser.add_argument('--target_size', type=int, default=64,
                        help='For .h5 input: crop/pad to this size')
    args = parser.parse_args()

    print(f"\n{'='*56}")
    print(f"  AutoPhaseNet3D inference")
    print(f"{'='*56}")
    print(f"  Input:  {args.input}")
    print(f"  Model:  {args.model}")
    print(f"  Mode:   {args.mode}")
    print(f"{'='*56}")

    diffraction, truth, voxel_nm = load_input(args.input, target_size=args.target_size)
    print(f"  Loaded diffraction volume: shape={diffraction.shape}")
    if truth is not None:
        print(f"  Ground truth available (simulated data)")

    enforce_os = not args.no_oversampling

    if args.mode == 'nn_only':
        r = nn_only_infer(diffraction, args.model, args.base_channels,
                          enforce_oversampling=enforce_os,
                          support_threshold=args.support_threshold)
        save_result(r, args.output)

    elif args.mode == 'refined':
        r = refined_infer(diffraction, args.model, args.base_channels,
                          n_raar=args.n_raar, n_er=args.n_er,
                          enforce_oversampling=enforce_os,
                          support_threshold=args.support_threshold)
        save_result(r, args.output)

    elif args.mode == 'compare':
        results = compare_methods(diffraction, args.model, truth,
                                    args.base_channels)
        # Save the refined (best) result
        save_result(results['refined'], args.output)

    print("\nDone.")
