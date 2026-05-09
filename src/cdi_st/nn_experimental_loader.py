"""
nn_experimental_loader.py — Load real BCDI experimental data from .h5 files.

Supports:
    - ID01 / BLISS format (ESRF)      — primary target
    - 34-ID-C format (APS)            — AutoPhaseNN paper format
    - Generic .h5 with a 3D dataset   — fallback

Key operations:
    1. Locate the 3D diffraction volume in the HDF5 tree
    2. Threshold hot pixels and beamstop
    3. Center the Bragg peak (center of mass)
    4. Crop / pad to the target grid size (64 or 128)
    5. Resample to the target oversampling ratio (~3.0, following AutoPhaseNN)

The resulting preprocessed volume can be passed to either:
    - nn_phase_retrieval.py (supervised model)
    - nn_autophase_infer.py (unsupervised model)
    - The unsupervised training loop (for fine-tuning)

Requires h5py. If you don't have it installed yet:
    pip install h5py

Usage:
    # Inspect the structure of an unknown .h5 file:
    python nn_experimental_loader.py --input scan_123.h5 --inspect

    # Load and preprocess, save as .npz:
    python nn_experimental_loader.py --input scan_123.h5 --output preprocessed.npz

    # Specify dataset path explicitly if auto-detect fails:
    python nn_experimental_loader.py --input scan_123.h5 \\
        --dataset_path /entry_0000/measurement/merlin/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 structure discovery
# ═══════════════════════════════════════════════════════════════════════════════

# Common paths where BCDI diffraction volumes live in different beamline formats
KNOWN_PATHS = [
    # ID01 / BLISS — most common for user
    "/entry_0000/measurement/merlin/data",
    "/entry_0000/measurement/eiger2M/data",
    "/entry_0000/measurement/mpx1x4/data",
    "/entry_0000/measurement/maxipix/data",
    # 34-ID-C (APS)
    "/entry1/instrument/detector/data",
    # Generic
    "/data",
    "/entry/data/data",
    "/intensity",
    "/diffraction",
]


def inspect_h5(path: str, max_depth: int = 6):
    """Print the full tree of an .h5 file to help locate the diffraction data."""
    try:
        import h5py
    except ImportError:
        print("h5py not installed. Run: pip install h5py")
        return

    print(f"\nStructure of {path}:")
    print("=" * 62)

    def _walk(name, obj):
        depth = name.count("/")
        if depth > max_depth:
            return
        indent = "  " * depth
        if hasattr(obj, "shape"):
            s = f"{indent}{name}  shape={obj.shape}  dtype={obj.dtype}"
            # Highlight likely candidates: 3D float/int arrays
            if len(obj.shape) == 3 and obj.shape[0] > 10:
                s += "  ← likely diffraction volume"
            elif len(obj.shape) == 3:
                s += "  ← 3D dataset"
            print(s)
        else:
            print(f"{indent}{name}/")

    with h5py.File(path, "r") as f:
        f.visititems(_walk)
    print("=" * 62)


def find_diffraction_dataset(h5_file):
    """
    Try known paths first, then scan for any suitable 3D dataset.

    Returns (path_str, dataset) or (None, None) if nothing found.
    """
    # Try known paths
    for p in KNOWN_PATHS:
        if p in h5_file:
            ds = h5_file[p]
            if hasattr(ds, "shape") and len(ds.shape) >= 3:
                return p, ds

    # Fall back: walk the tree and pick the largest 3D dataset
    candidates = []

    def _collect(name, obj):
        if hasattr(obj, "shape") and len(obj.shape) == 3:
            sz = int(np.prod(obj.shape))
            if sz > 10**5:  # arbitrary minimum size
                candidates.append((sz, name, obj))

    h5_file.visititems(_collect)
    if candidates:
        candidates.sort(reverse=True)
        _, name, ds = candidates[0]
        return name, ds

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing steps
# ═══════════════════════════════════════════════════════════════════════════════


def mask_detector_gaps(volume: np.ndarray, detector: str = "auto") -> np.ndarray:
    """
    Mask the inter-chip gaps of multi-chip BCDI detectors.

    Maxipix (ID01) is a 2×2 array of 256×256 chips with a gap of ~5 pixels
    between chips at rows/columns 255-260. These gaps appear as zeros and
    badly bias COM calculations and hot-pixel detection.

    The cdiutils / xrayutilities approach (which is what we follow here)
    is to FILL the gaps with the local median of surrounding voxels,
    not leave them as zeros. Zeros pull the COM toward the gap, while
    the median preserves the actual diffraction signal.

    Detector auto-detect:
        - 516×516 → Maxipix 2×2 (ID01)
        - 1062×1028 → Eiger 2M
        - other → use heuristic detection
    """

    out = volume.copy().astype(np.float32)
    Nz, Ny, Nx = out.shape

    if detector == "auto":
        if Ny == 516 and Nx == 516:
            detector = "maxipix"
        elif Ny >= 1000 and Nx >= 1000:
            detector = "eiger2M"
        else:
            detector = "unknown"

    if detector == "maxipix":
        gap_rows = list(range(255, 261))
        gap_cols = list(range(255, 261))
    elif detector == "eiger2M":
        # Eiger 2M has multiple module gaps; this covers the central one.
        # For full preprocessing of Eiger 2M, use cdiutils directly.
        gap_rows = list(range(513, 551))
        gap_cols = []
    else:
        # Heuristic: rows/cols where most voxels are zero or near-zero.
        # Two passes:
        #   - Strict (>95% zero): catches the original detector gap
        #   - Lenient (>80% near-zero): catches gaps after cropping/interpolation,
        #     where some bleed-in may have made the gap not literally zero
        eps = 1e-6
        # Reference for "near zero": 0.5% of the median nonzero value
        nz_global = out[out > eps]
        if len(nz_global) > 0:
            near_zero_thr = float(np.median(nz_global)) * 0.005
        else:
            near_zero_thr = eps
        # Check fraction of voxels that are AT or near zero per row/column
        is_zero = out <= near_zero_thr
        proj_y = is_zero.mean(axis=(0, 2))
        proj_x = is_zero.mean(axis=(0, 1))
        gap_rows = np.where(proj_y > 0.80)[0].tolist()
        gap_cols = np.where(proj_x > 0.80)[0].tolist()
        # Sanity: if "everything" is a gap, abort (means data is just empty)
        if len(gap_rows) > Ny * 0.5 or len(gap_cols) > Nx * 0.5:
            return out

    if not gap_rows and not gap_cols:
        return out

    gap_mask = np.zeros_like(out, dtype=bool)
    if gap_rows:
        gap_mask[:, [r for r in gap_rows if r < Ny], :] = True
    if gap_cols:
        gap_mask[:, :, [c for c in gap_cols if c < Nx]] = True

    # Fill gap voxels with the median of nearby NON-GAP voxels.
    # Implementation: iterate only over the gap voxels (they're a small fraction
    # of the data, ~2% for Maxipix), and for each one sample a neighborhood
    # in the same rocking frame.
    half_window = 9  # 19×19 window — wide enough to bridge a 6-pixel gap
    for z in range(Nz):
        gap_z = gap_mask[z]
        if not gap_z.any():
            continue
        non_gap = ~gap_z
        frame = out[z]
        gap_indices = np.argwhere(gap_z)
        for gy, gx in gap_indices:
            y0 = max(0, gy - half_window)
            y1 = min(Ny, gy + half_window + 1)
            x0 = max(0, gx - half_window)
            x1 = min(Nx, gx + half_window + 1)
            # Get the window's NON-gap voxels
            nbhd = frame[y0:y1, x0:x1]
            nbhd_mask = non_gap[y0:y1, x0:x1]
            valid = nbhd[nbhd_mask]
            if len(valid) > 0:
                out[z, gy, gx] = float(np.median(valid))
            # else: leave as is (rare edge case)

    return out


def remove_hot_pixels(volume: np.ndarray, threshold_sigma: float = 50.0) -> np.ndarray:
    """
    Remove hot pixels (cosmic rays, defective detector pixels) by clipping
    values that are MUCH higher than their local neighborhood.

    Note: BCDI detector data has very high dynamic range. The Bragg peak
    itself has intensity 1e4-1e6× higher than the fringes around it. So a
    sigma-based threshold needs to be VERY high (50+) to avoid clipping
    the peak. Earlier value of 20 was too aggressive and ate the Bragg peak.

    Better approach: use a 3-pixel median filter and only clip pixels that
    are 100× the median (single bright pixels surrounded by dark).
    """
    from scipy.ndimage import median_filter

    # 3D median filter — more robust against the peak itself
    local_median = median_filter(volume, size=3)
    # Only clip pixels that are >100× their immediate neighbors AND
    # not part of an extended bright structure
    safe_median = np.maximum(local_median, 1.0)  # avoid divide-by-zero
    ratio = volume / safe_median
    mask = (ratio > 100.0) & (volume > 100)  # bright spot far from neighbors
    cleaned = volume.copy()
    cleaned[mask] = local_median[mask]
    return cleaned


def find_bragg_peak_box(
    volume: np.ndarray, intensity_threshold_pct: float = 1.0, margin_voxels: int = 8
) -> tuple:
    """
    Locate the Bragg peak by finding the 3D bounding box of voxels above
    a threshold of the maximum intensity. Returns (z_slice, y_slice, x_slice).

    This is the standard approach in cdiutils, PyNX, BCDI-utilities:
    1. Find the brightest voxel
    2. Threshold around it (typically 1% of max — captures the peak + fringes)
    3. Use the bounding box of the thresholded region

    This is much better than COM-based centering, which is biased by
    detector noise and beam stops.
    """
    # Smooth lightly for robust peak finding (3-pixel Gaussian)
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(volume.astype(np.float32), sigma=2.0)

    peak_max = float(smoothed.max())
    if peak_max <= 0:
        # Empty volume — fallback to center
        cz, cy, cx = [s // 2 for s in volume.shape]
        return (
            slice(max(0, cz - 32), cz + 32),
            slice(max(0, cy - 32), cy + 32),
            slice(max(0, cx - 32), cx + 32),
        )

    # Find ALL voxels above threshold
    threshold = (intensity_threshold_pct / 100.0) * peak_max
    above = smoothed > threshold

    if not above.any():
        # Fall back: largest single voxel
        peak_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        cz, cy, cx = peak_idx
    else:
        # Bounding box of above-threshold region
        coords = np.argwhere(above)
        # Center is midpoint of bbox (more robust than COM if there's noise)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        cz = (z_min + z_max) // 2
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2

    return (cz, cy, cx)


def crop_around_peak(volume: np.ndarray, center: tuple, N: int) -> np.ndarray:
    """
    Crop a (N, N, N) box around the given center in `volume`.
    If the box extends beyond the array, pad with zeros (NOT wrap around).

    This replaces both `center_on_bragg_peak` (which used np.roll = wrap)
    and `crop_or_pad_to` (which centered on grid center, not peak center).
    """
    cz, cy, cx = center
    half = N // 2
    out = np.zeros((N, N, N), dtype=volume.dtype)

    # Compute valid src and dst slices for each axis
    def _sliced(c, src_size):
        src_start = max(0, c - half)
        src_end = min(src_size, c + half)
        dst_start = src_start - (c - half)  # how much padding on the left
        dst_end = dst_start + (src_end - src_start)
        return slice(src_start, src_end), slice(dst_start, dst_end)

    src_z, dst_z = _sliced(cz, volume.shape[0])
    src_y, dst_y = _sliced(cy, volume.shape[1])
    src_x, dst_x = _sliced(cx, volume.shape[2])

    out[dst_z, dst_y, dst_x] = volume[src_z, src_y, src_x]
    return out


def remove_beamstop_streaks(volume: np.ndarray) -> np.ndarray:
    """
    Remove beamstop / direct-beam streaks from BCDI data.

    These are bright lines along the rocking-axis direction (axis 0) that
    persist across all rocking frames — caused by the direct beam catching
    a beamstop edge or a saturated detector pixel.

    Critical: we ONLY check the detector plane (axes 1, 2). Streaks along
    the rocking axis are physical (Bragg fringes), so we never collapse
    along axis 0 to find "streaks" — that would destroy the actual signal.

    Detection: A streak is a (y, x) detector pixel where ALL rocking
    frames have intensity > 10× the global median. Real Bragg peaks
    have intensity at only some rocking angles.
    """
    out = volume.copy().astype(np.float32)
    nz = out[out > 0]
    if len(nz) == 0:
        return out
    global_med = float(np.median(nz))
    threshold = 10.0 * global_med

    Nz = out.shape[0]

    # For each (y, x) detector pixel, count how many rocking frames are bright.
    bright_per_pixel = (out > threshold).sum(axis=0)
    # A real Bragg peak appears in ~10-50% of rocking frames.
    # A streak appears in >80% of rocking frames.
    streak_mask_2d = bright_per_pixel > 0.80 * Nz

    n_streaks = int(streak_mask_2d.sum())
    if n_streaks == 0 or n_streaks > out.shape[1] * out.shape[2] * 0.05:
        # Either no streaks, or "too many" — safety: don't risk killing signal
        return out

    # Zero the streak pixels across all rocking frames
    streak_mask_3d = np.broadcast_to(streak_mask_2d[None, :, :], out.shape)
    out[streak_mask_3d] = 0
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main loading function
# ═══════════════════════════════════════════════════════════════════════════════


def load_h5_diffraction(
    path: str,
    dataset_path: str = None,
    target_size: int = 64,
    clean_hot_pixels: bool = True,
    center_peak: bool = True,
    beamstop_threshold: float = 0.0,
    max_frames: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load and preprocess a 3D diffraction volume from an .h5 file.

    Pipeline (matches cdiutils / BCDI-utilities standard):
        1. Locate dataset in HDF5 tree
        2. Convert to float32, clip negatives
        3. Remove beamstop/direct-beam streaks
        4. Remove hot pixels (conservative threshold to preserve Bragg peak)
        5. Locate Bragg peak by intensity bounding box
        6. Crop a (N, N, N) box AROUND the peak (not the array center)

    Parameters
    ----------
    path : str
        Path to .h5 file.
    dataset_path : str or None
        HDF5 path to the diffraction dataset. If None, auto-detect.
    target_size : int
        Output grid size N → returns (N, N, N) volume.
    clean_hot_pixels : bool
        Apply median filter to remove hot pixels.
    center_peak : bool
        Locate and crop around the Bragg peak.
    beamstop_threshold : float
        Values below this are set to 0 (removes beamstop artifacts).
    max_frames : int or None
        Limit the number of detector frames to use (helpful for memory).
    verbose : bool
        Print preprocessing diagnostics.

    Returns
    -------
    diffraction : ndarray [N, N, N]
        Preprocessed 3D diffraction volume (intensity, not amplitude).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install h5py")

    with h5py.File(path, "r") as f:
        # Find the dataset
        if dataset_path is not None:
            if dataset_path not in f:
                raise KeyError(f"Dataset '{dataset_path}' not found in {path}")
            ds = f[dataset_path]
            dpath = dataset_path
        else:
            dpath, ds = find_diffraction_dataset(f)
            if ds is None:
                raise ValueError(
                    f"Could not auto-detect diffraction data in {path}. "
                    "Run with --inspect to see structure, then use "
                    "--dataset_path to specify it explicitly."
                )

        if verbose:
            print(f"  Loaded dataset: {dpath}  shape={ds.shape}  dtype={ds.dtype}")

        # If data is 4D (e.g. scan points × detector), reshape or select
        if len(ds.shape) == 4:
            if ds.shape[1] == 1:
                ds_arr = ds[:, 0, :, :]
            else:
                ds_arr = ds[...]
        else:
            ds_arr = ds[...]

        # Optional frame limiting
        if max_frames is not None and ds_arr.shape[0] > max_frames:
            start = (ds_arr.shape[0] - max_frames) // 2
            ds_arr = ds_arr[start : start + max_frames]

        volume = np.asarray(ds_arr, dtype=np.float32)

    # ── Preprocessing pipeline ────────────────────────────────────────────
    if verbose:
        print(
            f"  Raw volume:    shape={volume.shape}  "
            f"min={volume.min():.2e} max={volume.max():.2e} "
            f"mean={volume.mean():.2e}"
        )

    # Remove negative values (sometimes present from background subtraction)
    volume = np.maximum(volume, 0)

    # CRITICAL: mask Maxipix / Eiger inter-chip gaps BEFORE other processing.
    # The gaps are zero rows/columns at fixed positions (e.g. 255-260 on
    # Maxipix). If left as zeros they:
    #   - Bias COM-based centering toward the gap position
    #   - Get misidentified as "streaks" by the streak detector
    #   - Break the Fourier transform during reconstruction
    # cdiutils and BCDI-utils both fill these gaps via local median.
    volume = mask_detector_gaps(volume, detector="auto")
    if verbose:
        print(
            f"  After gap mask: max={volume.max():.2e}  min_nonzero="
            f"{volume[volume > 0].min() if (volume > 0).any() else 0:.2e}"
        )

    # Remove beamstop (very low intensity region gets zeroed)
    if beamstop_threshold > 0:
        volume = np.where(volume < beamstop_threshold, 0, volume)

    # Remove direct beam / beamstop streaks (vertical/horizontal lines)
    volume = remove_beamstop_streaks(volume)

    # Clean hot pixels (very high threshold so we don't eat the Bragg peak)
    if clean_hot_pixels:
        volume = remove_hot_pixels(volume, threshold_sigma=50.0)

    # Locate the Bragg peak and crop around it
    if center_peak:
        peak_center = find_bragg_peak_box(volume)
        if verbose:
            print(f"  Bragg peak found at voxel: {peak_center}")
        volume = crop_around_peak(volume, peak_center, target_size)
    else:
        # Just crop the central (N, N, N) box
        offsets = [(s - target_size) // 2 for s in volume.shape]
        slicer = tuple(slice(max(0, o), max(0, o) + target_size) for o in offsets)
        cropped = volume[slicer]
        # Pad if too small
        if cropped.shape != (target_size, target_size, target_size):
            out = np.zeros((target_size, target_size, target_size), dtype=volume.dtype)
            for ax in range(3):
                pass  # already cropped, just place
            o = [(target_size - s) // 2 for s in cropped.shape]
            sl = tuple(slice(oi, oi + s) for oi, s in zip(o, cropped.shape))
            out[sl] = cropped
            volume = out
        else:
            volume = cropped

    if verbose:
        N = target_size
        peak_idx = np.unravel_index(np.argmax(volume), volume.shape)
        center = (N // 2, N // 2, N // 2)
        print(
            f"  Processed:     shape={volume.shape}  "
            f"peak_max={volume.max():.2e}  total_counts={volume.sum():.2e}"
        )
        print(f"  Peak index:    {peak_idx} (target center {center})")

    return volume


# ═══════════════════════════════════════════════════════════════════════════════
# Spec/EDF reader (ID01 native format, before BLISS h5)
# ═══════════════════════════════════════════════════════════════════════════════


def read_spec_scan(spec_path: str, scan_number: int) -> dict:
    """
    Read motor positions for a scan from a SPEC file.

    SPEC files are the native ID01 (and many other beamlines') metadata
    format used before BLISS/HDF5 became standard. They contain:
        - Header with motor names + initial positions ('motors' dict)
        - Per-scan motor sweeps (eta or phi rocking)
        - CCD frame numbers (mpx4inr column)

    This function returns the motor positions needed for q-space
    orthogonalization: eta, phi, nu, delta arrays + the rocking type.

    Requires the `spec` Python package:
        pip install spec
    Or alternatively `silx.io.specfile` which is more modern.

    Returns
    -------
    dict with keys:
        'eta', 'phi', 'nu', 'delta'    — motor angle arrays (degrees)
        'mpx4inr'                       — CCD frame numbers
        'rocking_type'                  — 'eta' or 'phi'
        'header_motors'                 — dict of all initial motor positions
    """
    try:
        # Try silx first (more modern, better maintained)
        from silx.io.specfile import SpecFile

        sf = SpecFile(spec_path)
        scan = sf[f"{scan_number}.1"]
        labels = scan.labels
        data = scan.data
        result = {label: data[i] for i, label in enumerate(labels)}
        # Header motors
        motor_names = scan.motor_names
        motor_positions = scan.motor_positions
        result["header_motors"] = dict(zip(motor_names, motor_positions))
    except ImportError:
        try:
            import spec

            h, d = spec.ReadSpec(spec_path, scan_number)
            result = dict(d)
            result["header_motors"] = dict(h.get("motors", {}))
        except ImportError:
            raise ImportError(
                "Need either silx (pip install silx) or spec (pip install spec) "
                "to read SPEC files."
            )

    # Determine rocking axis: which motor varies most over the scan?
    n_pts = len(result.get("mpx4inr", result.get(list(result.keys())[0])))
    rocking_type = None
    if "eta" in result and len(result["eta"]) == n_pts:
        eta_arr = np.asarray(result["eta"])
        if eta_arr.std() > 0.01:
            rocking_type = "eta"
    if rocking_type is None and "phi" in result and len(result["phi"]) == n_pts:
        phi_arr = np.asarray(result["phi"])
        if phi_arr.std() > 0.01:
            rocking_type = "phi"

    # Fill in fixed motors from header for rocking case
    h_motors = result["header_motors"]
    if rocking_type == "eta":
        result["phi"] = np.full(n_pts, h_motors.get("phi", 0.0))
        result["delta"] = np.full(
            n_pts, h_motors.get("del", h_motors.get("delta", 0.0))
        )
        result["nu"] = np.full(n_pts, h_motors.get("nu", 0.0))
    elif rocking_type == "phi":
        result["eta"] = np.full(n_pts, h_motors.get("eta", 0.0))
        result["delta"] = np.full(
            n_pts, h_motors.get("del", h_motors.get("delta", 0.0))
        )
        result["nu"] = np.full(n_pts, h_motors.get("nu", 0.0))

    result["rocking_type"] = rocking_type
    return result


def read_edf_stack(
    edf_template: str, frame_numbers: list, detector_shape: tuple = (516, 516)
) -> np.ndarray:
    """
    Read a stack of EDF (European Data Format) detector frames.

    EDF is the legacy ID01 detector format. Each frame is a single .edf or
    .edf.gz file, named like 'data_mpx4_00123.edf.gz'. This function loads
    a list of frames into a 3D volume (n_frames, det_y, det_x).

    Requires `xrayutilities` (pip install xrayutilities) which has a
    convenient EDF reader.

    Parameters
    ----------
    edf_template : str
        Filename template with one %d for frame number, e.g.
        '/data/dir/data_mpx4_%05d.edf.gz'
    frame_numbers : list of int
        Frame numbers to load (typically the 'mpx4inr' column from SPEC).
    detector_shape : tuple
        Expected (rows, cols) of each detector frame.

    Returns
    -------
    volume : ndarray (n_frames, det_y, det_x)
    """
    try:
        import xrayutilities as xu
    except ImportError:
        raise ImportError(
            "Need xrayutilities to read EDF files. Install with:\n"
            "  pip install xrayutilities\n"
            "Or convert your data to HDF5 with another tool first."
        )

    n = len(frame_numbers)
    volume = np.zeros((n,) + detector_shape, dtype=np.float32)
    for idx, frame in enumerate(frame_numbers):
        f = edf_template % int(frame)
        try:
            e = xu.io.EDFFile(f)
            volume[idx] = e.data.astype(np.float32)
        except Exception as ex:
            print(f"  Warning: could not read {f}: {ex}")
    return volume


def load_spec_edf_scan(
    spec_path: str,
    scan_number: int,
    edf_dir: str,
    edf_template_name: str = "data_mpx4_%05d.edf.gz",
    target_size: int = 64,
    detector_shape: tuple = (516, 516),
    detector: str = "maxipix",
    verbose: bool = True,
) -> dict:
    """
    Full ID01 spec+EDF loader: read motor positions from a SPEC file,
    load corresponding EDF detector frames, preprocess (gap-mask, hot pixels,
    streaks, peak finding), and return everything needed for reconstruction
    PLUS q-space orthogonalization (if xrayutilities is available).

    This is the ID01-native loading path that mirrors what cdiutils does.

    Parameters
    ----------
    spec_path : str
        Path to .spec file containing the scan header.
    scan_number : int
        SPEC scan number to load.
    edf_dir : str
        Directory containing the .edf.gz detector frames.
    edf_template_name : str
        Filename template (default: 'data_mpx4_%05d.edf.gz' for ID01 maxipix)
    target_size : int
        Output grid size N → returns (N, N, N) volume.
    detector_shape : tuple
        Raw detector dimensions (default 516×516 for Maxipix).
    detector : str
        Detector name for gap masking ('maxipix', 'eiger2M', 'auto').

    Returns
    -------
    dict with keys:
        'diffraction'    — preprocessed (target_size,)*3 volume
        'eta', 'phi', 'nu', 'delta'  — motor arrays (degrees)
        'rocking_type'   — 'eta' or 'phi'
        'frame_numbers'  — original CCD frame numbers
        'q_axes'         — (qx, qy, qz) arrays in 1/Å, IF xrayutilities available
        'voxel_size_nm'  — real-space voxel pitch (3-vector) IF q_axes present
    """
    import os

    # 1. Read SPEC header
    if verbose:
        print(f"Reading SPEC file: {spec_path}, scan {scan_number}")
    spec_data = read_spec_scan(spec_path, scan_number)
    rocking_type = spec_data["rocking_type"]
    if verbose:
        print(f"  Rocking axis: {rocking_type}")

    # 2. Load detector frames
    edf_template = os.path.join(edf_dir, edf_template_name)
    frame_numbers = np.asarray(spec_data["mpx4inr"]).astype(int)
    if verbose:
        print(f"  Loading {len(frame_numbers)} EDF frames from {edf_dir}")
    volume = read_edf_stack(edf_template, frame_numbers, detector_shape)
    if verbose:
        print(f"  Raw stack: shape={volume.shape}  max={volume.max():.2e}")

    # 3. Preprocess (gap-mask, hot pixels, streaks)
    volume = np.maximum(volume, 0)
    volume = mask_detector_gaps(volume, detector=detector)
    volume = remove_beamstop_streaks(volume)
    volume = remove_hot_pixels(volume, threshold_sigma=50.0)

    # 4. Find Bragg peak and crop
    peak_center = find_bragg_peak_box(volume)
    if verbose:
        print(f"  Bragg peak at: {peak_center}")
    cropped = crop_around_peak(volume, peak_center, target_size)

    result = {
        "diffraction": cropped.astype(np.float32),
        "eta": np.asarray(spec_data["eta"]),
        "phi": np.asarray(spec_data["phi"]),
        "nu": np.asarray(spec_data["nu"]),
        "delta": np.asarray(spec_data["delta"]),
        "rocking_type": rocking_type,
        "frame_numbers": frame_numbers,
        "header_motors": spec_data["header_motors"],
    }

    # 5. Q-space orthogonalization with xrayutilities (optional)
    try:
        import xrayutilities as xu

        if verbose:
            print("  Computing q-space coordinates with xrayutilities...")

        # Standard ID01 geometry from the user's lit_ID01_AOUT2021.py:
        #   sample circles: y- (eta), z- (phi)
        #   detector circles: z- (nu), y- (delta)
        #   primary beam: [1, 0, 0]
        # Detector calibration (defaults — user can override if needed):
        beam_energy_eV = 13000.0
        cch1, cch2 = 207.11, 167.86  # detector center pixels
        chpdeg = [406.3, 406.3]  # channels per degree

        qconv = xu.experiment.QConversion(["y-", "z-"], ["z-", "y-"], [1, 0, 0])
        hxrd = xu.experiment.HXRD([1, 0, 0], [0, 0, 1], en=beam_energy_eV, qconv=qconv)
        hxrd.Ang2Q.init_area(
            "z-",
            "y+",
            cch1=cch1,
            cch2=cch2,
            Nch1=detector_shape[0],
            Nch2=detector_shape[1],
            chpdeg1=chpdeg[0],
            chpdeg2=chpdeg[1],
        )

        qx, qy, qz = hxrd.Ang2Q.area(
            result["eta"], result["phi"], result["nu"], result["delta"]
        )
        # qx, qy, qz are 3D arrays of q-coordinates per pixel.
        # For voxel size estimation, take the mean step in each direction.
        # (Proper interpolation onto orthogonal grid would use Gridder3D.)
        dqx = float(np.mean(np.diff(qx, axis=0)))
        dqy = float(np.mean(np.diff(qy, axis=1)))
        dqz = float(np.mean(np.diff(qz, axis=2))) if qz.shape[2] > 1 else 1.0
        result["q_step_inv_A"] = (abs(dqx), abs(dqy), abs(dqz))
        # Real-space voxel size: dr = 2π / (N · dq), in nm
        # (10× factor converts Å to nm)
        N = target_size
        result["voxel_size_nm"] = np.array(
            [
                2 * np.pi / (N * abs(dqx) * 10) if abs(dqx) > 1e-12 else 1.0,
                2 * np.pi / (N * abs(dqy) * 10) if abs(dqy) > 1e-12 else 1.0,
                2 * np.pi / (N * abs(dqz) * 10) if abs(dqz) > 1e-12 else 1.0,
            ],
            dtype=np.float32,
        )
        if verbose:
            vn = result["voxel_size_nm"]
            print(f"  Voxel pitch: ({vn[0]:.3f}, {vn[1]:.3f}, {vn[2]:.3f}) nm")
    except ImportError:
        if verbose:
            print("  xrayutilities not installed — q-space conversion skipped.")
            print("  Install with: pip install xrayutilities")
    except Exception as e:
        if verbose:
            print(f"  q-space conversion failed: {e}")

    return result


def spec_edf_to_npz(
    spec_path: str,
    scan_number: int,
    edf_dir: str,
    npz_path: str,
    target_size: int = 64,
    edf_template_name: str = "data_mpx4_%05d.edf.gz",
):
    """
    One-call helper: convert ID01 spec+EDF to a reconstruction-ready .npz.
    """
    res = load_spec_edf_scan(
        spec_path=spec_path,
        scan_number=scan_number,
        edf_dir=edf_dir,
        edf_template_name=edf_template_name,
        target_size=target_size,
    )
    save_dict = {
        "diffraction": res["diffraction"],
        "amplitude": np.sqrt(np.maximum(res["diffraction"], 0)),
        "eta": res["eta"],
        "phi": res["phi"],
        "nu": res["nu"],
        "delta": res["delta"],
        "rocking_type": res["rocking_type"] or "unknown",
    }
    if "voxel_size_nm" in res:
        save_dict["voxel_size_nm"] = res["voxel_size_nm"]
    np.savez_compressed(npz_path, **save_dict)
    print(f"  Saved: {npz_path}")
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# Conversion script (experimental .h5 → training-compatible .npz)
# ═══════════════════════════════════════════════════════════════════════════════


def h5_to_npz(
    h5_path: str,
    npz_path: str,
    dataset_path: str = None,
    target_size: int = 64,
):
    """
    Convert an experimental .h5 file to an .npz compatible with the dataset
    loaders (for either inference or fine-tuning the unsupervised model).
    """
    diffraction = load_h5_diffraction(h5_path, dataset_path, target_size)

    # Save as .npz with the same structure as simulated files
    # (without phase_true/support, since those aren't known experimentally)
    np.savez_compressed(
        npz_path,
        diffraction=diffraction.astype(np.float32),
        amplitude=np.sqrt(diffraction).astype(np.float32),
    )
    print(f"  Saved: {npz_path}")

    # Also save metadata
    meta = {
        "source_file": str(h5_path),
        "source_dataset": dataset_path,
        "target_size": target_size,
        "is_experimental": True,
        "has_ground_truth": False,
    }
    meta_path = Path(npz_path).parent / (Path(npz_path).stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load BCDI diffraction data from experimental .h5 files"
    )
    parser.add_argument("--input", type=str, required=True, help="Input .h5 file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz file (if not given, just inspect)",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Just print the HDF5 structure and exit"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="HDF5 path to diffraction data (auto-detect if omitted)",
    )
    parser.add_argument("--target_size", type=int, default=64, help="Output grid size")
    parser.add_argument(
        "--max_depth", type=int, default=6, help="Max tree depth for --inspect"
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_h5(args.input, max_depth=args.max_depth)
        sys.exit(0)

    if args.output is None:
        print("No --output specified. Inspecting structure:")
        inspect_h5(args.input, max_depth=args.max_depth)
    else:
        h5_to_npz(args.input, args.output, args.dataset_path, args.target_size)
