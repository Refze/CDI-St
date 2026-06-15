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
import argparse, json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-load HDF5 compression plugins for Eiger/Lambda/Pilatus data.
#
# Eiger detectors (used at PETRA III P10/P21 and many other beamlines) write
# data with bitshuffle + LZ4 compression (filter ID 32008). This filter is
# NOT bundled with h5py — it lives in the `hdf5plugin` package.
#
# Without hdf5plugin, attempting to read a compressed Eiger dataset gives
# the cryptic error:
#       "Can't synchronously read data (can't open directory)"
# This is HDF5 looking for the plugin DLL/SO in a directory that doesn't
# exist on the user's machine.
#
# Importing hdf5plugin registers all common scientific compression filters
# (bitshuffle, LZ4, zstd, blosc, fcidecomp) with the HDF5 library, after
# which h5py can read the data transparently.
#
# We capture the availability status so we can give a USEFUL error message
# if the user tries to read a compressed file without hdf5plugin installed.
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import hdf5plugin  # noqa: F401  — registers LZ4/bitshuffle/zstd filters
    _HDF5_PLUGINS_AVAILABLE = True
    _HDF5_PLUGINS_VERSION = getattr(hdf5plugin, 'version', 'unknown')
except ImportError:
    _HDF5_PLUGINS_AVAILABLE = False
    _HDF5_PLUGINS_VERSION = None

# Filter IDs that REQUIRE hdf5plugin (h5py can't read them on its own)
_PLUGIN_FILTER_IDS = {
    32000: "LZF (32000) — actually built into h5py, but check",
    32001: "Blosc (32001)",
    32004: "LZ4 (32004)",
    32008: "Bitshuffle + LZ4 (32008) — Eiger detector default",
    32015: "Zstandard (32015)",
    32016: "FCIDECOMP (32016)",
}
# Filters built into h5py / HDF5 — don't need hdf5plugin:
_BUILTIN_FILTER_IDS = {0, 1, 2, 4, 32000}   # uncompressed, deflate, shuffle, szip, lzf


def _dataset_requires_plugin(dataset) -> Optional[str]:
    """
    Check if a dataset uses a compression filter that requires hdf5plugin.

    Returns
    -------
    None if the dataset uses only built-in filters.
    A descriptive string (e.g. "Bitshuffle + LZ4") if the dataset needs
    hdf5plugin and the filter is one of the known plugin-requiring filters.
    """
    try:
        plist = dataset.id.get_create_plist()
        for i in range(plist.get_nfilters()):
            f_info = plist.get_filter(i)
            fid = f_info[0]
            if fid in _BUILTIN_FILTER_IDS:
                continue
            if fid in _PLUGIN_FILTER_IDS:
                return _PLUGIN_FILTER_IDS[fid]
            return f"filter ID {fid}"
    except Exception:
        pass
    return None


def _is_compression_plugin_error(err: Exception) -> bool:
    """
    Heuristic: detect if an exception is the cryptic "missing HDF5 compression
    plugin" error. Symptom on Windows: 'Can't synchronously read data (can't
    open directory)'. On Linux: similar errors mentioning 'open directory' or
    'plugin'.
    """
    msg = str(err).lower()
    return (
        "open directory" in msg
        or "can't open plugin" in msg
        or "no filter registered" in msg
        or "filter" in msg and "not available" in msg
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 structure discovery
# ═══════════════════════════════════════════════════════════════════════════════

# Common paths where BCDI diffraction volumes live in different beamline formats
KNOWN_PATHS = [
    # ID01 / BLISS — most common for ESRF users
    '/entry_0000/measurement/merlin/data',
    '/entry_0000/measurement/eiger2M/data',
    '/entry_0000/measurement/mpx1x4/data',
    '/entry_0000/measurement/maxipix/data',
    # P10 / PETRA III (DESY) — NeXus-style
    '/entry/data/data',
    '/entry/instrument/detector/data',
    '/entry/instrument/eiger_4m/data',
    '/entry/instrument/eiger_500k/data',
    '/entry/instrument/eiger/data',
    '/entry/instrument/lambda/data',
    # CXI (Coherent X-ray Imaging) format
    '/entry_1/data_1/data',
    '/entry_1/instrument_1/detector_1/data',
    # 34-ID-C (APS)
    '/entry1/instrument/detector/data',
    # Generic
    '/data',
    '/intensity',
    '/diffraction',
]


def _find_p10_master(data_path: str) -> Optional[str]:
    """
    For a P10 'data file' like 'align_03_01698_data_000001.h5', look for the
    corresponding 'master.h5' in the same directory.

    P10 (and most modern Eiger setups) writes scans as one 'master' file plus
    one or more numbered data chunks. The master file contains a Virtual
    Dataset (VDS) that links the chunks together. Reading a single chunk
    directly often fails or yields partial data — the master is the correct
    entry point.

    Returns the path to the master file if found, else None.
    """
    import os, re
    if not data_path:
        return None
    base = os.path.basename(data_path)
    dirname = os.path.dirname(os.path.abspath(data_path))
    # P10 patterns we've seen in the wild:
    #   align_03_01698_data_000001.h5  ↔  align_03_01698_master.h5
    #   scan_0042_data_000123.h5       ↔  scan_0042_master.h5
    #   sample_0001_00042_data_000001.h5 ↔ sample_0001_00042_master.h5
    m = re.match(r'^(.*?)_data_\d+\.h5$', base, re.IGNORECASE)
    if not m:
        return None
    prefix = m.group(1)
    candidate = os.path.join(dirname, f"{prefix}_master.h5")
    if os.path.exists(candidate):
        return candidate
    # Some setups use ".nxs" extension instead
    candidate = os.path.join(dirname, f"{prefix}_master.nxs")
    if os.path.exists(candidate):
        return candidate
    return None


def parse_p10_fio(fio_path: str) -> dict:
    """
    Parse a P10 (DESY) '.fio' metadata file.

    .fio is a custom ASCII format used at PETRA III. Structure:

        !
        ! Comments
        !
        %c
         scan_command_here
        %p
         param_name = value
         ...
        %d
         Col 1 motor_or_counter_name DOUBLE
         Col 2 mpx4inr INTEGER
         ...
         value1 value2 ...
         value1 value2 ...

    Returns a dict with:
        scan_command : str
        params : dict[str, str/float]    # %p block
        columns : list[str]              # column names from %d block
        data : dict[str, np.ndarray]     # column data
    """
    import re
    result = {
        "scan_command": "",
        "params": {},
        "columns": [],
        "data": {},
    }
    try:
        with open(fio_path, "r", encoding="latin-1") as f:
            text = f.read()
    except Exception:
        return result

    # Section parser: split by %c / %p / %d markers
    section = None
    data_rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("!"):
            continue
        if s == "%c":
            section = "c"; continue
        if s == "%p":
            section = "p"; continue
        if s == "%d":
            section = "d"; continue
        if section == "c":
            # First non-comment line under %c is the scan command
            if not result["scan_command"]:
                result["scan_command"] = s
        elif section == "p":
            # name = value
            if "=" in s:
                k, _, v = s.partition("=")
                k = k.strip()
                v = v.strip()
                try:
                    v = float(v)
                except ValueError:
                    pass
                result["params"][k] = v
        elif section == "d":
            # Header lines: "Col N name TYPE", then data rows
            m = re.match(r"Col\s+(\d+)\s+(\S+)\s+\S+", s, re.IGNORECASE)
            if m:
                result["columns"].append(m.group(2))
            else:
                # Data row
                fields = s.split()
                if fields and all(_is_number(x) for x in fields):
                    data_rows.append([float(x) for x in fields])

    if data_rows and result["columns"]:
        arr = np.asarray(data_rows)
        ncols = min(arr.shape[1], len(result["columns"]))
        for i in range(ncols):
            result["data"][result["columns"][i]] = arr[:, i]
    return result


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def diagnose_vds(dataset, dataset_path: str) -> str:
    """
    Diagnose why a Virtual Dataset (VDS) read might fail.

    Returns a human-readable string listing the VDS sources and whether each
    source file is reachable. Used to translate cryptic HDF5 errors into
    actionable messages.
    """
    import os
    lines = [f"  Dataset '{dataset_path}' is a Virtual Dataset (VDS)."]
    try:
        sources = dataset.virtual_sources()
    except Exception as e:
        lines.append(f"  Could not enumerate VDS sources: {e}")
        return "\n".join(lines)

    if not sources:
        lines.append("  But it reports no VDS sources — file may be corrupted.")
        return "\n".join(lines)

    lines.append(f"  VDS references {len(sources)} source file(s):")
    n_missing = 0
    seen = set()
    for src in sources:
        try:
            src_file = src.file_name
            src_path = src.dset_name
        except AttributeError:
            try:
                src_file = src[0]
                src_path = src[1]
            except Exception:
                lines.append(f"    (unparseable source)")
                continue
        key = (src_file, src_path)
        if key in seen:
            continue
        seen.add(key)
        try:
            master_dir = os.path.dirname(os.path.abspath(dataset.file.filename))
        except Exception:
            master_dir = ""
        if not os.path.isabs(src_file):
            full_src = os.path.join(master_dir, src_file)
        else:
            full_src = src_file
        exists = os.path.exists(full_src)
        marker = "OK   " if exists else "MISS "
        if not exists:
            n_missing += 1
        lines.append(f"    [{marker}] {src_file}  ->  {src_path}")
    if n_missing:
        lines.append(f"\n  {n_missing} source file(s) are MISSING from "
                     f"the directory. The master file expects them next to "
                     f"itself. Copy them in, or open one of the existing "
                     f"chunk files directly.")
    return "\n".join(lines)


def _read_dataset_safely(ds, dpath: str, max_frames: int = None,
                          verbose: bool = False):
    """
    Read a dataset, with graceful fallback if it's a VDS whose sources fail,
    or if it requires a compression plugin we don't have.

    Strategy:
        0. If the dataset uses a compression filter that needs hdf5plugin
           and hdf5plugin is NOT installed, raise a clear error.
        1. If the dataset is a VDS, check whether all its source files exist.
           If any are missing, REFUSE the bulk read and raise a clear error.
        2. Otherwise, try bulk read.
        3. If bulk read fails with "can't open directory" (= compression
           plugin missing, or VDS source missing), report a useful error.
    """
    # Step 0: detect compression plugin requirements BEFORE attempting read
    filter_desc = _dataset_requires_plugin(ds)
    if filter_desc is not None and not _HDF5_PLUGINS_AVAILABLE:
        raise IOError(
            f"Dataset '{dpath}' is compressed with {filter_desc}.\n"
            f"This filter requires the `hdf5plugin` Python package,\n"
            f"which is NOT installed on your system.\n\n"
            f"FIX: install hdf5plugin and restart CDI-ST:\n"
            f"    pip install hdf5plugin\n\n"
            f"After installing, this dataset will read correctly. The\n"
            f"hdf5plugin package bundles the compression DLLs for Eiger,\n"
            f"Lambda, and other detectors used at synchrotron beamlines."
        )

    # Step 1: pre-check VDS source reachability
    if hasattr(ds, 'is_virtual') and ds.is_virtual:
        if not _vds_is_readable(ds):
            diag = diagnose_vds(ds, dpath)
            raise IOError(
                f"Dataset '{dpath}' is a Virtual Dataset with MISSING "
                f"source files. h5py would silently return ZEROS, so we "
                f"refuse the read to avoid producing garbage output.\n\n"
                f"{diag}\n\n"
                f"To proceed:\n"
                f"  - Open the chunk file's OWN data path "
                f"(usually /entry/data/data), not the master's VDS, OR\n"
                f"  - Provide all missing chunk files in the same directory, OR\n"
                f"  - Use load_p10_chunks_directly() to bypass VDS entirely."
            )

    # Step 2: bulk read
    try:
        if len(ds.shape) == 4:
            if ds.shape[1] == 1:
                ds_arr = ds[:, 0, :, :]
            else:
                ds_arr = ds[...]
        else:
            ds_arr = ds[...]
        if max_frames is not None and ds_arr.shape[0] > max_frames:
            start = (ds_arr.shape[0] - max_frames) // 2
            ds_arr = ds_arr[start:start + max_frames]
        return np.asarray(ds_arr, dtype=np.float32)
    except (OSError, IOError, RuntimeError) as e:
        bulk_err = str(e)
        # Triage the error:
        # 1. If it looks like a compression-plugin issue AND the dataset
        #    actually uses a plugin filter, raise the clear plugin error.
        if _is_compression_plugin_error(e):
            filter_desc = _dataset_requires_plugin(ds)
            if filter_desc is not None and not _HDF5_PLUGINS_AVAILABLE:
                raise IOError(
                    f"Dataset '{dpath}' uses {filter_desc} compression but\n"
                    f"hdf5plugin is not installed. h5py can decompress the\n"
                    f"metadata but not the actual data.\n\n"
                    f"FIX: install hdf5plugin and restart CDI-ST:\n"
                    f"    pip install hdf5plugin\n\n"
                    f"This is normal for Eiger / Lambda / Pilatus detector data."
                ) from e
            if filter_desc is not None and _HDF5_PLUGINS_AVAILABLE:
                # hdf5plugin IS installed but the filter still failed —
                # uncommon edge case (older hdf5plugin missing this filter)
                raise IOError(
                    f"Dataset '{dpath}' uses {filter_desc} compression and\n"
                    f"hdf5plugin {_HDF5_PLUGINS_VERSION} is installed but the\n"
                    f"filter cannot be loaded. Try upgrading:\n"
                    f"    pip install --upgrade hdf5plugin\n\n"
                    f"Original error: {e}"
                ) from e
        # 2. If it's a VDS source issue, fall through to the VDS recovery
        if "open directory" not in bulk_err and "external" not in bulk_err.lower():
            # Not VDS, not plugin — re-raise as-is
            raise

    # Step 3: VDS read failed at runtime -> diagnose
    diag = diagnose_vds(ds, dpath)
    if verbose:
        print(f"  Bulk read failed (VDS source missing). Diagnosing:")
        print(diag)
        print(f"  Attempting frame-by-frame read to skip broken sources...")

    # Try frame-by-frame to salvage what we can
    n_frames = ds.shape[0]
    salvaged = []
    skipped = 0
    for i in range(n_frames):
        try:
            if len(ds.shape) == 4:
                frame = ds[i, 0] if ds.shape[1] == 1 else ds[i]
            else:
                frame = ds[i]
            salvaged.append(np.asarray(frame, dtype=np.float32))
        except (OSError, IOError, RuntimeError):
            skipped += 1
            continue
    if not salvaged:
        raise IOError(
            f"Could not read any frames from {dpath}.\n\n{diag}\n\n"
            f"All {n_frames} frames are inaccessible. To fix this:\n"
            f"  - Either open the master file in CDI-ST (master finds chunks "
            f"automatically), OR\n"
            f"  - Copy ALL of the *_data_NNNNNN.h5 chunk files into the same "
            f"directory as this file, OR\n"
            f"  - If you only have one chunk file, give CDI-ST a dataset path "
            f"that lives INSIDE the chunk (not the master's VDS). The chunk "
            f"normally has its data at /entry/data/data."
        )
    if verbose and skipped:
        print(f"  Skipped {skipped} broken frames; recovered "
              f"{len(salvaged)} frames.")
    return np.stack(salvaged).astype(np.float32)


def inspect_h5(path: str, max_depth: int = 6):
    """Print the full tree of an .h5 file to help locate the diffraction data.

    Unlike a plain visititems() walk, this also enumerates EXTERNAL LINKS
    (the most common P10/Eiger pattern), which are otherwise invisible.
    """
    import os
    try:
        import h5py
    except ImportError:
        print("h5py not installed. Run: pip install h5py")
        return

    print(f"\nStructure of {path}:")
    print("=" * 62)
    base_dir = os.path.dirname(os.path.abspath(path))
    seen = set()

    def _show_links_in(grp, path_prefix, indent):
        """Enumerate keys including external/soft links."""
        try:
            keys = list(grp.keys())
        except Exception:
            return
        for k in keys:
            full = f"{path_prefix}/{k}"
            if full in seen:
                continue
            seen.add(full)
            try:
                link = grp.get(k, getlink=True)
            except Exception:
                link = None
            if isinstance(link, h5py.ExternalLink):
                tgt = link.filename
                tgt_full = os.path.join(base_dir, tgt) if not os.path.isabs(tgt) else tgt
                exists = "OK" if os.path.exists(tgt_full) else "MISSING"
                print(f"{indent}{full}  → external link to "
                      f"'{tgt}'::'{link.path}'  [{exists}]")
            elif isinstance(link, h5py.SoftLink):
                print(f"{indent}{full}  → soft link to '{link.path}'")
            else:
                # Real dataset or group
                try:
                    obj = grp[k]
                except Exception:
                    print(f"{indent}{full}  (unreadable)")
                    continue
                if isinstance(obj, h5py.Dataset):
                    s = f"{indent}{full}  shape={obj.shape}  dtype={obj.dtype}"
                    if len(obj.shape) == 3 and obj.shape[0] > 10:
                        s += "  ← likely diffraction volume"
                    elif len(obj.shape) == 3:
                        s += "  ← 3D dataset"
                    if hasattr(obj, 'is_virtual') and obj.is_virtual:
                        s += "  [VDS]"
                    print(s)
                elif isinstance(obj, h5py.Group):
                    depth = (full.count('/'))
                    print(f"{indent}{full}/")
                    if depth < max_depth:
                        _show_links_in(obj, full, indent + '  ')

    with h5py.File(path, 'r') as f:
        _show_links_in(f, "", "")
    print("=" * 62)


def _vds_is_readable(ds) -> bool:
    """
    Check if a VDS's source files are reachable WITHOUT actually reading.
    Returns True if not VDS, or if VDS and at least one source exists.
    """
    import os
    try:
        if not (hasattr(ds, 'is_virtual') and ds.is_virtual):
            return True   # not VDS, assume readable
        srcs = ds.virtual_sources()
        if not srcs:
            return True   # claims VDS but no sources -> let read decide
        master_dir = os.path.dirname(os.path.abspath(ds.file.filename))
        for src in srcs:
            try:
                src_file = src.file_name
            except AttributeError:
                try:
                    src_file = src[0]
                except Exception:
                    continue
            full = src_file if os.path.isabs(src_file) else os.path.join(master_dir, src_file)
            if os.path.exists(full):
                return True  # at least one source file is reachable
        return False  # VDS with all sources missing
    except Exception:
        return True   # be permissive on introspection errors


def enumerate_external_links(h5_file, group_path: str = "/entry/data") -> list:
    """
    Eiger / DESY HDF5 master files store frame data as a series of
    EXTERNAL LINKS inside `/entry/data/`. Each link has a name like
    `data_000001`, `data_000002`, ... and points to a separate chunk file.

    External links are NOT visited by `h5py.File.visititems()`, which is why
    `inspect_h5` shows `/entry/data/` as an empty group even though it
    contains hundreds of links.

    This function explicitly enumerates the group's keys (which DOES include
    external links) and returns information about each one.

    Returns
    -------
    list of dicts, one per entry:
        {
          'name':        'data_000001',
          'link_type':   'external' | 'soft' | 'dataset' | 'group',
          'target_file': '<scan>_data_000001.h5'  (only for external),
          'target_path': '/entry/data/data'       (only for external/soft),
        }
    """
    import h5py
    try:
        grp = h5_file[group_path]
    except (KeyError, OSError):
        return []
    if not isinstance(grp, h5py.Group):
        return []

    out = []
    try:
        keys = list(grp.keys())
    except Exception:
        return []

    for k in keys:
        try:
            link = grp.get(k, getlink=True)
        except Exception:
            link = None
        entry = {'name': k, 'link_type': 'dataset'}
        if isinstance(link, h5py.ExternalLink):
            entry['link_type'] = 'external'
            entry['target_file'] = link.filename
            entry['target_path'] = link.path
        elif isinstance(link, h5py.SoftLink):
            entry['link_type'] = 'soft'
            entry['target_path'] = link.path
        else:
            try:
                obj = grp[k]
                if isinstance(obj, h5py.Dataset):
                    entry['link_type'] = 'dataset'
                    entry['shape'] = obj.shape
                elif isinstance(obj, h5py.Group):
                    entry['link_type'] = 'group'
            except Exception:
                entry['link_type'] = 'unknown'
        out.append(entry)
    return out


def p10_scan_info(
    h5_path: str,
    group_path: str = "/entry/data",
) -> dict:
    """
    Cheap preview of a P10 master file: enumerate external links, probe
    each chunk for frame count + frame shape, and return a summary suitable
    for showing in a GUI before the user clicks Convert.

    This does NOT decompress any data — it only reads HDF5 metadata, so it
    is essentially instant even for 381-chunk datasets.

    Parameters
    ----------
    h5_path : str
        Path to the master .h5 file (or any chunk file in the set).
    group_path : str
        HDF5 group containing the external links. Default ``/entry/data``.

    Returns
    -------
    info : dict with keys
        - ``master``     : the resolved master file path
        - ``n_chunks``   : number of external links found
        - ``n_missing``  : how many chunk files don't exist on disk
        - ``n_frames``   : total frames across all readable chunks
        - ``frame_h``    : detector height (pixels)
        - ``frame_w``    : detector width (pixels)
        - ``dtype``      : numpy dtype string (e.g. 'uint32')
        - ``size_full_gb``: GB needed to load all frames at full detector
        - ``frames_per_chunk``: list of (chunk_name, n_frames_in_chunk)
    """
    import os, h5py
    # Resolve to master if a chunk was passed
    master = _find_p10_master(h5_path) or h5_path
    master_dir = os.path.dirname(os.path.abspath(master))

    with h5py.File(master, 'r') as f:
        links = enumerate_external_links(f, group_path)
    ext_links = [L for L in links if L['link_type'] == 'external']

    n_missing = 0
    frames_per_chunk = []
    n_frames = 0
    frame_h = frame_w = 0
    dtype_str = '?'
    for L in ext_links:
        tgt_file = L['target_file']
        tgt_path = L['target_path']
        chunk_full = (tgt_file if os.path.isabs(tgt_file)
                       else os.path.join(master_dir, tgt_file))
        if not os.path.exists(chunk_full):
            n_missing += 1
            frames_per_chunk.append((os.path.basename(chunk_full), 0))
            continue
        try:
            with h5py.File(chunk_full, 'r') as chunk:
                actual_path = (tgt_path if tgt_path in chunk
                               else _find_chunk_internal_data_path(chunk, hint_path=tgt_path))
                if actual_path is None:
                    continue
                ds = chunk[actual_path]
                shape = ds.shape
                n_f = shape[0] if len(shape) == 3 else 1
                h = shape[-2]; w = shape[-1]
                n_frames += n_f
                frame_h = max(frame_h, h)
                frame_w = max(frame_w, w)
                dtype_str = str(ds.dtype)
                frames_per_chunk.append((os.path.basename(chunk_full), n_f))
        except Exception:
            frames_per_chunk.append((os.path.basename(chunk_full), 0))
            continue

    size_full_gb = (n_frames * frame_h * frame_w * 4) / (1024 ** 3)
    return {
        'master': master,
        'n_chunks': len(ext_links),
        'n_missing': n_missing,
        'n_frames': n_frames,
        'frame_h': frame_h,
        'frame_w': frame_w,
        'dtype': dtype_str,
        'size_full_gb': size_full_gb,
        'frames_per_chunk': frames_per_chunk,
    }


def load_p10_from_external_links(
    master_path: str,
    group_path: str = "/entry/data",
    max_frames: int = None,
    frame_range: tuple = None,
    detector_roi: tuple = None,
    memory_budget_gb: float = 8.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Read P10 / Eiger data from a master file by manually resolving its
    EXTERNAL LINKS to chunk files.

    This is the correct strategy when the master file contains entries like
    `/entry/data/data_000001` that are external links to separate chunk files
    (the standard Eiger/DESY layout). Each chunk file contains contiguous
    detector data at `/entry/data/data`.

    The reader applies optional ``frame_range`` and ``detector_roi`` *during*
    the read, so reading e.g. 100 frames out of 3000 only allocates the
    100-frame volume — never the full 49-GB stack.

    Parameters
    ----------
    master_path : str
        Path to the master .h5 file (typically ``<scan>_master.h5``).
    group_path : str
        HDF5 group containing the external links. Default ``/entry/data``.
    max_frames : int or None
        Truncate the FINAL volume to this many central frames (legacy).
        Ignored if ``frame_range`` is also given.
    frame_range : tuple ``(start, stop)`` or None
        Read only frames ``[start, stop)`` (Python slice semantics) from the
        *global* frame index across all chunks. ``start=0, stop=None`` means
        "from start to end". This is applied chunk-by-chunk so memory usage
        scales with the requested range, not the total dataset size.
    detector_roi : tuple ``(y0, y1, x0, x1)`` or None
        Read only this detector region per frame. Indices are half-open
        like Python slices. Cuts memory by ``(2167*2070)/(roi area)`` for
        Eiger 4M data.
    memory_budget_gb : float
        Maximum acceptable memory footprint (in GB) for the concatenated
        volume. If the requested read would exceed this, the function
        raises ``MemoryError`` with a suggested ``frame_range`` and
        ``detector_roi`` instead of crashing the Python process.
    verbose : bool

    Returns
    -------
    volume : ndarray, shape (N_frames_selected, H, W), float32
    """
    import os, h5py
    master_dir = os.path.dirname(os.path.abspath(master_path))

    with h5py.File(master_path, 'r') as master:
        links = enumerate_external_links(master, group_path)

    if not links:
        raise IOError(
            f"No external links found in '{group_path}' of {master_path}.\n"
            f"This file may not be a P10 master file."
        )

    # Filter to only external links (skip groups, etc.)
    ext_links = [L for L in links if L['link_type'] == 'external']
    if verbose:
        print(f"  Master '{os.path.basename(master_path)}' has "
              f"{len(ext_links)} external link(s) to chunk files")
        for L in ext_links[:3]:
            tgt = L.get('target_file', '?')
            exists = os.path.exists(os.path.join(master_dir, tgt))
            mark = "OK" if exists else "MISS"
            print(f"    [{mark}] {L['name']} -> {tgt} :: {L.get('target_path', '?')}")
        if len(ext_links) > 3:
            print(f"    ... ({len(ext_links) - 3} more)")

    if not ext_links:
        raise IOError(
            f"Group '{group_path}' contains {len(links)} entries but none "
            f"are external links. Inspect the file structure to see what's "
            f"there.\nFirst few entries: "
            f"{[L['name'] + '(' + L['link_type'] + ')' for L in links[:5]]}"
        )

    # ── Pass 1 (cheap): probe each chunk for its frame count + dtype + shape
    # We open each file just to read the dataset metadata (essentially free
    # in HDF5), no data is decompressed yet.
    chunk_info = []   # list of (chunk_full, internal_path, n_frames, h, w, dtype)
    n_missing = 0
    for L in ext_links:
        tgt_file = L['target_file']
        tgt_path = L['target_path']
        chunk_full = (tgt_file if os.path.isabs(tgt_file)
                       else os.path.join(master_dir, tgt_file))
        if not os.path.exists(chunk_full):
            n_missing += 1
            continue
        try:
            with h5py.File(chunk_full, 'r') as chunk:
                if tgt_path in chunk:
                    ds = chunk[tgt_path]
                    actual_path = tgt_path
                else:
                    actual_path = _find_chunk_internal_data_path(
                        chunk, hint_path=tgt_path
                    )
                    if actual_path is None:
                        continue
                    ds = chunk[actual_path]
                shape = ds.shape
                if len(shape) == 2:
                    n_f, h, w = 1, shape[0], shape[1]
                elif len(shape) == 3:
                    n_f, h, w = shape
                else:
                    continue
                chunk_info.append((chunk_full, actual_path, n_f, h, w, ds.dtype))
        except Exception as e:
            if _is_compression_plugin_error(e) and not _HDF5_PLUGINS_AVAILABLE:
                raise IOError(
                    f"The chunk files use a compression filter that requires "
                    f"hdf5plugin (most likely bitshuffle+LZ4, standard for "
                    f"Eiger detectors).\n\n"
                    f"FIX: install hdf5plugin and restart CDI-ST:\n"
                    f"    pip install hdf5plugin\n\n"
                    f"After installing, this dataset will read correctly."
                ) from e
            continue

    if not chunk_info:
        raise IOError(
            f"Could not enumerate any chunks from master {master_path}.\n"
            f"  {n_missing} of {len(ext_links)} chunk files are MISSING from\n"
            f"  {master_dir}.\n\n"
            f"Copy all `<scan>_data_NNNNNN.h5` files into the same directory."
        )

    # ── Resolve frame_range against total frames
    n_total_frames = sum(ci[2] for ci in chunk_info)
    h0 = chunk_info[0][3]; w0 = chunk_info[0][4]
    if frame_range is None:
        global_start, global_stop = 0, n_total_frames
    else:
        global_start = max(0, int(frame_range[0]))
        global_stop = (n_total_frames if frame_range[1] is None
                       else min(n_total_frames, int(frame_range[1])))
    n_requested = max(0, global_stop - global_start)
    if n_requested == 0:
        raise ValueError(f"Empty frame range: {frame_range}")

    # ── Resolve detector_roi
    if detector_roi is None:
        y0, y1, x0, x1 = 0, h0, 0, w0
    else:
        y0, y1, x0, x1 = detector_roi
        y0, y1 = max(0, int(y0)), min(h0, int(y1))
        x0, x1 = max(0, int(x0)), min(w0, int(x1))
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"Empty detector ROI: {detector_roi}")

    # ── Memory budget check
    estimated_bytes = n_requested * (y1 - y0) * (x1 - x0) * 4   # float32
    estimated_gb = estimated_bytes / (1024 ** 3)
    if verbose:
        if frame_range is not None or detector_roi is not None:
            print(f"  Read plan: frames [{global_start}:{global_stop}] of "
                  f"{n_total_frames}, detector ROI [{y0}:{y1}, {x0}:{x1}] of "
                  f"[{h0}, {w0}] -> {estimated_gb:.2f} GB")
        else:
            print(f"  Read plan: all {n_total_frames} frames, full detector "
                  f"[{h0}, {w0}] -> {estimated_gb:.2f} GB")
    if estimated_gb > memory_budget_gb:
        # Suggest a sensible reduction
        # If frame_range can save us, prefer that. Otherwise suggest ROI too.
        suggest_n_frames = int(memory_budget_gb * (1024 ** 3) /
                                ((y1 - y0) * (x1 - x0) * 4))
        center = (global_start + global_stop) // 2
        sug_start = max(global_start, center - suggest_n_frames // 2)
        sug_stop = min(global_stop, sug_start + suggest_n_frames)
        suggestion = (
            f"\n\nSuggestion: pass frame_range=({sug_start}, {sug_stop}) to "
            f"load only the central {suggest_n_frames} frames around "
            f"the rocking-curve maximum (~{memory_budget_gb:.0f} GB)."
        )
        if (y1 - y0) * (x1 - x0) == h0 * w0:
            # Full detector — also suggest an ROI
            suggestion += (
                f"\nFor BCDI you typically only need a small ROI around the "
                f"Bragg peak — e.g., detector_roi=(<peak_y>-128, <peak_y>+128, "
                f"<peak_x>-128, <peak_x>+128)."
            )
        raise MemoryError(
            f"Reading the requested data would allocate ~{estimated_gb:.1f} GB "
            f"of RAM, which exceeds the memory_budget_gb={memory_budget_gb:.1f} "
            f"safety limit.{suggestion}\n\n"
            f"In the GUI: open the P10 dialog, set 'Frame range' and/or "
            f"'Detector ROI', and try again."
        )

    # ── Pass 2: read each chunk's slice within the requested frame range
    out = np.empty(
        (n_requested, y1 - y0, x1 - x0), dtype=np.float32
    )
    cursor = 0    # write position in `out`
    global_frame = 0    # running global frame index across chunks
    n_ok = 0
    for (chunk_full, internal_path, n_f, _h, _w, _dtype) in chunk_info:
        chunk_global_start = global_frame
        chunk_global_stop = global_frame + n_f
        global_frame = chunk_global_stop

        # Compute intersection with [global_start, global_stop)
        overlap_start = max(chunk_global_start, global_start)
        overlap_stop = min(chunk_global_stop, global_stop)
        if overlap_stop <= overlap_start:
            continue   # this chunk is entirely outside the requested range

        # Local indices within this chunk
        local_start = overlap_start - chunk_global_start
        local_stop = overlap_stop - chunk_global_start
        try:
            with h5py.File(chunk_full, 'r') as chunk:
                ds = chunk[internal_path]
                if len(ds.shape) == 2:
                    arr = ds[y0:y1, x0:x1][None, :, :]
                else:
                    arr = ds[local_start:local_stop, y0:y1, x0:x1]
                n_read = arr.shape[0]
                out[cursor:cursor + n_read] = arr.astype(np.float32, copy=False)
                cursor += n_read
                n_ok += 1
        except Exception as e:
            if _is_compression_plugin_error(e) and not _HDF5_PLUGINS_AVAILABLE:
                raise IOError(
                    f"The chunk files use a compression filter that requires "
                    f"hdf5plugin (most likely bitshuffle+LZ4, standard for "
                    f"Eiger detectors).\n\n"
                    f"FIX: install hdf5plugin and restart CDI-ST:\n"
                    f"    pip install hdf5plugin"
                ) from e
            if verbose:
                print(f"    skip {os.path.basename(chunk_full)} (error: {e})")
            continue

    if cursor == 0:
        raise IOError(
            f"Could not read any chunks. Files may be unreadable.\n"
            f"  Master:  {master_path}\n"
            f"  Range:   frames [{global_start}:{global_stop}]"
        )

    volume = out[:cursor]
    if verbose:
        print(f"  Successfully read {volume.shape[0]} frames from {n_ok} "
              f"chunk(s) via external links")
        if n_missing > 0:
            print(f"  WARNING: {n_missing} of {len(ext_links)} chunks missing")

    if frame_range is None and max_frames is not None and volume.shape[0] > max_frames:
        start = (volume.shape[0] - max_frames) // 2
        volume = volume[start:start + max_frames]
        if verbose:
            print(f"  Truncated to {max_frames} central frames")

    return volume


def find_diffraction_dataset(h5_file):
    """
    Try known paths first, then scan for any suitable 3D dataset.

    Skips Virtual Datasets whose source files are missing — picks readable
    contiguous datasets instead. This handles the P10 case where a chunk
    file's /entry/data/ group also contains VDS placeholders (`data_000001`,
    `data_000002`) pointing to *other* chunk files that the user doesn't
    have. The chunk's own contiguous data at `/entry/data/data` is preferred.

    Returns (path_str, dataset) or (None, None) if nothing found.
    """
    import h5py

    # Step 1: try the known paths directly
    for p in KNOWN_PATHS:
        try:
            if p in h5_file:
                obj = h5_file[p]
                if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 3:
                    # Skip broken VDS — prefer next candidate
                    if _vds_is_readable(obj):
                        return p, obj
        except (KeyError, OSError):
            continue

    # Step 2: full tree walk
    candidates_readable = []   # (size, name, dataset, prefer_score)
    candidates_broken_vds = [] # only used as last resort
    seen = set()

    def _walk(group, prefix=""):
        try:
            keys = list(group.keys())
        except (KeyError, OSError):
            return
        for k in keys:
            full = f"{prefix}/{k}" if prefix else f"/{k}"
            if full in seen:
                continue
            seen.add(full)
            try:
                obj = group[k]
            except (KeyError, OSError):
                continue
            if isinstance(obj, h5py.Dataset):
                if len(obj.shape) >= 3:
                    sz = int(np.prod(obj.shape))
                    if sz > 10 ** 4:
                        if _vds_is_readable(obj):
                            # Prefer datasets named exactly "data" (NeXus
                            # convention) over numbered VDS placeholders
                            score = 1 if k == "data" else 0
                            candidates_readable.append((score, sz, full, obj))
                        else:
                            candidates_broken_vds.append((sz, full, obj))
            elif isinstance(obj, h5py.Group):
                _walk(obj, full)

    _walk(h5_file)
    if candidates_readable:
        # Sort by (NeXus-name preference desc, size desc) - higher score first
        candidates_readable.sort(key=lambda c: (-c[0], -c[1]))
        _, _, name, ds = candidates_readable[0]
        return name, ds
    if candidates_broken_vds:
        # All candidates are broken VDS - return the largest one so the caller
        # can surface a helpful diagnostic error to the user
        candidates_broken_vds.sort(reverse=True)
        _, name, ds = candidates_broken_vds[0]
        return name, ds

    return None, None


def resolve_dataset_path(h5_file, user_path: str):
    """
    Resolve a user-provided HDF5 path to an actual Dataset.

    Handles the common confusion where the user types a Group path
    (e.g. '/entry/data') when they should have typed the Dataset path
    inside it (e.g. '/entry/data/data'). When the user's path lands on
    a group, this looks one level deeper for a child named 'data' or
    for any 3D+ dataset inside.

    Returns
    -------
    (resolved_path, dataset)  or  raises a helpful error.
    """
    import h5py
    if user_path not in h5_file:
        raise KeyError(
            f"Path '{user_path}' does not exist in the HDF5 file.\n"
            f"  Use 'Inspect HDF5 structure' to see what's available."
        )
    obj = h5_file[user_path]

    # If user typed a dataset directly, return it
    if isinstance(obj, h5py.Dataset):
        if len(obj.shape) < 2:
            raise ValueError(
                f"'{user_path}' is a {len(obj.shape)}D dataset — need at "
                f"least 2D detector frames (or 3D scan stack)."
            )
        return user_path, obj

    # User typed a group → look inside for the actual data
    if isinstance(obj, h5py.Group):
        # First: try a child called 'data' (the NeXus standard convention)
        if 'data' in obj:
            child = obj['data']
            if isinstance(child, h5py.Dataset) and len(child.shape) >= 2:
                full = (user_path.rstrip('/') + '/data')
                return full, child
        # Second: scan the immediate children for any 3D+ dataset
        for k in obj.keys():
            try:
                child = obj[k]
            except (KeyError, OSError):
                continue
            if isinstance(child, h5py.Dataset) and len(child.shape) >= 3:
                full = (user_path.rstrip('/') + '/' + k)
                return full, child
        # Third: walk deeper — maybe two levels under user's group
        for k in obj.keys():
            try:
                sub = obj[k]
            except (KeyError, OSError):
                continue
            if isinstance(sub, h5py.Group):
                for k2 in sub.keys():
                    try:
                        ds = sub[k2]
                    except (KeyError, OSError):
                        continue
                    if isinstance(ds, h5py.Dataset) and len(ds.shape) >= 3:
                        full = f"{user_path.rstrip('/')}/{k}/{k2}"
                        return full, ds

        # Nothing found inside the group — list what IS there to help the user
        children = []
        for k in obj.keys():
            try:
                c = obj[k]
                if isinstance(c, h5py.Dataset):
                    children.append(f"  {k}  [Dataset, shape={c.shape}]")
                else:
                    children.append(f"  {k}/  [Group]")
            except Exception:
                children.append(f"  {k}  [?]")
        listing = "\n".join(children) if children else "  (empty group)"
        raise ValueError(
            f"'{user_path}' is a Group, not a Dataset.\n"
            f"Contents of '{user_path}':\n{listing}\n\n"
            f"Click one of the entries above (e.g. add '/data' to your path)."
        )

    raise TypeError(f"'{user_path}' is a {type(obj).__name__}, expected Dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing steps
# ═══════════════════════════════════════════════════════════════════════════════

def mask_detector_gaps(volume: np.ndarray,
                          detector: str = 'auto') -> np.ndarray:
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
    from scipy.ndimage import median_filter

    out = volume.copy().astype(np.float32)
    Nz, Ny, Nx = out.shape

    if detector == 'auto':
        if Ny == 516 and Nx == 516:
            detector = 'maxipix'
        elif Ny >= 1000 and Nx >= 1000:
            detector = 'eiger2M'
        else:
            detector = 'unknown'

    if detector == 'maxipix':
        gap_rows = list(range(255, 261))
        gap_cols = list(range(255, 261))
    elif detector == 'eiger2M':
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
    half_window = 9   # 19×19 window — wide enough to bridge a 6-pixel gap
    for z in range(Nz):
        gap_z = gap_mask[z]
        if not gap_z.any():
            continue
        non_gap = ~gap_z
        frame = out[z]
        gap_indices = np.argwhere(gap_z)
        for (gy, gx) in gap_indices:
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


def find_bragg_peak_box(volume: np.ndarray,
                          intensity_threshold_pct: float = 1.0,
                          margin_voxels: int = 8) -> tuple:
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
        # Empty volume — fall back to center coordinates (NOT slices).
        # The caller expects (cz, cy, cx) as integers, and crop_around_peak
        # later subtracts half from these. Returning slices here caused a
        # "unsupported operand type 'slice' and 'int'" error downstream.
        cz, cy, cx = [s // 2 for s in volume.shape]
        return (cz, cy, cx)

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

    # Auto-resolve P10-style "_data_NNNNNN.h5" → corresponding "_master.h5".
    # Opening a data chunk directly often fails with HDF5 errors like
    # "Can't synchronously read data (can't open directory)" because the
    # chunk references a virtual dataset that lives in the master file.
    original_path = path
    master_candidate = _find_p10_master(path)
    if master_candidate is not None:
        if verbose:
            print(f"  Detected P10 data file → using master: {master_candidate}")
        path = master_candidate

    try:
        f = h5py.File(path, 'r')
    except OSError as e:
        msg = str(e)
        # Translate the cryptic HDF5 error into something actionable.
        if "open directory" in msg or "external" in msg.lower():
            raise IOError(
                f"Could not open {path}: this file uses a Virtual Dataset "
                f"(VDS) that links to other files in the same directory.\n\n"
                f"  - If you have a '_master.h5' file next to this one, open "
                f"that one instead — CDI-ST will auto-find the data chunks.\n"
                f"  - If files are missing from the directory, restore them or "
                f"copy ALL related files (master + data_NNNNNN) together.\n\n"
                f"Original HDF5 error: {msg}"
            ) from e
        raise

    try:
        # Find the dataset
        if dataset_path is not None:
            # User provided an explicit path — but they might have given a
            # Group rather than a Dataset (the common confusion with NeXus
            # files where '/entry/data' is a Group containing the actual
            # 'data' Dataset). resolve_dataset_path handles both.
            dpath, ds = resolve_dataset_path(f, dataset_path)
            if dpath != dataset_path and verbose:
                print(f"  Note: '{dataset_path}' is a Group, resolved to "
                      f"Dataset at '{dpath}'")
        else:
            dpath, ds = find_diffraction_dataset(f)
            if ds is None:
                # Print the file structure to help debugging
                try:
                    print("\n  File structure (for debugging):")
                    def _walk(name, obj):
                        if hasattr(obj, 'shape'):
                            print(f"    {name}  shape={obj.shape}  dtype={obj.dtype}")
                    f.visititems(_walk)
                except Exception:
                    pass
                f.close()
                raise ValueError(
                    f"Could not auto-detect diffraction data in {path}. "
                    f"Run with inspect_h5() to see structure, then pass "
                    f"dataset_path='/path/to/data' explicitly."
                )

        if verbose:
            print(f"  Loaded dataset: {dpath}  shape={ds.shape}  dtype={ds.dtype}")
            # If it's a VDS, mention that proactively (helps diagnose later)
            try:
                if hasattr(ds, 'is_virtual') and ds.is_virtual:
                    n_src = len(ds.virtual_sources())
                    print(f"  This is a Virtual Dataset with {n_src} external "
                          f"source file(s).")
            except Exception:
                pass

        # Read with graceful VDS-fallback. If sources are missing, this
        # tries frame-by-frame and raises a helpful error if nothing readable.
        volume = _read_dataset_safely(ds, dpath, max_frames=max_frames,
                                       verbose=verbose)
    finally:
        f.close()

    # ── Preprocessing pipeline ────────────────────────────────────────────
    if verbose:
        print(f"  Raw volume:    shape={volume.shape}  "
              f"min={volume.min():.2e} max={volume.max():.2e} "
              f"mean={volume.mean():.2e}")

    # Remove negative values (sometimes present from background subtraction)
    volume = np.maximum(volume, 0)

    # CRITICAL: mask Maxipix / Eiger inter-chip gaps BEFORE other processing.
    # The gaps are zero rows/columns at fixed positions (e.g. 255-260 on
    # Maxipix). If left as zeros they:
    #   - Bias COM-based centering toward the gap position
    #   - Get misidentified as "streaks" by the streak detector
    #   - Break the Fourier transform during reconstruction
    # cdiutils and BCDI-utils both fill these gaps via local median.
    volume = mask_detector_gaps(volume, detector='auto')
    if verbose:
        print(f"  After gap mask: max={volume.max():.2e}  min_nonzero="
              f"{volume[volume > 0].min() if (volume > 0).any() else 0:.2e}")

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
        print(f"  Processed:     shape={volume.shape}  "
              f"peak_max={volume.max():.2e}  total_counts={volume.sum():.2e}")
        print(f"  Peak index:    {peak_idx} (target center {center})")

    return volume


# ═══════════════════════════════════════════════════════════════════════════════
# Spec/EDF reader (ID01 native format, before BLISS h5)
# ═══════════════════════════════════════════════════════════════════════════════

def list_spec_scans(spec_path: str) -> list:
    """
    Return a list of all scans in a SPEC file with summary info, so the GUI
    can show the user which scan number corresponds to which frame range.

    Returns a list of dicts with keys:
        scan_number  — the #S N integer (1, 2, 3, ...)
        command      — the scan command, e.g. "ascan eta 33.5 34.5 100 1"
        n_points     — number of measurement points
        frame_min    — first CCD frame number (from mpx4inr column)
        frame_max    — last CCD frame number
    """
    out = []
    try:
        from silx.io.specfile import SpecFile
        sf = SpecFile(spec_path)
        try:
            keys = list(sf.keys())
        except Exception:
            keys = [f"{i+1}.1" for i in range(len(sf))]
        for key in keys:
            try:
                scan = sf[key]
                try:
                    scan_num = int(str(key).split(".")[0])
                except ValueError:
                    continue
                cmd = ""
                try:
                    header = scan.header
                    if header and len(header) > 0:
                        first = header[0]
                        if first.startswith("#S"):
                            parts = first.split(None, 2)
                            cmd = parts[2] if len(parts) > 2 else ""
                except Exception:
                    pass
                frame_min = frame_max = None
                n_points = 0
                try:
                    labels = list(scan.labels)
                    if "mpx4inr" in labels:
                        try:
                            frames = scan.data_column_by_name("mpx4inr")
                        except Exception:
                            i = labels.index("mpx4inr")
                            d = scan.data
                            frames = d[:, i] if d.shape[1] == len(labels) else d[i]
                        frames = np.asarray(frames)
                        if frames.size > 0:
                            frame_min = int(frames.min())
                            frame_max = int(frames.max())
                            n_points = int(frames.size)
                except Exception:
                    pass
                out.append({
                    "scan_number": scan_num,
                    "command": cmd,
                    "n_points": n_points,
                    "frame_min": frame_min,
                    "frame_max": frame_max,
                })
            except Exception:
                continue
    except ImportError:
        # Without silx, do a simple text parse of the SPEC file
        with open(spec_path, "r", encoding="latin-1") as f:
            current_scan = None
            in_data = False
            mpx_col = None
            for line in f:
                if line.startswith("#S "):
                    if current_scan is not None:
                        out.append(current_scan)
                    parts = line.split(None, 2)
                    try:
                        n = int(parts[1])
                    except (IndexError, ValueError):
                        n = -1
                    current_scan = {
                        "scan_number": n,
                        "command": parts[2].strip() if len(parts) > 2 else "",
                        "n_points": 0,
                        "frame_min": None,
                        "frame_max": None,
                    }
                    in_data = False
                    mpx_col = None
                elif line.startswith("#L ") and current_scan is not None:
                    labels_for_scan = line[3:].strip().split()
                    if "mpx4inr" in labels_for_scan:
                        mpx_col = labels_for_scan.index("mpx4inr")
                    in_data = True
                elif in_data and current_scan is not None and not line.startswith("#"):
                    fields = line.strip().split()
                    if mpx_col is not None and mpx_col < len(fields):
                        try:
                            v = int(float(fields[mpx_col]))
                            if current_scan["frame_min"] is None or v < current_scan["frame_min"]:
                                current_scan["frame_min"] = v
                            if current_scan["frame_max"] is None or v > current_scan["frame_max"]:
                                current_scan["frame_max"] = v
                            current_scan["n_points"] += 1
                        except ValueError:
                            pass
            if current_scan is not None:
                out.append(current_scan)
    return out


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
        # IMPORTANT: modern silx (>=0.5) returns data with shape
        # (nlines, ncolumns). To get the column for label i, we need
        # data[:, i], NOT data[i]. Using data[i] gives one measurement
        # point's row of values across all columns — completely
        # misaligned with the label list.
        # Use scan.data_column_by_name() when available (robust); fall
        # back to direct indexing with shape detection.
        result = {}
        for i, label in enumerate(labels):
            col = None
            try:
                col = scan.data_column_by_name(label)
            except Exception:
                if data.ndim == 2:
                    if data.shape[1] == len(labels):
                        col = data[:, i]              # (nlines, ncols)
                    else:
                        col = data[i]                 # (ncols, nlines) legacy
                else:
                    col = data
            result[label] = np.asarray(col)
        # Header motors
        motor_names = scan.motor_names
        motor_positions = scan.motor_positions
        result['header_motors'] = dict(zip(motor_names, motor_positions))
    except ImportError:
        try:
            import spec
            h, d = spec.ReadSpec(spec_path, scan_number)
            result = dict(d)
            result['header_motors'] = dict(h.get('motors', {}))
        except ImportError:
            raise ImportError(
                "Need either silx (pip install silx) or spec (pip install spec) "
                "to read SPEC files."
            )

    # Determine rocking axis: which motor varies most over the scan?
    n_pts = len(result.get('mpx4inr', result.get(list(result.keys())[0])))
    rocking_type = None
    if 'eta' in result and len(result['eta']) == n_pts:
        eta_arr = np.asarray(result['eta'])
        if eta_arr.std() > 0.01:
            rocking_type = 'eta'
    if rocking_type is None and 'phi' in result and len(result['phi']) == n_pts:
        phi_arr = np.asarray(result['phi'])
        if phi_arr.std() > 0.01:
            rocking_type = 'phi'

    # Fill in fixed motors from header for rocking case
    h_motors = result['header_motors']
    if rocking_type == 'eta':
        result['phi'] = np.full(n_pts, h_motors.get('phi', 0.0))
        result['delta'] = np.full(n_pts, h_motors.get('del', h_motors.get('delta', 0.0)))
        result['nu'] = np.full(n_pts, h_motors.get('nu', 0.0))
    elif rocking_type == 'phi':
        result['eta'] = np.full(n_pts, h_motors.get('eta', 0.0))
        result['delta'] = np.full(n_pts, h_motors.get('del', h_motors.get('delta', 0.0)))
        result['nu'] = np.full(n_pts, h_motors.get('nu', 0.0))

    result['rocking_type'] = rocking_type
    return result


def read_edf_stack(edf_template: str, frame_numbers: list,
                    detector_shape: tuple = (516, 516)) -> np.ndarray:
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
    n_loaded = 0
    failed_examples = []
    for idx, frame in enumerate(frame_numbers):
        f = edf_template % int(frame)
        try:
            e = xu.io.EDFFile(f)
            volume[idx] = e.data.astype(np.float32)
            n_loaded += 1
        except Exception as ex:
            if len(failed_examples) < 3:
                failed_examples.append((f, str(ex)))

    if n_loaded == 0:
        # Every single frame failed to load. Give a clear, actionable error
        # rather than letting the rest of the pipeline crash on a zero volume.
        examples = "\n".join(f"  - {p} : {e}" for p, e in failed_examples)
        raise FileNotFoundError(
            f"Could not load ANY of the {n} EDF frames "
            f"(frame numbers {min(frame_numbers)}-{max(frame_numbers)}).\n\n"
            f"First few failures:\n{examples}\n\n"
            f"Likely causes:\n"
            f"  1. The chosen SPEC scan references frame numbers that don't\n"
            f"     match the EDF files in your folder. Use 'Browse scans…'\n"
            f"     to find the scan whose mpx4inr range matches your files.\n"
            f"  2. The filename template (e.g. '{edf_template}') is wrong\n"
            f"     for your data (check: %05d zero-pad, .edf vs .edf.gz).\n"
            f"  3. The EDF directory path is incorrect."
        )
    if n_loaded < n:
        print(f"  Warning: loaded {n_loaded}/{n} frames; "
              f"{n - n_loaded} missing (left as zero)")
    return volume


def load_spec_edf_scan(
    spec_path: str,
    scan_number: int,
    edf_dir: str,
    edf_template_name: str = "data_mpx4_%05d.edf.gz",
    target_size: int = 64,
    detector_shape: tuple = (516, 516),
    detector: str = 'maxipix',
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
    rocking_type = spec_data['rocking_type']
    if verbose:
        print(f"  Rocking axis: {rocking_type}")

    # 2. Load detector frames
    edf_template = os.path.join(edf_dir, edf_template_name)
    frame_numbers = np.asarray(spec_data['mpx4inr']).astype(int)
    if verbose:
        print(f"  Scan {scan_number} references frames "
              f"{int(frame_numbers.min())}-{int(frame_numbers.max())} "
              f"({len(frame_numbers)} total)")
        print(f"  Loading from {edf_dir}")
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
        'diffraction': cropped.astype(np.float32),
        'eta': np.asarray(spec_data['eta']),
        'phi': np.asarray(spec_data['phi']),
        'nu': np.asarray(spec_data['nu']),
        'delta': np.asarray(spec_data['delta']),
        'rocking_type': rocking_type,
        'frame_numbers': frame_numbers,
        'header_motors': spec_data['header_motors'],
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
        chpdeg = [406.3, 406.3]      # channels per degree

        qconv = xu.experiment.QConversion(
            ['y-', 'z-'], ['z-', 'y-'], [1, 0, 0]
        )
        hxrd = xu.experiment.HXRD([1, 0, 0], [0, 0, 1],
                                    en=beam_energy_eV, qconv=qconv)
        hxrd.Ang2Q.init_area('z-', 'y+',
                              cch1=cch1, cch2=cch2,
                              Nch1=detector_shape[0], Nch2=detector_shape[1],
                              chpdeg1=chpdeg[0], chpdeg2=chpdeg[1])

        qx, qy, qz = hxrd.Ang2Q.area(result['eta'], result['phi'],
                                       result['nu'], result['delta'])
        # qx, qy, qz are 3D arrays of q-coordinates per pixel.
        # For voxel size estimation, take the mean step in each direction.
        # (Proper interpolation onto orthogonal grid would use Gridder3D.)
        dqx = float(np.mean(np.diff(qx, axis=0)))
        dqy = float(np.mean(np.diff(qy, axis=1)))
        dqz = float(np.mean(np.diff(qz, axis=2))) if qz.shape[2] > 1 else 1.0
        result['q_step_inv_A'] = (abs(dqx), abs(dqy), abs(dqz))
        # Real-space voxel size: dr = 2π / (N · dq), in nm
        # (10× factor converts Å to nm)
        N = target_size
        result['voxel_size_nm'] = np.array([
            2 * np.pi / (N * abs(dqx) * 10) if abs(dqx) > 1e-12 else 1.0,
            2 * np.pi / (N * abs(dqy) * 10) if abs(dqy) > 1e-12 else 1.0,
            2 * np.pi / (N * abs(dqz) * 10) if abs(dqz) > 1e-12 else 1.0,
        ], dtype=np.float32)
        if verbose:
            vn = result['voxel_size_nm']
            print(f"  Voxel pitch: ({vn[0]:.3f}, {vn[1]:.3f}, {vn[2]:.3f}) nm")
    except ImportError:
        if verbose:
            print("  xrayutilities not installed — q-space conversion skipped.")
            print("  Install with: pip install xrayutilities")
    except Exception as e:
        if verbose:
            print(f"  q-space conversion failed: {e}")

    return result


def _find_chunk_internal_data_path(h5_file, hint_path: str = None) -> Optional[str]:
    """
    For a P10-style chunk file, find the path to its own contiguous data.

    Different P10 configurations store data at different paths inside the
    chunk:
      - /entry/data/data            (NeXus standard)
      - /entry/data/data_000001     (P10 with numbered datasets — common)
      - /entry/instrument/eiger_4m/data
      - /entry/instrument/eiger_500k/data
      - /entry/instrument/detector/data
      - /entry_1/data_1/data        (CXI)
      - /entry/data_NNNN            (some BLISS setups)

    This function scans the file and returns the path that:
      (1) is a real Dataset (not a Group, not a broken VDS)
      (2) is 3D with substantial size (> 10k voxels)
      (3) has reachable data (or is contiguous, not VDS)
    """
    import h5py

    # If hint_path is provided and valid, try it first
    if hint_path and hint_path in h5_file:
        try:
            obj = h5_file[hint_path]
            if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                if _vds_is_readable(obj):
                    return hint_path
        except Exception:
            pass

    # Try common P10/NeXus paths first
    for p in [
        '/entry/data/data',
        '/entry/instrument/eiger_4m/data',
        '/entry/instrument/eiger_500k/data',
        '/entry/instrument/detector/data',
        '/entry/instrument/lambda/data',
        '/entry_1/data_1/data',
        '/entry_1/instrument_1/detector_1/data',
    ]:
        try:
            if p in h5_file:
                obj = h5_file[p]
                if (isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2
                        and _vds_is_readable(obj)):
                    return p
        except Exception:
            continue

    # Scan inside /entry/data/ for ANY 3D dataset
    # (this catches the data_NNNNNN naming convention)
    for parent_path in ['/entry/data', '/entry/instrument']:
        if parent_path not in h5_file:
            continue
        parent = h5_file[parent_path]
        if not isinstance(parent, h5py.Group):
            continue
        candidates = []
        for k in parent.keys():
            try:
                obj = parent[k]
            except Exception:
                continue
            if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                if _vds_is_readable(obj):
                    sz = int(np.prod(obj.shape))
                    candidates.append((sz, f"{parent_path}/{k}"))
            elif isinstance(obj, h5py.Group):
                # One more level for e.g. /entry/instrument/eiger_4m/data
                for k2 in obj.keys():
                    try:
                        ds = obj[k2]
                    except Exception:
                        continue
                    if (isinstance(ds, h5py.Dataset) and len(ds.shape) >= 2
                            and _vds_is_readable(ds)):
                        sz = int(np.prod(ds.shape))
                        candidates.append((sz, f"{parent_path}/{k}/{k2}"))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

    # Last resort — full tree walk
    fallback = None
    fallback_size = 0
    def _walk(g, prefix=""):
        nonlocal fallback, fallback_size
        try:
            keys = list(g.keys())
        except Exception:
            return
        for k in keys:
            full = f"{prefix}/{k}" if prefix else f"/{k}"
            try:
                obj = g[k]
            except Exception:
                continue
            if isinstance(obj, h5py.Dataset):
                if (len(obj.shape) >= 2 and _vds_is_readable(obj)):
                    sz = int(np.prod(obj.shape))
                    if sz > fallback_size:
                        fallback_size = sz
                        fallback = full
            elif isinstance(obj, h5py.Group):
                _walk(obj, full)
    _walk(h5_file)
    return fallback


def load_p10_chunks_directly(
    one_h5_path: str,
    chunk_dataset_path: str = "/entry/data/data",
    cxi_dataset_path: str = "/entry_1/data_1/data",
    max_frames: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    BYPASS VDS — directly open each `*_data_NNNNNN.h5` chunk in the directory
    and concatenate their contiguous data. Auto-discovers the actual dataset
    path inside each chunk (P10 sometimes stores data at /entry/data/data,
    sometimes at /entry/data/data_NNNNNN, depending on the acquisition mode).

    Falls back to .cxi if no chunks are found.

    Parameters
    ----------
    one_h5_path : str
        Any file in the dataset's directory — chunk, master, or CXI.
    chunk_dataset_path : str
        HINT for the path inside each chunk. If None or the path doesn't
        exist in a particular chunk, auto-discovers an appropriate one.
    cxi_dataset_path : str
        Path inside .cxi files (for the CXI fallback).
    max_frames : int or None
        Truncate to at most this many frames (centered).
    verbose : bool

    Returns
    -------
    volume : ndarray  shape (N_frames_total, H, W)  float32
    """
    import os, re, glob
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install h5py")

    dirname = os.path.dirname(os.path.abspath(one_h5_path))
    base = os.path.basename(one_h5_path)

    # Derive the dataset stem
    stem_match = re.match(
        r'^(.+?)_(?:master|data_\d+)\.(?:h5|nxs|cxi)$',
        base, flags=re.IGNORECASE,
    )
    if stem_match:
        stem = stem_match.group(1)
    else:
        stem = re.sub(r'\.(h5|nxs|cxi)$', '', base, flags=re.IGNORECASE)

    # Find all chunk files
    chunk_pattern = os.path.join(dirname, f"{stem}_data_*.h5")
    chunks = sorted(glob.glob(chunk_pattern))
    if not chunks:
        looser = os.path.join(dirname, f"{stem}*data_*.h5")
        chunks = sorted(glob.glob(looser))
    chunks = [c for c in chunks if not c.endswith("_master.h5")]

    if verbose:
        print(f"  Directory scan: stem='{stem}', found {len(chunks)} chunk file(s)")
        for c in chunks[:5]:
            print(f"    - {os.path.basename(c)}")
        if len(chunks) > 5:
            print(f"    ... ({len(chunks) - 5} more)")

    if chunks:
        # PASS 1: discover the actual path inside each chunk
        chunk_paths = []   # list of (file_path, discovered_internal_path)
        for c in chunks:
            try:
                with h5py.File(c, 'r') as f:
                    p = _find_chunk_internal_data_path(f, chunk_dataset_path)
                    if p is not None:
                        chunk_paths.append((c, p))
                        if verbose and len(chunk_paths) <= 3:
                            print(f"    Found data in {os.path.basename(c)} "
                                  f"at '{p}'")
            except Exception as e:
                if verbose:
                    print(f"    Skipping {os.path.basename(c)}: {e}")
                continue

        if not chunk_paths:
            raise IOError(
                f"None of the {len(chunks)} chunk files contain accessible "
                f"data. Inspect the file structure to find the correct path."
            )

        # PASS 2: determine reference shape from first chunk
        with h5py.File(chunk_paths[0][0], 'r') as f:
            ds = f[chunk_paths[0][1]]
            ref_shape = ds.shape[1:] if len(ds.shape) >= 3 else ds.shape

        # PASS 3: read each chunk and concatenate
        salvaged_volumes = []
        plugin_error_seen = False
        plugin_error_filter = None
        for c, p in chunk_paths:
            try:
                with h5py.File(c, 'r') as f:
                    ds = f[p]
                    # Before reading, check if this chunk needs hdf5plugin
                    # and we don't have it — bail out IMMEDIATELY rather than
                    # spam 381 identical errors
                    if not _HDF5_PLUGINS_AVAILABLE:
                        needs = _dataset_requires_plugin(ds)
                        if needs is not None:
                            plugin_error_seen = True
                            plugin_error_filter = needs
                            break
                    arr = ds[...]
                    if len(arr.shape) == 2:
                        arr = arr[None, :, :]
                    salvaged_volumes.append(arr.astype(np.float32))
            except Exception as e:
                # Detect compression-plugin error at read time too
                if _is_compression_plugin_error(e) and not _HDF5_PLUGINS_AVAILABLE:
                    plugin_error_seen = True
                    plugin_error_filter = _dataset_requires_plugin(
                        h5py.File(c, 'r')[p]
                    ) if True else "compression filter"
                    # Try to read the filter info — but fall back gracefully
                    try:
                        with h5py.File(c, 'r') as f:
                            plugin_error_filter = (_dataset_requires_plugin(f[p])
                                                    or "an unsupported filter")
                    except Exception:
                        plugin_error_filter = "an unsupported compression filter"
                    break
                if verbose:
                    print(f"    Failed reading {os.path.basename(c)}: {e}")

        # If we bailed out due to a missing plugin, give the actionable error
        if plugin_error_seen:
            raise IOError(
                f"The chunk files use {plugin_error_filter} compression, but\n"
                f"hdf5plugin is NOT installed on your system. This is the\n"
                f"standard compression for Eiger detectors at PETRA III P10.\n\n"
                f"FIX: install hdf5plugin and restart CDI-ST:\n"
                f"    pip install hdf5plugin\n\n"
                f"After installing, this dataset will read correctly.\n"
                f"hdf5plugin is a small package (~10 MB) that bundles the\n"
                f"compression DLLs for Eiger, Lambda, Pilatus, and other\n"
                f"detectors used at synchrotron beamlines."
            )

        if not salvaged_volumes:
            raise IOError(
                f"All chunk reads failed. Files exist but data is unreadable."
            )

        # Concatenate
        try:
            volume = np.concatenate(salvaged_volumes, axis=0)
        except ValueError as e:
            # Shapes don't match — pad to common size or fail loudly
            shapes = [v.shape for v in salvaged_volumes]
            raise IOError(
                f"Chunk shapes are inconsistent: {shapes}\n"
                f"Cannot concatenate. {e}"
            )

        if verbose:
            print(f"  Successfully concatenated {volume.shape[0]} frames "
                  f"from {len(chunk_paths)} chunk(s)")

    else:
        # No chunks found - try CXI file in same directory
        cxi_candidates = sorted(glob.glob(os.path.join(dirname, f"{stem}*.cxi")))
        if not cxi_candidates:
            cxi_candidates = sorted(glob.glob(os.path.join(dirname, "*.cxi")))
        if cxi_candidates:
            cxi_path = cxi_candidates[0]
            if verbose:
                print(f"  No data chunks found. Falling back to CXI file: "
                      f"{os.path.basename(cxi_path)}")
            with h5py.File(cxi_path, 'r') as f:
                p = _find_chunk_internal_data_path(f, cxi_dataset_path)
                if p is None:
                    raise IOError(
                        f"CXI file {cxi_path} has no recognizable detector "
                        f"dataset. Use 'Inspect HDF5 structure' to see what's "
                        f"inside."
                    )
                volume = np.asarray(f[p][...], dtype=np.float32)
                if verbose:
                    print(f"  Loaded CXI data from '{p}': shape={volume.shape}")
        else:
            raise IOError(
                f"No chunk files (*_data_*.h5) and no .cxi files found in "
                f"{dirname}. Cannot reconstruct dataset.\n"
                f"Looked for: {chunk_pattern}"
            )

    # Optional max_frames truncation
    if max_frames is not None and volume.shape[0] > max_frames:
        start = (volume.shape[0] - max_frames) // 2
        volume = volume[start:start + max_frames]
        if verbose:
            print(f"  Truncated to {max_frames} central frames")

    return volume


def _preprocess_p10_volume(raw_volume: np.ndarray, target_size: int = 64,
                            verbose: bool = True) -> np.ndarray:
    """
    Apply the same preprocessing as load_h5_diffraction's tail, but starting
    from a raw frame stack (N_frames, H, W) — the output of
    load_p10_chunks_directly. Used as the second half of the direct-chunk
    fallback in p10_h5_to_npz.
    """
    volume = np.maximum(raw_volume.astype(np.float32), 0)
    if verbose:
        print(f"  Raw volume:    shape={volume.shape}  "
              f"min={volume.min():.2e} max={volume.max():.2e}")

    # Mask detector chip gaps
    try:
        volume = mask_detector_gaps(volume, detector='auto')
    except Exception as e:
        if verbose:
            print(f"  Detector gap masking skipped: {e}")

    # Remove hot pixels
    try:
        volume = remove_hot_pixels(volume)
    except Exception as e:
        if verbose:
            print(f"  Hot pixel removal skipped: {e}")

    # Remove beamstop streaks (only on detector axes)
    try:
        volume = remove_beamstop_streaks(volume)
    except Exception as e:
        if verbose:
            print(f"  Beamstop streak removal skipped: {e}")

    # Find Bragg peak via the robust box-finder
    try:
        peak_idx = find_bragg_peak_box(volume)
    except Exception:
        peak_idx = tuple(s // 2 for s in volume.shape)

    # Crop a target_size cube around the peak
    volume = crop_around_peak(volume, peak_idx, target_size)

    if verbose:
        print(f"  Cropped:       shape={volume.shape}  "
              f"peak at {peak_idx}, max={volume.max():.2e}")
    return volume


def p10_h5_to_npz(
    h5_path: str,
    npz_path: str,
    fio_path: str = None,
    target_size: int = 64,
    dataset_path: str = None,
    verbose: bool = True,
    force_direct_chunks: bool = False,
    frame_range: tuple = None,
    detector_roi: tuple = None,
    memory_budget_gb: float = 8.0,
):
    """
    Load a P10 (PETRA III / DESY) BCDI scan and save as a .npz that
    CDI-ST reconstruction can use directly.

    Robust to several P10 quirks:
      1. Master file with external links → reads chunks directly (Eiger default)
      2. Single chunk with broken VDS placeholders → reads its own data
      3. Bitshuffle+LZ4 compressed data → requires ``hdf5plugin`` (auto-detected)
      4. Large multi-thousand-frame scans → use ``frame_range`` / ``detector_roi``
         to read only a subset (avoids 49-GB memory blow-ups)
      5. CXI (.cxi) files in the directory → used as fallback if no chunks

    Parameters
    ----------
    h5_path : str
        Any P10 file: master, data chunk, or .cxi.
    npz_path : str
        Output .npz filename.
    fio_path : str or None
        Optional .fio metadata path (motor positions). Auto-found if None.
    target_size : int
        Output cube size (target_size³). The cropped output is centered on
        the Bragg peak found in the loaded volume.
    dataset_path : str or None
        Explicit HDF5 path to detector data. Auto-detected if None.
    verbose : bool
    force_direct_chunks : bool
        If True, skip the master/VDS attempt and go straight to enumerating
        chunk files in the directory.
    frame_range : tuple ``(start, stop)`` or None
        Read only frames in ``[start, stop)`` (across all chunks). Useful for
        long rocking-curve scans where you only need the frames around the
        peak. Defaults to ``None`` (all frames).
    detector_roi : tuple ``(y0, y1, x0, x1)`` or None
        Read only this detector region per frame. Indices are Python slices.
        Defaults to ``None`` (full detector).
    memory_budget_gb : float
        Maximum estimated memory for the read. If exceeded, the function
        raises ``MemoryError`` with a suggested ``frame_range`` instead
        of crashing. Default 8 GB.
    """
    import os

    diffraction = None
    primary_error = None
    external_link_error = None

    # ── Strategy 0: detect if this is an Eiger/P10 MASTER with external links.
    # If so, this is by far the most reliable read path — bypasses h5py's
    # link-following entirely by opening chunk files directly.
    # When external links exist, we ALWAYS prefer this strategy because
    # the master's link-following mechanism can fail silently (returning
    # zeros, partial data, or "frames inaccessible" errors) on Windows or
    # with non-ASCII paths.
    has_external_links = False
    if not force_direct_chunks:
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                for grp_path in ['/entry/data', '/entry/instrument/detector/data',
                                  '/entry_1/data_1']:
                    if grp_path not in f:
                        continue
                    links = enumerate_external_links(f, grp_path)
                    if any(L['link_type'] == 'external' for L in links):
                        has_external_links = True
                        if verbose:
                            print(f"  Detected {len([L for L in links if L['link_type']=='external'])} "
                                  f"external link(s) in '{grp_path}' — this is an Eiger/P10 master file.")
                        break
        except Exception:
            pass

    if has_external_links:
        # External links exist — try resolving them. The FIRST group we
        # detected with external links is the right one; only try alternatives
        # if that one specifically returned "No external links found".
        first_ext_error = None
        for ext_grp_path in ['/entry/data', '/entry/instrument/detector/data',
                              '/entry_1/data_1']:
            try:
                raw_volume = load_p10_from_external_links(
                    h5_path, group_path=ext_grp_path,
                    max_frames=None,
                    frame_range=frame_range,
                    detector_roi=detector_roi,
                    memory_budget_gb=memory_budget_gb,
                    verbose=verbose,
                )
                diffraction = _preprocess_p10_volume(
                    raw_volume, target_size=target_size, verbose=verbose,
                )
                break
            except MemoryError:
                # Re-raise memory errors — user needs to subset their range
                raise
            except IOError as inner:
                # If the failure was "no external links here", keep trying other
                # group paths. If the failure was "chunks missing" — surface
                # that immediately, it's the actionable error.
                msg = str(inner)
                if "No external links found" in msg:
                    if first_ext_error is None:
                        first_ext_error = inner
                    continue
                else:
                    # Real error (chunks missing, unreadable, etc) — surface it
                    external_link_error = inner
                    break
        else:
            external_link_error = first_ext_error

    # ── Strategy 1: master/VDS read via load_h5_diffraction ──────────
    # Only used when there are no external links (or external-link strategy
    # already failed and we still need to try something).
    if diffraction is None and not force_direct_chunks:
        try:
            diffraction = load_h5_diffraction(
                h5_path, dataset_path=dataset_path,
                target_size=target_size, verbose=verbose,
            )
        except Exception as e:
            primary_error = e
            if verbose:
                print(f"\n  Master/VDS read failed: {e}")
                print(f"  Trying direct chunk enumeration...\n")

    # ── Strategy 2: direct chunk enumeration (last resort) ───────────
    if diffraction is None:
        try:
            # IMPORTANT: do NOT pass the user's dataset_path to the chunk
            # reader if it's a VDS-like path (e.g. '/entry/data/data_000002').
            # The fallback is meant to bypass VDS entirely — so we read
            # each chunk's OWN data at the canonical chunk-internal location.
            import re
            user_path_is_vds_like = (dataset_path is not None and
                bool(re.search(r'data_\d+\s*$', dataset_path)))
            if user_path_is_vds_like:
                chunk_path = "/entry/data/data"
                if verbose:
                    print(f"  Note: '{dataset_path}' is a VDS/external-link path — "
                          f"chunk fallback will read '/entry/data/data' "
                          f"from each chunk instead.")
            else:
                chunk_path = dataset_path or "/entry/data/data"
            raw_volume = load_p10_chunks_directly(
                h5_path,
                chunk_dataset_path=chunk_path,
                max_frames=None,
                verbose=verbose,
            )
            # Now run the same preprocessing pipeline that load_h5_diffraction
            # would have applied to the bulk-read array
            diffraction = _preprocess_p10_volume(
                raw_volume, target_size=target_size, verbose=verbose,
            )
        except Exception as e2:
            # All strategies failed - aggregate errors for clearest diagnostic
            err_parts = []
            if external_link_error is not None:
                err_parts.append(f"  External-link resolution:\n    "
                                  f"{external_link_error}")
            if primary_error is not None:
                err_parts.append(f"  Master/VDS read:\n    {primary_error}")
            err_parts.append(f"  Direct chunk enumeration:\n    {e2}")
            raise IOError(
                "Failed to load P10 data via all available methods:\n\n"
                + "\n\n".join(err_parts) + "\n\n"
                "Possible solutions:\n"
                "  - Make sure ALL `<scan>_data_NNNNNN.h5` chunk files are\n"
                "    in the SAME directory as the master file\n"
                "  - Use 'Inspect HDF5 structure' to see what's inside —\n"
                "    look for entries marked '→ external link to ...'\n"
                "  - For Eiger master files, the data chunks must be on disk\n"
                "  - Try opening a single chunk file directly"
            ) from e2

    # ── .fio metadata parsing ────────────────────────────────────────
    fio_data = None
    if fio_path is None:
        # Look for sibling .fio file with the same stem as the master
        base = os.path.basename(h5_path)
        dirname = os.path.dirname(os.path.abspath(h5_path))
        import re
        stem = re.sub(r'_(master|data_\d+)\.h5$', '', base, flags=re.IGNORECASE)
        stem = re.sub(r'\.h5$', '', stem)
        for cand in [
            os.path.join(dirname, stem + ".fio"),
            os.path.join(dirname, base.replace(".h5", ".fio")),
            os.path.join(dirname, "..", stem + ".fio"),
        ]:
            if os.path.exists(cand):
                fio_path = cand
                break
    if fio_path is not None and os.path.exists(fio_path):
        if verbose:
            print(f"  Reading metadata: {fio_path}")
        fio_data = parse_p10_fio(fio_path)
        if verbose and fio_data.get("scan_command"):
            print(f"  Scan: {fio_data['scan_command']}")

    # Save .npz
    save_dict = {
        "diffraction": diffraction.astype(np.float32),
        "source_file": str(h5_path),
        "beamline": "P10",
    }
    if fio_data is not None:
        save_dict["scan_command"] = fio_data.get("scan_command", "")
        # Common P10 motors that BCDI cares about
        for motor in ["omega", "chi", "phi", "theta", "tth", "eta", "mu",
                      "del", "delta", "gamma", "nu", "energy"]:
            if motor in fio_data["data"]:
                arr = fio_data["data"][motor]
                save_dict[f"motor_{motor}"] = np.asarray(arr, dtype=np.float32)
            elif motor in fio_data["params"]:
                save_dict[f"motor_{motor}_static"] = float(fio_data["params"][motor])
    np.savez_compressed(npz_path, **save_dict)
    if verbose:
        print(f"  Saved {npz_path}  "
              f"shape={diffraction.shape}  max={diffraction.max():.2e}")

    return {
        "diffraction": diffraction,
        "voxel_size_nm": None,
        "source_file": h5_path,
        "fio": fio_data,
    }


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
        'diffraction': res['diffraction'],
        'amplitude': np.sqrt(np.maximum(res['diffraction'], 0)),
        'eta': res['eta'],
        'phi': res['phi'],
        'nu': res['nu'],
        'delta': res['delta'],
        'rocking_type': res['rocking_type'] or 'unknown',
    }
    if 'voxel_size_nm' in res:
        save_dict['voxel_size_nm'] = res['voxel_size_nm']
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
        'source_file': str(h5_path),
        'source_dataset': dataset_path,
        'target_size': target_size,
        'is_experimental': True,
        'has_ground_truth': False,
    }
    meta_path = Path(npz_path).parent / (Path(npz_path).stem + '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load BCDI diffraction data from experimental .h5 files'
    )
    parser.add_argument('--input', type=str, required=True, help='Input .h5 file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .npz file (if not given, just inspect)')
    parser.add_argument('--inspect', action='store_true',
                        help='Just print the HDF5 structure and exit')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='HDF5 path to diffraction data (auto-detect if omitted)')
    parser.add_argument('--target_size', type=int, default=64,
                        help='Output grid size')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Max tree depth for --inspect')
    args = parser.parse_args()

    if args.inspect:
        inspect_h5(args.input, max_depth=args.max_depth)
        sys.exit(0)

    if args.output is None:
        print("No --output specified. Inspecting structure:")
        inspect_h5(args.input, max_depth=args.max_depth)
    else:
        h5_to_npz(args.input, args.output, args.dataset_path, args.target_size)
