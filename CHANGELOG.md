# Changelog

All notable changes to CDI-ST are documented here.

## [Unreleased]

## [0.2.1] — 2026-06-15

### Added
- **PETRA III P10 / DESY Eiger compressed-data support.** Auto-imports
  `hdf5plugin` at module load so bitshuffle + LZ4 compressed datasets
  (the Eiger default at DESY P10 and many other beamlines) read
  transparently. `hdf5plugin>=4.0` is now a runtime dependency.
- **HDF5 external-link resolution.** The P10 loader now enumerates and
  resolves external links in master files instead of relying on h5py's
  default link-following, which was unreliable on Windows. Each chunk is
  opened directly to bypass any path-mangling. Master files with hundreds
  of external links to chunk files (the standard Eiger layout) work out
  of the box.
- **Scan range selection for P10** (new in the P10 converter dialog) —
  similar to the ID01 SPEC+EDF scan browser. Avoids loading 49 GB of
  detector frames into RAM when you only need the rocking-curve maximum:
  - **Probe scan** button reads chunk metadata only (no decompression),
    instantly reporting total frame count, detector shape, and estimated
    full-load memory footprint.
  - **Frame range** (`from`, `to`) reads only the requested global frame
    range across all chunks. Loaded chunk-by-chunk so memory scales with
    the chosen range, not the total dataset size.
  - **Detector ROI** reads a centered square ROI from each frame instead
    of the full detector. For BCDI you typically only need ~256×256
    around the Bragg peak — cuts memory by `(2167*2070)/(roi²)` for
    Eiger 4M data.
  - **Memory budget** (default 8 GB) — the loader refuses reads that
    would exceed this and tells you which `frame_range` would fit,
    instead of crashing the Python process with an OOM error.
- **Programmatic API additions** to `nn_experimental_loader`:
  `p10_scan_info()` for cheap metadata previews, plus new
  `frame_range`, `detector_roi`, `memory_budget_gb` parameters on
  `load_p10_from_external_links()` and `p10_h5_to_npz()`.

### Fixed
- Cryptic "Can't synchronously read data (can't open directory)" error
  on P10 / Eiger files. The message was HDF5 looking for the compression
  plugin DLL in a non-existent directory — now caught and converted to
  a clear "install hdf5plugin" instruction.
- AutoPhase_NN tab missing `_on_epoch` slot (introduced in 0.2.0 during
  the live-curve refactor; AutoPhase_NN's `epoch_done` signal had no
  receiver and emitted unhandled errors). Restored.
- CDI_NN `KeyError: 'phase_true'` — `BCDIDataset.__getitem__` now exposes
  the GUI's expected keys (`phase_true`, `amplitude`) alongside the legacy
  ones (`target_phase`, `input`).
- `BCDIPhaseLoss` accepts both the GUI's descriptive kwargs (`alpha_amp`,
  `beta_phase`, `gamma_diff`, `diff_amp`) and the original short ones
  (`alpha`, `beta`, `gamma`, `amplitude`). Either spelling now works.
- Training Stop button no longer crashes the program. `QThread.terminate()`
  was killing PyTorch mid-`backward()` and corrupting CUDA / autograd
  state — replaced with a cooperative `request_stop()` flag that the
  training loop checks at batch / epoch boundaries, then exits cleanly
  with a saved checkpoint.

### Changed
- Live training curve in AutoPhase_NN and CDI_NN — the dashed "running"
  line updates every ~2 s during training so users can see progress
  immediately on slow CPU runs (rather than staring at a blank plot
  for an entire epoch).
- `inspect_h5()` now properly shows external links with their target
  files and `[OK]` / `[MISSING]` markers. The empty-looking `/entry/data/`
  group on Eiger masters is no longer misleading.


## [0.1.3] — 2026-05-20

### Fixed
- Fixed exporting VTI to Paraview

### Added
- SPEC+EDF converter: Added option to convert multiple scans


## [0.1.2] — 2026-05-20

### Fixed
- SPEC+EDF converter: scan listing, frame range display, clear errors when EDFs fail to load
- silx data column indexing
- `find_bragg_peak_box` slice/int crash on empty volumes

### Added
- "Browse scans…" button in SPEC+EDF dialog

## [0.1.1] — 2026-05-20

### Fixed
- SPEC+EDF converter crash with "slice - int" error when EDF files could not be loaded (`find_bragg_peak_box` now returns coordinates instead of slices in the zero-volume fallback)
- silx data column indexing was wrong — modern silx returns `data.shape == (nlines, ncolumns)`, so `data[i]` was the i-th measurement row, not the i-th column. Now uses `scan.data_column_by_name(label)` with shape-aware fallback
- Clear error message when no EDF frames can be loaded (lists likely causes instead of crashing later)
- SPEC+EDF converter now prints the frame range of the chosen scan before loading

### Added
- New "Browse scans…" button in SPEC+EDF dialog: lists every scan in the SPEC file with its CCD frame range, so users can pick the scan matching their downloaded EDF files
- New `list_spec_scans()` helper in `nn_experimental_loader.py`

### Changed
- Renamed output field "Output .npz:" to "Output file (.npz):" with clarifying placeholder and tooltip


## [0.1.0] — Alpha — 2026-05-08

### Added
- **Initial public release** of CDI-ST as a packaged Python application.
- BCDI Simulation module (Material → Beam → Results)
  - Material presets: Si, Ge, Pt, Au, SiC, GaAs, and more
  - CIF import for arbitrary crystals
  - Particle shapes: cube, sphere, cylinder, hexagonal prism, octahedron, dodecahedron
  - Strain fields: radial gradient, edge dislocation, random
  - Configurable line dislocations: edge / screw / mixed with arbitrary Burgers vector
  - 3D atomic lattice viewer (Plotly)
  - Detector presets: Maxipix 2×2, Eiger 2M, custom rectangular
  - Analytical and FFT simulation paths
- BCDI Data Analysis module (Generate Data → AutoPhase_NN → CDI_NN → Reconstruction → 3D Viewer)
  - Training-data generator with random material/shape/size/strain/dislocation
  - Toggles for randomized strain and dislocations
  - AutoPhaseNet3D (unsupervised, dual-decoder, physics-driven loss)
  - PhaseUNet3D (supervised, BCDI loss with α/β/γ weights)
  - Inference: NN-only, refined (HIO+RAAR+ER), ensemble, compare
  - Iterative phase retrieval: HIO (β=0.9), RAAR (β=0.9), ER, with shrink-wrap
  - Experimental data loader: HDF5, ID01 SPEC+EDF, q-space orthogonalization
  - Detector chip-gap masking, hot pixel removal, beamstop streak removal
  - 3D viewer with point cloud / isosurface / surface-only modes
  - Per-voxel opacity weighted by intensity
  - VTI export for ParaView
- Splash screen and theme-matched launcher
- In-app Reports & Suggestions feature
- Cross-platform support (Linux, macOS, Windows) via PyQt6
- MIT license, full source on GitHub

### Known limitations (alpha)
- Q-space orthogonalization is ID01-specific (can be extended)
- Some checkpoints from earlier prototypes lack `grid_size` metadata
- Mask-aware modulus constraint not yet implemented (gaps are filled instead)

[Unreleased]: https://github.com/SAIDIsoufiane/CDI-St/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SAIDIsoufiane/CDI-St/releases/tag/v0.1.0
