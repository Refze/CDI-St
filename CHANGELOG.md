# Changelog

All notable changes to CDI-ST are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/Refze/CDI-St/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Refze/CDI-St/releases/tag/v0.1.0
