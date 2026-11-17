# Changelog

## [0.3.3] - 2026-11-17

### Fixed

- **Some missing reflections for hexagonal and rhombohedral materials.** 

- A reflection typed as any member of its family is now found. Previously only
  the stored representative and its Friedel mate matched, so `(0 -1 10)` in
  corundum reported "not found" even though the peak was in the table under
  another label.
- Window no longer open larger than the screen.
- The Beam tab's control column scrolls instead of being squeezed
- `BCDISimulator.simulate()` returned `None` when called without a progress
  callback: `if progress_cb: progress_cb(100);return ...etc` put the `return` inside
  the `if`. Present since 0.2.2 and invisible from the GUI, which always passes a
  callback, but it made the simulator unusable from a script.

### Added

- **Reconstruction without a neural network.** A `HIO/RAAR/ER only` mode runs the
  standard iterative algorithms from a random start with no checkpoint file, and
  without importing PyTorch at all, so it works on a machine that cannot run a
  network.
- **Render quality control in the 3D viewer**, chosen before rendering: Draft
  (15,000 points), Standard (60,000, the default), High (200,000) and Maximum
  (1,000,000). Higher settings also smooth the isosurface less, so density and
  strain stay distinguishable. Measured on a 128^3 reconstruction with 268,000
  support voxels: 0.4 s, 0.5 s, 1.4 s and 1.4 s respectively. The old viewer was
  fixed at 15,000 points and 4,000 interior points, with no way to change them.
- **Auto-update view** toggle. With it on, changing any viewer setting re-renders
  without pressing "Update view". Changes are coalesced for 450 ms so dragging a
  slider does not start a render on every step.

### Changed

- Turbo is the default colormap in the 3D viewer; magma and cividis added.
- Reconstruction modes are named in plain language and name the networks:
  "AutoPhaseNet -> HIO/RAAR/ER refinement", "AutoPhaseNet alone (fast, rough)",
  "HIO/RAAR/ER only - no neural network", "Both networks averaged
  (AutoPhaseNet + PhaseUNet3D)". `nn_only` and `ensemble` are no longer shown to
  the user; the internal keys are unchanged so saved settings keep working.
- The checkpoint field is disabled when a mode needs no network.
- Simulation tabs are named "Material", "Beam", "Results" without numbering.

## [0.3.2] - 2026-11-08

### Fixed

- The launcher window was missing: `main()` opened the simulation window
  directly, so the data-analysis tabs were unreachable from the installed
  package. The v0.3.0 rewrite of `bcdi_gui.py` had dropped `Launcher`,
  `CDIStSplash`, `MW_Analysis` and `ReportsDialog`. All four are restored, and
  `MW_Sim` keeps its Reports & Suggestions button.

## [0.3.1] - 2026-11-08

### Fixed

- `ModuleNotFoundError: No module named 'bcdi_core'` on launch. The modules
  imported each other as top-level names, which works when the files sit
  together in one folder but not from inside the `cdi_st` package. Imports are
  now package-relative, with a fallback so the files still run directly from a
  plain folder.

## [0.3.0] - 2026-11-08

Faceted particles, photon-budget noise, detector/sample space, standalone phase
retrieval and training diagnostics.

With thanks to **Dr. Gerards Hinsley**, whose review of v0.2.2 prompted most of
the changes in this release: the facet options, the photon-count control, the
detector-space toggle, the standalone reconstruction path, the training log, and
the corrections to the noise model and colour-scale controls.

### Added

**Simulation**
- Crystallographic facet shapes built from a Wulff half-space construction
  (`facets.py`): octahedron, cuboctahedron, truncated octahedron, truncated cube,
  rhombic dodecahedron, tunable Wulff, and hexagonal prism / bipyramid /
  truncated bipyramid. Facet normals come from the reciprocal lattice, so the
  construction is correct for every crystal system.
- Facet multiplicities control: a ratio of facet distances, off by default,
  enabled by a tick box, with a plain-language tooltip giving the shape each
  value produces.
- Particle size entered directly in nm; supercell multiples follow from the
  lattice vectors. Cell limit raised from 2500 to 100000 per axis, with a size
  notice above 2M atoms and a confirmation above 10M.
- Photon number: the pattern is scaled to a chosen number of photons before
  Poisson sampling, so the counting statistics mean something. The field accepts
  scientific notation (1e9, 2.5e12).
- Detector / sample space toggle. `ortho.det_matrix` builds the real
  (dq/dpx, dq/dpy, dq/drocking) transform; the Results tab switches between the
  orthogonal lab q grid and the raw skewed detector grid and reports the skew.
- Custom 2theta with a live readout: the offset from the nearest reflection, the
  resulting dq, and the peak shift in both simulation voxels and real detector
  pixels, with a warning when the peak leaves the frame.
- Prismatic dislocation loop alongside the existing line dislocations.
- Separate simulation grid control, independent of the detector panel size.
- Visualization controls (bonds, strain, dislocation) grouped together.

**Data processing**
- read reconstructions from `.npz`, `.npy`, `.h5`, `.cxi`,
  `.nxs` and `.mat`, including PyNX `/entry_1/image_1/data` and amplitude/phase
  pairs, so work reconstructed outside CDI-St can be loaded.
- per-step training log (`.log` + `.jsonl`) that flags NaN/Inf
  losses, non-finite gradients and weights, and loss spikes, then dumps input
  statistics and the likely cause. `scan_dataset()` pre-flights a training set.

### Changed

- Read noise is applied only to integrating detectors. Hybrid photon-counting
  detectors (EIGER, PILATUS, MAXIPIX, MERLIN, TIMEPIX) have Poisson statistics
  only, and the Readout option disables itself when one is selected.
- Noise controls are named per effect (Poisson / Readout / Air / Dead px /
  Hot px) instead of a single "Add noise", and all are off by default, so the
  ideal pattern is what you see until you ask for noise.
- Default simulation grid raised from 128 to 192, giving 2.25x the pixels in the
  2D slices for about 2 s per simulation.
- The simulation panel is resizable; the fixed 520 px left column is a splitter.
- Frame transform uses cubic interpolation (0.7% round-trip error, against 3.6%
  for linear).

### Fixed

- Poisson noise was on by default and applied the photon budget unconditionally,
  which quantised 83% of the volume to zero counts and erased the weak fringes
  in the 3D peak and the 2D slices. Noise is now opt-in.
- The simulation grid was capped by the detector panel size, so a 128 x 128
  panel forced 128^3 slices regardless of the grid setting. Panel geometry
  (exported as metadata) and simulation grid are now independent.
- Reflections are labelled by their canonical Friedel mate, so selecting (1 1 1)
  no longer displays (-1 -1 -1). Both label the same physical peak; the hkl
  search simply reached the negative representative first. The raw index is kept
  as `hkl_raw`.
- `select_reflection` returns full-precision d, 2theta, q, |F|^2 and LP. It
  previously used the four-decimal values rounded for the table.
- Half-voxel registration error in the detector/lab transform. An orthogonal
  transform is now an exact no-op for any q-grid convention.
- Shrinkwrap could inflate the support until the algorithm fit the noise (89% of
  the array at 1e5 counts). An absolute cap keeps reconstructions stable from
  1e3 to 1e8 photons.
- Selecting a large detector preset set the transform size to the full panel:
  Eiger2 4M attempted a 2068^3 complex FFT, about 140 GB.
- The facet ratio control was not read when building, and the large-supercell
  notice was missing.

## [0.2.2] — 2026-06-18

### Fixed
- **ID01 SPEC+EDF conversion now tolerates missing edge frames.**
  Acquisitions are sometimes started a frame late or stopped a frame
  early, so a SPEC scan can list frames 262–662 (401 points) while the
  EDF directory only has 263–661 (399 files). The converter now probes
  the disk first, loads only the EDF files that actually exist, and
  trims the per-frame motor arrays (`eta`, `phi`, `nu`, `delta`,
  `mpx4inr`) to stay in lock-step with the loaded data. A clear note
  identifies which leading / trailing / interior frames were skipped.
  New helper `_filter_existing_edf_frames()` exposes this filtering for
  programmatic use.

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
- **Scan range selection for P10** (new in the P10 converter dialog)
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
  - **Memory budget** (default 8 GB) the loader refuses reads that
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
- Live training curve in AutoPhase_NN and CDI_NN the dashed "running"
  line updates every ~2 s during training so users can see progress
  immediately on slow CPU runs (rather than staring at a blank plot
  for an entire epoch).
- `inspect_h5()` now properly shows external links with their target
  files and `[OK]` / `[MISSING]` markers. The empty-looking `/entry/data/`
  group on Eiger masters is no longer misleading.

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
