# Changelog

## [0.3.4] - 2026-11-23

### Fixed

- **R-factor was not comparable between modes, and could look good for a failed reconstruction.**
- **R is now shown with two references**, so a number such as 0.12 can be judged:
  the noise level (the R this result would reach if it were exactly right, from
  Poisson statistics of the data) and the R of a featureless sphere (no phase
  retrieval at all). The result is labelled *fits the data to the noise level*,
  *partial fit*, *not reconstructed*, or *data too noisy to judge* when even a
  featureless sphere fits within the noise. Checked on 56 cases from 10^2 counts to
  noise-free data with no wrong label.
- **Crash after an HIO/RAAR/ER only reconstruction** (`cannot convert float NaN to integer`). 
  The object was centred but its support was not, so the two could stop
  overlapping. They now move together, and the display falls back to the array
  centre when there is nothing to weigh. Phase-ramp removal in the classical path
  is now exact.
- **AutoPhase_NN alone lost most of the network's phase.** Centring the object
  shifted the amplitude first and then masked the phase with the already shifted
  box, so wherever the network placed the object off centre its phase was set to
  zero (36 to 100 % of the object in tests). The refinement modes started from this
  flat phase too. The object is now moved as one complex array by whole voxels,
  which leaves |F| unchanged. Refined results are as good as before.

### Changed

- 3D viewer render styles: **Point cloud** (default), **Isosurface**, a smooth
  solid whose cut face, with the clip plane, shows the field inside; **Surface
  only**, the outer skin, hollow when cut.
- **Render quality changes only how finely the object is drawn.** The values, the
  colour scale, the iso level and the smoothing are identical at every level;
  quality sets the mesh resolution and, for very large reconstructions, how many
  points are drawn.
- **Live preview** replaces Auto-update view: a quick draft while you drag, full
  quality when you stop, rotation and zoom kept.
- Reconstruction modes name the networks as their tabs do: AutoPhase_NN, CDI_NN.

### Known issues

- Some real P10 scans are converting incorrectly; under investigation...

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
  (1,000,000).

### Changed

- Turbo is the default colormap in the 3D viewer; magma and cividis added.
- The checkpoint field is disabled when a mode needs no network.

## [0.3.2] - 2026-11-08

### Fixed

- The launcher window was missing: `main()` opened the simulation window
  directly, so the data-analysis tabs were unreachable from the installed
  package. The v0.3.0 rewrite of `bcdi_gui.py` had dropped `Launcher`,
  `CDIStSplash`, `MW_Analysis` and `ReportsDialog`. All four are restored, and
  `MW_Sim` keeps its Reports & Suggestions button.

## [0.3.1] - 2026-11-08

### Fixed

- `ModuleNotFoundError: No module named 'bcdi_core'` on launch. Imports are
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

### Changed

- Read noise is applied only to integrating detectors. Hybrid photon-counting
  detectors (EIGER, PILATUS, MAXIPIX, MERLIN, TIMEPIX) have Poisson statistics
  only, and the Readout option disables itself when one is selected.
- Noise controls are named per effect.
- Default simulation grid raised from 128 to 192, giving 2.25x the pixels in the
  2D slices for about 2 s per simulation.
- Frame transform uses cubic interpolation (0.7% round-trip error, against 3.6%
  for linear).

### Fixed

- The simulation grid was capped by the detector panel size, so a 128 x 128
  panel forced 128^3 slices regardless of the grid setting.
- Reflections are labelled by their canonical Friedel mate, so selecting (1 1 1)
  no longer displays (-1 -1 -1).
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

## [0.2.2] - 2026-06-18

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

## [0.2.1] - 2026-06-15

### Added
- **PETRA III P10 / DESY Eiger compressed-data support.** Auto-imports
  `hdf5plugin` at module load so bitshuffle + LZ4 compressed datasets
  (the Eiger default at DESY P10 and many other beamlines) read
  transparently. `hdf5plugin>=4.0` is now a runtime dependency.
- **HDF5 external-link resolution.** The P10 loader now enumerates and
  resolves external links in master files instead of relying on h5py's
  default link-following, which was unreliable on Windows.
- **Scan range selection for P10** (new in the P10 converter dialog)
  similar to the ID01 SPEC+EDF scan browser. Avoids loading 49 GB of
  detector frames into RAM when you only need the rocking-curve maximum

### Fixed
- AutoPhase_NN tab missing `_on_epoch` slot..
- Training Stop button no longer crashes the program.

### Changed
- Live training curve in AutoPhase_NN and CDI_NN.
- `inspect_h5()` now properly shows external links with their target
  files and `[OK]` / `[MISSING]` markers.

## [0.1.0] - Alpha - 2026-05-08

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
  - AutoPhaseNN (unsupervised, dual-decoder, physics-driven loss)
  - CDI_NN (supervised, BCDI loss with α/β/γ weights)
  - Inference: NN-only, refined (HIO+RAAR+ER), ensemble, compare
  - Iterative phase retrieval: HIO (β=0.9), RAAR (β=0.9), ER, with shrink-wrap
  - Experimental data loader: HDF5, ID01 SPEC+EDF, q-space orthogonalization
  - Detector chip-gap masking, hot pixel removal, beamstop streak removal
  - 3D viewer with point cloud / isosurface / surface-only modes
  - Per-voxel opacity weighted by intensity
  - VTI export for ParaView
- Splash screen and theme-matched launcher
- Cross-platform support (Linux, macOS, Windows) via PyQt6

### Known limitations (alpha)
- Q-space orthogonalization is only ID01-specific
- Some checkpoints from earlier prototypes lack `grid_size` metadata
- Mask-aware modulus constraint not yet implemented (gaps are filled instead)
- VTI export is exporting a 2D slice, will be fixed.
