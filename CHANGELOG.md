# Changelog

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
- Photon budget: the pattern is scaled to a chosen number of photons before
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
- `recon.py` — standalone ER / HIO / RAAR phase retrieval with shrinkwrap,
  bad-pixel masking, chi-squared history, centring, phase-ramp removal and PRTF.
  No network needed.
- `recon.load_recon` — reads reconstructions from `.npz`, `.npy`, `.h5`, `.cxi`,
  `.nxs` and `.mat`, including PyNX `/entry_1/image_1/data` and amplitude/phase
  pairs, so work reconstructed outside CDI-St can be loaded.
- `trainlog.py` — per-step training log (`.log` + `.jsonl`) that flags NaN/Inf
  losses, non-finite gradients and weights, and loss spikes, then dumps input
  statistics and the likely cause. `scan_dataset()` pre-flights a training set.
- `viz3d.py` — 14 colormaps including Turbo; clip plane with a filled cut face
  showing the data on the cut layer; explicit colour-scale scope (whole volume /
  rendered shell / current slice) with a label stating which was used.

### Changed

- Read noise is applied only to integrating detectors. Hybrid photon-counting
  detectors (EIGER, PILATUS, MAXIPIX, MERLIN, TIMEPIX) have Poisson statistics
  only, and the Readout option disables itself when one is selected. A
  `CCD (integrating)` preset was added for the case where read noise applies.
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
