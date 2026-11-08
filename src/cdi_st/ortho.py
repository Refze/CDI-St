from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates

LAB_AXES = {'x': np.array([1., 0, 0]), 'y': np.array([0, 1., 0]), 'z': np.array([0, 0, 1.])}


def det_matrix(lam_ang, two_theta_deg, pix_um, det_dist_m, rock_step_deg,
               gamma_deg=0.0, rock_axis='z'):
    """Columns of M are dq/d(px), dq/d(py), dq/d(rocking step), in 1/A.

    Beam along +x, vertical +z. two_theta is the in-plane scattering angle,
    gamma the out-of-plane one. A non-orthogonal M is why raw BCDI data must be
    interpolated onto a regular grid before the object is Fourier-inverted.
    """
    k = 2 * np.pi / lam_ang
    d, g = np.radians(two_theta_deg), np.radians(gamma_deg)
    nf = np.array([np.cos(g) * np.cos(d), np.cos(g) * np.sin(d), np.sin(g)])
    eh = np.array([-np.sin(d), np.cos(d), 0.])
    ev = np.array([-np.sin(g) * np.cos(d), -np.sin(g) * np.sin(d), np.cos(g)])
    qB = k * (nf - np.array([1., 0, 0]))
    s = k * (pix_um * 1e-6) / det_dist_m
    dqr = -np.cross(LAB_AXES[rock_axis], qB) * np.radians(rock_step_deg)
    return np.column_stack([s * eh, s * ev, dqr]), qB


def skew(M):
    """0 for an orthogonal sampling grid; rises as the grid shears."""
    C = M / np.linalg.norm(M, axis=0, keepdims=True)
    G = np.abs(C.T @ C); np.fill_diagonal(G, 0.0)
    return float(G.max())


def _match_scale(M, dq):
    """Rescale M to the lab voxel size so only the shear, not the zoom, is shown."""
    n = np.linalg.norm(M, axis=0)
    if not np.all(n > 0): return M
    return M * (float(np.mean(dq)) / float(np.mean(n)))


def lab_to_det(vol, q, M, order=1, match_scale=True):
    """Resample an orthogonal q-space volume onto the skewed detector grid."""
    qx, qy, qz = q['qx'], q['qy'], q['qz']
    dq = np.array([qx[1] - qx[0], qy[1] - qy[0], qz[1] - qz[0]])
    if match_scale: M = _match_scale(M, dq)
    n = vol.shape
    q0 = np.array([qx[0], qy[0], qz[0]])
    c0 = -q0 / dq                      # fractional index where q = 0
    I, J, K = np.meshgrid(*[np.arange(s) - c0[a] for a, s in enumerate(n)], indexing='ij')
    P = np.stack([I.ravel(), J.ravel(), K.ravel()])
    Q = M @ P
    c = np.stack([Q[a] / dq[a] + c0[a] for a in range(3)])
    out = map_coordinates(vol, c, order=order, mode='constant', cval=0.0)
    return out.reshape(n).astype(np.float32)


def det_to_lab(vol, q, M, order=1, match_scale=True):
    """Orthogonalise detector-frame data onto the regular lab q grid."""
    qx, qy, qz = q['qx'], q['qy'], q['qz']
    dq = np.array([qx[1] - qx[0], qy[1] - qy[0], qz[1] - qz[0]])
    if match_scale: M = _match_scale(M, dq)
    q0 = np.array([qx[0], qy[0], qz[0]])
    c0 = -q0 / dq
    Minv = np.linalg.inv(M)
    QX, QY, QZ = np.meshgrid(qx, qy, qz, indexing='ij')
    Q = np.stack([QX.ravel(), QY.ravel(), QZ.ravel()])
    P = Minv @ Q
    c = np.stack([P[a] + c0[a] for a in range(3)])
    out = map_coordinates(vol, c, order=order, mode='constant', cval=0.0)
    return out.reshape(vol.shape).astype(np.float32)


def apply_photons(vol, n_photons, mode='total', poisson=True, seed=None):
    """Scale to a photon budget, then draw counting statistics.

    mode 'total' sets the integrated counts in the volume, 'peak' the brightest voxel.
    """
    v = np.asarray(vol, np.float64)
    s = v.sum() if mode == 'total' else v.max()
    if s <= 0: return np.zeros(v.shape, np.float32)
    v = v * (float(n_photons) / s)
    if poisson:
        v = np.random.default_rng(seed).poisson(np.clip(v, 0, 1e15)).astype(np.float64)
    return v.astype(np.float32)


def detector_noise(vol, dark_cps=0.0, exposure_s=1.0, readout_e=0.0,
                   dead_frac=0.0, hot_frac=0.0, seed=None):
    """Instrument effects applied after the photon budget.

    readout_e is meaningful only for integrating (CCD/CMOS) detectors. Hybrid
    photon-counting detectors (EIGER, PILATUS, MAXIPIX, MERLIN, TIMEPIX) have no
    read noise -- leave it at zero and use dark_cps for their small dark rate.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(vol, np.float64).copy()
    if dark_cps > 0: v += rng.poisson(dark_cps * exposure_s, v.shape)
    if readout_e > 0: v = np.maximum(v + rng.normal(0, readout_e, v.shape), 0)
    if dead_frac > 0:
        i = rng.choice(v.size, int(dead_frac * v.size), replace=False); v.flat[i] = 0
    if hot_frac > 0:
        i = rng.choice(v.size, int(hot_frac * v.size), replace=False)
        v.flat[i] = max(v.max(), 1.0) * rng.uniform(1, 5, i.size)
    return v.astype(np.float32)
