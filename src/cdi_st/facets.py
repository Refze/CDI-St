from __future__ import annotations
import numpy as np
from itertools import permutations
from scipy.optimize import linprog

_CUBIC = {'100': (1, 0, 0), '110': (1, 1, 0), '111': (1, 1, 1), '311': (3, 1, 1)}
_PRISM = [(1, 0, 0), (0, 1, 0), (-1, 1, 0), (-1, 0, 0), (0, -1, 0), (1, -1, 0)]
_BASAL = [(0, 0, 1), (0, 0, -1)]
_PYR = [(h, k, l) for (h, k, _) in _PRISM for l in (1, -1)]

# name -> {facet family: relative distance from centre}; larger d shrinks that facet
SHAPES = {
    'cube':             {'100': 1.0},
    'octahedron':       {'111': 1.0},
    'cuboctahedron':    {'111': 1.0, '100': 0.82},
    'trunc_octahedron': {'111': 1.0, '100': 1.10},
    'trunc_cube':       {'100': 1.0, '111': 0.90},
    'rhombic_dodeca':   {'110': 1.0},
    'wulff':            {'111': 1.0, '100': 1.15},
    'hex_prism':        {'prism': 1.0, 'basal': 1.0},
    'hex_bipyramid':    {'prism': 1.0, 'pyr': 0.95, 'basal': 1.3},
    'hex_trunc':        {'prism': 1.0, 'pyr': 1.15, 'basal': 0.95},
}
HEX_SHAPES = [k for k, v in SHAPES.items() if 'prism' in v]
CUBIC_SHAPES = [k for k, v in SHAPES.items() if 'prism' not in v]


def _family(fam):
    if fam == 'prism': return list(_PRISM)
    if fam == 'basal': return list(_BASAL)
    if fam == 'pyr': return list(_PYR)
    g = _CUBIC[fam]; out = set()
    for p in permutations(g):
        for s in np.ndindex(2, 2, 2):
            out.add(tuple(((-1) ** s[i]) * p[i] for i in range(3)))
    return sorted(out)


def normals(shape, B, wulff_ratio=1.15):
    """Unit facet normals (Cartesian) and relative distances. B rows = reciprocal vectors."""
    spec = dict(SHAPES[shape])
    if shape == 'wulff': spec = {'111': 1.0, '100': float(wulff_ratio)}
    N, D = [], []
    for fam, d in spec.items():
        for hkl in _family(fam):
            n = np.asarray(hkl, float) @ B
            nn = np.linalg.norm(n)
            if nn < 1e-9: continue
            N.append(n / nn); D.append(d)
    return np.asarray(N), np.asarray(D)


def _support(N, D, u):
    r = linprog(-u, A_ub=N, b_ub=D, bounds=[(None, None)] * 3)
    return float(-r.fun) if r.success else np.inf


def extents(N, D):
    """Bounding-box extent of the unit polyhedron along x, y, z."""
    e = np.empty(3)
    for a in range(3):
        u = np.zeros(3); u[a] = 1.0
        e[a] = _support(N, D, u) + _support(N, D, -u)
    return e


def scale_to_size(N, D, size_ang):
    """Scale factor R so the polyhedron's largest extent equals size_ang."""
    e = extents(N, D); m = np.nanmax(e[np.isfinite(e)]) if np.isfinite(e).any() else 0.
    return float(size_ang / m) if m > 0 else 1.0


def mask(pos, centre, N, D, R):
    return np.all((pos - centre) @ N.T <= R * D[None, :], axis=1)


def grid_mask(X, Y, Z, N, D, R):
    """Facet mask on a meshgrid (used for the simulator support function)."""
    m = np.ones(X.shape, bool)
    for n, d in zip(N, D):
        m &= (n[0] * X + n[1] * Y + n[2] * Z) <= R * d
    return m
