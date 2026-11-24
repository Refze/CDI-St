from __future__ import annotations
import os
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass

_F = lambda a: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a)))
_Fi = lambda a: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(a)))


def parse_recipe(s):
    """'HIO:200,ER:50' -> [('HIO',200),('ER',50)]"""
    out = []
    for part in str(s).replace(';', ',').split(','):
        part = part.strip()
        if not part: continue
        a, _, n = part.partition(':')
        a = a.strip().upper()
        if a not in ('ER', 'HIO', 'RAAR'): raise ValueError(f'unknown algorithm {a}')
        out.append((a, int(n) if n else 50))
    return out


def auto_support(data, thr=0.08):
    ac = np.abs(_Fi(np.sqrt(np.maximum(data, 0))))
    return ac > thr * ac.max()


def shrinkwrap(rho, sigma, thr):
    a = gaussian_filter(np.abs(rho).astype(np.float64), sigma)
    return a > thr * a.max()


def _scale(pred, meas, w=None):
    # maximum-likelihood scale for counts: model and data hold the same total
    # intensity. Fitting the scale on amplitudes is biased low when most voxels
    # are zero, and normalising each pattern by its own maximum lets one voxel set it.
    p = np.asarray(pred, np.float64); m = np.asarray(meas, np.float64)
    w = 1. if w is None else w
    return np.sqrt((w * m * m).sum() / max((w * p * p).sum(), 1e-300))


def rfac(pred, meas, w=None):
    # R = sum| k|F| - A | / sum A, with k from _scale. Every mode reports this.
    p = np.asarray(pred, np.float64); m = np.asarray(meas, np.float64)
    k = _scale(p, m, w); ww = 1. if w is None else w
    return float((ww * np.abs(k * p - m)).sum() / max((ww * m).sum(), 1e-300))


def rfactor(F, amp, w=None):
    return rfac(np.abs(F), amp, w)


def _is_counts(I):
    nz = I[I > 0]
    return nz.size > 0 and I.max() < 2 ** 31 and np.mean(np.abs(nz - np.round(nz)) < 1e-3) > 0.8


def _r0(pred, w, n, seed):
    rng = np.random.default_rng(seed)
    return float(np.mean([rfac(pred, np.sqrt(rng.poisson(pred * pred)), w) for _ in range(n)]))


def _sample(I, n_max, seed):
    if I.size <= n_max: return None, None
    f = I.ravel(); top = np.argpartition(f, -n_max // 2)[-n_max // 2:]
    keep = np.ones(f.size, bool); keep[top] = False; rest = np.flatnonzero(keep)
    pick = np.random.default_rng(seed).choice(rest, n_max - top.size, replace=False)
    w = np.concatenate([np.ones(top.size), np.full(pick.size, rest.size / pick.size)])
    return np.concatenate([top, pick]), w


def fit_quality(rho, data, mask=None, draws=3, seed=0, n_max=400_000):
    I = np.nan_to_num(np.maximum(np.asarray(data, np.float64), 0))
    W = None if mask is None else (~np.asarray(mask, bool)).astype(np.float64)
    A = np.sqrt(I); P = np.abs(_F(rho))
    R = rfac(P, A, W)
    idx, ws = _sample(I, n_max, seed)
    g = [2 * np.pi * (np.arange(n) - n // 2) / n for n in I.shape]
    if idx is None:
        a, p_, w = A.ravel(), P.ravel(), (None if W is None else W.ravel())
        q = np.sqrt(sum(x * x for x in np.meshgrid(*g, indexing='ij'))).ravel()
    else:
        a, p_ = A.ravel()[idx], P.ravel()[idx]
        w = ws if W is None else ws * W.ravel()[idx]
        u = np.unravel_index(idx, I.shape); q = np.sqrt(sum(g[k][u[k]] ** 2 for k in range(3)))
    M = _scale(p_, a, w) * p_
    Rb, Mb = np.inf, None
    for r_ in np.geomspace(1.5, min(I.shape) / 3, 24):
        x = np.maximum(q * r_, 1e-9)
        Fs = np.abs(np.where(x < 1e-4, 1., 3 * (np.sin(x) - x * np.cos(x)) / x ** 3))
        rr = rfac(Fs, a, w)
        if rr < Rb: Rb, Mb = rr, Fs * _scale(Fs, a, w)
    out = dict(R=R, R0=None, R_sphere=Rb, counts=_is_counts(I))
    if out['counts']:
        R0 = _r0(M, w, draws, seed); out['R0'] = R0
        if Rb / _r0(Mb, w, draws, seed + 1) <= 1.3: v = 'data too noisy to judge'
        elif R / R0 <= 1.2: v = 'fits the data to the noise level'
        elif R <= 0.15 * Rb: v = 'fits the data'
        elif R >= 0.8 * Rb: v = 'not reconstructed'
        else: v = 'partial fit'
    else:
        v = 'fits the data' if R <= 0.15 * Rb else ('not reconstructed' if R >= 0.8 * Rb else 'partial fit')
    out['verdict'] = v
    return out


def retrieve(data, support=None, recipe='HIO:200,ER:50', beta=0.9, bad_mask=None,
             sw_every=20, sw_sigma=(3.0, 1.0), sw_thr=0.2, sw_max_frac=0.35, seed=0, cb=None):
    raw = np.asarray(data, np.float64)
    fin = np.isfinite(raw)
    if not fin.all():
        bad_mask = ~fin if bad_mask is None else (np.asarray(bad_mask, bool) | ~fin)
    data = np.where(fin, np.maximum(raw, 0), 0.)
    if data.sum() <= 0: raise ValueError('diffraction volume is empty or entirely invalid')
    amp = np.sqrt(data)
    w = None if bad_mask is None else ~np.asarray(bad_mask, bool)
    steps = parse_recipe(recipe)
    total = sum(n for _, n in steps)
    rng = np.random.default_rng(seed)
    sup = auto_support(data) if support is None else np.asarray(support, bool)
    if not sup.any(): sup = auto_support(data)

    rho = _Fi(amp * np.exp(2j * np.pi * rng.random(amp.shape))) * sup
    errs, it = [], 0
    s0, s1 = sw_sigma
    stop = False
    for algo, n in steps:
        for _ in range(n):
            F = _F(rho)
            e = rfactor(F, amp, w)
            if not np.isfinite(e):
                stop = True; break
            errs.append(e)
            ph = np.exp(1j * np.angle(F))
            Fm = amp * ph if bad_mask is None else np.where(bad_mask, F, amp * ph)
            rp = _Fi(Fm)
            prev = rho
            if algo == 'ER':
                rho = rp * sup
            elif algo == 'HIO':
                rho = np.where(sup, rp, rho - beta * rp)
            else:
                rho = np.where(sup, rp, beta * rho + (1 - 2 * beta) * rp)
            if not np.isfinite(rho).all():
                rho = prev; stop = True; break
            it += 1
            if sw_every and it % sw_every == 0:
                sig = s0 + (s1 - s0) * (it / max(total, 1))
                new = shrinkwrap(rho, sig, sw_thr)
                # reject a degenerate or runaway support: unguarded, noisy data
                # bloats the support until the algorithm just fits the noise
                if 8 < new.sum() < sw_max_frac * new.size: sup = new
            if cb is not None: cb(it, algo, errs[-1], float(sup.mean()))
        if stop: break

    # the object and its support must move together, or the support no longer
    # covers the object and anything weighted by it sums to zero
    rho, sup = centre(rho * sup, sup)
    if not np.isfinite(rho).all() or not (np.abs(rho) > 0).any():
        raise RuntimeError('phase retrieval collapsed to an empty object: check the input '
                           'volume, or try fewer HIO iterations')
    # final R on the support-constrained object, as the network paths report it
    errs.append(rfactor(_F(rho), amp, w))
    return rho, sup, np.asarray(errs)


def centre(rho, sup=None):
    a = np.abs(rho)
    if not np.isfinite(a).all() or a.sum() <= 0:
        return rho if sup is None else (rho, sup)
    c = np.round(np.asarray(center_of_mass(a)) - np.asarray(rho.shape) // 2).astype(int)
    rho = np.roll(rho, tuple(-c), axis=(0, 1, 2))
    return rho if sup is None else (rho, np.roll(np.asarray(sup), tuple(-c), axis=(0, 1, 2)))


def remove_ramp(rho, sup=None):
    m = (np.abs(rho) > 0.1 * np.abs(rho).max()) if sup is None else np.asarray(sup, bool)
    if m.sum() < 16 or not np.isfinite(rho).all(): return rho
    k = []
    for ax in range(3):
        a = np.moveaxis(rho, ax, 0); mm = np.moveaxis(m, ax, 0)
        p = a[1:] * np.conj(a[:-1]); w = mm[1:] & mm[:-1]
        k.append(float(np.angle(p[w].sum())) if w.any() else 0.)
    g = np.meshgrid(*[np.arange(n, dtype=float) - n // 2 for n in rho.shape], indexing='ij')
    return rho * np.exp(-1j * (k[0] * g[0] + k[1] * g[1] + k[2] * g[2]))


def prtf(rho, data, nbins=40):
    """Single-run PRTF vs |q| (shell-averaged). Call on the raw retrieve() output:
    remove_ramp shifts the pattern in q and would drag the PRTF down."""
    amp = np.sqrt(np.maximum(np.asarray(data, np.float64), 0))
    F = np.abs(_F(rho))
    g = np.meshgrid(*[np.arange(s) - s // 2 for s in data.shape], indexing='ij')
    r = np.sqrt(sum(x.astype(float) ** 2 for x in g))
    b = np.clip((r / (r.max() + 1e-12) * nbins).astype(int), 0, nbins - 1)
    num = np.bincount(b.ravel(), F.ravel(), nbins)
    den = np.bincount(b.ravel(), amp.ravel(), nbins)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(den > 0, num / den, np.nan)
    return r.max() * (np.arange(nbins) + 0.5) / nbins, p


# ---------------------------------------------------------------- external I/O

_OBJ_KEYS = ['obj', 'object', 'rho', 'reconstruction', 'recon', 'complex_object',
             'image', 'data', 'result']
_H5_PATHS = ['/entry_1/image_1/data', '/entry_1/data_1/data', '/entry/data/data',
             '/object', '/obj', '/reconstruction', '/rho', '/data']


def load_recon(path):
    """Load a reconstruction produced elsewhere (PyNX, bcdi, cdiutils, MATLAB...).

    Returns (complex array, metadata dict). Amplitude/phase pairs are recombined.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.asarray(np.load(path)), {'source': path}
    if ext == '.npz':
        z = np.load(path, allow_pickle=True)
        meta = {'source': path, 'keys': list(z.keys())}
        for k in _OBJ_KEYS:
            if k in z and np.iscomplexobj(z[k]): return np.asarray(z[k]), meta
        amp = next((z[k] for k in ('amp', 'amplitude', 'modulus') if k in z), None)
        ph = next((z[k] for k in ('phase', 'phi', 'angle') if k in z), None)
        if amp is not None:
            return np.asarray(amp) * np.exp(1j * np.asarray(ph if ph is not None else 0)), meta
        for k in _OBJ_KEYS:
            if k in z and np.asarray(z[k]).ndim == 3:
                return np.asarray(z[k]).astype(complex), meta
        raise KeyError(f'no object array found; keys={list(z.keys())}')
    if ext in ('.h5', '.hdf5', '.cxi', '.nxs'):
        import h5py
        with h5py.File(path, 'r') as f:
            meta = {'source': path, 'attrs': dict(f.attrs)}
            for p in _H5_PATHS:
                if p in f:
                    a = np.asarray(f[p])
                    if a.ndim == 3: return a.astype(complex), meta
            found = []
            f.visit(lambda n: found.append(n))
            for n in found:
                try: d = f[n]
                except Exception: continue
                if hasattr(d, 'shape') and len(getattr(d, 'shape', ())) == 3:
                    return np.asarray(d).astype(complex), {**meta, 'path': n}
        raise KeyError('no 3-D dataset found in file')
    if ext == '.mat':
        try:
            from scipy.io import loadmat
            m = loadmat(path)
            for k in _OBJ_KEYS + list(m.keys()):
                if k.startswith('__'): continue
                a = np.asarray(m.get(k, []))
                if a.ndim == 3: return a.astype(complex), {'source': path, 'key': k}
        except NotImplementedError:
            import h5py
            with h5py.File(path, 'r') as f:
                for n in f:
                    a = np.asarray(f[n])
                    if a.ndim == 3: return a.astype(complex), {'source': path, 'key': n}
        raise KeyError('no 3-D array in .mat file')
    raise ValueError(f'unsupported format: {ext}')


def save_recon(path, rho, support=None, errs=None, **meta):
    d = {'obj': np.asarray(rho)}
    if support is not None: d['support'] = np.asarray(support)
    if errs is not None: d['errors'] = np.asarray(errs)
    for k, v in meta.items(): d[k] = np.asarray(v)
    if path.endswith(('.h5', '.cxi')):
        import h5py
        with h5py.File(path, 'w') as f:
            for k, v in d.items(): f.create_dataset(k, data=v, compression='gzip')
    else:
        np.savez_compressed(path, **d)
    return path
