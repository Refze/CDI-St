from __future__ import annotations
import os
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass, shift as ndshift

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


def chi2(F, amp, w=None):
    d = np.abs(F) - amp
    if w is not None: d = d * w
    n = (amp ** 2).sum()
    return float((d ** 2).sum() / n) if n > 0 else np.inf


def retrieve(data, support=None, recipe='HIO:200,ER:50', beta=0.9, bad_mask=None,
             sw_every=20, sw_sigma=(3.0, 1.0), sw_thr=0.2, sw_max_frac=0.35, seed=0, cb=None):
    """Iterative phase retrieval on a centred 3-D intensity volume.

    data      intensity (not amplitude), Bragg peak at the array centre
    bad_mask  True where the detector value is invalid -- those pixels float free
    recipe    e.g. 'RAAR:300,HIO:100,ER:50'
    sw_thr    shrinkwrap threshold on the smoothed amplitude (0.2 is a safe default)
    cb(i, algo, err, support_frac) is called every iteration if given
    Returns (rho, support, errors).
    """
    data = np.asarray(data, np.float64)
    amp = np.sqrt(np.maximum(data, 0))
    steps = parse_recipe(recipe)
    total = sum(n for _, n in steps)
    rng = np.random.default_rng(seed)
    sup = auto_support(data) if support is None else np.asarray(support, bool)
    if not sup.any(): sup = auto_support(data)

    rho = _Fi(amp * np.exp(2j * np.pi * rng.random(amp.shape))) * sup
    errs, it = [], 0
    s0, s1 = sw_sigma

    for algo, n in steps:
        for _ in range(n):
            F = _F(rho)
            errs.append(chi2(F, amp, None if bad_mask is None else ~bad_mask))
            ph = np.exp(1j * np.angle(F))
            Fm = amp * ph if bad_mask is None else np.where(bad_mask, F, amp * ph)
            rp = _Fi(Fm)
            if algo == 'ER':
                rho = rp * sup
            elif algo == 'HIO':
                rho = np.where(sup, rp, rho - beta * rp)
            else:
                rho = np.where(sup, rp, beta * rho + (1 - 2 * beta) * rp)
            it += 1
            if sw_every and it % sw_every == 0:
                sig = s0 + (s1 - s0) * (it / max(total, 1))
                new = shrinkwrap(rho, sig, sw_thr)
                # reject a degenerate or runaway support: unguarded, noisy data
                # bloats the support until the algorithm just fits the noise
                if 8 < new.sum() < sw_max_frac * new.size: sup = new
            if cb is not None: cb(it, algo, errs[-1], float(sup.mean()))

    rho = centre(rho * sup)
    return rho, sup, np.asarray(errs)


def centre(rho):
    a = np.abs(rho)
    if a.max() <= 0: return rho
    c = np.asarray(center_of_mass(a)) - np.asarray(rho.shape) / 2.0
    return ndshift(rho.real, -c, order=1) + 1j * ndshift(rho.imag, -c, order=1)


def remove_ramp(rho, sup=None):
    """Strip the linear phase ramp (an arbitrary real-space origin shift)."""
    m = (np.abs(rho) > 0.1 * np.abs(rho).max()) if sup is None else np.asarray(sup, bool)
    if m.sum() < 16: return rho
    idx = np.argwhere(m).astype(float)
    ph = np.unwrap(np.angle(rho[m]))
    A = np.column_stack([idx, np.ones(len(idx))])
    c, *_ = np.linalg.lstsq(A, ph, rcond=None)
    g = np.meshgrid(*[np.arange(s, dtype=float) for s in rho.shape], indexing='ij')
    ramp = c[0] * g[0] + c[1] * g[1] + c[2] * g[2] + c[3]
    return rho * np.exp(-1j * ramp)


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
