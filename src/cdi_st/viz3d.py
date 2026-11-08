from __future__ import annotations
import numpy as np

BCDI = [[0, '#030d1e'], [.12, '#0a2a6e'], [.30, '#1060c8'], [.50, '#30b0e8'],
        [.65, '#e8d830'], [.80, '#e87010'], [.92, '#d01808'], [1., '#ffffff']]

CMAPS = {'BCDI': BCDI, 'Turbo': 'Turbo', 'Inferno': 'Inferno', 'Viridis': 'Viridis',
         'Plasma': 'Plasma', 'Magma': 'Magma', 'Cividis': 'Cividis', 'Jet': 'Jet',
         'Hot': 'Hot', 'Rainbow': 'Rainbow', 'Greys': 'Greys', 'Blackbody': 'Blackbody',
         'Electric': 'Electric', 'Twilight': 'Twilight'}
CMAP_NAMES = list(CMAPS)
# cyclic maps belong on phase, not intensity
PHASE_CMAPS = ['Twilight', 'HSV', 'Phase']

SCOPES = {'volume': 'whole volume (global min/max)',
          'shell': 'rendered shell only (between iso-min and iso-max)',
          'slice': 'current clip slice only'}


def limits(vol, scope='volume', iso_lo=None, iso_hi=None, clip_index=None, clip_axis=0):
    v = np.asarray(vol)
    if scope == 'shell' and iso_lo is not None:
        m = v >= iso_lo
        if iso_hi is not None: m &= v <= iso_hi
        s = v[m]
    elif scope == 'slice' and clip_index is not None:
        s = np.take(v, int(clip_index), axis=clip_axis).ravel()
    else:
        scope, s = 'volume', v.ravel()
    s = s[np.isfinite(s)]
    if s.size == 0: s = v.ravel()[np.isfinite(v.ravel())]
    lo, hi = (float(s.min()), float(s.max())) if s.size else (0.0, 1.0)
    return lo, hi, f'auto scale from {SCOPES.get(scope, scope)}: [{lo:.3g}, {hi:.3g}]'


def clip_volume(vol, axis=0, index=None, side='below'):
    v = np.asarray(vol, np.float32).copy()
    n = v.shape[axis]
    i = n // 2 if index is None else int(np.clip(index, 0, n - 1))
    keep = [slice(None)] * 3
    keep[axis] = slice(i, None) if side == 'below' else slice(None, i + 1)
    out = np.full_like(v, np.nan)
    out[tuple(keep)] = v[tuple(keep)]
    return out, i


def cut_face(fig, vol, q, axis, index, cmap, lo, hi, opacity=1.0):
    import plotly.graph_objects as go
    qx, qy, qz = q['qx'], q['qy'], q['qz']
    sl = np.take(np.asarray(vol), int(index), axis=axis)
    if axis == 0:
        A, B = np.meshgrid(qy, qz, indexing='ij')
        X = np.full(A.shape, qx[int(index)]); Y, Z = A, B
    elif axis == 1:
        A, B = np.meshgrid(qx, qz, indexing='ij')
        Y = np.full(A.shape, qy[int(index)]); X, Z = A, B
    else:
        A, B = np.meshgrid(qx, qy, indexing='ij')
        Z = np.full(A.shape, qz[int(index)]); X, Y = A, B
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=sl, colorscale=CMAPS.get(cmap, cmap),
                             cmin=lo, cmax=hi, showscale=False, opacity=opacity,
                             name='cut', hovertemplate='%{surfacecolor:.3g}<extra>cut</extra>'))
    return fig


def volume_fig(vol, q, cmap='BCDI', iso_lo_pct=75, iso_hi_pct=100, opacity=1.0,
               surfaces=21, scope='volume', clip=None, clip_fill=True, title=''):
    """
    clip = (axis, index, side) or None.
    """
    import plotly.graph_objects as go
    v = np.asarray(vol, np.float32)
    vmax = float(np.nanmax(v)) if np.isfinite(v).any() else 1.0
    lo_i, hi_i = vmax * iso_lo_pct / 100.0, vmax * iso_hi_pct / 100.0
    ci = clip[1] if clip else None
    ca = clip[0] if clip else 0
    lo, hi, label = limits(v, scope, lo_i, hi_i, ci, ca)
    shown = v
    if clip:
        shown, ci = clip_volume(v, clip[0], clip[1], clip[2])
    qx, qy, qz = q['qx'], q['qy'], q['qz']
    QX, QY, QZ = np.meshgrid(qx, qy, qz, indexing='ij')
    fig = go.Figure()
    fig.add_trace(go.Volume(x=QX.ravel(), y=QY.ravel(), z=QZ.ravel(),
                            value=np.nan_to_num(shown, nan=-1e30).ravel(),
                            isomin=lo_i, isomax=hi_i, opacity=opacity,
                            surface_count=int(surfaces), colorscale=CMAPS.get(cmap, cmap),
                            cmin=lo, cmax=hi, showscale=True,
                            colorbar=dict(thickness=14, len=.9, tickfont=dict(size=10)),
                            caps=dict(x_show=False, y_show=False, z_show=False)))
    if clip and clip_fill:
        cut_face(fig, v, q, ca, ci, cmap, lo, hi)
    ax = dict(backgroundcolor='#080a0e', gridcolor='rgba(255,255,255,.09)',
              showbackground=True, color='#c0c4cc', zeroline=False)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0d0e12',
                      scene=dict(xaxis=dict(**ax, title='qx(1/A)'), yaxis=dict(**ax, title='qy(1/A)'),
                                 zaxis=dict(**ax, title='qz(1/A)'), aspectmode='data',
                                 camera=dict(eye=dict(x=1.4, y=1.4, z=.7))),
                      font=dict(family='Consolas,monospace', size=11, color='#c0c4cc'),
                      title=dict(text=title, font=dict(size=12), x=.5, xanchor='center'),
                      margin=dict(l=0, r=0, t=36, b=0))
    return fig, label
