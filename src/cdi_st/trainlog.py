from __future__ import annotations
import json, os, time
from datetime import datetime
import numpy as np


def _finite_stats(a):
    a = np.asarray(a, np.float64).ravel()
    n = a.size
    bad_n = int(np.isnan(a).sum()); bad_i = int(np.isinf(a).sum())
    f = a[np.isfinite(a)]
    return {'n': n, 'nan': bad_n, 'inf': bad_i,
            'min': float(f.min()) if f.size else None,
            'max': float(f.max()) if f.size else None,
            'absmax': float(np.abs(f).max()) if f.size else None,
            'mean': float(f.mean()) if f.size else None,
            'std': float(f.std()) if f.size else None}


class TrainLog:
    """Writes <run>.log (human) and <run>.jsonl (per-step metrics) next to each other.

    tl = TrainLog('runs', 'autophase_v3')
    ...
    loss.backward()
    if tl.step(ep, it, loss, model=net, lr=lr, inputs=x) != 'ok':
        opt.zero_grad(); continue        # skip the poisoned update
    opt.step()
    """

    def __init__(self, outdir, run=None, grad_clip_warn=1e3, loss_jump=50.0, echo=True):
        os.makedirs(outdir, exist_ok=True)
        run = run or datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.run = run
        self.log_path = os.path.join(outdir, run + '.log')
        self.jsonl_path = os.path.join(outdir, run + '.jsonl')
        self.grad_clip_warn = grad_clip_warn
        self.loss_jump = loss_jump
        self.echo = echo
        self.t0 = time.time()
        self.n_bad = 0
        self.last_loss = None
        self._lf = open(self.log_path, 'a', buffering=1)
        self._jf = open(self.jsonl_path, 'a', buffering=1)
        self.write(f'=== {run} started {datetime.now().isoformat(timespec="seconds")} ===')

    def write(self, msg):
        line = f'[{time.time() - self.t0:8.1f}s] {msg}'
        self._lf.write(line + '\n')
        if self.echo: print(line, flush=True)

    def config(self, **kw):
        self.write('config ' + json.dumps(kw, default=str))

    def _grad_norm(self, model):
        tot, bad = 0.0, []
        for n, p in model.named_parameters():
            g = getattr(p, 'grad', None)
            if g is None: continue
            g = g.detach()
            if not bool(g.isfinite().all()): bad.append(n)
            tot += float(g.float().pow(2).sum())
        return tot ** 0.5, bad

    def _param_check(self, model):
        return [n for n, p in model.named_parameters()
                if not bool(p.detach().isfinite().all())]

    def step(self, epoch, it, loss, model=None, lr=None, inputs=None, extra=None):
        """Returns 'ok', 'nan', 'inf' or 'spike'. Dumps a diagnosis on anything but 'ok'."""
        lv = float(loss.detach()) if hasattr(loss, 'detach') else float(loss)
        rec = {'t': round(time.time() - self.t0, 2), 'epoch': epoch, 'it': it, 'loss': lv}
        if lr is not None: rec['lr'] = float(lr)

        gn, gbad = (None, [])
        if model is not None:
            try: gn, gbad = self._grad_norm(model)
            except Exception as e: self.write(f'grad inspect failed: {e!r}')
            if gn is not None: rec['grad_norm'] = gn

        status = 'ok'
        if lv != lv: status = 'nan'
        elif lv in (float('inf'), float('-inf')): status = 'inf'
        elif gbad: status = 'nan'
        elif self.last_loss is not None and abs(lv) > self.loss_jump * max(abs(self.last_loss), 1e-12):
            status = 'spike'
        elif gn is not None and gn > self.grad_clip_warn:
            self.write(f'ep{epoch} it{it} large grad norm {gn:.3e} (loss {lv:.4e})')

        rec['status'] = status
        if extra: rec.update(extra)
        self._jf.write(json.dumps(rec) + '\n')

        if status != 'ok':
            self.n_bad += 1
            self.write(f'--- {status.upper()} at epoch {epoch} it {it}: loss={lv!r} '
                       f'grad_norm={gn} lr={lr}')
            if gbad: self.write(f'    non-finite grads in: {gbad[:12]}')
            if model is not None:
                pbad = self._param_check(model)
                if pbad: self.write(f'    non-finite weights in: {pbad[:12]}')
            if inputs is not None:
                for name, arr in self._as_arrays(inputs):
                    s = _finite_stats(arr)
                    if s['nan'] or s['inf'] or (s['absmax'] or 0) > 1e8:
                        self.write(f'    input {name}: {s}')
            self.write('    likely causes: log/sqrt/div of 0 in the loss, un-normalised '
                       'intensities, lr too high, fp16 overflow, or an empty support mask')
        self.last_loss = lv if status == 'ok' else self.last_loss
        return status

    @staticmethod
    def _as_arrays(x, prefix='x'):
        out = []
        if isinstance(x, dict): items = list(x.items())
        elif isinstance(x, (list, tuple)): items = [(f'{prefix}{i}', v) for i, v in enumerate(x)]
        else: items = [(prefix, x)]
        for n, v in items:
            try:
                out.append((n, v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)))
            except Exception:
                pass
        return out

    def epoch_end(self, epoch, **metrics):
        self.write(f'epoch {epoch} ' + ' '.join(f'{k}={v:.5g}' for k, v in metrics.items()))
        self._jf.write(json.dumps({'epoch': epoch, 'summary': metrics,
                                   't': round(time.time() - self.t0, 2)}) + '\n')

    def close(self):
        self.write(f'=== finished, {self.n_bad} bad step(s), log: {self.log_path} ===')
        self._lf.close(); self._jf.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def scan_dataset(arrays, names=None):
    """Pre-flight check: report non-finite or absurdly scaled training data."""
    names = names or [f'array{i}' for i in range(len(arrays))]
    bad = []
    for n, a in zip(names, arrays):
        s = _finite_stats(a)
        if s['nan'] or s['inf'] or s['absmax'] is None or s['absmax'] > 1e8:
            bad.append((n, s))
    return bad
