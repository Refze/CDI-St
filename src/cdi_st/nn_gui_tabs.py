"""
nn_gui_tabs.py — Three new tabs for the BCDI GUI:
    T4: NN Training — choose data, parameters, and train with live progress
    T5: BCDI Reconstruction — load .npz/.h5 and reconstruct with 4 figures
    T6: 3D Reconstruction Viewer — interactive 3D with phase/strain/density
"""

import sys, os, tempfile, json, time
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QSize
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QProgressBar, QPlainTextEdit, QFrame, QCheckBox, QScrollArea,
    QFileDialog, QSplitter, QSlider, QMessageBox, QTabWidget, QApplication,
    QDialog, QTableWidget, QTableWidgetItem
)
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    _HAS_WEB = True
except ImportError:
    _HAS_WEB = False

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

MPL_DARK = {
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#000005',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
    'axes.titlecolor': '#e6edf3', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'text.color': '#e6edf3', 'font.size': 9
}


def _dbl(lo, hi, v, d, suf=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setDecimals(d); s.setValue(v); s.setMinimumWidth(100)
    if suf:
        s.setSuffix(suf)
    return s


def _ph(m):
    return (f'<html><head><style>html,body{{height:100%;margin:0;background:#0a0d12;'
            f'color:#8b949e}}.w{{height:100%;display:flex;align-items:center;'
            f'justify-content:center}}.i{{padding:24px;border:1px dashed #30363d;'
            f'border-radius:8px}}</style></head><body><div class="w">'
            f'<div class="i">{m}</div></div></body></html>')


# ═══════════════════════════════════════════════════════════════════════════════
# Worker threads
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingWorker(QThread):
    """Run data generation + NN training in a background thread."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    epoch_done = pyqtSignal(int, float, float)  # epoch, train_loss, val_loss
    running_loss = pyqtSignal(float, float)  # (fractional_epoch, running_train_loss)
    finished = pyqtSignal(str)  # model path
    failed = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._should_stop = False

    def request_stop(self):
        """Cooperative stop: training loop checks this flag and exits gracefully.

        Safer than QThread.terminate(), which kills the thread mid-operation
        and corrupts CUDA context / PyTorch autograd state.
        """
        self._should_stop = True

    def _apply_noise(self, measured, p):
        """Apply on-the-fly experimental noise to a batch of measured magnitudes."""
        import torch as _t
        # Convert magnitude → intensity
        intensity = measured ** 2

        # Air scatter: uniform background
        if p.get('noise_air', 0) > 0:
            intensity = intensity + p['noise_air']

        # Poisson (photon shot noise)
        if p.get('noise_poisson', False):
            intensity = _t.poisson(intensity.clamp(min=0))

        # Readout: Gaussian electronic noise
        if p.get('noise_readout', 0) > 0:
            intensity = intensity + _t.randn_like(intensity) * p['noise_readout']

        # Dead pixels: random fraction zeroed out
        dead = p.get('noise_dead', 0)
        if dead > 0:
            mask = (_t.rand_like(intensity) > dead).float()
            intensity = intensity * mask

        # Clamp non-negative and convert back to magnitude
        intensity = intensity.clamp(min=0)
        return _t.sqrt(intensity)

    def run(self):
        try:
            import torch
            from cdi_st.nn_autophase_model import AutoPhaseNet3D, PhysicsForwardModel, UnsupervisedBCDILoss, count_parameters
            from cdi_st.nn_autophase_train import UnsupervisedBCDIDataset
            from torch.utils.data import DataLoader, random_split
            from torch.amp import GradScaler, autocast
            import torch.optim as optim

            p = self.p
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log.emit(f"Device: {device}")
            if device.type == 'cuda':
                self.log.emit(f"GPU: {torch.cuda.get_device_name()}")

            # --- Step 1: Verify data ---
            data_dir = Path(p['data_dir'])
            npz_files = sorted(data_dir.glob("sample_*.npz"))
            h5_files = sorted(data_dir.glob("*.h5"))
            n_files = len(npz_files) + len(h5_files)
            if n_files == 0:
                self.failed.emit(f"No .npz or .h5 files in {data_dir}")
                return
            self.log.emit(f"Found {n_files} files in {data_dir}")

            # --- Step 2: Build dataset ---
            full_ds = UnsupervisedBCDIDataset(str(data_dir), augment=False,
                                               grid_size=p['grid_size'])
            n = len(full_ds)
            n_val = max(1, int(n * 0.15))
            n_train = n - n_val
            train_ds, val_ds = random_split(
                full_ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            train_ds.dataset = UnsupervisedBCDIDataset(str(data_dir), augment=True,
                                                         grid_size=p['grid_size'])
            train_loader = DataLoader(train_ds, batch_size=p['batch_size'],
                                       shuffle=True, num_workers=0, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=p['batch_size'],
                                     shuffle=False, num_workers=0)
            self.log.emit(f"Train: {n_train}  Val: {n_val}  Batch: {p['batch_size']}")

            # --- Step 3: Build model ---
            model = AutoPhaseNet3D(base_channels=p['base_channels'],
                                    enforce_oversampling=p['enforce_oversampling']).to(device)
            physics = PhysicsForwardModel(threshold=p['threshold']).to(device)
            loss_fn = UnsupervisedBCDILoss(support_smoothness=p['support_smoothness'],
                                             tv_phase=p['tv_phase'])
            n_params = count_parameters(model)
            self.log.emit(f"Model: {n_params:,} parameters, base_ch={p['base_channels']}")

            optimizer = optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
            scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

            # Resume
            start_epoch = 0
            best_val = float('inf')
            if p.get('resume') and os.path.exists(p['resume']):
                ckpt = torch.load(p['resume'], map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                start_epoch = ckpt.get('epoch', 0)
                best_val = ckpt.get('val_loss', float('inf'))
                self.log.emit(f"Resumed from epoch {start_epoch}")

            out_dir = Path(p['output_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)
            no_improve = 0

            # --- Step 4: Training loop ---
            # Compute total number of batch iterations across all epochs
            # for accurate intra-epoch progress reporting.
            n_train_batches = max(1, len(train_loader))
            n_val_batches = max(1, len(val_loader))
            batches_per_epoch = n_train_batches + n_val_batches
            total_iters = (p['epochs'] - start_epoch) * batches_per_epoch
            iter_count = 0

            import time as _time
            last_emit = _time.time()

            for epoch in range(start_epoch, p['epochs']):
                if self._should_stop:
                    self.log.emit(
                        f"Stop requested. Saving checkpoint at end of epoch "
                        f"{epoch}/{p['epochs']}..."
                    )
                    break
                model.train()
                t0 = time.time()
                train_total = 0; train_n = 0
                for batch_idx, batch in enumerate(train_loader):
                    if self._should_stop:
                        break
                    inp = batch['input'].to(device)
                    measured = batch['measured'].to(device)

                    # Apply on-the-fly noise to make model robust to real data
                    if p.get('apply_noise', False):
                        with torch.no_grad():
                            measured = self._apply_noise(measured, p)
                            # Recompute log-normalized input from noisy measurement
                            log_mag = torch.log10(measured + 1.0)
                            scale = log_mag.amax(dim=(1,2,3,4), keepdim=True).clamp(min=1e-6)
                            inp = log_mag / scale

                    optimizer.zero_grad(set_to_none=True)
                    with autocast('cuda', enabled=(device.type == 'cuda')):
                        amp, phase = model(inp)
                        pred_diff, support = physics(amp, phase)
                        losses = loss_fn(pred_diff, measured, amp, phase, support)
                    scaler.scale(losses['total']).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    train_total += losses['total'].item()
                    train_n += 1

                    # Intra-epoch progress: update at most every 2 seconds so
                    # the GUI thread isn't flooded with signal traffic.
                    iter_count += 1
                    now = _time.time()
                    if now - last_emit > 2.0:
                        pct = int(100 * iter_count / max(total_iters, 1))
                        self.progress.emit(min(pct, 99))
                        # Also a brief log line every few seconds so user
                        # sees the trainer is alive even mid-epoch.
                        running_loss_val = train_total / max(train_n, 1)
                        frac_ep = epoch + (batch_idx + 1) / n_train_batches
                        self.running_loss.emit(frac_ep, running_loss_val)
                        self.log.emit(
                            f"  Ep {epoch+1}/{p['epochs']}  "
                            f"batch {batch_idx+1}/{n_train_batches}  "
                            f"loss={running_loss_val:.5f}"
                        )
                        last_emit = now

                # Validate
                model.eval()
                val_total = 0; val_n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        inp = batch['input'].to(device)
                        measured = batch['measured'].to(device)
                        with autocast('cuda', enabled=(device.type == 'cuda')):
                            amp, phase = model(inp)
                            pred_diff, support = physics(amp, phase)
                            losses = loss_fn(pred_diff, measured, amp, phase, support)
                        val_total += losses['total'].item()
                        val_n += 1
                        iter_count += 1

                train_loss = train_total / max(train_n, 1)
                val_loss = val_total / max(val_n, 1)
                scheduler.step(val_loss)
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]['lr']

                self.log.emit(
                    f"Epoch {epoch+1}/{p['epochs']}  "
                    f"train={train_loss:.5f}  val={val_loss:.5f}  "
                    f"lr={lr_now:.1e}  ({elapsed:.1f}s)"
                )
                self.epoch_done.emit(epoch + 1, train_loss, val_loss)
                self.progress.emit(int(100 * (epoch + 1) / p['epochs']))

                # Checkpoint
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'val_loss': best_val,
                        'grid_size': p['grid_size'],
                        'base_channels': p['base_channels'],
                        'enforce_oversampling': p['enforce_oversampling'],
                    }, out_dir / 'best_model.pt')
                    self.log.emit(f"  ★ New best model saved (val={best_val:.5f})")
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= p['patience']:
                        self.log.emit("Early stopping.")
                        break

            self.finished.emit(str(out_dir / 'best_model.pt'))

        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class SupervisedTrainingWorker(QThread):
    """
    Train the SUPERVISED PhaseUNet3D model.

    Unlike AutoPhaseNet (unsupervised, only diffraction), this needs
    ground-truth phase_true AND support in each .npz file. The data
    generator already produces these — they're ignored when training
    AutoPhaseNet.
    """
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    epoch_done = pyqtSignal(int, float, float)
    running_loss = pyqtSignal(float, float)  # (fractional_epoch, running_train_loss)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._should_stop = False

    def request_stop(self):
        """Cooperative stop — training loop checks this and exits cleanly."""
        self._should_stop = True

    def run(self):
        try:
            import torch
            import torch.optim as optim
            from torch.amp import GradScaler, autocast
            from torch.utils.data import DataLoader, random_split
            from cdi_st.nn_phase_model import PhaseUNet3D, BCDIPhaseLoss, count_parameters
            from cdi_st.nn_dataset import BCDIDataset

            p = self.p
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log.emit(f"Device: {device}")
            if device.type == 'cuda':
                self.log.emit(f"GPU: {torch.cuda.get_device_name()}")

            data_dir = Path(p['data_dir'])
            files = sorted(data_dir.glob("sample_*.npz"))
            if len(files) == 0:
                self.failed.emit(f"No sample_*.npz files in {data_dir}")
                return

            # Verify ground truth keys exist (needed for supervised training)
            import numpy as np
            try:
                test = np.load(files[0])
                if 'phase_true' not in test or 'support' not in test:
                    self.failed.emit(
                        "Supervised training requires .npz files with "
                        "'phase_true' and 'support' keys. Files generated "
                        "by the 'Generate Data' tab include these. Files "
                        "exported from the simulation tab do NOT — those "
                        "can only be used for unsupervised AutoPhaseNet."
                    )
                    return
            except Exception as e:
                self.failed.emit(f"Could not verify dataset: {e}")
                return

            self.log.emit(f"Found {len(files)} files in {data_dir}")

            # Build dataset
            full_ds = BCDIDataset(str(data_dir), augment=False, grid_size=p['grid_size'])
            n = len(full_ds)
            n_val = max(1, int(n * 0.15))
            n_train = n - n_val
            train_ds, val_ds = random_split(
                full_ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            train_ds.dataset = BCDIDataset(str(data_dir), augment=True,
                                             grid_size=p['grid_size'])
            train_loader = DataLoader(train_ds, batch_size=p['batch_size'],
                                       shuffle=True, num_workers=0, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=p['batch_size'],
                                     shuffle=False, num_workers=0)
            self.log.emit(f"Train: {n_train}  Val: {n_val}  Batch: {p['batch_size']}")

            # Model + loss
            # NOTE: BCDIPhaseLoss uses the original (alpha/beta/gamma)
            # keyword names. We deliberately do NOT use the descriptive
            # names (alpha_amp/beta_phase/gamma_diff) here, so this GUI
            # works with ANY version of nn_phase_model.py without requiring
            # the user to update both files in lockstep.
            model = PhaseUNet3D(in_channels=1, base_channels=p['base_channels']).to(device)
            loss_fn = BCDIPhaseLoss(
                alpha=p.get('alpha_amp', p.get('alpha', 1.0)),
                beta=p.get('beta_phase', p.get('beta', 1.0)),
                gamma=p.get('gamma_diff', p.get('gamma', 0.5)),
            )
            n_params = count_parameters(model)
            self.log.emit(f"PhaseUNet3D: {n_params:,} parameters, base_ch={p['base_channels']}")

            optimizer = optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                factor=0.5, patience=5)
            scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

            start_epoch = 0
            best_val = float('inf')
            if p.get('resume') and os.path.exists(p['resume']):
                ckpt = torch.load(p['resume'], map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                start_epoch = ckpt.get('epoch', 0)
                best_val = ckpt.get('val_loss', float('inf'))
                self.log.emit(f"Resumed from epoch {start_epoch}")

            out_dir = Path(p['output_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)
            no_improve = 0

            # Training loop
            # Intra-epoch progress reporting: emit progress at most every 2s
            n_train_batches = max(1, len(train_loader))
            n_val_batches = max(1, len(val_loader))
            total_iters = (p['epochs'] - start_epoch) * (n_train_batches + n_val_batches)
            iter_count = 0
            import time as _time
            last_emit = _time.time()

            for epoch in range(start_epoch, p['epochs']):
                if self._should_stop:
                    self.log.emit(
                        f"Stop requested. Saving checkpoint at end of epoch "
                        f"{epoch}/{p['epochs']}..."
                    )
                    break
                model.train()
                t0 = time.time()
                train_total = 0; train_n = 0

                for batch_idx, batch in enumerate(train_loader):
                    if self._should_stop:
                        break
                    inp = batch['input'].to(device)
                    phase_true = batch['phase_true'].to(device)
                    support = batch['support'].to(device)
                    diff_amp = batch['amplitude'].to(device)

                    optimizer.zero_grad(set_to_none=True)
                    with autocast('cuda', enabled=(device.type == 'cuda')):
                        phase_pred = model(inp)
                        losses = loss_fn(
                            phase_pred=phase_pred,
                            phase_true=phase_true,
                            support=support,
                            amplitude=diff_amp,
                        )
                    scaler.scale(losses['total']).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    train_total += losses['total'].item()
                    train_n += 1

                    iter_count += 1
                    now = _time.time()
                    if now - last_emit > 2.0:
                        pct = int(100 * iter_count / max(total_iters, 1))
                        self.progress.emit(min(pct, 99))
                        running_loss = train_total / max(train_n, 1)
                        # Fractional epoch for x-axis of live curve
                        frac_ep = epoch + (batch_idx + 1) / n_train_batches
                        self.running_loss.emit(frac_ep, running_loss)
                        self.log.emit(
                            f"  Ep {epoch+1}/{p['epochs']}  "
                            f"batch {batch_idx+1}/{n_train_batches}  "
                            f"loss={running_loss:.5f}"
                        )
                        last_emit = now

                # Validate
                model.eval()
                val_total = 0; val_n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        inp = batch['input'].to(device)
                        phase_true = batch['phase_true'].to(device)
                        support = batch['support'].to(device)
                        diff_amp = batch['amplitude'].to(device)
                        with autocast('cuda', enabled=(device.type == 'cuda')):
                            phase_pred = model(inp)
                            losses = loss_fn(
                                phase_pred=phase_pred,
                                phase_true=phase_true,
                                support=support,
                                amplitude=diff_amp,
                            )
                        val_total += losses['total'].item()
                        val_n += 1
                        iter_count += 1

                train_loss = train_total / max(train_n, 1)
                val_loss = val_total / max(val_n, 1)
                scheduler.step(val_loss)
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]['lr']

                self.log.emit(
                    f"Epoch {epoch+1}/{p['epochs']}  "
                    f"train={train_loss:.5f}  val={val_loss:.5f}  "
                    f"lr={lr_now:.1e}  ({elapsed:.1f}s)"
                )
                self.epoch_done.emit(epoch + 1, train_loss, val_loss)
                self.progress.emit(int(100 * (epoch + 1) / p['epochs']))

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'val_loss': best_val,
                        'grid_size': p['grid_size'],
                        'base_channels': p['base_channels'],
                    }, out_dir / 'best_model.pt')
                    self.log.emit(f"  \u2605 New best (val={best_val:.5f})")
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= p['patience']:
                        self.log.emit("Early stopping.")
                        break

            self.finished.emit(str(out_dir / 'best_model.pt'))

        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class ReconstructionWorker(QThread):
    """Run NN inference + RAAR refinement in background."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(dict)  # result dict with all arrays
    failed = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params

    def run(self):
        try:
            p = self.p
            self.log.emit(f"Loading {p['input_path']}...")
            self.progress.emit(5)

            # Load input
            from cdi_st.nn_autophase_infer import (
                load_input, nn_only_infer, refined_infer, ensemble_infer
            )

            # Read model checkpoint to find its trained grid size (if available)
            import torch as _torch
            try:
                _ckpt_meta = _torch.load(p['model_path'], map_location='cpu', weights_only=False)
                model_grid = _ckpt_meta.get('grid_size', None)
                model_channels = _ckpt_meta.get('base_channels', None)
                if model_grid:
                    self.log.emit(f"Model trained at grid {model_grid}\u00b3, base_channels={model_channels}")
                # Old checkpoints without grid_size metadata: the model is
                # fully convolutional so it runs at the input's native size.
                # Don't log a warning — that's the normal state for v0.1.x
                # checkpoints. The check is silent unless something actually
                # goes wrong (handled later in inference).
                del _ckpt_meta
            except Exception:
                model_grid = None

            diffraction, truth, voxel_nm = load_input(
                p['input_path'],
                target_size=model_grid if model_grid else p.get('grid_size', 64),
            )
            N_in = diffraction.shape[0]
            if voxel_nm is not None:
                self.log.emit(f"Input voxel pitch: {np.asarray(voxel_nm).mean():.3f} nm/voxel")

            # Only resample when the checkpoint EXPLICITLY says what grid it expects
            if model_grid and N_in != model_grid:
                self.log.emit(
                    f"Input volume is {N_in}\u00b3, model expects {model_grid}\u00b3 \u2014 "
                    f"resampling to match"
                )
                from scipy.ndimage import zoom
                factor = model_grid / N_in
                diffraction = zoom(diffraction, factor, order=1)
                if voxel_nm is not None:
                    voxel_nm = np.asarray(voxel_nm) / factor
                if truth is not None:
                    truth['phase_true'] = zoom(truth['phase_true'], factor, order=1)
                    truth['support'] = (zoom(truth['support'].astype(np.float32),
                                              factor, order=1) > 0.5).astype(np.float32)
                self.log.emit(f"Resampled volume: {diffraction.shape}")
            else:
                self.log.emit(f"Volume: {diffraction.shape}  max={diffraction.max():.2e}")

            self.progress.emit(15)

            mode = p['mode']
            if mode == 'nn_only':
                self.log.emit("AutoPhaseNet forward pass (no refinement)...")
                result = nn_only_infer(
                    diffraction, p['model_path'],
                    base_channels=p.get('base_channels', 32),
                    support_threshold=p.get('support_threshold', 0.05),
                )
            elif mode == 'refined':
                n_hio = p.get('n_hio', 50)
                self.log.emit(
                    f"AutoPhaseNet \u2192 HIO({n_hio}) \u2192 RAAR({p['n_raar']}) \u2192 ER({p['n_er']})..."
                )
                result = refined_infer(
                    diffraction, p['model_path'],
                    base_channels=p.get('base_channels', 32),
                    n_raar=p['n_raar'], n_er=p['n_er'], n_hio=n_hio,
                    support_threshold=p.get('support_threshold', 0.05),
                )
            elif mode in ('ensemble', 'ensemble+refine'):
                m2 = p.get('model_path2')
                if not m2:
                    raise RuntimeError("Ensemble mode requires Model 2 path")
                self.log.emit(f"Running ensemble (AutoPhaseNet + supervised)...")
                refine = (mode == 'ensemble+refine')
                result = ensemble_infer(
                    diffraction,
                    autophase_model=p['model_path'],
                    supervised_model=m2,
                    base_channels_autophase=p.get('base_channels', 32),
                    base_channels_supervised=p.get('base_channels', 32),
                    refine=refine,
                    n_hio=p.get('n_hio', 50),
                    n_raar=p['n_raar'],
                    n_er=p['n_er'],
                )
            else:
                raise RuntimeError(f"Unknown mode: {mode}")
            self.progress.emit(90)

            # Build output dict
            out = {
                'object_3d': result.object_3d,
                'amplitude': result.amplitude,
                'phase': result.phase,
                'support': result.support,
                'error_metric': result.error_metric,
                'method': result.method,
                'elapsed': result.elapsed_seconds,
                'diffraction': diffraction,
                'voxel_size_nm': voxel_nm,  # may be None for experimental data
            }
            if truth is not None:
                out['phase_true'] = truth['phase_true']
                out['support_true'] = truth['support']

            self.log.emit(
                f"Done in {result.elapsed_seconds:.2f}s  "
                f"R-factor={result.error_metric[-1]:.4f}"
            )
            self.progress.emit(100)
            self.done.emit(out)

        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 (4 in main GUI) — Generate Training Data
# ═══════════════════════════════════════════════════════════════════════════════

class DataGenWorker(QThread):
    """Background worker for generating training samples."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    sample_ready = pyqtSignal(int, dict)   # sample_id, preview dict
    finished_all = pyqtSignal(int)         # number of samples generated
    failed = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            from cdi_st.nn_data_generator import generate_single_sample
            import numpy as np
            from pathlib import Path

            p = self.p
            out_dir = Path(p['output_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)

            n = p['num_samples']
            generated = 0
            preview_every = max(1, n // 25)  # at most 25 previews shown

            self.log.emit(f"Generating {n} samples in {out_dir}")
            self.log.emit(f"Grid: {p['grid_size']}\u00b3   Material: {p['material'] or 'random'}")
            if p['vary_size']:
                self.log.emit(f"Varying particle size: {p['size_min']}-{p['size_max']} supercell mult")

            for i in range(n):
                if self._stopped:
                    self.log.emit("Stopped by user.")
                    break

                rng = np.random.default_rng(p['seed'] + i)

                try:
                    sample = generate_single_sample(
                        sample_id=i,
                        grid_size=p['grid_size'],
                        rng=rng,
                        add_noise=p['add_noise'],
                        fixed_material=p['material'] if p['material'] else None,
                        randomize_dislocation=p.get('randomize_dislocation', True),
                        randomize_strain=p.get('randomize_strain', True),
                    )
                    if sample is None:
                        continue

                    # Save .npz
                    fpath = out_dir / f"sample_{i:05d}.npz"
                    np.savez_compressed(
                        fpath,
                        amplitude=sample['amplitude'],
                        phase_true=sample['phase_true'],
                        support=sample['support'],
                        diffraction=sample['diffraction'],
                    )
                    generated += 1

                    # Emit preview every N samples
                    if i % preview_every == 0 or i == n - 1:
                        m = sample['metadata']
                        # Build a small preview dict (avoid emitting big arrays often)
                        preview = {
                            'sample_id': i,
                            'support_slice': sample['support'][:, :, sample['support'].shape[2] // 2].copy(),
                            'phase_slice': sample['phase_true'][:, :, sample['phase_true'].shape[2] // 2].copy(),
                            'diff_slice': np.log10(sample['diffraction'][:, :, sample['diffraction'].shape[2] // 2] + 1).copy(),
                            'meta': m,
                        }
                        self.sample_ready.emit(i, preview)

                    if (i + 1) % 5 == 0 or i == n - 1:
                        m = sample['metadata']
                        self.log.emit(
                            f"  [{generated}/{n}] {m['material']} {m['shape']} "
                            f"hkl={m['hkl']} strain={m['strain_type']}"
                        )

                except Exception as ex:
                    self.log.emit(f"  [skip] sample {i}: {ex}")

                self.progress.emit(int(100 * (i + 1) / n))

            self.finished_all.emit(generated)

        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class T_Gen(QWidget):
    """Tab: Generate Training Data."""

    def __init__(self):
        super().__init__()
        self._worker = None
        self._previews = []  # list of (sample_id, preview_dict)
        self._ui()

    def _ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left: controls ────────────────────────────────────────────────
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFixedWidth(380)
        inner = QWidget()
        ll = QVBoxLayout(inner)
        ll.setSpacing(6)
        ll.setContentsMargins(4, 4, 8, 4)

        title = QLabel("Generate Training Data")
        title.setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700")
        ll.addWidget(title)

        desc = QLabel(
            "Each sample is a (shape, phase, diffraction) triplet generated by\n"
            "randomizing crystal shape, size, strain, and reflection."
        )
        desc.setStyleSheet("color:#8b949e;font-size:8pt")
        desc.setWordWrap(True)
        ll.addWidget(desc)

        # Output directory
        og = QGroupBox("Output directory")
        ov = QHBoxLayout(og)
        self.out_dir = QLineEdit("./training_data")
        self.out_dir.setToolTip("Where to save the generated .npz samples")
        ov.addWidget(self.out_dir, 1)
        ob = QPushButton("Browse")
        ob.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px")
        ob.setMaximumWidth(60)
        ob.clicked.connect(self._browse_out)
        ov.addWidget(ob)
        ll.addWidget(og)

        # Quantity
        qg = QGroupBox("Quantity")
        qf = QFormLayout(qg)
        qf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.num_samples = QSpinBox()
        self.num_samples.setRange(1, 100000)
        self.num_samples.setValue(2000)
        self.num_samples.setToolTip(
            "Number of training samples to generate.\n"
            "More samples = better generalization.\n"
            "Recommended: 2000+ for production training."
        )
        qf.addRow("Number of samples:", self.num_samples)

        self.grid_size = QSpinBox()
        self.grid_size.setRange(16, 256)
        self.grid_size.setValue(64)
        self.grid_size.setSingleStep(16)
        self.grid_size.setToolTip("Detector grid size N → diffraction volume is N\u00b3")
        qf.addRow("Grid size:", self.grid_size)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 100000)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("Random seed for reproducibility")
        qf.addRow("Seed:", self.seed_spin)

        ll.addWidget(qg)

        # Material
        mg = QGroupBox("Material")
        mv = QVBoxLayout(mg)
        self.material_combo = QComboBox()
        self.material_combo.addItem("(random — all materials)")
        try:
            from cdi_st.bcdi_core import MATERIAL_PRESETS
            for name in sorted(MATERIAL_PRESETS.keys()):
                self.material_combo.addItem(name)
        except ImportError:
            pass
        self.material_combo.setToolTip(
            "Lock to one material or randomize.\n"
            "Random = better generalization across elements.\n"
            "Specific = better accuracy for that material only."
        )
        mv.addWidget(self.material_combo)
        ll.addWidget(mg)

        # Particle size variation
        sg = QGroupBox("Particle size variation")
        sf = QFormLayout(sg)
        sf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.vary_size = QCheckBox("Randomize particle size")
        self.vary_size.setChecked(True)
        self.vary_size.setToolTip(
            "If checked, each sample gets a random size between\n"
            "the min and max supercell multipliers below.\n"
            "If unchecked, all samples use the default size from bcdi_core."
        )
        sf.addRow(self.vary_size)

        self.size_min = QSpinBox()
        self.size_min.setRange(5, 100)
        self.size_min.setValue(10)
        self.size_min.setToolTip("Minimum supercell multiplier per axis (smaller = smaller particle)")
        sf.addRow("Min supercell:", self.size_min)

        self.size_max = QSpinBox()
        self.size_max.setRange(5, 100)
        self.size_max.setValue(30)
        self.size_max.setToolTip("Maximum supercell multiplier per axis")
        sf.addRow("Max supercell:", self.size_max)

        ll.addWidget(sg)

        # Noise
        ng = QGroupBox("Add experimental noise")
        nv = QVBoxLayout(ng)
        self.add_noise_check = QCheckBox("Add noise to ~30% of samples")
        self.add_noise_check.setChecked(True)
        self.add_noise_check.setToolTip(
            "Apply Poisson + readout + air-scatter noise to a random\n"
            "subset of samples to simulate real experimental data.\n"
            "Helps the model generalize to noisy measurements."
        )
        nv.addWidget(self.add_noise_check)
        ll.addWidget(ng)

        # Randomization toggles
        rg = QGroupBox("Randomization")
        rv = QVBoxLayout(rg)
        self.rand_disloc_check = QCheckBox("Random dislocations")
        self.rand_disloc_check.setChecked(True)
        self.rand_disloc_check.setToolTip(
            "If enabled, ~40% of samples include a random line dislocation\n"
            "(edge / screw / mixed) at a random position.\n"
            "Disable to generate only clean, defect-free crystals."
        )
        rv.addWidget(self.rand_disloc_check)
        self.rand_strain_check = QCheckBox("Random strain")
        self.rand_strain_check.setChecked(True)
        self.rand_strain_check.setToolTip(
            "If enabled, ~60% of samples include random strain\n"
            "(radial gradient, edge dislocation, or random field).\n"
            "Disable to generate only strain-free crystals."
        )
        rv.addWidget(self.rand_strain_check)
        ll.addWidget(rg)

        # Buttons
        self.gen_btn = QPushButton("\u25b6 Generate Samples")
        self.gen_btn.setStyleSheet("background:#1f6feb;min-height:36px;font-size:11pt;font-weight:600")
        self.gen_btn.clicked.connect(self._start_generation)
        ll.addWidget(self.gen_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background:#da3633;min-height:24px;font-size:9pt")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_generation)
        ll.addWidget(self.stop_btn)

        self.pg = QProgressBar()
        ll.addWidget(self.pg)

        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setStyleSheet("color:#8b949e;font-size:9pt")
        self.status_lbl.setWordWrap(True)
        ll.addWidget(self.status_lbl)

        ll.addStretch()
        sc.setWidget(inner)
        root.addWidget(sc)

        # ── Right: live preview grid + log ────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)

        rv.addWidget(QLabel("Live preview (sampled snapshots of generation)"))
        with matplotlib.rc_context(MPL_DARK):
            self.preview_fig = Figure(figsize=(10, 7), dpi=110, tight_layout=True, facecolor='#0a0d12')
        self.preview_canvas = FigureCanvas(self.preview_fig)
        self.preview_canvas.setMinimumHeight(380)
        fr = QFrame()
        fr.setFrameShape(QFrame.Shape.StyledPanel)
        fl = QVBoxLayout(fr)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.preview_canvas, 1)
        rv.addWidget(fr, 1)

        rv.addWidget(QLabel("Generation log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(170)
        rv.addWidget(self.log)

        root.addWidget(right, 1)

        # Initial empty preview
        self._draw_previews()

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output directory")
        if d:
            self.out_dir.setText(d)

    def _start_generation(self):
        out = self.out_dir.text().strip()
        if not out:
            QMessageBox.warning(self, "Error", "Specify an output directory.")
            return

        material_idx = self.material_combo.currentIndex()
        material = None if material_idx == 0 else self.material_combo.currentText()

        if self.size_min.value() >= self.size_max.value() and self.vary_size.isChecked():
            QMessageBox.warning(self, "Error", "Min supercell must be < Max supercell.")
            return

        params = {
            'output_dir': out,
            'num_samples': self.num_samples.value(),
            'grid_size': self.grid_size.value(),
            'seed': self.seed_spin.value(),
            'material': material,
            'vary_size': self.vary_size.isChecked(),
            'size_min': self.size_min.value(),
            'size_max': self.size_max.value(),
            'add_noise': self.add_noise_check.isChecked(),
            'randomize_dislocation': self.rand_disloc_check.isChecked(),
            'randomize_strain': self.rand_strain_check.isChecked(),
        }

        self._previews = []
        self.log.clear()
        self.pg.setValue(0)
        self.gen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText("Generating...")

        self._worker = DataGenWorker(params)
        self._worker.log.connect(self.log.appendPlainText)
        self._worker.progress.connect(self.pg.setValue)
        self._worker.sample_ready.connect(self._on_sample_ready)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop_generation(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        self.gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_sample_ready(self, sid, preview):
        self._previews.append((sid, preview))
        # Keep only the last 12 previews (4×3 grid)
        if len(self._previews) > 12:
            self._previews = self._previews[-12:]
        self._draw_previews()

    def _draw_previews(self):
        with matplotlib.rc_context(MPL_DARK):
            self.preview_fig.clear()

            if not self._previews:
                ax = self.preview_fig.add_subplot(111)
                ax.set_facecolor('#0a0d12')
                ax.text(0.5, 0.5,
                        "Click 'Generate Samples' to start.\n\n"
                        "Generated lattice previews will appear here.",
                        ha='center', va='center', fontsize=12, color='#8b949e',
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                self.preview_canvas.draw()
                return

            # Show up to 12 previews in a 3x4 grid (each preview is 1 wide column)
            n = len(self._previews)
            cols = min(4, n)
            rows = (n + cols - 1) // cols
            for i, (sid, prev) in enumerate(self._previews):
                ax = self.preview_fig.add_subplot(rows, cols, i + 1)
                ax.set_facecolor('#000005')
                # Show the diffraction in log scale (most informative single image)
                ax.imshow(prev['diff_slice'], cmap='jet', origin='lower', aspect='equal')
                m = prev['meta']
                shape = m.get('shape', '?')
                mat = m.get('material', '?')
                hkl = m.get('hkl', '?')
                ax.set_title(f"#{sid}: {mat} {shape}\nhkl={hkl}",
                             fontsize=7, color='#e6edf3')
                ax.tick_params(labelsize=5)
                ax.set_xticks([]); ax.set_yticks([])

        self.preview_canvas.draw()

    def _on_finished(self, n):
        self.gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText(f"\u2713 Generated {n} samples successfully.")
        self.log.appendPlainText(f"\nDone — {n} samples saved to {self.out_dir.text()}")

    def _on_failed(self, msg):
        self.gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("Failed.")
        self.log.appendPlainText(f"\nERROR:\n{msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — NN Training
# ═══════════════════════════════════════════════════════════════════════════════

class T4(QWidget):
    """Training tab: data directory, hyperparameters, live training curve."""

    def __init__(self):
        super().__init__()
        self._worker = None
        self._train_losses = []
        self._val_losses = []
        self._ui()

    def _ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left panel: controls ──────────────────────────────────────────
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFixedWidth(400)
        inner = QWidget()
        ll = QVBoxLayout(inner)
        ll.setSpacing(6)
        ll.setContentsMargins(4, 4, 8, 4)

        title = QLabel("AutoPhaseNet Training (unsupervised)")
        title.setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700")
        ll.addWidget(title)

        info = QLabel(
            "Trains the UNSUPERVISED dual-decoder model from Yao et al. 2022.\n"
            "Predicts both amplitude and phase. Uses only diffraction data\n"
            "(ground-truth phase/support are NOT required)."
        )
        info.setStyleSheet("color:#8b949e;font-size:8pt")
        info.setWordWrap(True)
        ll.addWidget(info)

        # Data directory
        dg = QGroupBox("Training Data")
        dv = QVBoxLayout(dg)
        dr = QHBoxLayout()
        self.data_dir = QLineEdit()
        self.data_dir.setPlaceholderText("/path/to/training_data")
        dr.addWidget(self.data_dir, 1)
        browse = QPushButton("Browse")
        browse.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px")
        browse.setMaximumWidth(60)
        browse.clicked.connect(self._browse_data)
        dr.addWidget(browse)
        dv.addLayout(dr)
        self.data_info = QLabel("Select a directory with .npz or .h5 files")
        self.data_info.setStyleSheet("color:#8b949e;font-size:9pt")
        self.data_info.setWordWrap(True)
        dv.addWidget(self.data_info)
        ll.addWidget(dg)

        # Model path (optional resume)
        mg = QGroupBox("Resume from checkpoint (optional)")
        mv = QHBoxLayout(mg)
        self.resume_path = QLineEdit()
        self.resume_path.setPlaceholderText("checkpoints/best_model.pt")
        mv.addWidget(self.resume_path, 1)
        rb = QPushButton("Browse")
        rb.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px")
        rb.setMaximumWidth(60)
        rb.clicked.connect(self._browse_resume)
        mv.addWidget(rb)
        ll.addWidget(mg)

        # Hyperparameters
        hg = QGroupBox("Hyperparameters")
        hf = QFormLayout(hg)
        hf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        hf.setVerticalSpacing(5)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(60)
        self.epochs_spin.setToolTip(
            "Number of complete passes over the training set.\n"
            "Typical: 50-100 for good results.\n"
            "More epochs = better fit but longer training."
        )
        hf.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        self.batch_spin.setToolTip(
            "Number of samples processed per gradient update.\n"
            "Larger = more stable training but more GPU memory.\n"
            "8 fits ~8GB GPU at 64³, drop to 4 if OOM."
        )
        hf.addRow("Batch size:", self.batch_spin)

        self.lr_spin = _dbl(1e-6, 0.1, 1e-3, 6)
        self.lr_spin.setToolTip(
            "Learning rate for AdamW optimizer.\n"
            "1e-3 = standard for fresh training.\n"
            "1e-5 = good for fine-tuning a pretrained model."
        )
        hf.addRow("Learning rate:", self.lr_spin)

        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(8, 128)
        self.channels_spin.setValue(32)
        self.channels_spin.setToolTip(
            "Width of the U-Net base layer (doubles per level).\n"
            "16 = 0.5M params (light, fits 4GB GPU)\n"
            "32 = 2.3M params (recommended, fits 8GB)\n"
            "64 = 9M params (heavy, needs ≥16GB)"
        )
        hf.addRow("Base channels:", self.channels_spin)

        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(16, 256)
        self.grid_spin.setValue(64)
        self.grid_spin.setSingleStep(16)
        self.grid_spin.setToolTip(
            "Diffraction volume size N (cube N×N×N).\n"
            "Must match your data files. 64 is standard for BCDI."
        )
        hf.addRow("Grid size:", self.grid_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(3, 100)
        self.patience_spin.setValue(20)
        self.patience_spin.setToolTip(
            "Stop training if validation loss does not improve\n"
            "for this many consecutive epochs (early stopping)."
        )
        hf.addRow("Patience:", self.patience_spin)

        ll.addWidget(hg)

        # Regularizers
        rg = QGroupBox("Regularizers")
        rf = QFormLayout(rg)
        rf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        rf.setVerticalSpacing(5)

        self.smooth_spin = _dbl(0, 1, 0.005, 4)
        self.smooth_spin.setToolTip(
            "Penalty on fragmented support shape.\n"
            "Higher = forces smoother, more compact support.\n"
            "Set to 0 if you want fragmented features (e.g. dislocations)."
        )
        rf.addRow("Support smooth:", self.smooth_spin)

        self.tv_spin = _dbl(0, 1, 0.005, 4)
        self.tv_spin.setToolTip(
            "Total Variation on phase = smoother strain fields.\n"
            "0 = preserve sharp phase jumps (dislocations).\n"
            "0.005-0.05 = realistic strain smoothness."
        )
        rf.addRow("TV phase:", self.tv_spin)

        self.thresh_spin = _dbl(0, 1, 0.1, 2)
        self.thresh_spin.setToolTip(
            "Support shape threshold on predicted amplitude.\n"
            "Voxels with |ρ| above this fraction of the max\n"
            "are considered inside the crystal."
        )
        rf.addRow("Threshold:", self.thresh_spin)

        self.os_check = QCheckBox("Enforce oversampling (zero-pad N/2)")
        self.os_check.setChecked(True)
        self.os_check.setToolTip(
            "Constrain object to fit in central N/2 box.\n"
            "Required by Miao oversampling theorem for phase retrieval.\n"
            "Disable only for special cases."
        )
        rf.addRow(self.os_check)

        ll.addWidget(rg)

        # Noise simulation (helps training generalize to real experiments)
        ng = QGroupBox("Add experimental noise to training samples")
        nf = QFormLayout(ng)
        nf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        nf.setVerticalSpacing(5)

        self.noise_check = QCheckBox("Apply noise during training")
        self.noise_check.setChecked(False)
        self.noise_check.setToolTip(
            "Apply random Poisson/readout/air-scatter noise\n"
            "on-the-fly during training. Makes the model\n"
            "more robust to real experimental data."
        )
        nf.addRow(self.noise_check)

        self.noise_poisson = QCheckBox("Poisson (photon shot noise)")
        self.noise_poisson.setChecked(True)
        self.noise_poisson.setToolTip(
            "Simulates photon counting statistics.\n"
            "Always present in real measurements."
        )
        nf.addRow(self.noise_poisson)

        self.noise_readout = _dbl(0, 100, 2.0, 1)
        self.noise_readout.setToolTip(
            "Detector electronic readout noise (RMS counts).\n"
            "Typical Pilatus/Eiger: 1-5 counts."
        )
        nf.addRow("Readout noise:", self.noise_readout)

        self.noise_airscatter = _dbl(0, 1000, 50.0, 1)
        self.noise_airscatter.setToolTip(
            "Air scatter background (counts/pixel).\n"
            "Typical synchrotron: 10-200 counts."
        )
        nf.addRow("Air scatter:", self.noise_airscatter)

        self.noise_dead = _dbl(0, 0.1, 0.005, 4)
        self.noise_dead.setToolTip(
            "Fraction of dead/hot detector pixels (0-0.1).\n"
            "Typical real detectors: 0.001-0.01."
        )
        nf.addRow("Dead pixels:", self.noise_dead)

        ll.addWidget(ng)

        # Output directory
        og = QGroupBox("Output")
        of_ = QHBoxLayout(og)
        self.out_dir = QLineEdit("./checkpoints_autophase")
        of_.addWidget(self.out_dir, 1)
        ob = QPushButton("Browse")
        ob.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px")
        ob.setMaximumWidth(60)
        ob.clicked.connect(self._browse_out)
        of_.addWidget(ob)
        ll.addWidget(og)

        # Launch button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet("background:#1f6feb;min-height:34px;font-size:11pt")
        self.train_btn.clicked.connect(self._start_training)
        ll.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background:#da3633;min-height:26px;font-size:9pt")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        ll.addWidget(self.stop_btn)

        self.pg = QProgressBar()
        ll.addWidget(self.pg)

        ll.addStretch()
        sc.setWidget(inner)
        root.addWidget(sc)

        # ── Right panel: training log + live curve ────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)

        # Training curve
        rv.addWidget(QLabel("Training curve"))
        with matplotlib.rc_context(MPL_DARK):
            self.fig_curve = Figure(figsize=(6, 3.5), dpi=130, tight_layout=True)
        self.canvas_curve = FigureCanvas(self.fig_curve)
        self.canvas_curve.setMinimumHeight(260)
        fr = QFrame()
        fr.setFrameShape(QFrame.Shape.StyledPanel)
        fl = QVBoxLayout(fr)
        fl.setContentsMargins(0, 0, 0, 0)
        tb = NavigationToolbar(self.canvas_curve, fr)
        tb.setStyleSheet("background:#161b22")
        fl.addWidget(tb)
        fl.addWidget(self.canvas_curve, 1)
        rv.addWidget(fr, 1)

        # Log
        rv.addWidget(QLabel("Training log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(220)
        rv.addWidget(self.log)

        root.addWidget(right, 1)

    def _browse_data(self):
        d = QFileDialog.getExistingDirectory(self, "Select training data directory")
        if d:
            self.data_dir.setText(d)
            npz = len(list(Path(d).glob("sample_*.npz")))
            h5 = len(list(Path(d).glob("*.h5")))
            self.data_info.setText(f"Found {npz} .npz and {h5} .h5 files ({npz + h5} total)")

    def _browse_resume(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", "", "PyTorch (*.pt)")
        if f:
            self.resume_path.setText(f)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.out_dir.setText(d)

    def _start_training(self):
        data_dir = self.data_dir.text().strip()
        if not data_dir or not Path(data_dir).exists():
            QMessageBox.warning(self, "Error", "Select a valid data directory")
            return

        params = {
            'data_dir': data_dir,
            'output_dir': self.out_dir.text().strip(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'lr': self.lr_spin.value(),
            'base_channels': self.channels_spin.value(),
            'grid_size': self.grid_spin.value(),
            'patience': self.patience_spin.value(),
            'support_smoothness': self.smooth_spin.value(),
            'tv_phase': self.tv_spin.value(),
            'threshold': self.thresh_spin.value(),
            'enforce_oversampling': self.os_check.isChecked(),
            # Noise params (applied during dataloader augmentation)
            'apply_noise': self.noise_check.isChecked(),
            'noise_poisson': self.noise_poisson.isChecked(),
            'noise_readout': self.noise_readout.value(),
            'noise_air': self.noise_airscatter.value(),
            'noise_dead': self.noise_dead.value(),
        }
        resume = self.resume_path.text().strip()
        if resume and os.path.exists(resume):
            params['resume'] = resume

        self._train_losses = []
        self._val_losses = []
        self.log.clear()
        self.pg.setValue(0)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._worker = TrainingWorker(params)
        self._worker.log.connect(self.log.appendPlainText)
        self._worker.progress.connect(self.pg.setValue)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.running_loss.connect(self._on_running_loss)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop_training(self):
        if self._worker and self._worker.isRunning():
            # Cooperative stop: ask the worker to break out of the training
            # loop at the next batch boundary. The worker will save its
            # checkpoint and emit 'finished' cleanly.
            #
            # Do NOT call self._worker.terminate() — that kills the thread
            # mid-batch and corrupts CUDA / autograd state, crashing the GUI.
            self._worker.request_stop()
            self.log.appendPlainText(
                "Stop requested. Worker will finish current batch and save "
                "a checkpoint. This may take a few seconds..."
            )
            self.stop_btn.setEnabled(False)
            # train_btn re-enabled by _on_finished / _on_failed
            return
        # If worker isn't running, just reset buttons
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_epoch(self, epoch, train_loss, val_loss):
        """Called when a full epoch finishes — append the official point."""
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        # Live points are now superseded by the official epoch point
        self._live_x = []
        self._live_y = []
        self._draw_curve()

    def _on_running_loss(self, fractional_epoch, loss):
        """Live mid-epoch loss point for users to see something is happening."""
        if not hasattr(self, '_live_x'):
            self._live_x = []
            self._live_y = []
        self._live_x.append(fractional_epoch)
        self._live_y.append(loss)
        # Keep only the last 200 points so we don't slow down rendering
        if len(self._live_x) > 200:
            self._live_x = self._live_x[-200:]
            self._live_y = self._live_y[-200:]
        self._draw_curve()

    def _draw_curve(self):
        with matplotlib.rc_context(MPL_DARK):
            self.fig_curve.clear()
            ax = self.fig_curve.add_subplot(111)
            ax.set_facecolor('#000005')
            # Choose log vs linear: log breaks when there's a single point or
            # when losses cover < 1 decade; linear is safer for early epochs
            n_epochs = len(self._train_losses)
            use_log = n_epochs >= 3 and max(self._train_losses) / max(
                min(self._train_losses), 1e-12) > 3.0
            plot = ax.semilogy if use_log else ax.plot

            # Live curve (mid-epoch running loss) — light, dashed, behind
            if hasattr(self, '_live_x') and self._live_x:
                plot(self._live_x, self._live_y, color='#3fb950',
                     linewidth=1, alpha=0.45, linestyle='--', label='Running')

            # Per-epoch curves
            if n_epochs > 0:
                epochs = list(range(1, n_epochs + 1))
                plot(epochs, self._train_losses, color='#3fb950',
                     linewidth=2, marker='o', markersize=4, label='Train')
                plot(epochs, self._val_losses, color='#f0883e',
                     linewidth=2, marker='s', markersize=4, label='Val')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training curve', fontsize=10)
            if (hasattr(self, '_live_x') and self._live_x) or n_epochs > 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2)
        self.canvas_curve.draw()

    def _on_finished(self, model_path):
        self.log.appendPlainText(f"\n✓ Training complete. Model: {model_path}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_failed(self, msg):
        self.log.appendPlainText(f"\n✗ FAILED: {msg}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4_Sup — Supervised PhaseUNet3D Training
# ═══════════════════════════════════════════════════════════════════════════════

class T4_Sup(QWidget):
    """
    Training tab for the SUPERVISED PhaseUNet3D model.

    Required dataset: .npz files with keys
        'amplitude', 'phase_true', 'support', 'diffraction'
    Files generated by the 'Generate Data' tab include all of these.
    Files exported from the simulation tab include only 'diffraction_volume'
    and cannot be used for supervised training.

    Output: best_model.pt in the chosen output directory. Use this as
    'Model 2' in the Reconstruct tab for ensemble mode.
    """

    def __init__(self):
        super().__init__()
        self._worker = None
        self._train_losses = []
        self._val_losses = []
        self._ui()

    def _ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left panel: controls ──────────────────────────────────────────
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFixedWidth(400)
        inner = QWidget()
        ll = QVBoxLayout(inner)
        ll.setSpacing(6)
        ll.setContentsMargins(4, 4, 8, 4)

        title = QLabel("Supervised PhaseUNet3D Training")
        title.setStyleSheet("color:#bf8700;font-size:13pt;font-weight:700")
        ll.addWidget(title)

        info = QLabel(
            "Trains a SUPERVISED model that learns from ground-truth\n"
            "phase and support labels (single-decoder, phase only).\n"
            "Use the resulting checkpoint as 'Model 2' in the Reconstruct\n"
            "tab to enable ensemble inference."
        )
        info.setStyleSheet("color:#8b949e;font-size:8pt")
        info.setWordWrap(True)
        ll.addWidget(info)

        # Data dir
        dg = QGroupBox("Training Data")
        dv = QVBoxLayout(dg)
        dr = QHBoxLayout()
        self.data_dir = QLineEdit()
        self.data_dir.setPlaceholderText("/path/to/training_data")
        self.data_dir.setToolTip(
            "Directory containing .npz files with phase_true and support "
            "(generated by the 'Generate Data' tab — NOT simulation exports)."
        )
        dr.addWidget(self.data_dir, 1)
        browse = QPushButton("Browse")
        browse.setStyleSheet("background:#bf8700;padding:5px 10px;min-height:20px")
        browse.setMaximumWidth(60)
        browse.clicked.connect(self._browse_data)
        dr.addWidget(browse)
        dv.addLayout(dr)
        self.data_info = QLabel("Select a directory with sample_*.npz files")
        self.data_info.setStyleSheet("color:#8b949e;font-size:9pt")
        self.data_info.setWordWrap(True)
        dv.addWidget(self.data_info)
        ll.addWidget(dg)

        # Resume
        mg = QGroupBox("Resume from checkpoint (optional)")
        mv = QHBoxLayout(mg)
        self.resume_path = QLineEdit()
        self.resume_path.setPlaceholderText("checkpoints_supervised/best_model.pt")
        mv.addWidget(self.resume_path, 1)
        rb = QPushButton("Browse")
        rb.setStyleSheet("background:#bf8700;padding:5px 10px;min-height:20px")
        rb.setMaximumWidth(60)
        rb.clicked.connect(self._browse_resume)
        mv.addWidget(rb)
        ll.addWidget(mg)

        # Hyperparameters
        hg = QGroupBox("Hyperparameters")
        hf = QFormLayout(hg)
        hf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        hf.setVerticalSpacing(5)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(60)
        self.epochs_spin.setToolTip("Total epochs. 50-100 typical for supervised training.")
        hf.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        self.batch_spin.setToolTip("Batch size. 8 fits ~8GB GPU at 64\u00b3.")
        hf.addRow("Batch size:", self.batch_spin)

        self.lr_spin = _dbl(1e-6, 0.1, 1e-3, 6)
        self.lr_spin.setToolTip("Learning rate. 1e-3 typical for fresh training.")
        hf.addRow("Learning rate:", self.lr_spin)

        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(8, 128)
        self.channels_spin.setValue(32)
        self.channels_spin.setToolTip(
            "Network width. 32 = ~2M params (recommended for ensemble use).\n"
            "Should match base_channels of the AutoPhaseNet model for fair ensembling."
        )
        hf.addRow("Base channels:", self.channels_spin)

        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(16, 256)
        self.grid_spin.setValue(64)
        self.grid_spin.setSingleStep(16)
        self.grid_spin.setToolTip("Grid size. Must match training data.")
        hf.addRow("Grid size:", self.grid_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(3, 100)
        self.patience_spin.setValue(20)
        self.patience_spin.setToolTip("Early stopping patience.")
        hf.addRow("Patience:", self.patience_spin)

        ll.addWidget(hg)

        # Loss weights (specific to PhaseUNet3D / BCDIPhaseLoss)
        wg = QGroupBox("Loss weights")
        wf = QFormLayout(wg)
        wf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        wf.setVerticalSpacing(5)

        self.alpha_spin = _dbl(0, 10, 1.0, 3)
        self.alpha_spin.setToolTip(
            "Weight for amplitude/support match.\n"
            "Higher = stronger emphasis on getting the support shape right."
        )
        wf.addRow("\u03b1 amplitude:", self.alpha_spin)

        self.beta_spin = _dbl(0, 10, 1.0, 3)
        self.beta_spin.setToolTip(
            "Weight for phase prediction inside support.\n"
            "Higher = stronger emphasis on phase accuracy."
        )
        wf.addRow("\u03b2 phase:", self.beta_spin)

        self.gamma_spin = _dbl(0, 10, 0.5, 3)
        self.gamma_spin.setToolTip(
            "Weight for diffraction-magnitude consistency.\n"
            "Connects the supervised model to the physics constraint."
        )
        wf.addRow("\u03b3 diffraction:", self.gamma_spin)

        ll.addWidget(wg)

        # Output
        og = QGroupBox("Output")
        of_ = QHBoxLayout(og)
        self.out_dir = QLineEdit("./checkpoints_supervised")
        self.out_dir.setToolTip(
            "Where to save the best checkpoint. Use this path as 'Model 2'\n"
            "in the Reconstruct tab for ensemble inference."
        )
        of_.addWidget(self.out_dir, 1)
        ob = QPushButton("Browse")
        ob.setStyleSheet("background:#bf8700;padding:5px 10px;min-height:20px")
        ob.setMaximumWidth(60)
        ob.clicked.connect(self._browse_out)
        of_.addWidget(ob)
        ll.addWidget(og)

        # Buttons
        self.train_btn = QPushButton("Start Supervised Training")
        self.train_btn.setStyleSheet("background:#bf8700;min-height:34px;font-size:11pt;font-weight:600")
        self.train_btn.clicked.connect(self._start_training)
        ll.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background:#da3633;min-height:26px;font-size:9pt")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        ll.addWidget(self.stop_btn)

        self.pg = QProgressBar()
        ll.addWidget(self.pg)

        ll.addStretch()
        sc.setWidget(inner)
        root.addWidget(sc)

        # ── Right panel: training curve + log ─────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)

        rv.addWidget(QLabel("Training curve (supervised)"))
        with matplotlib.rc_context(MPL_DARK):
            self.fig_curve = Figure(figsize=(6, 3.5), dpi=130, tight_layout=True)
        self.canvas_curve = FigureCanvas(self.fig_curve)
        self.canvas_curve.setMinimumHeight(260)
        fr = QFrame()
        fr.setFrameShape(QFrame.Shape.StyledPanel)
        fl = QVBoxLayout(fr)
        fl.setContentsMargins(0, 0, 0, 0)
        tb = NavigationToolbar(self.canvas_curve, fr)
        tb.setStyleSheet("background:#161b22")
        fl.addWidget(tb)
        fl.addWidget(self.canvas_curve, 1)
        rv.addWidget(fr, 1)

        rv.addWidget(QLabel("Training log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(220)
        rv.addWidget(self.log)

        root.addWidget(right, 1)

    def _browse_data(self):
        d = QFileDialog.getExistingDirectory(self, "Select training data directory")
        if d:
            self.data_dir.setText(d)
            files = list(Path(d).glob("sample_*.npz"))
            if files:
                # Verify ground truth keys
                try:
                    test = np.load(files[0])
                    has_truth = ('phase_true' in test and 'support' in test)
                except Exception:
                    has_truth = False
                status = "\u2713 ground truth found" if has_truth else "\u2717 missing phase_true/support"
                self.data_info.setText(f"Found {len(files)} .npz files \u2014 {status}")
            else:
                self.data_info.setText("No sample_*.npz files found")

    def _browse_resume(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", "", "PyTorch (*.pt)")
        if f:
            self.resume_path.setText(f)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.out_dir.setText(d)

    def _start_training(self):
        data_dir = self.data_dir.text().strip()
        if not data_dir or not Path(data_dir).exists():
            QMessageBox.warning(self, "Error", "Select a valid data directory")
            return

        params = {
            'data_dir': data_dir,
            'output_dir': self.out_dir.text().strip(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'lr': self.lr_spin.value(),
            'base_channels': self.channels_spin.value(),
            'grid_size': self.grid_spin.value(),
            'patience': self.patience_spin.value(),
            'alpha_amp': self.alpha_spin.value(),
            'beta_phase': self.beta_spin.value(),
            'gamma_diff': self.gamma_spin.value(),
        }
        resume = self.resume_path.text().strip()
        if resume and os.path.exists(resume):
            params['resume'] = resume

        self._train_losses = []
        self._val_losses = []
        self.log.clear()
        self.pg.setValue(0)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._worker = SupervisedTrainingWorker(params)
        self._worker.log.connect(self.log.appendPlainText)
        self._worker.progress.connect(self.pg.setValue)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.running_loss.connect(self._on_running_loss)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop_training(self):
        if self._worker and self._worker.isRunning():
            # Cooperative stop — see comment in T4_Sup._stop_training
            self._worker.request_stop()
            self.log.appendPlainText(
                "Stop requested. Worker will finish current batch and save "
                "a checkpoint. This may take a few seconds..."
            )
            self.stop_btn.setEnabled(False)
            return
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_epoch(self, epoch, train_loss, val_loss):
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._live_x = []
        self._live_y = []
        self._draw_curve()

    def _on_running_loss(self, fractional_epoch, loss):
        """Live mid-epoch loss point."""
        if not hasattr(self, '_live_x'):
            self._live_x = []
            self._live_y = []
        self._live_x.append(fractional_epoch)
        self._live_y.append(loss)
        if len(self._live_x) > 200:
            self._live_x = self._live_x[-200:]
            self._live_y = self._live_y[-200:]
        self._draw_curve()

    def _draw_curve(self):
        with matplotlib.rc_context(MPL_DARK):
            self.fig_curve.clear()
            ax = self.fig_curve.add_subplot(111)
            ax.set_facecolor('#000005')
            n_epochs = len(self._train_losses)
            use_log = n_epochs >= 3 and max(self._train_losses) / max(
                min(self._train_losses), 1e-12) > 3.0
            plot = ax.semilogy if use_log else ax.plot

            if hasattr(self, '_live_x') and self._live_x:
                plot(self._live_x, self._live_y, color='#3fb950',
                     linewidth=1, alpha=0.45, linestyle='--', label='Running')

            if n_epochs > 0:
                epochs = list(range(1, n_epochs + 1))
                plot(epochs, self._train_losses, color='#3fb950',
                     linewidth=2, marker='o', markersize=4, label='Train')
                plot(epochs, self._val_losses, color='#bf8700',
                     linewidth=2, marker='s', markersize=4, label='Val')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Unsupervised training curve', fontsize=10)
            if (hasattr(self, '_live_x') and self._live_x) or n_epochs > 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2)
        self.canvas_curve.draw()

    def _on_finished(self, model_path):
        self.log.appendPlainText(f"\n\u2713 Training complete. Model: {model_path}")
        self.log.appendPlainText("\nUse this path as 'Model 2' in the Reconstruct tab\nfor ensemble inference.")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_failed(self, msg):
        self.log.appendPlainText(f"\n\u2717 FAILED: {msg}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BCDI Reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

class T5(QWidget):
    """Reconstruction tab: load data, run NN + refinement, show 4 figures."""

    recon_done = pyqtSignal(dict)  # emitted when reconstruction finishes (for T6)

    def __init__(self):
        super().__init__()
        self._worker = None
        self._result = None
        self._ui()

    def _ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── Top bar: input + model + launch ───────────────────────────────
        title = QLabel("BCDI Reconstruction")
        title.setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700")
        root.addWidget(title)

        top = QGroupBox("Input")
        tg = QGridLayout(top)
        tg.setHorizontalSpacing(8)
        tg.setVerticalSpacing(5)

        tg.addWidget(QLabel("Data file:"), 0, 0, Qt.AlignmentFlag.AlignRight)
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText(".npz (simulated/preprocessed) or .h5 (experimental)")
        tg.addWidget(self.input_path, 0, 1)
        # Browse + SPEC button row
        ib_row = QHBoxLayout()
        ib_row.setSpacing(4)
        ib = QPushButton("Browse")
        ib.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px")
        ib.setMaximumWidth(70)
        ib.clicked.connect(self._browse_input)
        ib_row.addWidget(ib)
        spec_btn = QPushButton("SPEC+EDF…")
        spec_btn.setStyleSheet("background:#bf8700;padding:5px 8px;min-height:20px;font-size:9pt")
        spec_btn.setMaximumWidth(95)
        spec_btn.setToolTip(
            "Convert ID01-style SPEC + EDF data to .npz.\n"
            "Reads motor positions from the SPEC file and detector\n"
            "frames from the .edf.gz files, then preprocesses\n"
            "(gap-mask, hot pixels, peak finding, q-space).\n"
            "Requires xrayutilities for q-space conversion."
        )
        spec_btn.clicked.connect(self._convert_spec_edf)
        ib_row.addWidget(spec_btn)
        p10_btn = QPushButton("P10 .h5…")
        p10_btn.setStyleSheet("background:#bf8700;padding:5px 8px;min-height:20px;font-size:9pt")
        p10_btn.setMaximumWidth(75)
        p10_btn.setToolTip(
            "Convert PETRA III P10 (DESY) data to .npz.\n"
            "Pick either the _master.h5 file or any _data_NNNNNN.h5 chunk —\n"
            "the loader auto-finds the master and reads the linked frames.\n"
            "Reads motor positions from the .fio metadata file if present."
        )
        p10_btn.clicked.connect(self._convert_p10_h5)
        ib_row.addWidget(p10_btn)
        ib_widget = QWidget()
        ib_widget.setLayout(ib_row)
        ib_widget.setMaximumWidth(245)
        tg.addWidget(ib_widget, 0, 2)

        tg.addWidget(QLabel("Model:"), 1, 0, Qt.AlignmentFlag.AlignRight)
        self.model_path = QLineEdit("checkpoints_autophase/best_model.pt")
        self.model_path.setToolTip("AutoPhaseNet (unsupervised) checkpoint")
        tg.addWidget(self.model_path, 1, 1)
        mb = QPushButton("Browse")
        mb.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px")
        mb.setMaximumWidth(60)
        mb.clicked.connect(self._browse_model)
        tg.addWidget(mb, 1, 2)

        # Optional second model for ensemble
        tg.addWidget(QLabel("Model 2 (ensemble):"), 2, 0, Qt.AlignmentFlag.AlignRight)
        self.model_path2 = QLineEdit()
        self.model_path2.setPlaceholderText("(optional) supervised PhaseUNet3D checkpoint")
        self.model_path2.setToolTip(
            "Optional 2nd model for ensemble mode.\n"
            "Should be a supervised PhaseUNet3D checkpoint trained on\n"
            "(diffraction, phase_true, support) triplets."
        )
        tg.addWidget(self.model_path2, 2, 1)
        mb2 = QPushButton("Browse")
        mb2.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px")
        mb2.setMaximumWidth(60)
        mb2.clicked.connect(self._browse_model2)
        tg.addWidget(mb2, 2, 2)

        # Mode + params row
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["refined", "nn_only", "ensemble", "ensemble+refine"])
        self.mode_combo.setToolTip(
            "Reconstruction strategy:\n"
            "\u2022 nn_only: just AutoPhaseNet forward pass (~100ms, rough)\n"
            "\u2022 refined: AutoPhaseNet \u2192 HIO \u2192 RAAR \u2192 ER (most common)\n"
            "\u2022 ensemble: combine AutoPhaseNet + supervised PhaseUNet3D\n"
            "\u2022 ensemble+refine: ensemble \u2192 HIO/RAAR/ER\n\n"
            "Ensemble modes need Model 2 to be set (supervised checkpoint).\n"
            "If 'refined' is worse than 'nn_only', automatic fallback to NN output."
        )
        pr.addWidget(self.mode_combo)
        pr.addWidget(QLabel("HIO:"))
        self.hio_spin = QSpinBox()
        self.hio_spin.setRange(0, 1000)
        self.hio_spin.setValue(50)
        self.hio_spin.setToolTip(
            "HIO (Hybrid Input-Output, Fienup 1982) iterations.\n"
            "Aggressive feedback that escapes local minima.\n"
            "Standard BCDI pipeline runs HIO before RAAR.\n"
            "50-200 typical (set to 0 to skip)."
        )
        pr.addWidget(self.hio_spin)
        pr.addWidget(QLabel("RAAR:"))
        self.raar_spin = QSpinBox()
        self.raar_spin.setRange(0, 1000)
        self.raar_spin.setValue(50)
        self.raar_spin.setToolTip(
            "RAAR (Relaxed Averaged Alternating Reflections, Luke 2005).\n"
            "Smoother convergence than HIO. Run after HIO. 50-100 typical."
        )
        pr.addWidget(self.raar_spin)
        pr.addWidget(QLabel("ER:"))
        self.er_spin = QSpinBox()
        self.er_spin.setRange(0, 500)
        self.er_spin.setValue(20)
        self.er_spin.setToolTip(
            "Error Reduction (ER) polishing iterations after RAAR.\n"
            "Tightens the final solution. 20-50 typical."
        )
        pr.addWidget(self.er_spin)
        pr.addWidget(QLabel("Ch:"))
        self.ch_spin = QSpinBox()
        self.ch_spin.setRange(8, 128)
        self.ch_spin.setValue(32)
        self.ch_spin.setToolTip(
            "Base channels of the trained model.\n"
            "MUST match the model checkpoint (32 = default 2.3M params)."
        )
        pr.addWidget(self.ch_spin)
        pr.addStretch()
        tg.addLayout(pr, 3, 0, 1, 3)

        root.addWidget(top)

        # Launch + progress
        lr = QHBoxLayout()
        self.launch_btn = QPushButton("Launch Reconstruction")
        self.launch_btn.setStyleSheet("background:#1f6feb;min-height:32px;font-size:10pt;font-weight:600")
        self.launch_btn.clicked.connect(self._launch)
        lr.addWidget(self.launch_btn)
        self.pg = QProgressBar()
        lr.addWidget(self.pg)
        root.addLayout(lr)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#8b949e;font-size:9pt")
        root.addWidget(self.status_lbl)

        # ── 4 figure sub-tabs (each gets full space) ─────────────────────
        self.fig_tabs = QTabWidget()
        self.fig_tabs.setStyleSheet(
            "QTabBar::tab{background:#161b22;color:#8b949e;padding:6px 14px;min-width:80px;font-size:9pt}"
            "QTabBar::tab:selected{background:#0d1117;color:#3fb950;border-bottom:2px solid #3fb950}"
        )

        fig_names = [
            ("Bragg Peak", "Accumulated 2D Bragg peak projections"),
            ("Density/Phase/Strain", "Electron density, phase and strain along 3 directions"),
            ("Convergence", "Phase retrieval R-factor convergence"),
            ("Ground Truth", "Reconstruction vs ground truth comparison"),
        ]

        self.figs = []
        self.canvases = []
        with matplotlib.rc_context(MPL_DARK):
            for tab_name, tooltip in fig_names:
                fig = Figure(figsize=(10, 6), dpi=110, tight_layout=True)
                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(350)

                page = QWidget()
                pl = QVBoxLayout(page)
                pl.setContentsMargins(0, 0, 0, 0)
                pl.setSpacing(2)

                # Toolbar + export button row
                bar = QHBoxLayout()
                tb = NavigationToolbar(canvas, page)
                tb.setStyleSheet("background:#161b22")
                bar.addWidget(tb)
                bar.addStretch()
                exp_btn = QPushButton("Export PNG")
                exp_btn.setStyleSheet("background:#4f98a3;padding:4px 10px;min-height:20px;font-size:8pt")
                exp_btn.clicked.connect(lambda _, f=fig, n=tab_name: self._export_fig(f, n))
                bar.addWidget(exp_btn)
                pl.addLayout(bar)
                pl.addWidget(canvas, 1)

                self.fig_tabs.addTab(page, tab_name)
                self.fig_tabs.setTabToolTip(self.fig_tabs.count() - 1, tooltip)
                self.figs.append(fig)
                self.canvases.append(canvas)

        root.addWidget(self.fig_tabs, 1)

    def _browse_input(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select diffraction data", "",
            "All supported (*.npz *.h5);;NumPy (*.npz);;HDF5 (*.h5)"
        )
        if f:
            self.input_path.setText(f)

    def _convert_p10_h5(self):
        """Convert PETRA III P10 .h5 (master or data chunk) + .fio → .npz."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Convert P10 .h5 + .fio to .npz")
        dlg.setMinimumWidth(560)
        layout = QVBoxLayout(dlg)

        intro = QLabel(
            "<b>PETRA III P10 (DESY) BCDI data conversion</b><br>"
            "<span style='color:#8b949e;font-size:9pt'>"
            "Pick either the <tt>_master.h5</tt> file or any "
            "<tt>_data_NNNNNN.h5</tt> chunk — the loader auto-finds the "
            "master. The companion <tt>.fio</tt> metadata file (motor "
            "positions, scan command) is detected automatically if it "
            "lives in the same directory.</span>"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # H5 path
        h5_row = QHBoxLayout()
        h5_le = QLineEdit()
        h5_le.setPlaceholderText("…/align_03_01698_master.h5  (or any data_*.h5)")
        h5_row.addWidget(h5_le, 1)
        h5_browse = QPushButton("…"); h5_browse.setMaximumWidth(30)
        def _pick_h5():
            f, _ = QFileDialog.getOpenFileName(
                self, "Pick P10 .h5 file (master or data chunk)",
                "", "HDF5 (*.h5 *.nxs)"
            )
            if f:
                h5_le.setText(f)
                # Try to auto-find the .fio sibling
                self._autodetect_p10_fio(f, fio_le)
        h5_browse.clicked.connect(_pick_h5)
        h5_row.addWidget(h5_browse)
        h5_w = QWidget(); h5_w.setLayout(h5_row)
        form.addRow("HDF5 file:", h5_w)

        # FIO path (optional)
        fio_row = QHBoxLayout()
        fio_le = QLineEdit()
        fio_le.setPlaceholderText("(optional) …/align_03_01698.fio")
        fio_le.setToolTip(
            "Optional .fio metadata file with motor positions and scan command.\n"
            "If left empty, CDI-ST looks for a sibling .fio file with the\n"
            "same name stem as the master."
        )
        fio_row.addWidget(fio_le, 1)
        fio_browse = QPushButton("…"); fio_browse.setMaximumWidth(30)
        def _pick_fio():
            f, _ = QFileDialog.getOpenFileName(
                self, "Pick .fio metadata file (optional)",
                "", "FIO (*.fio);;All (*)"
            )
            if f:
                fio_le.setText(f)
        fio_browse.clicked.connect(_pick_fio)
        fio_row.addWidget(fio_browse)
        fio_w = QWidget(); fio_w.setLayout(fio_row)
        form.addRow("Metadata (.fio):", fio_w)

        # Inspect button — show the HDF5 structure to help debug
        inspect_btn = QPushButton("Inspect HDF5 structure…")
        inspect_btn.setStyleSheet("background:#30363d;padding:5px 10px;min-height:20px;font-size:9pt")
        inspect_btn.setToolTip(
            "Print the dataset tree of the chosen file so you can see\n"
            "where the detector data lives. Useful when auto-detection fails."
        )
        def _do_inspect():
            p = h5_le.text().strip()
            if not p or not os.path.exists(p):
                QMessageBox.warning(self, "No file", "Pick the .h5 file first.")
                return
            try:
                import io as _io, contextlib as _cl
                from cdi_st.nn_experimental_loader import inspect_h5, _find_p10_master
                buf = _io.StringIO()
                master = _find_p10_master(p) or p
                with _cl.redirect_stdout(buf):
                    inspect_h5(master)
                text = buf.getvalue()
            except Exception as e:
                text = f"Error reading {p}:\n{e}"
            tree_dlg = QDialog(dlg)
            tree_dlg.setWindowTitle("HDF5 structure")
            tree_dlg.resize(720, 460)
            tv = QVBoxLayout(tree_dlg)
            txt = QPlainTextEdit(text); txt.setReadOnly(True)
            txt.setStyleSheet("font-family:Consolas, monospace; font-size:9pt;")
            tv.addWidget(txt)
            close = QPushButton("Close"); close.clicked.connect(tree_dlg.accept)
            tv.addWidget(close)
            tree_dlg.exec()
        inspect_btn.clicked.connect(_do_inspect)
        form.addRow("", inspect_btn)

        # Optional explicit dataset path
        ds_le = QLineEdit()
        ds_le.setPlaceholderText("(optional) e.g. /entry/data/data")
        ds_le.setToolTip(
            "Explicit HDF5 path to the detector data, if auto-detect fails.\n"
            "Use 'Inspect HDF5 structure' first to see what's available."
        )
        form.addRow("Dataset path:", ds_le)

        # ─── NEW in 0.2.1: scan probe + frame range + detector ROI ────────
        # For long Eiger scans (e.g. 381 chunks × 10 frames at 2167×2070), loading
        # everything would require ~50 GB of RAM. Let the user preview the scan,
        # then pick a frame subrange and/or detector ROI to keep memory sane.

        # "Probe scan" — read all chunk metadata to populate frame counts.
        probe_btn = QPushButton("Probe scan (read metadata only)…")
        probe_btn.setStyleSheet("background:#30363d;padding:5px 10px;min-height:20px;font-size:9pt")
        probe_btn.setToolTip(
            "Open the master file and enumerate its chunk files (no data\n"
            "is decompressed — this is essentially instant). Populates the\n"
            "Frame range and shows an estimated memory footprint."
        )

        # Scan info label
        info_lbl = QLabel(
            "<span style='color:#8b949e;font-size:9pt'>"
            "Click 'Probe scan' to see total frame count and memory needed."
            "</span>"
        )
        info_lbl.setWordWrap(True)

        # Frame range
        fr_row = QHBoxLayout()
        fr_start = QSpinBox(); fr_start.setRange(0, 999999); fr_start.setValue(0)
        fr_start.setToolTip("First frame index (inclusive) to load. 0 = beginning of scan.")
        fr_stop = QSpinBox(); fr_stop.setRange(0, 999999); fr_stop.setValue(0)
        fr_stop.setSpecialValueText("(end)")
        fr_stop.setToolTip(
            "Last frame index (exclusive) to load. 0 means 'all frames'.\n"
            "For BCDI rocking curves you typically want only ~100-300 frames\n"
            "around the maximum of the rocking curve."
        )
        fr_row.addWidget(QLabel("from")); fr_row.addWidget(fr_start)
        fr_row.addWidget(QLabel("to")); fr_row.addWidget(fr_stop)
        fr_row.addStretch(1)
        fr_w = QWidget(); fr_w.setLayout(fr_row)

        # Detector ROI (full / centered)
        roi_row = QHBoxLayout()
        roi_check = QCheckBox("Enable")
        roi_check.setToolTip(
            "Read only a region of interest from each detector frame.\n"
            "Reduces memory and disk by (full / ROI²)×. For BCDI you usually\n"
            "only need ~256×256 around the Bragg peak."
        )
        roi_size = QSpinBox(); roi_size.setRange(32, 4096)
        roi_size.setValue(256); roi_size.setSingleStep(32); roi_size.setSuffix(" px")
        roi_size.setToolTip("Square ROI side length, centered on detector center.")
        roi_size.setEnabled(False)
        roi_check.toggled.connect(roi_size.setEnabled)
        roi_row.addWidget(roi_check)
        roi_row.addWidget(QLabel("size:"))
        roi_row.addWidget(roi_size)
        roi_row.addStretch(1)
        roi_w = QWidget(); roi_w.setLayout(roi_row)

        # Memory budget
        mem_spin = QDoubleSpinBox()
        mem_spin.setRange(0.5, 256.0); mem_spin.setValue(8.0); mem_spin.setSingleStep(1.0)
        mem_spin.setSuffix(" GB")
        mem_spin.setToolTip(
            "Maximum RAM the loader is allowed to use. If the requested read\n"
            "would exceed this, the loader refuses and tells you to shrink the\n"
            "frame range or detector ROI — rather than crashing Python with\n"
            "an out-of-memory error."
        )

        # State variable that gets set by Probe scan
        scan_state = {'info': None}

        def _do_probe():
            p = h5_le.text().strip()
            if not p or not os.path.exists(p):
                QMessageBox.warning(self, "No file", "Pick the .h5 file first.")
                return
            try:
                from cdi_st.nn_experimental_loader import p10_scan_info
                info = p10_scan_info(p)
                scan_state['info'] = info
                # Wire up the spinboxes to the actual frame count
                fr_start.setRange(0, max(info['n_frames'] - 1, 0))
                fr_stop.setRange(0, info['n_frames'])
                fr_stop.setValue(info['n_frames'])   # default: read all
                roi_size.setMaximum(min(info['frame_h'], info['frame_w']))
                # Display
                miss = (f", <span style='color:#da3633'>{info['n_missing']} MISSING</span>"
                        if info['n_missing'] else "")
                info_lbl.setText(
                    f"<span style='color:#3fb950;font-size:9pt'>"
                    f"Scan: <b>{info['n_frames']}</b> frames × "
                    f"<b>{info['frame_h']}×{info['frame_w']}</b> "
                    f"(<tt>{info['dtype']}</tt>) "
                    f"across {info['n_chunks']} chunk(s){miss}"
                    f"<br>Full read needs <b>{info['size_full_gb']:.2f} GB</b> "
                    f"as float32. Set Frame range and/or Detector ROI "
                    f"to load a subset."
                    f"</span>"
                )
            except Exception as e:
                info_lbl.setText(
                    f"<span style='color:#da3633'>Probe failed: {e}</span>"
                )

        probe_btn.clicked.connect(_do_probe)
        form.addRow("", probe_btn)
        form.addRow("", info_lbl)
        form.addRow("Frame range:", fr_w)
        form.addRow("Detector ROI:", roi_w)
        form.addRow("Memory budget:", mem_spin)

        # Target size
        size_spin = QSpinBox()
        size_spin.setRange(32, 256); size_spin.setValue(64); size_spin.setSingleStep(16)
        size_spin.setToolTip("Cropped output grid: target_size³ centered on the Bragg peak.")
        form.addRow("Target size:", size_spin)

        # Output file
        out_row = QHBoxLayout()
        out_le = QLineEdit()
        out_le.setPlaceholderText("e.g. p10_recon_input.npz")
        out_row.addWidget(out_le, 1)
        out_browse = QPushButton("…"); out_browse.setMaximumWidth(30)
        def _pick_out():
            f, _ = QFileDialog.getSaveFileName(self, "Save .npz", "p10_scan.npz", "NumPy (*.npz)")
            if f: out_le.setText(f)
        out_browse.clicked.connect(_pick_out)
        out_row.addWidget(out_browse)
        out_w = QWidget(); out_w.setLayout(out_row)
        form.addRow("Output file (.npz):", out_w)

        layout.addLayout(form)

        status = QLabel("")
        status.setWordWrap(True)
        layout.addWidget(status)

        btn_row = QHBoxLayout(); btn_row.addStretch()
        cancel_btn = QPushButton("Cancel"); cancel_btn.clicked.connect(dlg.reject); btn_row.addWidget(cancel_btn)
        convert_btn = QPushButton("Convert")
        convert_btn.setStyleSheet("background:#1f6feb;padding:6px 16px;font-weight:600")
        btn_row.addWidget(convert_btn)
        layout.addLayout(btn_row)

        def _do_convert():
            h5_path = h5_le.text().strip()
            out_path = out_le.text().strip()
            if not h5_path or not out_path:
                status.setText("<span style='color:#da3633'>HDF5 file and output file are required.</span>")
                return
            if not os.path.exists(h5_path):
                status.setText(f"<span style='color:#da3633'>HDF5 file not found: {h5_path}</span>")
                return
            convert_btn.setEnabled(False)
            status.setText("Converting…")
            QApplication.processEvents()
            try:
                from cdi_st.nn_experimental_loader import p10_h5_to_npz
                ds_path = ds_le.text().strip() or None
                fio_path = fio_le.text().strip() or None
                # Compute frame_range from spinboxes (0,0) means "all"
                fr = None
                if not (fr_start.value() == 0 and fr_stop.value() == 0):
                    fr = (fr_start.value(),
                          fr_stop.value() if fr_stop.value() > 0 else None)
                # Compute detector_roi if enabled
                roi = None
                if roi_check.isChecked() and scan_state['info'] is not None:
                    n = roi_size.value()
                    h, w = scan_state['info']['frame_h'], scan_state['info']['frame_w']
                    cy, cx = h // 2, w // 2
                    half = n // 2
                    roi = (max(0, cy - half), min(h, cy + half),
                           max(0, cx - half), min(w, cx + half))
                result = p10_h5_to_npz(
                    h5_path=h5_path,
                    npz_path=out_path,
                    fio_path=fio_path,
                    target_size=size_spin.value(),
                    dataset_path=ds_path,
                    frame_range=fr,
                    detector_roi=roi,
                    memory_budget_gb=mem_spin.value(),
                    verbose=True,
                )
                shape = result['diffraction'].shape
                peak = result['diffraction'].max()
                fio_msg = ""
                if result.get('fio') and result['fio'].get('scan_command'):
                    fio_msg = f"<br>Scan: {result['fio']['scan_command'][:80]}"
                status.setText(
                    f"<span style='color:#3fb950'>\u2713 Saved {out_path}</span>"
                    f"<br><span style='color:#8b949e;font-size:9pt'>"
                    f"Volume: {shape[0]}\u00b3, peak max={peak:.1e}{fio_msg}</span>"
                )
                self.input_path.setText(out_path)
                cancel_btn.setText("Close")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(tb)
                status.setText(
                    f"<span style='color:#da3633'>\u2717 Conversion failed:</span><br>"
                    f"<span style='font-size:9pt'>{str(e)[:300]}</span>"
                )
            finally:
                convert_btn.setEnabled(True)

        convert_btn.clicked.connect(_do_convert)
        dlg.exec()

    def _autodetect_p10_fio(self, h5_path: str, fio_line_edit):
        """Try to find a sibling .fio file when an .h5 is picked."""
        import os, re
        if not h5_path:
            return
        base = os.path.basename(h5_path)
        dirname = os.path.dirname(os.path.abspath(h5_path))
        stem = re.sub(r'_(master|data_\d+)\.h5$', '', base, flags=re.IGNORECASE)
        stem = re.sub(r'\.h5$', '', stem)
        for cand in [
            os.path.join(dirname, stem + ".fio"),
            os.path.join(dirname, base.replace(".h5", ".fio")),
            os.path.join(dirname, "..", stem + ".fio"),
        ]:
            if os.path.exists(cand):
                fio_line_edit.setText(os.path.abspath(cand))
                return

    def _convert_spec_edf(self):
        """Open dialog to convert ID01 SPEC+EDF data into a reconstruction-ready .npz."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Convert SPEC + EDF to .npz")
        dlg.setMinimumWidth(520)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)

        info = QLabel(
            "<b>ID01 SPEC + EDF \u2192 .npz converter</b><br>"
            "<span style='color:#8b949e;font-size:9pt'>"
            "Reads motor positions from a SPEC file and detector frames "
            "from .edf.gz files. Performs gap masking, hot-pixel removal, "
            "Bragg peak finding, and (if xrayutilities is installed) "
            "q-space orthogonalization for proper voxel-pitch calibration."
            "</span>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # SPEC file
        spec_row = QHBoxLayout()
        spec_le = QLineEdit()
        spec_le.setPlaceholderText("/path/to/sample.spec")
        spec_row.addWidget(spec_le, 1)
        spec_browse = QPushButton("…")
        spec_browse.setMaximumWidth(30)
        def _pick_spec():
            f, _ = QFileDialog.getOpenFileName(self, "Select SPEC file", "", "SPEC (*.spec);;All files (*)")
            if f:
                spec_le.setText(f)
        spec_browse.clicked.connect(_pick_spec)
        spec_row.addWidget(spec_browse)
        spec_widget = QWidget()
        spec_widget.setLayout(spec_row)
        form.addRow("SPEC file:", spec_widget)

        # Scan number + Browse scans helper
        scan_row = QHBoxLayout()
        scan_spin = QSpinBox()
        scan_spin.setRange(1, 99999)
        scan_spin.setValue(1)
        scan_row.addWidget(scan_spin, 1)
        scan_browse = QPushButton("Browse scans…")
        scan_browse.setMaximumWidth(115)
        scan_browse.setToolTip(
            "List all scans in the SPEC file with their CCD frame ranges.\n"
            "You can select ONE or MULTIPLE scans:\n"
            "  • Single scan → output is one .npz file\n"
            "  • Multiple scans (Ctrl/Shift click) → output becomes a directory,\n"
            "    each scan saved as scan_<N>_recon_input.npz"
        )
        scan_row.addWidget(scan_browse)
        scan_widget = QWidget()
        scan_widget.setLayout(scan_row)
        form.addRow("Scan number:", scan_widget)

        scan_info = QLabel("")
        scan_info.setStyleSheet("color:#8b949e;font-size:9pt;background:transparent")
        scan_info.setWordWrap(True)
        form.addRow("", scan_info)

        # State: list of currently-selected scans for batch conversion
        # When length is 1, single-scan mode (uses scan_spin + out_le).
        # When length > 1, batch mode (uses scan_list + out_le as directory).
        scan_state = {"selected": []}

        def _list_scans():
            sp = spec_le.text().strip()
            if not sp or not os.path.exists(sp):
                QMessageBox.warning(
                    self, "Pick a SPEC file first",
                    "Please select the SPEC file before browsing scans."
                )
                return
            try:
                from cdi_st.nn_experimental_loader import list_spec_scans
                scans = list_spec_scans(sp)
            except Exception as e:
                QMessageBox.warning(self, "Cannot read SPEC file", str(e))
                return
            if not scans:
                QMessageBox.information(
                    self, "No scans found",
                    "The SPEC file appears to contain no scans."
                )
                return
            chooser = QDialog(dlg)
            chooser.setWindowTitle("SPEC scans (select one or many)")
            chooser.resize(820, 480)
            cl = QVBoxLayout(chooser)
            hint = QLabel(
                "Select ONE scan to convert just that scan, OR Ctrl+click / "
                "Shift+click to select MULTIPLE scans for batch conversion. "
                "Pick scans whose CCD frame range covers the EDF files you have."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#8b949e;font-size:9pt")
            cl.addWidget(hint)
            tbl = QTableWidget(len(scans), 4)
            tbl.setHorizontalHeaderLabels(["Scan #", "Frames", "Points", "Command"])
            tbl.horizontalHeader().setStretchLastSection(True)
            # MULTI-select enabled (Ctrl/Shift to extend)
            tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            tbl.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
            tbl.verticalHeader().setVisible(False)
            for r, s in enumerate(scans):
                tbl.setItem(r, 0, QTableWidgetItem(str(s["scan_number"])))
                fr = (f"{s['frame_min']}–{s['frame_max']}"
                      if s["frame_min"] is not None else "—")
                tbl.setItem(r, 1, QTableWidgetItem(fr))
                tbl.setItem(r, 2, QTableWidgetItem(str(s["n_points"])))
                tbl.setItem(r, 3, QTableWidgetItem(s["command"]))
            tbl.resizeColumnsToContents()
            cl.addWidget(tbl, 1)

            sel_label = QLabel("0 scan(s) selected")
            sel_label.setStyleSheet("color:#8b949e;font-size:9pt")
            cl.addWidget(sel_label)
            def _on_sel_changed():
                n = len(tbl.selectionModel().selectedRows())
                sel_label.setText(f"{n} scan(s) selected")
            tbl.selectionModel().selectionChanged.connect(lambda *_: _on_sel_changed())

            br = QHBoxLayout()
            br.addStretch()
            cancel = QPushButton("Cancel")
            cancel.clicked.connect(chooser.reject)
            br.addWidget(cancel)
            use = QPushButton("Use selected scan(s)")
            use.setStyleSheet("background:#1f6feb;padding:6px 14px;font-weight:600")
            def _accept():
                rows = tbl.selectionModel().selectedRows()
                if not rows:
                    return
                indices = sorted(r.row() for r in rows)
                selected_scans = [scans[i] for i in indices]
                scan_state["selected"] = selected_scans
                # Update UI
                scan_spin.setValue(int(selected_scans[0]["scan_number"]))
                if len(selected_scans) == 1:
                    s = selected_scans[0]
                    fr_txt = (f"frames {s['frame_min']}–{s['frame_max']}"
                              if s["frame_min"] is not None else "frames unknown")
                    scan_info.setText(
                        f"<b>1 scan:</b> #{s['scan_number']} ({fr_txt}, "
                        f"{s['n_points']} pts).  {s['command'][:60]}"
                    )
                else:
                    nums = ", ".join(str(s["scan_number"]) for s in selected_scans)
                    scan_info.setText(
                        f"<b>{len(selected_scans)} scans</b> selected: {nums}.<br>"
                        f"<span style='color:#bf8700'>"
                        f"Output is now a DIRECTORY. Each scan saves as "
                        f"<tt>scan_&lt;N&gt;_recon_input.npz</tt> inside it.</span>"
                    )
                chooser.accept()
            use.clicked.connect(_accept)
            tbl.doubleClicked.connect(lambda *_: _accept())
            br.addWidget(use)
            cl.addLayout(br)
            chooser.exec()
        scan_browse.clicked.connect(_list_scans)

        # EDF directory
        edf_row = QHBoxLayout()
        edf_le = QLineEdit()
        edf_le.setPlaceholderText("/path/to/mpx/")
        edf_row.addWidget(edf_le, 1)
        edf_browse = QPushButton("…")
        edf_browse.setMaximumWidth(30)
        def _pick_edf():
            d = QFileDialog.getExistingDirectory(self, "Select EDF directory")
            if d:
                edf_le.setText(d)
        edf_browse.clicked.connect(_pick_edf)
        edf_row.addWidget(edf_browse)
        edf_widget = QWidget()
        edf_widget.setLayout(edf_row)
        form.addRow("EDF directory:", edf_widget)

        # Filename template
        tpl_le = QLineEdit("data_mpx4_%05d.edf.gz")
        tpl_le.setToolTip(
            "Filename pattern with %d for the frame number.\n"
            "ID01 maxipix default: data_mpx4_%05d.edf.gz"
        )
        form.addRow("Filename template:", tpl_le)

        # Detector
        det_combo = QComboBox()
        det_combo.addItems(["maxipix (516×516)", "eiger2M (1062×1028)", "auto"])
        form.addRow("Detector:", det_combo)

        # Target size
        size_spin = QSpinBox()
        size_spin.setRange(16, 256)
        size_spin.setValue(64)
        size_spin.setSingleStep(16)
        form.addRow("Output size:", size_spin)

        # Output .npz path
        out_row = QHBoxLayout()
        out_le = QLineEdit()
        out_le.setPlaceholderText("e.g. scan46_recon_input.npz")
        out_le.setToolTip(
            "Filename for the converted output. The whole 3D diffraction\n"
            "volume + motor angles + frame numbers are stored in ONE .npz file.\n"
            "Pick any name and location; the file will be created (or overwritten)."
        )
        out_row.addWidget(out_le, 1)
        out_browse = QPushButton("…")
        out_browse.setMaximumWidth(30)
        def _pick_out():
            # If multiple scans are selected, pick a directory; otherwise pick a file
            sel = scan_state.get("selected", [])
            if len(sel) > 1:
                d = QFileDialog.getExistingDirectory(
                    self, "Pick output directory for batch conversion"
                )
                if d:
                    out_le.setText(d)
            else:
                f, _ = QFileDialog.getSaveFileName(
                    self, "Save converted .npz", "scan.npz", "NumPy (*.npz)"
                )
                if f:
                    out_le.setText(f)
        out_browse.clicked.connect(_pick_out)
        out_row.addWidget(out_browse)
        out_widget = QWidget()
        out_widget.setLayout(out_row)
        form.addRow("Output file (.npz):", out_widget)

        layout.addLayout(form)

        # Status label
        status = QLabel("")
        status.setWordWrap(True)
        status.setStyleSheet("color:#8b949e;font-size:9pt")
        layout.addWidget(status)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)
        convert_btn = QPushButton("Convert")
        convert_btn.setStyleSheet("background:#bf8700;padding:6px 14px;font-weight:600")
        btn_row.addWidget(convert_btn)
        layout.addLayout(btn_row)

        def _do_convert():
            spec_path = spec_le.text().strip()
            edf_dir = edf_le.text().strip()
            out_path = out_le.text().strip()
            if not all([spec_path, edf_dir, out_path]):
                status.setText("<span style='color:#da3633'>All paths are required.</span>")
                return
            if not os.path.exists(spec_path):
                status.setText(f"<span style='color:#da3633'>SPEC file not found: {spec_path}</span>")
                return
            if not os.path.isdir(edf_dir):
                status.setText(f"<span style='color:#da3633'>EDF directory not found: {edf_dir}</span>")
                return

            det_str = det_combo.currentText().split()[0]
            det_shape = (516, 516) if det_str == 'maxipix' else (1062, 1028)

            # Determine batch mode from the scan_state set by "Browse scans"
            selected = scan_state.get("selected", [])
            if len(selected) <= 1:
                # Single-scan mode (either from spinner or single-row selection)
                scan_numbers = [scan_spin.value()]
                # out_path is interpreted as a file
                if os.path.isdir(out_path):
                    # User pointed at a folder — auto-name inside it
                    out_targets = [os.path.join(
                        out_path, f"scan_{scan_numbers[0]}_recon_input.npz"
                    )]
                else:
                    out_targets = [out_path]
            else:
                # Batch mode — out_path is a directory
                scan_numbers = [int(s["scan_number"]) for s in selected]
                if not os.path.isdir(out_path):
                    # Auto-create if it doesn't exist
                    try:
                        os.makedirs(out_path, exist_ok=True)
                    except Exception as e:
                        status.setText(
                            f"<span style='color:#da3633'>Output directory "
                            f"could not be created: {e}</span>"
                        )
                        return
                out_targets = [
                    os.path.join(out_path, f"scan_{n}_recon_input.npz")
                    for n in scan_numbers
                ]

            convert_btn.setEnabled(False)
            from cdi_st.nn_experimental_loader import spec_edf_to_npz
            n_total = len(scan_numbers)
            n_ok = 0
            failures = []
            last_result_path = None

            for i, (sn, tgt) in enumerate(zip(scan_numbers, out_targets), start=1):
                status.setText(
                    f"Converting scan {sn}… ({i}/{n_total})"
                )
                QApplication.processEvents()
                try:
                    result = spec_edf_to_npz(
                        spec_path=spec_path,
                        scan_number=sn,
                        edf_dir=edf_dir,
                        npz_path=tgt,
                        target_size=size_spin.value(),
                        edf_template_name=tpl_le.text().strip(),
                    )
                    n_ok += 1
                    last_result_path = tgt
                except Exception as e:
                    import traceback
                    print(f"[scan {sn}] failed: {e}")
                    print(traceback.format_exc())
                    failures.append((sn, str(e)))

            # Final status summary
            if n_ok == n_total:
                # All succeeded
                if n_total == 1:
                    status.setText(
                        f"<span style='color:#3fb950'>\u2713 Saved {out_targets[0]}</span>"
                    )
                    self.input_path.setText(out_targets[0])
                else:
                    status.setText(
                        f"<span style='color:#3fb950'>\u2713 Converted {n_ok}/"
                        f"{n_total} scans into {out_path}</span><br>"
                        f"<span style='color:#8b949e;font-size:9pt'>"
                        f"Pick one .npz with the input file selector to load it."
                        f"</span>"
                    )
                    # Pre-fill input_path with the first one for convenience
                    self.input_path.setText(out_targets[0])
            elif n_ok > 0:
                fail_txt = ", ".join(f"#{sn}" for sn, _ in failures[:3])
                status.setText(
                    f"<span style='color:#bf8700'>⚠ {n_ok}/{n_total} scans "
                    f"succeeded. Failed: {fail_txt}{'…' if len(failures) > 3 else ''}</span>"
                )
                if last_result_path:
                    self.input_path.setText(last_result_path)
            else:
                first_err = failures[0][1] if failures else "unknown error"
                status.setText(
                    f"<span style='color:#da3633'>\u2717 All conversions failed.</span><br>"
                    f"<span style='font-size:9pt'>First error: {first_err[:200]}</span>"
                )

            cancel_btn.setText("Close")
            convert_btn.setEnabled(True)

        convert_btn.clicked.connect(_do_convert)
        dlg.exec()

    def _browse_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select AutoPhaseNet model", "", "PyTorch (*.pt)")
        if f:
            self.model_path.setText(f)

    def _browse_model2(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select supervised model", "", "PyTorch (*.pt)")
        if f:
            self.model_path2.setText(f)

    def _launch(self):
        inp = self.input_path.text().strip()
        model = self.model_path.text().strip()
        model2 = self.model_path2.text().strip()
        mode = self.mode_combo.currentText()

        if not inp or not os.path.exists(inp):
            QMessageBox.warning(self, "Error", "Select a valid input file")
            return
        if not model or not os.path.exists(model):
            QMessageBox.warning(self, "Error", "Select a valid AutoPhaseNet checkpoint")
            return
        if mode in ('ensemble', 'ensemble+refine'):
            if not model2 or not os.path.exists(model2):
                QMessageBox.warning(
                    self, "Error",
                    "Ensemble mode requires Model 2 (supervised PhaseUNet3D checkpoint)."
                )
                return

        params = {
            'input_path': inp,
            'model_path': model,
            'model_path2': model2 if model2 else None,
            'mode': self.mode_combo.currentText(),
            'n_raar': self.raar_spin.value(),
            'n_er': self.er_spin.value(),
            'n_hio': self.hio_spin.value(),
            'base_channels': self.ch_spin.value(),
            'support_threshold': 0.05,
        }

        self.launch_btn.setEnabled(False)
        self.pg.setValue(0)
        self.status_lbl.setText("Running...")

        self._worker = ReconstructionWorker(params)
        self._worker.log.connect(lambda s: self.status_lbl.setText(s))
        self._worker.progress.connect(self.pg.setValue)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result):
        self._result = result
        self.launch_btn.setEnabled(True)
        self.status_lbl.setText(
            f"Done in {result['elapsed']:.2f}s  R={result['error_metric'][-1]:.4f}  "
            f"method={result['method']}"
        )
        self._draw_all(result)
        self.recon_done.emit(result)

    def _on_failed(self, msg):
        self.launch_btn.setEnabled(True)
        self.status_lbl.setText(f"FAILED: {msg[:120]}")

    # ── Drawing functions ─────────────────────────────────────────────────

    def _remove_phase_offset(self, phase, support):
        mask = support > 0.5
        if mask.sum() == 0:
            return phase
        mean = np.angle(np.mean(np.exp(1j * phase[mask])))
        return np.angle(np.exp(1j * (phase - mean)))

    def _export_fig(self, fig, name):
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {name}", f"bcdi_{name.lower().replace(' ','_')}.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if path:
            fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
            self.status_lbl.setText(f"Exported: {path}")

    def _draw_all(self, r):
        self._draw_bragg(r)
        self._draw_density_phase(r)
        self._draw_convergence(r)
        self._draw_comparison(r)

    def _draw_bragg(self, r):
        """Fig 0: Accumulated 2D Bragg peak from the diffraction data."""
        diff = r['diffraction']
        with matplotlib.rc_context(MPL_DARK):
            fig = self.figs[0]
            fig.clear()
            axes = fig.subplots(1, 3)
            fig.suptitle('Accumulated 2D Bragg peak projections', fontsize=11, color='#4f98a3')
            proj_xy = np.log10(diff.sum(axis=2).T + 1)
            proj_xz = np.log10(diff.sum(axis=1).T + 1)
            proj_yz = np.log10(diff.sum(axis=0).T + 1)
            for ax, data, title in zip(axes, [proj_xy, proj_xz, proj_yz],
                                        ['Σ along z (qx-qy)', 'Σ along y (qx-qz)', 'Σ along x (qy-qz)']):
                ax.set_facecolor('#000005')
                im = ax.imshow(data, cmap='jet', origin='lower', aspect='equal')
                ax.set_title(title, fontsize=9)
                ax.tick_params(labelsize=7)
                fig.colorbar(im, ax=ax, shrink=0.7)
        self.canvases[0].draw()

    def _draw_density_phase(self, r):
        """Fig 1: Electron density, phase, strain along 3 orthogonal slices."""
        amp = r['amplitude']
        support = r['support']
        phase = self._remove_phase_offset(r['phase'], support)
        N = amp.shape[0]

        # Slice through the center-of-mass of the support, NOT the geometric
        # center of the volume. Phase retrieval has translation freedom, so
        # the object can sit anywhere in the volume. Slicing at N//2 may miss
        # it entirely. Slicing at the COM always shows the object's middle.
        sup_mask = (support > 0.5)
        if sup_mask.sum() > 5:
            from scipy.ndimage import center_of_mass
            com = center_of_mass(amp * sup_mask)
            cx, cy, cz = (int(round(com[0])), int(round(com[1])), int(round(com[2])))
            cx = max(0, min(N - 1, cx))
            cy = max(0, min(N - 1, cy))
            cz = max(0, min(N - 1, cz))
        else:
            cx = cy = cz = N // 2

        with matplotlib.rc_context(MPL_DARK):
            fig = self.figs[1]
            fig.clear()
            fig.suptitle(
                f'Electron density | Phase | Strain  (slices at COM = ({cx}, {cy}, {cz}))',
                fontsize=11, color='#4f98a3'
            )
            # GridSpec with FIXED column widths: 3 equal image columns + 1 thin
            # colorbar column. This is the key to the layout being stable: the
            # colorbar lives in its own column, so it never pushes the image
            # axes around or causes the layout to reflow when matplotlib
            # adjusts the figure during a zoom or pan operation.
            gs = fig.add_gridspec(
                3, 4,
                width_ratios=[1.0, 1.0, 1.0, 0.05],
                wspace=0.30,
                hspace=0.40,
                left=0.06, right=0.93, top=0.92, bottom=0.06,
            )
            axes = np.empty((3, 3), dtype=object)
            cax = [None, None, None]   # colorbar axes per row
            for r in range(3):
                for c in range(3):
                    axes[r, c] = fig.add_subplot(gs[r, c])
                cax[r] = fig.add_subplot(gs[r, 3])

            slicers = [
                ('XY', lambda a: a[:, :, cz]),
                ('XZ', lambda a: a[:, cy, :]),
                ('YZ', lambda a: a[cx, :, :]),
            ]

            # Row 0: amplitude (auto-scale to data range)
            amp_max = amp.max()
            for col, (name, getter) in enumerate(slicers):
                ax = axes[0, col]
                ax.set_facecolor('#000005')
                sl = getter(amp)
                im = ax.imshow(sl, cmap='hot', origin='lower', aspect='equal',
                               vmin=0, vmax=amp_max if amp_max > 0 else 1)
                ax.set_title(f'|ρ| {name}', fontsize=9, color='#e6edf3')
                ax.tick_params(labelsize=6)
                if col == 2:
                    fig.colorbar(im, cax=cax[0])
                    cax[0].tick_params(labelsize=7, colors='#8b949e')

            # Row 1: phase (masked to support, full [-π, π] range)
            for col, (name, getter) in enumerate(slicers):
                ax = axes[1, col]
                ax.set_facecolor('#000005')
                sup_sl = getter(support) > 0.5
                sl = getter(phase) * sup_sl
                im = ax.imshow(sl, cmap='twilight', vmin=-np.pi, vmax=np.pi,
                               origin='lower', aspect='equal')
                ax.set_title(f'φ {name}', fontsize=9, color='#e6edf3')
                ax.tick_params(labelsize=6)
                if col == 2:
                    cb = fig.colorbar(im, cax=cax[1])
                    cb.set_ticks([-np.pi, 0, np.pi])
                    cb.set_ticklabels(['-π', '0', 'π'])
                    cax[1].tick_params(labelsize=7, colors='#8b949e')

            # Row 2: strain magnitude (gradient of phase inside support)
            for col, (name, getter) in enumerate(slicers):
                ax = axes[2, col]
                ax.set_facecolor('#000005')
                sup_sl = getter(support) > 0.5
                ph_slice = getter(phase) * sup_sl
                grad_x, grad_y = np.gradient(ph_slice)
                strain_mag = np.sqrt(grad_x**2 + grad_y**2) * sup_sl
                vmax_s = np.percentile(strain_mag[sup_sl], 95) if sup_sl.sum() > 10 else 1
                im = ax.imshow(strain_mag, cmap='inferno', origin='lower', aspect='equal',
                               vmin=0, vmax=max(vmax_s, 1e-6))
                ax.set_title(f'|∇φ| {name}', fontsize=9, color='#e6edf3')
                ax.tick_params(labelsize=6)
                if col == 2:
                    fig.colorbar(im, cax=cax[2])
                    cax[2].tick_params(labelsize=7, colors='#8b949e')

        self.canvases[1].draw()

    def _draw_convergence(self, r):
        """Fig 2: R-factor convergence."""
        errors = np.asarray(r['error_metric'])
        method = r.get('method', '')
        with matplotlib.rc_context(MPL_DARK):
            fig = self.figs[2]
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor('#000005')

            if len(errors) > 1:
                ax.semilogy(errors, color='#3fb950', linewidth=2, label='R-factor')
                ax.axhline(errors[-1], color='#f0883e', linestyle='--', alpha=0.7,
                           label=f'Final R={errors[-1]:.4f}')
                ax.set_xlabel('Iteration', fontsize=10)
                ax.set_ylabel('R-factor (log scale)', fontsize=10)
            else:
                # Single point — display as a clear annotated value, not a bar
                ax.text(0.5, 0.55,
                        f"R = {errors[0]:.4f}",
                        ha='center', va='center', fontsize=32, color='#3fb950',
                        fontweight='bold', transform=ax.transAxes)
                ax.text(0.5, 0.30,
                        f"NN-only mode (no iterative refinement)",
                        ha='center', va='center', fontsize=11, color='#8b949e',
                        transform=ax.transAxes)
                ax.text(0.5, 0.20,
                        f"Method: {method}",
                        ha='center', va='center', fontsize=9, color='#8b949e',
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_title('Phase retrieval convergence', fontsize=11, color='#4f98a3')
                self.canvases[2].draw()
                return

            ax.set_title('Phase retrieval convergence', fontsize=11, color='#4f98a3')
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=9)
        self.canvases[2].draw()

    def _draw_comparison(self, r):
        """Fig 3: Reconstruction vs ground truth (if available)."""
        with matplotlib.rc_context(MPL_DARK):
            fig = self.figs[3]
            fig.clear()

            if 'phase_true' not in r:
                ax = fig.add_subplot(111)
                ax.set_facecolor('#000005')
                ax.text(0.5, 0.5,
                        'No ground truth available\n\n'
                        'This is experimental data — ground truth\n'
                        'phase is unknown. Check diffraction match\n'
                        'in the Bragg Peak tab instead.',
                        ha='center', va='center', fontsize=12, color='#8b949e',
                        transform=ax.transAxes)
                ax.set_title('Ground truth comparison', fontsize=11, color='#4f98a3')
                self.canvases[3].draw()
                return

            support = r['support']
            phase_recon = self._remove_phase_offset(r['phase'], support)
            sup_true = r.get('support_true', support)
            phase_true = self._remove_phase_offset(r['phase_true'], sup_true)

            N = phase_recon.shape[0]
            # Slice through the COM of the reconstruction (not the geometric
            # center). Same reasoning as in _draw_density_phase.
            sup_mask_full = (support > 0.5)
            if sup_mask_full.sum() > 5:
                from scipy.ndimage import center_of_mass
                com = center_of_mass(r['amplitude'] * sup_mask_full)
                c = max(0, min(N - 1, int(round(com[2]))))
            else:
                c = N // 2
            kw = dict(cmap='twilight', vmin=-np.pi, vmax=np.pi,
                      origin='lower', aspect='equal')

            axes = fig.subplots(2, 3)
            fig.suptitle('Reconstruction vs Ground Truth', fontsize=11, color='#4f98a3')

            # Row 0: phase comparison
            pt = phase_true[:, :, c] * (sup_true[:, :, c] > 0.5)
            pr = phase_recon[:, :, c] * (support[:, :, c] > 0.5)
            common = (support[:, :, c] > 0.5) & (sup_true[:, :, c] > 0.5)
            pe = np.angle(np.exp(1j * (phase_recon[:, :, c] - phase_true[:, :, c]))) * common

            for ax, data, title in zip(axes[0], [pt, pr, pe],
                                        ['Ground truth φ', 'Reconstructed φ', 'Phase error']):
                ax.set_facecolor('#000005')
                im = ax.imshow(data, **kw)
                ax.set_title(title, fontsize=9, color='#e6edf3')
                ax.tick_params(labelsize=6)
            cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.7)
            cb.set_ticks([-np.pi, 0, np.pi])
            cb.set_ticklabels(['-π', '0', 'π'])

            # Row 1: amplitude comparison
            amp_true_sl = sup_true[:, :, c].astype(np.float32)
            amp_recon_sl = r['amplitude'][:, :, c]
            amp_diff = np.abs(amp_recon_sl - amp_true_sl)
            vmax_a = max(amp_true_sl.max(), amp_recon_sl.max(), 1e-6)
            kw_a = dict(cmap='hot', vmin=0, vmax=vmax_a, origin='lower', aspect='equal')

            for ax, data, title in zip(axes[1], [amp_true_sl, amp_recon_sl, amp_diff],
                                        ['Support (truth)', 'Reconstructed |ρ|', '|ρ| difference']):
                ax.set_facecolor('#000005')
                ax.imshow(data, **kw_a)
                ax.set_title(title, fontsize=9, color='#e6edf3')
                ax.tick_params(labelsize=6)

            # Metrics
            mask3d = (support > 0.5) & (sup_true > 0.5)
            if mask3d.sum() > 10:
                perr = np.angle(np.exp(1j * (phase_recon - phase_true)))
                rmse = np.sqrt(np.mean(perr[mask3d]**2))
                pt_vals = phase_true[mask3d]
                if pt_vals.std() > 1e-6:
                    corr = np.corrcoef(phase_recon[mask3d], pt_vals)[0, 1]
                    fig.text(0.5, 0.01,
                             f'Phase RMSE = {rmse:.3f} rad    Correlation = {corr:.3f}',
                             ha='center', fontsize=10, color='#3fb950')
                else:
                    fig.text(0.5, 0.01,
                             f'Phase RMSE = {rmse:.3f} rad    (constant GT phase — no strain)',
                             ha='center', fontsize=10, color='#f0883e')

        self.canvases[3].draw()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — 3D Reconstruction Viewer (matplotlib-based, no CDN required)
# ═══════════════════════════════════════════════════════════════════════════════

class T6(QWidget):
    """3D interactive viewer using matplotlib (no Three.js dependency)."""

    def __init__(self):
        super().__init__()
        self._result = None
        self._ui()

    def _ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left: scrollable controls panel ───────────────────────────────
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFixedWidth(360)
        sc.setStyleSheet("QScrollArea{border:none}")
        ctrl = QWidget()
        cv = QVBoxLayout(ctrl)
        cv.setSpacing(8)
        cv.setContentsMargins(4, 4, 12, 4)

        title = QLabel("3D Viewer")
        title.setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700")
        cv.addWidget(title)

        self.info_lbl = QLabel("No reconstruction loaded.")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet("color:#8b949e;font-size:9pt")
        cv.addWidget(self.info_lbl)

        # ── Big launch button ─────────────────────────────────────────────
        self.view_btn = QPushButton("▶  View Reconstruction")
        self.view_btn.setStyleSheet(
            "background:#1f6feb;min-height:38px;font-size:11pt;font-weight:700;padding:8px 14px"
        )
        self.view_btn.setEnabled(False)
        self.view_btn.setToolTip("Render the 3D reconstruction with current settings")
        self.view_btn.clicked.connect(self._render_now)
        cv.addWidget(self.view_btn)

        self.update_btn = QPushButton("⟳ Update view")
        self.update_btn.setStyleSheet("background:#4f98a3;min-height:26px;font-size:9pt")
        self.update_btn.setEnabled(False)
        self.update_btn.setToolTip("Re-render after changing display settings")
        self.update_btn.clicked.connect(self._render_now)
        cv.addWidget(self.update_btn)

        # Display mode
        mg = QGroupBox("Display mode (what to color)")
        mv = QVBoxLayout(mg)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Electron density |ρ|", "Phase φ", "Strain |∇φ|"])
        self.mode_combo.setToolTip(
            "What scalar field to display:\n"
            "• Electron density: |ρ(r)| (the reconstructed object shape)\n"
            "• Phase: φ(r) (related to lattice displacement)\n"
            "• Strain: |∇φ| (gradient of phase, highlights defects)"
        )
        mv.addWidget(self.mode_combo)
        cv.addWidget(mg)

        # Render style
        rg = QGroupBox("Render style")
        rgv = QVBoxLayout(rg)
        self.render_combo = QComboBox()
        self.render_combo.addItems([
            "Point cloud (fast, voxels)",
            "Isosurface (smooth, ParaView-like)",
            "Surface only (no interior)",
        ])
        self.render_combo.setToolTip(
            "How to render the 3D object:\n"
            "• Point cloud: every voxel as a colored sphere (fast, but messy)\n"
            "• Isosurface: marching-cubes mesh of |ρ| boundary, painted with the\n"
            "  selected scalar field — clean surface like ParaView\n"
            "• Surface only: same but draws the boundary voxels only (faster)"
        )
        rgv.addWidget(self.render_combo)
        cv.addWidget(rg)

        # Export to ParaView (.vti)
        vg = QGroupBox("Export to ParaView")
        vv = QVBoxLayout(vg)
        self.vti_btn = QPushButton("Export VTI / VTK file…")
        self.vti_btn.setStyleSheet("background:#bf8700;padding:6px 10px;min-height:26px;font-size:9pt")
        self.vti_btn.setEnabled(False)
        self.vti_btn.setToolTip(
            "Save reconstruction as a .vti (VTK image) file.\n"
            "Open it in ParaView for industry-standard volume rendering,\n"
            "isosurface extraction, dislocation tracing, and publication\n"
            "quality figures. ParaView is FREE: paraview.org"
        )
        self.vti_btn.clicked.connect(self._export_vti)
        vv.addWidget(self.vti_btn)
        info = QLabel("VTI files contain |ρ|, φ, and strain as scalar fields\nwith proper voxel-pitch metadata in nm.")
        info.setStyleSheet("color:#6e7681;font-size:8pt;font-style:italic")
        info.setWordWrap(True)
        vv.addWidget(info)
        cv.addWidget(vg)

        # Isosurface threshold
        tg = QGroupBox("Isosurface threshold")
        tv = QVBoxLayout(tg)
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(1, 100)
        self.thresh_slider.setValue(10)
        self.thresh_slider.setToolTip(
            "Show only voxels with |ρ| above this fraction of the maximum.\n"
            "Lower values = more voxels visible (use ~5-15% for cubes,\n"
            "20-40% for spheres or noisy reconstructions)."
        )
        self.thresh_slider.valueChanged.connect(self._update_labels)
        tv.addWidget(self.thresh_slider)
        self.thresh_lbl = QLabel("20% of max amplitude")
        self.thresh_lbl.setStyleSheet("color:#8b949e;font-size:9pt")
        tv.addWidget(self.thresh_lbl)
        cv.addWidget(tg)

        # Point size
        sg = QGroupBox("Voxel size")
        sv = QVBoxLayout(sg)
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(5, 100)
        self.size_slider.setValue(30)
        self.size_slider.setToolTip("Size of each voxel in the 3D scatter plot")
        self.size_slider.valueChanged.connect(self._update_labels)
        sv.addWidget(self.size_slider)
        self.size_lbl = QLabel("30")
        self.size_lbl.setStyleSheet("color:#8b949e;font-size:9pt")
        sv.addWidget(self.size_lbl)
        cv.addWidget(sg)

        # Opacity
        og = QGroupBox("Opacity")
        ov = QVBoxLayout(og)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setToolTip(
            "Voxel transparency. Lower opacity reveals\n"
            "internal structure like dislocations."
        )
        self.opacity_slider.valueChanged.connect(self._update_labels)
        ov.addWidget(self.opacity_slider)
        self.opacity_lbl = QLabel("0.70")
        self.opacity_lbl.setStyleSheet("color:#8b949e;font-size:9pt")
        ov.addWidget(self.opacity_lbl)
        cv.addWidget(og)

        # Colormap
        cg = QGroupBox("Colormap")
        cv2 = QVBoxLayout(cg)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["twilight", "hot", "inferno", "viridis", "jet", "coolwarm", "plasma"])
        self.cmap_combo.setToolTip(
            "Color scheme. 'twilight' is best for phase\n"
            "(diverging through zero); 'hot'/'inferno' for amplitude;\n"
            "'coolwarm' for strain."
        )
        cv2.addWidget(self.cmap_combo)
        cv.addWidget(cg)

        # Scale limits (vmin / vmax) for current display mode
        slg = QGroupBox("Color scale limits")
        slv = QVBoxLayout(slg)
        slv.setSpacing(6)

        self.auto_scale_check = QCheckBox("Auto (use data range)")
        self.auto_scale_check.setChecked(True)
        self.auto_scale_check.setToolTip(
            "Automatically use the data's 5th-95th percentile range.\n"
            "Uncheck to set custom limits below."
        )
        self.auto_scale_check.toggled.connect(self._on_autoscale_toggled)
        slv.addWidget(self.auto_scale_check)

        # vmin: label on top, spinbox below — more readable than form layout
        vmin_lbl = QLabel("Minimum value (vmin):")
        vmin_lbl.setStyleSheet("color:#8b949e;font-size:9pt;margin-top:4px")
        slv.addWidget(vmin_lbl)
        self.vmin_spin = _dbl(-1e6, 1e6, 0.0, 4)
        self.vmin_spin.setEnabled(False)
        self.vmin_spin.setMinimumHeight(28)
        self.vmin_spin.setMinimumWidth(150)
        self.vmin_spin.setToolTip(
            "Minimum value mapped to start of colormap.\n"
            "For phase: typically -3.1416 (-π)\n"
            "For density/strain: typically 0"
        )
        slv.addWidget(self.vmin_spin)

        vmax_lbl = QLabel("Maximum value (vmax):")
        vmax_lbl.setStyleSheet("color:#8b949e;font-size:9pt;margin-top:4px")
        slv.addWidget(vmax_lbl)
        self.vmax_spin = _dbl(-1e6, 1e6, 1.0, 4)
        self.vmax_spin.setEnabled(False)
        self.vmax_spin.setMinimumHeight(28)
        self.vmax_spin.setMinimumWidth(150)
        self.vmax_spin.setToolTip(
            "Maximum value mapped to end of colormap.\n"
            "For phase: typically 3.1416 (+π)"
        )
        slv.addWidget(self.vmax_spin)

        # When mode changes, suggest sensible defaults
        self.mode_combo.currentIndexChanged.connect(self._suggest_scale_defaults)

        cv.addWidget(slg)

        # Clip plane
        clg = QGroupBox("Clip plane (see inside)")
        clv = QVBoxLayout(clg)
        clv.setSpacing(6)

        self.clip_check = QCheckBox("Enable clip")
        self.clip_check.setToolTip(
            "Cut the object along the selected axis to expose internal\n"
            "phase or strain features (useful for revealing dislocations)."
        )
        clv.addWidget(self.clip_check)

        # Plane axis selection — label on top
        plane_lbl = QLabel("Cut along axis:")
        plane_lbl.setStyleSheet("color:#8b949e;font-size:9pt;margin-top:4px")
        clv.addWidget(plane_lbl)
        self.clip_axis_combo = QComboBox()
        self.clip_axis_combo.addItems([
            "X axis  (cuts in YZ plane)",
            "Y axis  (cuts in XZ plane)",
            "Z axis  (cuts in XY plane)",
        ])
        self.clip_axis_combo.setCurrentIndex(2)
        self.clip_axis_combo.setMinimumHeight(28)
        self.clip_axis_combo.setToolTip(
            "Which axis to slice along:\n"
            "\u2022 X axis: cut in the YZ plane (vertical wall)\n"
            "\u2022 Y axis: cut in the XZ plane\n"
            "\u2022 Z axis: cut in the XY plane (horizontal floor — default)"
        )
        clv.addWidget(self.clip_axis_combo)

        # Direction
        dir_lbl = QLabel("Keep voxels:")
        dir_lbl.setStyleSheet("color:#8b949e;font-size:9pt;margin-top:4px")
        clv.addWidget(dir_lbl)
        self.clip_dir_combo = QComboBox()
        self.clip_dir_combo.addItems([
            "below the plane",
            "above the plane",
        ])
        self.clip_dir_combo.setMinimumHeight(28)
        self.clip_dir_combo.setToolTip(
            "Which side of the cut plane to keep visible.\n"
            "Flip this to expose the other side of the crystal."
        )
        clv.addWidget(self.clip_dir_combo)

        pos_lbl = QLabel("Plane position:")
        pos_lbl.setStyleSheet("color:#8b949e;font-size:9pt;margin-top:4px")
        clv.addWidget(pos_lbl)
        self.clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_slider.setRange(0, 100)
        self.clip_slider.setValue(50)
        self.clip_slider.setMinimumHeight(20)
        self.clip_slider.setToolTip(
            "Position of clip plane along the chosen axis.\n"
            "0 = at min coord, 100 = at max coord, 50 = middle."
        )
        clv.addWidget(self.clip_slider)

        self.clip_pos_lbl = QLabel("position: 50%")
        self.clip_pos_lbl.setStyleSheet("color:#8b949e;font-size:8pt")
        clv.addWidget(self.clip_pos_lbl)
        self.clip_slider.valueChanged.connect(
            lambda v: self.clip_pos_lbl.setText(f"position: {v}%")
        )

        cv.addWidget(clg)

        # Status
        note = QLabel("After changing controls,\nclick 'Update view'.")
        note.setStyleSheet("color:#6e7681;font-size:8pt;font-style:italic")
        note.setWordWrap(True)
        cv.addWidget(note)

        # Export button
        self.export_btn = QPushButton("Export PNG")
        self.export_btn.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:24px;font-size:9pt")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_view)
        cv.addWidget(self.export_btn)

        cv.addStretch()
        sc.setWidget(ctrl)
        root.addWidget(sc)

        # ── Right: matplotlib 3D canvas (always works, no CDN) ─────────────
        right = QFrame()
        right.setFrameShape(QFrame.Shape.StyledPanel)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        with matplotlib.rc_context(MPL_DARK):
            self.fig3d = Figure(figsize=(10, 8), dpi=110, tight_layout=False, facecolor='#0a0d12')

        self.canvas3d = FigureCanvas(self.fig3d)
        toolbar3d = NavigationToolbar(self.canvas3d, right)
        toolbar3d.setStyleSheet("background:#161b22")
        rv.addWidget(toolbar3d)
        rv.addWidget(self.canvas3d, 1)
        root.addWidget(right, 1)

        # Mouse interaction matching the lattice (Plotly) view:
        #   - left drag        → rotate (matplotlib default)
        #   - right drag       → pan
        #   - scroll wheel     → zoom in/out
        #   - double-click     → reset view
        # We hook matplotlib's event system to add wheel zoom and right-click pan.
        self._init_view_state = None     # saved on first plot for reset
        self._pan_state = None
        self.canvas3d.mpl_connect('scroll_event', self._on_3d_scroll)
        self.canvas3d.mpl_connect('button_press_event', self._on_3d_press)
        self.canvas3d.mpl_connect('button_release_event', self._on_3d_release)
        self.canvas3d.mpl_connect('motion_notify_event', self._on_3d_motion)

        # Initial blank view with helpful message
        self._show_message(
            "Run a reconstruction in Tab 5,\nthen click 'View Reconstruction' here."
        )

    # ── Interactive controls (zoom, pan, reset) ───────────────────────────
    def _current_3d_axis(self):
        """Return the active 3D axes if present, else None."""
        if not hasattr(self, 'fig3d') or self.fig3d is None:
            return None
        for ax in self.fig3d.axes:
            if hasattr(ax, 'get_proj') and hasattr(ax, 'get_zlim'):
                return ax
        return None

    def _on_3d_scroll(self, event):
        """Mouse wheel → zoom in/out by adjusting the axis limits around their center."""
        ax = self._current_3d_axis()
        if ax is None:
            return
        factor = 0.85 if event.button == 'up' else 1.15
        for getlim, setlim in [
            (ax.get_xlim, ax.set_xlim),
            (ax.get_ylim, ax.set_ylim),
            (ax.get_zlim, ax.set_zlim),
        ]:
            lo, hi = getlim()
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * factor
            setlim(mid - half, mid + half)
        self.canvas3d.draw_idle()

    def _on_3d_press(self, event):
        """Right-click press → start panning. Double-click left → reset view."""
        if event.dblclick and event.button == 1:
            self._reset_3d_view()
            return
        if event.button == 3:
            ax = self._current_3d_axis()
            if ax is None:
                return
            self._pan_state = {
                'x': event.x,
                'y': event.y,
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim(),
                'zlim': ax.get_zlim(),
            }

    def _on_3d_release(self, event):
        if event.button == 3:
            self._pan_state = None

    def _on_3d_motion(self, event):
        """Right-drag → translate the view limits."""
        if self._pan_state is None or event.x is None or event.y is None:
            return
        ax = self._current_3d_axis()
        if ax is None:
            return
        st = self._pan_state
        dx_pix = event.x - st['x']
        dy_pix = event.y - st['y']
        # Convert to data units using axis ranges and figure size
        w, h = self.canvas3d.get_width_height()
        scale_x = (st['xlim'][1] - st['xlim'][0]) / max(w, 1)
        scale_y = (st['ylim'][1] - st['ylim'][0]) / max(h, 1)
        # Map screen drag to a 3D camera-relative shift. Simpler model:
        # drag x → shift X, drag y → shift Y.
        ax.set_xlim(st['xlim'][0] - dx_pix * scale_x, st['xlim'][1] - dx_pix * scale_x)
        ax.set_ylim(st['ylim'][0] + dy_pix * scale_y, st['ylim'][1] + dy_pix * scale_y)
        self.canvas3d.draw_idle()

    def _reset_3d_view(self):
        """Restore the saved initial view state (limits + camera angles)."""
        ax = self._current_3d_axis()
        if ax is None or self._init_view_state is None:
            return
        st = self._init_view_state
        ax.set_xlim(*st['xlim'])
        ax.set_ylim(*st['ylim'])
        ax.set_zlim(*st['zlim'])
        try:
            ax.view_init(elev=st['elev'], azim=st['azim'])
        except Exception:
            pass
        self.canvas3d.draw_idle()

    def _save_3d_view_state(self):
        """Save the current view as the 'home' for reset (called after each draw)."""
        ax = self._current_3d_axis()
        if ax is None:
            return
        try:
            self._init_view_state = {
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim(),
                'zlim': ax.get_zlim(),
                'elev': ax.elev,
                'azim': ax.azim,
            }
        except Exception:
            self._init_view_state = None

    def _show_message(self, msg, color='#8b949e'):
        """Display a centered text message in the canvas."""
        with matplotlib.rc_context(MPL_DARK):
            self.fig3d.clear()
            ax = self.fig3d.add_subplot(111)
            ax.set_facecolor('#0a0d12')
            ax.text(0.5, 0.5, msg, ha='center', va='center',
                    fontsize=14, color=color, transform=ax.transAxes,
                    multialignment='center')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        self.canvas3d.draw()

    def _update_labels(self):
        """Update slider labels (no re-render)."""
        self.thresh_lbl.setText(f"{self.thresh_slider.value()}% of max amplitude")
        self.opacity_lbl.setText(f"{self.opacity_slider.value()/100:.2f}")
        self.size_lbl.setText(f"{self.size_slider.value()}")

    def _on_autoscale_toggled(self, checked):
        """Enable/disable manual vmin/vmax spinboxes."""
        self.vmin_spin.setEnabled(not checked)
        self.vmax_spin.setEnabled(not checked)
        if not checked:
            # When user disables auto-scale, populate spinboxes with current data range
            self._suggest_scale_defaults()

    def _suggest_scale_defaults(self):
        """When mode changes, set sensible vmin/vmax for the new field."""
        idx = self.mode_combo.currentIndex()
        if idx == 1:                     # Phase
            self.vmin_spin.setValue(-3.1416)
            self.vmax_spin.setValue(3.1416)
        else:                            # Density or strain — use current data
            if self._result is not None:
                if idx == 0:
                    arr = self._result['amplitude']
                else:
                    # Strain: gradient of phase
                    phase = self._result['phase']
                    sup = self._result['support'] > 0.5
                    if sup.sum() > 0:
                        mean = np.angle(np.mean(np.exp(1j * phase[sup])))
                        phase = np.angle(np.exp(1j * (phase - mean)))
                    gx, gy, gz = np.gradient(phase * sup)
                    arr = np.sqrt(gx**2 + gy**2 + gz**2)
                # Use 5th-95th percentile for robust default
                vmin = float(np.percentile(arr, 5))
                vmax = float(np.percentile(arr, 95))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                self.vmin_spin.setValue(vmin)
                self.vmax_spin.setValue(vmax)
            else:
                self.vmin_spin.setValue(0.0)
                self.vmax_spin.setValue(1.0)

    def set_result(self, result: dict):
        """Called from T5 when reconstruction finishes."""
        self._result = result
        support = result['support']
        n_vox = int((support > 0.5).sum())
        n_total = support.size
        amp_max = float(result['amplitude'].max())
        sup_pct = 100.0 * n_vox / n_total

        # Warn if support is suspicious
        warning = ""
        if sup_pct > 50:
            warning = "<br><b style='color:#f85149'>⚠ Support fills > 50% of grid.<br>Reconstruction may be unconstrained.</b>"
        elif n_vox < 50:
            warning = "<br><b style='color:#f85149'>⚠ Very few support voxels.<br>Reconstruction may have failed.</b>"

        self.info_lbl.setText(
            f"<b style='color:#3fb950'>✓ Reconstruction loaded</b><br>"
            f"Volume: {result['amplitude'].shape[0]}³<br>"
            f"Support: {n_vox:,} voxels ({sup_pct:.1f}%)<br>"
            f"R-factor: {result['error_metric'][-1]:.4f}<br>"
            f"|ρ|_max: {amp_max:.3g}<br>"
            f"Method: {result['method']}{warning}"
        )
        self.view_btn.setEnabled(True)
        self.update_btn.setEnabled(True)
        self.vti_btn.setEnabled(True)
        self._show_message(
            "✓ Reconstruction loaded\n\n"
            "Click 'View Reconstruction'\nto render the 3D object.",
            color='#3fb950'
        )

    def _render_now(self):
        """Render the 3D view with current settings."""
        if self._result is None:
            self._show_message("No data loaded.\nRun a reconstruction in Tab 5 first.",
                               color='#f85149')
            return
        self._update_labels()
        self._show_message("Building 3D scene...", color='#8b949e')
        QApplication.processEvents()  # update UI

        try:
            self._draw_3d()
            self.export_btn.setEnabled(True)
        except DataQualityError as e:
            self._show_message(f"Cannot render this reconstruction:\n\n{e}",
                               color='#f0883e')
            self.export_btn.setEnabled(False)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"3D render error:\n{tb}")
            self._show_message(f"Render error:\n{str(e)[:200]}",
                               color='#f85149')
            self.export_btn.setEnabled(False)

    def _draw_3d(self):
        """Build the 3D plot with selectable render style."""
        r = self._result
        mode_idx = self.mode_combo.currentIndex()
        render_idx = self.render_combo.currentIndex()  # 0=points, 1=isosurface, 2=surface only
        threshold = self.thresh_slider.value() / 100.0
        opacity = self.opacity_slider.value() / 100.0
        cmap = self.cmap_combo.currentText()
        clip_enabled = self.clip_check.isChecked()
        clip_pos = self.clip_slider.value() / 100.0
        point_size = float(self.size_slider.value())

        support = r['support']
        amp = r['amplitude']
        N = amp.shape[0]

        # Sanity checks
        n_vox = int((support > 0.5).sum())
        n_total = support.size
        if n_vox > 0.85 * n_total:
            raise DataQualityError(
                f"Support fills {100*n_vox/n_total:.0f}% of the grid.\n"
                f"This means RAAR converged to a trivial solution.\n"
                f"Try retraining or use mode='nn_only'."
            )
        if n_vox < 20:
            raise DataQualityError(
                f"Only {n_vox} support voxels — reconstruction failed.\n"
                f"Try mode='nn_only' or train the model further."
            )

        # ── Compute physical-strain field properly (not just gradient norm) ──
        # Phase displacement-corrected and smoothed before gradient
        from scipy.ndimage import gaussian_filter
        sup_mask = (support > 0.5)

        # Build scalar field
        if mode_idx == 0:
            scalar = amp.copy()
            cb_label = "|ρ(r)| (electron density)"
        elif mode_idx == 1:
            phase = r['phase'].copy()
            if sup_mask.sum() > 0:
                mean = np.angle(np.mean(np.exp(1j * phase[sup_mask])))
                phase = np.angle(np.exp(1j * (phase - mean)))
            scalar = phase * sup_mask
            cb_label = "φ(r) phase [rad]"
        else:
            phase = r['phase'].copy()
            if sup_mask.sum() > 0:
                mean = np.angle(np.mean(np.exp(1j * phase[sup_mask])))
                phase = np.angle(np.exp(1j * (phase - mean)))
            # Smooth phase BEFORE taking gradient — reduces voxel-noise
            # in the strain field. σ=0.7 voxels keeps real features.
            phase_masked = phase * sup_mask
            phase_smoothed = gaussian_filter(phase_masked, sigma=0.7)
            gx, gy, gz = np.gradient(phase_smoothed)
            strain = np.sqrt(gx**2 + gy**2 + gz**2)
            # Erode the support by 1 voxel for the strain so we don't show
            # boundary-gradient artifacts (∇φ is ill-defined at the edge)
            try:
                from scipy.ndimage import binary_erosion
                strain_mask = binary_erosion(sup_mask, iterations=1)
                strain = strain * strain_mask
            except ImportError:
                strain = strain * sup_mask
            scalar = strain
            cb_label = "|∇φ(r)| (strain magnitude)"

        # Voxel pitch
        voxel_nm = r.get('voxel_size_nm', None)
        if voxel_nm is not None and np.asarray(voxel_nm).size >= 1:
            vn = np.asarray(voxel_nm).flatten()
            sx = sy = sz = float(vn[0]) if len(vn) == 1 else None
            if sx is None:
                sx, sy, sz = float(vn[0]), float(vn[1]), float(vn[2])
            axis_units = "nm"
        else:
            sx = sy = sz = 1.0
            axis_units = "voxels"

        # Color limits — same logic as before
        def _compute_limits(values_for_scale, mode_idx):
            if self.auto_scale_check.isChecked():
                if mode_idx == 1:
                    # Phase is bounded by [-π, π] by definition. Use the full
                    # range so colormap mapping is consistent across views.
                    return -float(np.pi), float(np.pi)
                elif mode_idx == 2:
                    nz = values_for_scale[values_for_scale > 1e-9]
                    if len(nz) > 10:
                        return float(np.percentile(nz, 50)), float(np.percentile(nz, 99))
                    return 0.0, max(float(values_for_scale.max()), 1e-6)
                else:
                    vmin = float(np.percentile(values_for_scale, 5))
                    vmax = float(np.percentile(values_for_scale, 95))
                    if abs(vmax - vmin) < 1e-12:
                        vmax = vmin + 1.0
                    return vmin, vmax
            else:
                v0 = float(self.vmin_spin.value())
                v1 = float(self.vmax_spin.value())
                if v1 <= v0:
                    v1 = v0 + 1e-6
                return v0, v1

        from mpl_toolkits.mplot3d import Axes3D  # noqa
        with matplotlib.rc_context(MPL_DARK):
            self.fig3d.clear()
            ax = self.fig3d.add_subplot(111, projection='3d')
            ax.set_facecolor('#0a0d12')

            # ────────────────────────────────────────────────────────────
            # ISOSURFACE / SURFACE-ONLY rendering using marching cubes
            # ────────────────────────────────────────────────────────────
            if render_idx in (1, 2):
                try:
                    from skimage import measure
                except ImportError:
                    raise DataQualityError(
                        "Isosurface rendering requires scikit-image.\n"
                        "Install it:  pip install scikit-image\n"
                        "Or switch render style to 'Point cloud'."
                    )

                # Build the isosurface from amplitude (always — that's the shape).
                # Smoothing differs by render mode:
                #   - Isosurface: more aggressive smoothing → polished look
                #   - Surface only: less smoothing → preserves mesh detail
                amp_max = max(float(amp.max()), 1e-12)
                iso_value = threshold * amp_max
                if render_idx == 1:    # Isosurface: smoother
                    amp_smooth = gaussian_filter(amp, sigma=1.5)
                else:                  # Surface only: less smooth
                    amp_smooth = gaussian_filter(amp, sigma=0.4)

                if amp_smooth.max() < iso_value:
                    raise DataQualityError(
                        f"No voxels above isosurface threshold {threshold:.0%}.\n"
                        f"Lower the threshold."
                    )

                try:
                    mc_result = measure.marching_cubes(
                        amp_smooth, level=iso_value, spacing=(sx, sy, sz),
                        gradient_direction='descent', allow_degenerate=False,
                    )
                    verts = np.asarray(mc_result[0]).reshape(-1, 3)
                    faces = np.asarray(mc_result[1]).reshape(-1, 3).astype(np.int64)
                except Exception as e:
                    raise DataQualityError(
                        f"Marching cubes failed: {e}\n"
                        f"Try a different threshold or switch to point cloud mode."
                    )

                if len(verts) < 3 or len(faces) < 1:
                    raise DataQualityError(
                        f"Isosurface produced no triangles "
                        f"({len(verts)} verts, {len(faces)} faces).\n"
                        f"Lower the threshold or use point cloud mode."
                    )

                # Center vertices around origin
                verts_c = verts.copy()
                verts_c[:, 0] -= (N / 2) * sx
                verts_c[:, 1] -= (N / 2) * sy
                verts_c[:, 2] -= (N / 2) * sz

                # Apply clip plane to vertices/faces
                if clip_enabled:
                    clip_axis = self.clip_axis_combo.currentIndex()
                    clip_above = (self.clip_dir_combo.currentIndex() == 1)
                    clip_pos_phys = (clip_pos - 0.5) * N * (sx, sy, sz)[clip_axis]
                    if clip_above:
                        keep_v = verts_c[:, clip_axis] > clip_pos_phys
                    else:
                        keep_v = verts_c[:, clip_axis] < clip_pos_phys
                    keep_f = keep_v[faces].all(axis=1)
                    if keep_f.sum() == 0:
                        raise DataQualityError(
                            "No faces remain after clip. Move the slider."
                        )
                    faces = faces[keep_f]

                # If somehow faces became empty after clip, abort cleanly
                if len(faces) < 1:
                    raise DataQualityError(
                        "No faces remain to render."
                    )

                # Sample the SCALAR FIELD at vertex positions to color the surface.
                from scipy.ndimage import map_coordinates
                # `verts` from marching_cubes is in PHYSICAL units (we passed
                # spacing=(sx,sy,sz)). For map_coordinates we need voxel indices.
                vertex_voxels = np.column_stack([
                    verts[:, 0] / max(sx, 1e-12),
                    verts[:, 1] / max(sy, 1e-12),
                    verts[:, 2] / max(sz, 1e-12),
                ]).astype(np.float64)
                vertex_scalar = map_coordinates(
                    scalar, vertex_voxels.T, order=1, mode='constant', cval=0.0
                )
                vertex_scalar = np.asarray(vertex_scalar).reshape(-1)

                # Compute color limits from the scalar at vertices
                vmin, vmax = _compute_limits(vertex_scalar, mode_idx)

                # Mean of 3 vertex scalars per face
                face_scalar = vertex_scalar[faces].mean(axis=1)
                face_scalar_norm = np.clip(
                    (face_scalar - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0
                )
                cmap_obj = matplotlib.colormaps[cmap]
                face_colors = np.asarray(cmap_obj(face_scalar_norm))
                # Defensive: ensure shape (N_faces, 4)
                if face_colors.ndim != 2 or face_colors.shape[0] != len(faces):
                    raise DataQualityError(
                        f"Color array shape mismatch: {face_colors.shape} "
                        f"expected ({len(faces)}, 4)"
                    )

                # Per-face alpha weighted by amplitude (NOT the colored field).
                # Sample amplitude at vertices, average per face, normalize.
                # Same formula as in point cloud: alpha = opacity ** (1/amp_norm).
                vertex_amp = map_coordinates(
                    amp, vertex_voxels.T, order=1, mode='constant', cval=0.0
                )
                vertex_amp = np.asarray(vertex_amp).reshape(-1)
                face_amp = vertex_amp[faces].mean(axis=1)
                face_amp_min = float(face_amp.min()) if len(face_amp) else 0.0
                face_amp_max = float(face_amp.max()) if len(face_amp) else 1.0
                if face_amp_max - face_amp_min > 1e-12:
                    face_amp_norm = (face_amp - face_amp_min) / (face_amp_max - face_amp_min)
                else:
                    face_amp_norm = np.ones_like(face_amp)
                face_amp_norm = np.clip(face_amp_norm, 0.05, 1.0)
                if opacity >= 0.999:
                    per_face_alpha = np.ones_like(face_amp_norm)
                else:
                    safe_op = max(opacity, 1e-6)
                    per_face_alpha = np.clip(
                        safe_op ** (1.0 / face_amp_norm), 0.0, 1.0
                    )
                face_colors[:, 3] = per_face_alpha

                # Edge color
                if render_idx == 2:
                    edge_colors = face_colors[:, :3] * 0.5
                else:
                    edge_colors = 'none'

                # Build the triangle vertex array explicitly as a (N_faces, 3, 3)
                # to avoid any matplotlib internal broadcast surprises.
                tri_verts = verts_c[faces]   # shape (N_faces, 3, 3)
                tri_verts = np.asarray(tri_verts).reshape(len(faces), 3, 3)

                # Apply manual lighting per face for a "polished surface" look,
                # ONLY for isosurface (render_idx=1). Surface-only mode keeps
                # flat colors so mesh edges read clearly.
                # We compute the cosine of the angle between the face normal and
                # a fixed light direction, then blend the face color toward
                # white where the surface points at the light.
                if render_idx == 1 and len(faces) > 0:
                    v0 = tri_verts[:, 0]
                    v1 = tri_verts[:, 1]
                    v2 = tri_verts[:, 2]
                    n = np.cross(v1 - v0, v2 - v0)
                    n_len = np.linalg.norm(n, axis=1, keepdims=True)
                    n = n / np.maximum(n_len, 1e-12)
                    light_dir = np.array([0.4, 0.4, 0.8], dtype=np.float32)
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    cos_a = np.clip(np.abs(n @ light_dir), 0.0, 1.0)
                    # Phong-ish: ambient + diffuse
                    intensity = 0.45 + 0.55 * cos_a
                    rgb = face_colors[:, :3] * intensity[:, None]
                    face_colors_lit = face_colors.copy()
                    face_colors_lit[:, :3] = np.clip(rgb, 0.0, 1.0)
                else:
                    face_colors_lit = face_colors

                # Render via Poly3DCollection. We disable matplotlib's internal
                # shading (shade=True can crash on degenerate meshes); the
                # manual per-face lighting above provides the smooth-surface look.
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                mesh = Poly3DCollection(
                    tri_verts,
                    facecolors=face_colors_lit,
                    edgecolors=edge_colors,
                    linewidths=0.1 if render_idx == 2 else 0,
                    shade=False,
                )
                ax.add_collection3d(mesh)

                # ── For isosurface mode, add an internal point cloud ──────
                # The user wants the isosurface "filled with inside data" —
                # voxels INSIDE the iso shell, colored by the scalar field,
                # showing through the (possibly translucent) surface. This
                # gives a volumetric look distinct from "Surface only".
                if render_idx == 1:
                    # Sample voxels above 1.4×iso_value (well inside the shell)
                    deep_mask = amp_smooth > min(iso_value * 1.4, amp_max * 0.95)
                    if deep_mask.sum() > 30:
                        deep_coords = np.argwhere(deep_mask)
                        # Sub-sample to keep the plot responsive
                        max_internal = 4000
                        if len(deep_coords) > max_internal:
                            sub_idx = np.random.default_rng(0).choice(
                                len(deep_coords), max_internal, replace=False
                            )
                            deep_coords = deep_coords[sub_idx]
                        # Apply the same clip plane to internal points
                        if clip_enabled:
                            ca = self.clip_axis_combo.currentIndex()
                            cabove = (self.clip_dir_combo.currentIndex() == 1)
                            cpos_idx = int(clip_pos * N)
                            if cabove:
                                k = deep_coords[:, ca] > cpos_idx
                            else:
                                k = deep_coords[:, ca] < cpos_idx
                            deep_coords = deep_coords[k]
                        if len(deep_coords) > 0:
                            dcx = (deep_coords[:, 0] - N / 2) * sx
                            dcy = (deep_coords[:, 1] - N / 2) * sy
                            dcz = (deep_coords[:, 2] - N / 2) * sz
                            dval = scalar[deep_coords[:, 0], deep_coords[:, 1],
                                            deep_coords[:, 2]]
                            d_norm = np.clip(
                                (dval - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0
                            )
                            d_rgba = np.asarray(cmap_obj(d_norm))
                            # Internal points use a softer alpha so they don't
                            # crowd the surface but remain visible
                            d_rgba[:, 3] = 0.25 * opacity
                            ax.scatter(dcx, dcy, dcz, c=d_rgba,
                                        s=max(point_size * 0.4, 1.5),
                                        edgecolors='none', depthshade=False)

                # Dummy mappable for colorbar (represents the scalar field)
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                sm = matplotlib.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
                sm.set_array([])

                # Title
                tri_count = len(faces)
                ax.set_title(f"{cb_label}    {tri_count:,} triangles",
                             color='#4f98a3', fontsize=11)
                cb = self.fig3d.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
                cb.set_label(cb_label, color='#e6edf3', fontsize=9)
                cb.ax.tick_params(colors='#8b949e', labelsize=8)
                # Phase mode: show π and -π instead of numeric 3.14
                if mode_idx == 1:
                    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                    cb.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

            else:
                # ────────────────────────────────────────────────────────────
                # POINT CLOUD rendering (legacy)
                # ────────────────────────────────────────────────────────────
                amp_max = max(float(amp.max()), 1e-12)
                amp_norm = amp / amp_max
                vox_mask = amp_norm > threshold
                if vox_mask.sum() == 0:
                    raise DataQualityError(
                        f"No voxels above threshold {threshold:.0%}."
                    )
                coords = np.argwhere(vox_mask)
                values = scalar[vox_mask]
                # Sample amplitude at the SAME voxels — needed for per-voxel
                # alpha that weights low-intensity points more than high.
                amp_at_voxels = amp[vox_mask]

                if clip_enabled:
                    clip_axis = self.clip_axis_combo.currentIndex()
                    clip_above = (self.clip_dir_combo.currentIndex() == 1)
                    clip_pos_idx = int(clip_pos * N)
                    if clip_above:
                        keep = coords[:, clip_axis] > clip_pos_idx
                    else:
                        keep = coords[:, clip_axis] < clip_pos_idx
                    coords = coords[keep]
                    values = values[keep]
                    amp_at_voxels = amp_at_voxels[keep]
                    if len(coords) == 0:
                        raise DataQualityError("No voxels remain after clip plane.")

                max_pts = 15000
                if len(coords) > max_pts:
                    idx = np.random.default_rng(0).choice(len(coords), max_pts, replace=False)
                    coords = coords[idx]
                    values = values[idx]
                    amp_at_voxels = amp_at_voxels[idx]

                cx = (coords[:, 0] - N / 2) * sx
                cy = (coords[:, 1] - N / 2) * sy
                cz = (coords[:, 2] - N / 2) * sz

                vmin, vmax = _compute_limits(values, mode_idx)

                # Per-voxel alpha: low-intensity voxels fade much more than
                # high-intensity ones when the user lowers the opacity slider.
                # Formula: alpha = opacity ** (1/amp_norm)
                #   amp_norm = 1.0 (peak)   → alpha = opacity        (linear)
                #   amp_norm = 0.5          → alpha = opacity^2      (drops faster)
                #   amp_norm = 0.1          → alpha = opacity^10     (drops VERY fast)
                # This mimics how a ParaView volume rendering shows a bright
                # core through a translucent envelope as opacity decreases.
                amp_min = float(amp_at_voxels.min())
                amp_max_v = float(amp_at_voxels.max())
                if amp_max_v - amp_min > 1e-12:
                    amp_norm = (amp_at_voxels - amp_min) / (amp_max_v - amp_min)
                else:
                    amp_norm = np.ones_like(amp_at_voxels)
                # Clamp to avoid huge exponents (would underflow to 0 too fast)
                amp_norm = np.clip(amp_norm, 0.05, 1.0)
                if opacity >= 0.999:
                    per_alpha = np.ones_like(amp_norm)
                else:
                    safe_op = max(opacity, 1e-6)
                    per_alpha = np.clip(safe_op ** (1.0 / amp_norm), 0.0, 1.0)

                # Build per-point RGBA from the colormap, then override alpha
                cmap_obj = matplotlib.colormaps[cmap]
                norm_v = np.clip(
                    (values - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0
                )
                rgba = np.asarray(cmap_obj(norm_v))
                rgba[:, 3] = per_alpha

                sc = ax.scatter(cx, cy, cz, c=rgba,
                                 s=point_size,
                                 edgecolors='none', depthshade=True)
                ax.set_title(f"{cb_label}    {len(coords):,} voxels",
                             color='#4f98a3', fontsize=11)
                # Build a separate scalar mappable for the colorbar (so it
                # represents the scalar field, not the per-voxel alpha).
                norm_obj = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                sm = matplotlib.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
                sm.set_array([])
                cb = self.fig3d.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
                cb.set_label(cb_label, color='#e6edf3', fontsize=9)
                cb.ax.tick_params(colors='#8b949e', labelsize=8)
                # Phase mode: show π and -π instead of numeric 3.14
                if mode_idx == 1:
                    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                    cb.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

            # ── Common axis setup ────────────────────────────────────────
            half_x = (N / 2) * sx
            half_y = (N / 2) * sy
            half_z = (N / 2) * sz
            ax.set_xlim(-half_x, half_x)
            ax.set_ylim(-half_y, half_y)
            ax.set_zlim(-half_z, half_z)
            ax.set_xlabel(f'x [{axis_units}]', color='#8b949e', fontsize=9)
            ax.set_ylabel(f'y [{axis_units}]', color='#8b949e', fontsize=9)
            ax.set_zlabel(f'z [{axis_units}]', color='#8b949e', fontsize=9)
            ax.tick_params(colors='#8b949e', labelsize=8)

            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_facecolor((0.04, 0.05, 0.07, 0.6))
                pane.set_edgecolor('#30363d')

        self.canvas3d.draw()
        # Save the just-rendered view as the "home" for double-click-to-reset
        self._save_3d_view_state()

    def _export_view(self):
        """Export the current 3D view as PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export 3D view", "bcdi_3d_view.png", "PNG (*.png)"
        )
        if path:
            self.fig3d.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0a0d12')

    def _export_vti(self):
        """Export reconstruction as ParaView-compatible .vti file."""
        if self._result is None:
            return

        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export to ParaView",
            "bcdi_reconstruction.vti",
            "VTK ImageData (*.vti);;Legacy VTK (*.vtk)",
        )
        if not path:
            return

        try:
            self._write_vti(path)
            QMessageBox.information(
                self, "Export complete",
                f"<b>Saved:</b><br><tt>{path}</tt><br><br>"
                f"<b>To view the FULL 3D object in ParaView:</b><br>"
                f"1. Open the .vti file in ParaView<br>"
                f"2. In the <b>Pipeline Browser</b> click the file → "
                f"<b>Apply</b> (green button)<br>"
                f"3. Change <b>Representation</b> from 'Outline' to "
                f"<b>'Volume'</b> (top-left dropdown)<br>"
                f"4. Change <b>Coloring</b> from 'support' to "
                f"<b>'amplitude'</b> (the second dropdown)<br>"
                f"5. Adjust opacity via <b>Edit Color Map</b> if needed<br><br>"
                f"<i>Note: if you see only a flat slice, you're in 'Slice' "
                f"mode. Switch to 'Volume' representation to see the full 3D "
                f"reconstruction.</i><br><br>"
                f"<b>Available scalar fields in the file:</b><br>"
                f"  • <tt>amplitude</tt>      — |ρ(r)|, GUI threshold applied<br>"
                f"  • <tt>amplitude_full</tt> — |ρ(r)|, no threshold<br>"
                f"  • <tt>phase</tt>           — φ(r), NaN outside support<br>"
                f"  • <tt>strain</tt>          — |∇φ(r)|, NaN outside support<br>"
                f"  • <tt>support</tt>         — binary mask (0 or 1)<br><br>"
                f"Free download: https://www.paraview.org",
            )
        except Exception as e:
            import traceback
            QMessageBox.critical(
                self, "Export failed",
                f"Could not write VTI file:\n\n{e}\n\n"
                f"{traceback.format_exc()[:300]}",
            )

    def _write_vti(self, path):
        """
        Write a VTK ImageData (.vti) file containing the reconstruction.

        Includes scalar arrays:
            amplitude         — |ρ(r)| with current GUI threshold applied
            amplitude_full    — |ρ(r)| raw (no threshold)
            phase             — φ(r), centered relative to support mean
            strain            — |∇φ(r)|, smoothed and edge-eroded
            support           — binary support mask (0 or 1)

        Spacing is in nm if voxel_size_nm is known, else 1 (voxels).
        Phase and strain are masked to the support: outside the support
        they are set to NaN so ParaView correctly hides them when slicing
        or doing volume rendering.

        The 'amplitude' array masks values below the GUI threshold to zero.
        This way when you open the file in ParaView, the default isosurface
        and volume rendering show the real reconstructed object instead of
        a blob washed out by below-threshold noise. The raw amplitude is
        still available as 'amplitude_full' if you want the original values.
        """
        from scipy.ndimage import gaussian_filter
        r = self._result
        amp_raw = r['amplitude'].astype(np.float32)
        phase_raw = r['phase'].astype(np.float32)
        sup = r['support'].astype(np.float32)
        sup_mask = sup > 0.5

        # Apply current GUI isosurface threshold to the EXPORTED amplitude.
        # This is what the user sees in the 3D viewer — match it.
        threshold = self.thresh_slider.value() / 100.0
        amp_max = max(float(amp_raw.max()), 1e-12)
        thr_value = threshold * amp_max
        # Soft mask: zero below threshold, keep value above
        amp_thresholded = np.where(amp_raw >= thr_value, amp_raw, 0.0).astype(np.float32)

        # Phase relative to support mean
        if sup_mask.sum() > 0:
            mean = np.angle(np.mean(np.exp(1j * phase_raw[sup_mask])))
            phase_centered = np.angle(np.exp(1j * (phase_raw - mean))).astype(np.float32)
        else:
            phase_centered = phase_raw

        # Phase: NaN outside support so ParaView hides those voxels
        phase_export = phase_centered.copy()
        phase_export[~sup_mask] = np.nan

        # Strain (smoothed + edge-eroded for cleaner visualization)
        phase_for_strain = phase_centered * sup_mask
        phase_smooth = gaussian_filter(phase_for_strain, sigma=0.7)
        gx, gy, gz = np.gradient(phase_smooth)
        strain = np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32)
        try:
            from scipy.ndimage import binary_erosion
            sm = binary_erosion(sup_mask, iterations=1)
            # NaN outside the eroded support
            strain_export = strain.copy()
            strain_export[~sm] = np.nan
        except ImportError:
            strain_export = strain.copy()
            strain_export[~sup_mask] = np.nan

        # Voxel spacing
        voxel_nm = r.get('voxel_size_nm', None)
        if voxel_nm is not None and np.asarray(voxel_nm).size >= 1:
            vn = np.asarray(voxel_nm).flatten()
            if len(vn) == 1:
                spacing = (float(vn[0]), float(vn[0]), float(vn[0]))
            else:
                spacing = (float(vn[0]), float(vn[1]), float(vn[2]))
        else:
            spacing = (1.0, 1.0, 1.0)

        N = amp_raw.shape[0]
        origin = (-(N / 2) * spacing[0],
                  -(N / 2) * spacing[1],
                  -(N / 2) * spacing[2])

        arrays_to_write = [
            ('amplitude',      amp_thresholded),
            ('amplitude_full', amp_raw),
            ('phase',          phase_export),
            ('strain',         strain_export),
            ('support',        sup),
        ]

        # Try VTK Python first (most reliable), fall back to manual XML
        try:
            import vtk
            from vtk.util import numpy_support
            self._write_vti_vtk(path, arrays_to_write, spacing, origin, vtk, numpy_support)
        except ImportError:
            self._write_vti_manual(path, arrays_to_write, spacing, origin)

    def _write_vti_vtk(self, path, arrays_to_write, spacing, origin, vtk, numpy_support):
        """Write VTI using the official VTK library (best quality)."""
        first_arr = arrays_to_write[0][1]
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(first_arr.shape[0], first_arr.shape[1], first_arr.shape[2])
        image_data.SetSpacing(*spacing)
        image_data.SetOrigin(*origin)

        for name, arr in arrays_to_write:
            # VTK expects flat, Fortran-ordered for ImageData
            flat = arr.astype(np.float32).ravel(order='F')
            vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True)
            vtk_arr.SetName(name)
            image_data.GetPointData().AddArray(vtk_arr)

        # Set first array as the active scalar
        image_data.GetPointData().SetActiveScalars(arrays_to_write[0][0])

        if path.endswith('.vtk'):
            writer = vtk.vtkStructuredPointsWriter()
        else:
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetCompressorTypeToZLib()
        writer.SetFileName(path)
        writer.SetInputData(image_data)
        writer.Write()

    def _write_vti_manual(self, path, arrays_to_write, spacing, origin):
        """
        Write VTI without the vtk library. Uses base64-encoded uncompressed
        binary with a simple UInt32 length header — the most compatible format
        ParaView supports. Avoids zlib block-based encoding (which some
        ParaView builds parse incorrectly with a single large block, causing
        the data to appear as a flat slice).
        """
        import base64, struct
        first_arr = arrays_to_write[0][1]
        # Numpy shape is (Nx, Ny, Nz) for the reconstruction volume.
        # VTK ImageData uses (Nx, Ny, Nz) as its Dimensions, with data laid
        # out in Fortran order (X fastest, then Y, then Z).
        Nx, Ny, Nz = first_arr.shape

        def _encode(arr):
            # 1. Cast to float32 in Fortran order (X varies fastest)
            flat = np.ascontiguousarray(
                arr.astype(np.float32).transpose(0, 1, 2).ravel(order='F')
            ).tobytes()
            # 2. Prepend a UInt32 byte-count header (uncompressed VTK format)
            header = struct.pack('<I', len(flat))
            # 3. Base64-encode for embedding in XML
            return base64.b64encode(header + flat).decode('ascii')

        arrays_xml = ""
        for name, arr in arrays_to_write:
            data_b64 = _encode(arr)
            arrays_xml += (
                f'      <DataArray type="Float32" Name="{name}" '
                f'format="binary" NumberOfComponents="1">\n'
                f'        {data_b64}\n'
                f'      </DataArray>\n'
            )

        first_name = arrays_to_write[0][0]
        # No `compressor` attribute → uncompressed binary, simpler parsing
        xml = f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt32">
  <ImageData WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}" Origin="{origin[0]} {origin[1]} {origin[2]}" Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">
    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">
      <PointData Scalars="{first_name}">
{arrays_xml}      </PointData>
    </Piece>
  </ImageData>
</VTKFile>
'''
        with open(path, 'w', encoding='utf-8') as f:
            f.write(xml)


class DataQualityError(Exception):
    """Raised when the reconstruction is not viewable due to data issues."""
    pass
