from __future__ import annotations
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, List

from .nn_phase_model import PhaseUNet3D



@dataclass
class ReconstructionResult:
    object_3d: np.ndarray          # Complex object ρ·e^{iφ}, shape [N,N,N]
    phase: np.ndarray              # Phase field, shape [N,N,N]
    amplitude: np.ndarray          # |ρ|, shape [N,N,N]
    support: np.ndarray            # Final support mask, shape [N,N,N]
    error_metric: List[float]      # R-factor at each iteration
    method: str = ''               # 'nn_only', 'classical', 'hybrid'
    elapsed_seconds: float = 0.0   # Wall time
    n_iterations: int = 0          # Total iterations



def _fft3(x):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))


def _ifft3(x):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x)))


def _r_factor(measured_amp, current_amp):
    num = np.sum(np.abs(measured_amp - current_amp))
    den = np.sum(np.abs(measured_amp))
    return num / max(den, 1e-12)


def _modulus_projection(obj, measured_amplitude):
    F = _fft3(obj)
    phase = np.angle(F)
    F_constrained = measured_amplitude * np.exp(1j * phase)
    return _ifft3(F_constrained)


def _support_projection(obj, support, mode='er', beta=0.9, obj_prev=None):
    inside = support > 0.5

    if mode == 'er':
        # Error Reduction: zero outside support
        result = obj.copy()
        result[~inside] = 0
        return result

    elif mode == 'hio':
        # Inside support: keep modulus-projected value
        # Outside support: x_n - β · P_M x_n
        if obj_prev is None:
            # First iter: same as ER
            result = obj.copy()
            result[~inside] = 0
            return result
        result = np.empty_like(obj)
        result[inside] = obj[inside]
        result[~inside] = obj_prev[~inside] - beta * obj[~inside]
        return result

    elif mode == 'raar':
        # x_{n+1} = β/2 (R_S R_M + I) x_n + (1-β) P_M x_n
        #
        # Computing R_S R_M x_n:
        #   P_M x_n is `obj` (passed in, modulus-projected)
        #   R_M x_n = 2 P_M x_n - x_n = 2*obj - obj_prev
        # Then R_S applied to it: R_S y = 2 P_S y - y
        #
        # We need obj_prev (previous iterate x_n)
        if obj_prev is None:
            # First iter: fall back to ER
            result = obj.copy()
            result[~inside] = 0
            return result

        # R_M(x_n) = 2 P_M(x_n) - x_n = 2*obj - obj_prev
        rm = 2 * obj - obj_prev

        # P_S(rm)
        ps_rm = rm.copy()
        ps_rm[~inside] = 0

        # R_S(rm) = 2 P_S(rm) - rm
        rs_rm = 2 * ps_rm - rm

        # x_{n+1} = β/2 (R_S R_M + I) x_n + (1-β) P_M x_n
        #        = β/2 (rs_rm + obj_prev) + (1-β) obj
        result = (beta / 2.0) * (rs_rm + obj_prev) + (1 - beta) * obj
        return result

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _shrink_wrap(obj, support, threshold=0.10, sigma=2.0, max_support_fraction=0.30):
    from scipy.ndimage import gaussian_filter
    amp = np.abs(obj)
    blurred = gaussian_filter(amp, sigma=sigma)
    new_support = (blurred > threshold * blurred.max()).astype(np.float32)

    # Hard cap: prevent support from growing to fill the entire grid
    n_voxels = new_support.sum()
    max_voxels = max_support_fraction * new_support.size
    if n_voxels > max_voxels:
        # Too many voxels — keep only the brightest fraction
        # by raising the effective threshold
        flat = blurred.flatten()
        # Find threshold value that keeps exactly max_voxels
        sorted_vals = np.partition(flat, -int(max_voxels))[-int(max_voxels):]
        adaptive_threshold = sorted_vals.min()
        new_support = (blurred >= adaptive_threshold).astype(np.float32)

    return new_support


def run_phase_retrieval(
    measured_amplitude: np.ndarray,
    initial_phase: np.ndarray = None,
    initial_amplitude: np.ndarray = None,
    support: np.ndarray = None,
    n_raar: int = 100,
    n_er: int = 20,
    n_hio: int = 0,
    beta: float = 0.9,
    use_shrink_wrap: bool = False,
    shrink_wrap_interval: int = 50,
    shrink_wrap_threshold: float = 0.10,
    shrink_wrap_sigma: float = 2.0,
    max_support_fraction: float = 0.30,
    algorithm: str = 'hybrid',
    progress_cb=None,
) -> ReconstructionResult:

    N = measured_amplitude.shape[0]
    rng = np.random.default_rng(42)


    if support is None:
        auto_corr = np.abs(_ifft3(measured_amplitude ** 2))
        ac_max = auto_corr.max()
        # Try increasingly strict thresholds until support is at most 30% of grid
        # (15% × 2 since autocorr is 2× object linearly, so 8× by volume → use 30% as safe upper)
        n_total = auto_corr.size
        for thr in [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]:
            sup_try = (auto_corr > thr * ac_max)
            if sup_try.sum() < n_total * 0.30:
                support = sup_try.astype(np.float32)
                break
        else:
            # Fallback: keep top 15% of voxels
            n_keep = int(n_total * 0.15)
            t = np.partition(auto_corr.ravel(), -n_keep)[-n_keep]
            support = (auto_corr >= t).astype(np.float32)
    support = support.astype(np.float32)

    # ── Initialize complex object — preserve NN amplitude if available ───
    if initial_phase is None:
        initial_phase = rng.uniform(-np.pi, np.pi, (N, N, N)).astype(np.float32)

    if initial_amplitude is not None:
        # Use NN-predicted amplitude (already correctly scaled in nn_only_infer)
        amp_seed = np.maximum(initial_amplitude, 0).astype(np.float32)
    else:
        # Seed amplitude from |IFFT(measured)| but normalize
        # so the FFT magnitude matches measured_amplitude.max() exactly
        rough = np.abs(_ifft3(measured_amplitude))
        rough = rough * (support > 0.5)
        # Scale to match measured magnitude
        seed_obj = rough * np.exp(1j * (initial_phase if initial_phase is not None else 0))
        F_seed = _fft3(seed_obj)
        F_seed_max = float(np.abs(F_seed).max())
        meas_max = float(measured_amplitude.max())
        if F_seed_max > 1e-12 and meas_max > 0:
            amp_seed = rough * (meas_max / F_seed_max)
        else:
            amp_seed = rough

    obj = (support > 0.5) * amp_seed * np.exp(1j * initial_phase)

    # Track best solution
    best_obj = obj.copy()
    best_support = support.copy()
    best_r = float('inf')

    errors = []

    # Determine schedule
    if algorithm == 'er':
        schedule = ['er'] * n_er
    elif algorithm == 'hio':
        schedule = ['hio'] * n_hio + ['er'] * n_er
    elif algorithm == 'raar':
        schedule = ['raar'] * n_raar + ['er'] * n_er
    elif algorithm == 'hybrid':
        # Standard hybrid: HIO (escape minima) → RAAR (smooth) → ER (polish)
        schedule = ['hio'] * n_hio + ['raar'] * n_raar + ['er'] * n_er
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    total_iter = max(len(schedule), 1)

    for i, mode in enumerate(schedule):
        # Fourier modulus projection: g'_n = P_M(g_n)
        obj_mod = _modulus_projection(obj, measured_amplitude)

        # Compute error from the modulus-projected iterate
        F_current = _fft3(obj)
        r = _r_factor(measured_amplitude, np.abs(F_current))
        errors.append(r)

        # Track best
        if r < best_r:
            best_r = r
            best_obj = obj.copy()
            best_support = support.copy()

        # Support projection: pass CURRENT iterate as `obj_prev` argument
        # (HIO/RAAR formulae need g_n, not g_{n-1}).
        obj_new = _support_projection(
            obj_mod, support, mode=mode, beta=beta, obj_prev=obj
        )

        obj = obj_new

        # Shrink-wrap support update (only during HIO/RAAR phase, not ER)
        if (use_shrink_wrap
                and (i + 1) % shrink_wrap_interval == 0
                and i > 0
                and mode != 'er'):
            support = _shrink_wrap(
                obj, support,
                threshold=shrink_wrap_threshold,
                sigma=shrink_wrap_sigma,
                max_support_fraction=max_support_fraction,
            )

        if progress_cb:
            progress_cb(i / total_iter)

    # Final ER pass on best object to clean up
    final_obj = best_obj.copy()
    final_support = best_support.copy()
    for i in range(5):
        final_obj = _modulus_projection(final_obj, measured_amplitude)
        final_obj = _support_projection(final_obj, final_support, mode='er')

    # Final R-factor on the cleaned best
    F_final = _fft3(final_obj)
    final_r = _r_factor(measured_amplitude, np.abs(F_final))
    errors.append(final_r)

    if progress_cb:
        progress_cb(1.0)

    return ReconstructionResult(
        object_3d=final_obj,
        phase=np.angle(final_obj),
        amplitude=np.abs(final_obj),
        support=final_support,
        error_metric=errors,
        n_iterations=total_iter + 5,
    )



class HybridPhaseRetriever:


    def __init__(
        self,
        model_path: str = None,
        base_channels: int = 32,
        device: str = None,
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        if model_path is not None:
            self._load_model(model_path, base_channels)

    def _load_model(self, path: str, base_channels: int):
        """Load trained U-Net from checkpoint."""
        self.model = PhaseUNet3D(in_channels=1, base_channels=base_channels)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded model from {path} (device={self.device})")

    @torch.no_grad()
    def predict_phase_nn(self, diffraction: np.ndarray) -> np.ndarray:

        if self.model is None:
            raise RuntimeError("No model loaded. Provide model_path.")

        # Preprocess: amplitude → log → normalize
        amplitude = np.sqrt(np.maximum(diffraction, 0)).astype(np.float32)
        log_amp = np.log10(amplitude + 1.0)
        scale = log_amp.max()
        if scale > 0:
            log_amp = log_amp / scale

        # To tensor: [N,N,N] → [1, 1, N, N, N]
        x = torch.from_numpy(log_amp[np.newaxis, np.newaxis]).float().to(self.device)

        # Predict
        phase_norm = self.model(x)  # [1, 1, N, N, N], range [-1, 1]

        # Convert back: [-1, 1] → [-π, π]
        phase = phase_norm.cpu().numpy()[0, 0] * np.pi

        return phase.astype(np.float32)

    def reconstruct(
        self,
        diffraction: np.ndarray,
        support: np.ndarray = None,
        mode: str = 'hybrid',
        n_raar: int = 100,
        n_er: int = 20,
        beta: float = 0.87,
        use_shrink_wrap: bool = False,
        progress_cb=None,
    ) -> ReconstructionResult:

        t0 = time.time()
        amplitude = np.sqrt(np.maximum(diffraction, 0)).astype(np.float32)

        if mode == 'nn_only':

            phase = self.predict_phase_nn(diffraction)

            if support is None:
                auto_corr = np.abs(_ifft3(amplitude ** 2))
                support = (auto_corr > 0.04 * auto_corr.max()).astype(np.float32)

            obj = support * np.exp(1j * phase)
            F = _fft3(obj)
            r = _r_factor(amplitude, np.abs(F))

            result = ReconstructionResult(
                object_3d=obj,
                phase=phase * support,
                amplitude=np.abs(obj),
                support=support,
                error_metric=[r],
                method='nn_only',
                elapsed_seconds=time.time() - t0,
                n_iterations=0,
            )

        elif mode == 'hybrid':

            print("  [1/3] NN phase prediction...")
            initial_phase = self.predict_phase_nn(diffraction)

            print(f"  [2/3] RAAR refinement ({n_raar} iterations)...")
            print(f"  [3/3] ER polish ({n_er} iterations)...")

            result = run_phase_retrieval(
                measured_amplitude=amplitude,
                initial_phase=initial_phase,
                support=support,
                n_raar=n_raar,
                n_er=n_er,
                beta=beta,
                use_shrink_wrap=use_shrink_wrap,
                progress_cb=progress_cb,
            )
            result.method = 'hybrid'
            result.elapsed_seconds = time.time() - t0

        elif mode == 'classical':

            # Use more iterations to compensate for random start
            if n_raar < 200:
                n_raar = 500
            if n_er < 50:
                n_er = 100

            print(f"  Classical: {n_raar} RAAR + {n_er} ER (random start)")

            result = run_phase_retrieval(
                measured_amplitude=amplitude,
                initial_phase=None,  # Random start
                support=support,
                n_raar=n_raar,
                n_er=n_er,
                beta=beta,
                use_shrink_wrap=use_shrink_wrap,
                progress_cb=progress_cb,
            )
            result.method = 'classical'
            result.elapsed_seconds = time.time() - t0

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'hybrid', 'nn_only', or 'classical'.")

        print(f"  Done in {result.elapsed_seconds:.2f}s  "
              f"R-factor={result.error_metric[-1]:.4f}")

        return result

    def compare_methods(
        self,
        diffraction: np.ndarray,
        support: np.ndarray = None,
    ) -> dict:

        results = {}

        print("\n── NN-only reconstruction ──")
        results['nn_only'] = self.reconstruct(
            diffraction, support, mode='nn_only'
        )

        print("\n── Hybrid (NN + 100 RAAR + 20 ER) ──")
        results['hybrid'] = self.reconstruct(
            diffraction, support, mode='hybrid',
            n_raar=100, n_er=20,
        )

        print("\n── Classical (random + 500 RAAR + 100 ER) ──")
        results['classical'] = self.reconstruct(
            diffraction, support, mode='classical',
            n_raar=500, n_er=100,
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"{'Method':<15} {'R-factor':>10} {'Time (s)':>10} {'Iterations':>12}")
        print(f"{'-'*60}")
        for name, r in results.items():
            print(
                f"{name:<15} {r.error_metric[-1]:>10.4f} "
                f"{r.elapsed_seconds:>10.2f} {r.n_iterations:>12d}"
            )
        print(f"{'='*60}")

        return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BCDI phase retrieval')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to .npz file with diffraction data')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to trained U-Net checkpoint')
    parser.add_argument('--mode', type=str, default='hybrid',
                        choices=['hybrid', 'nn_only', 'classical', 'compare'],
                        help='Reconstruction mode')
    parser.add_argument('--output', type=str, default='reconstruction.npz',
                        help='Output file')
    parser.add_argument('--n_raar', type=int, default=100)
    parser.add_argument('--n_er', type=int, default=20)
    parser.add_argument('--base_channels', type=int, default=32)
    args = parser.parse_args()

    # Load data
    data = np.load(args.input)
    diffraction = data['diffraction']
    support = data.get('support', None)

    # Create retriever
    model_path = args.model if args.mode != 'classical' else None
    retriever = HybridPhaseRetriever(
        model_path=model_path,
        base_channels=args.base_channels,
    )

    if args.mode == 'compare':
        results = retriever.compare_methods(diffraction, support)
        # Save hybrid result
        r = results['hybrid']
    else:
        r = retriever.reconstruct(
            diffraction, support,
            mode=args.mode,
            n_raar=args.n_raar,
            n_er=args.n_er,
        )

    # Save result
    np.savez_compressed(
        args.output,
        object_real=np.real(r.object_3d),
        object_imag=np.imag(r.object_3d),
        phase=r.phase,
        amplitude=r.amplitude,
        support=r.support,
        error_metric=np.array(r.error_metric),
    )
    print(f"\nSaved to {args.output}")
