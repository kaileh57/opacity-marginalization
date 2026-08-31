"""Forward-model discrepancy ("floor") model for misspecification-robust amortized NPE.

The exoJAX forward model has a line-shape/broadening floor concentrated in the red carbon bands
(CO2 4.0-4.35 um, CO/CH4 4.35-4.80 um): a gold-standard exact fit to a benchmark T8 leaves a
self-normalized residual of rms ~0.3 there (cores over-absorbed, wings under-absorbed), versus ~0
elsewhere. Training the NPE only on clean simulated spectra makes it over-trust those bands on REAL
data -> spurious precision and spurious "detections". This module injects realizations of that floor
into the (self-normalized) training spectra so the network learns the carbon bands are unreliable and
widens its posteriors there appropriately (a simulation-augmentation form of robust SBI / a learned
model-discrepancy term in the Kennedy-O'Hagan sense).

The empirical floor template is the benchmark exact-fit residual (self-normalized), so the injected
discrepancy has the right shape, amplitude, and wavelength localization by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _smooth(x: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Cheap Gaussian smoothing along the last axis (no SciPy dependency)."""
    if sigma_pix <= 0:
        return x
    radius = int(max(1, round(3 * sigma_pix)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_pix) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode="same")


class FloorDiscrepancy:
    """Heteroscedastic carbon-band floor model that produces additive perturbations in
    self-normalized spectrum space.

    A draw is: a random multiple of the systematic template shape (cores over-absorbed) PLUS
    correlated ("line-like") stochastic noise whose per-wavelength amplitude follows the template
    envelope. Both are zero outside the carbon bands by construction (the template is ~0 there).
    """

    def __init__(
        self,
        template: np.ndarray,
        amplitude: float = 1.0,
        prob: float = 0.6,
        corr_len_pix: float = 3.0,
        jitter_frac: float = 0.8,
        env_smooth_pix: float = 5.0,
    ) -> None:
        t = np.asarray(template, dtype=np.float32)
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        self.template = t
        self.length = int(t.shape[0])
        # per-wavelength amplitude envelope of the floor (smoothed |residual|), normalized to unit peak
        env = _smooth(np.abs(t), env_smooth_pix)
        peak = float(env.max()) if env.max() > 0 else 1.0
        self.envelope = (env / peak).astype(np.float32)
        self.amplitude = float(amplitude)
        self.prob = float(prob)
        self.corr_len_pix = float(corr_len_pix)
        self.jitter_frac = float(jitter_frac)

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "FloorDiscrepancy":
        return cls(np.load(Path(path)), **kw)

    def resampled(self, length: int) -> "FloorDiscrepancy":
        """Return a copy with the template/envelope linearly resampled to `length` (handles dataset
        downsampling / a different spectral grid)."""
        if length == self.length:
            return self
        xp = np.linspace(0.0, 1.0, self.length)
        xq = np.linspace(0.0, 1.0, length)
        new = FloorDiscrepancy.__new__(FloorDiscrepancy)
        new.template = np.interp(xq, xp, self.template).astype(np.float32)
        new.envelope = np.interp(xq, xp, self.envelope).astype(np.float32)
        new.length = int(length)
        new.amplitude = self.amplitude
        new.prob = self.prob
        new.corr_len_pix = self.corr_len_pix
        new.jitter_frac = self.jitter_frac
        return new

    def draw_numpy(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return (n, length) additive perturbations (float32). Rows not selected (prob) are zero."""
        L = self.length
        out = np.zeros((n, L), dtype=np.float32)
        sel = rng.random(n) < self.prob
        if not sel.any():
            return out
        m = int(sel.sum())
        # systematic component: random signed multiple of the template shape
        a = rng.normal(0.0, self.amplitude, size=(m, 1)).astype(np.float32)
        systematic = a * self.template[None, :]
        # stochastic component: correlated noise shaped by the floor envelope
        white = rng.standard_normal((m, L)).astype(np.float32)
        radius = int(max(1, round(3 * self.corr_len_pix)))
        k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / self.corr_len_pix) ** 2).astype(np.float32)
        k /= k.sum()
        corr = np.stack([np.convolve(w, k, mode="same") for w in white])
        corr /= (corr.std(axis=1, keepdims=True) + 1e-8)
        stochastic = (self.jitter_frac * self.amplitude) * self.envelope[None, :] * corr
        out[sel] = systematic + stochastic
        return out

    def perturb_batch(self, spectra, rng: np.random.Generator):
        """Add a floor draw to a torch batch of self-normalized spectra, shape (B, 1, L) or (B, L)."""
        if torch is None:
            raise ImportError("PyTorch required for perturb_batch.")
        arr = spectra
        squeeze = arr.dim() == 3
        B = arr.shape[0]
        L = arr.shape[-1]
        model = self if L == self.length else self.resampled(L)
        delta = torch.from_numpy(model.draw_numpy(B, rng)).to(arr.device, arr.dtype)
        if squeeze:
            delta = delta[:, None, :]
        return arr + delta
