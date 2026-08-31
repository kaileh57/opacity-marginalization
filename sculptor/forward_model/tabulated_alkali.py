"""Tabulated atomic-alkali opacity operator (Allard Na/K resonance wings).

exojax provides atomic line databases (AdbVald/AdbKurucz) but ONLY Voigt/Lorentzian pressure
broadening (Unsoeld, Kurucz-Avrett, vdW) - it has NO Allard non-Voigt resonance-wing treatment,
which is physically required for the Na D / K I 0.77um doublets whose H2-broadened far wings shape
the L/early-T near-IR continuum (and thus the gravity/radius inference the binary screen depends on).

This operator ingests a pre-computed Allard cross-section table (built by
scripts/build_alkali_opacity_table.py from petitRADTRANS's 23Na__NewAllard / 39K__Allard) already
resampled onto the track nu grid, and exposes the SAME interface as an exojax OpaPremodit operator:
`xsmatrix(temperatures_K, pressures_bar) -> (n_layers, n_nu)` cross-sections in cm^2/molecule. Because
ExoJAXOpacityBackend.compute_cross_sections_jax dispatches on `hasattr(op, "xsmatrix")`, this plugs in
with no change to the compute loop; the RT weights it by the atomic Na/K number density from chemistry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class TabulatedAlkaliOpa:
    """Bilinear (T, log10 P) interpolation of a pre-tabulated alkali cross-section onto each layer."""

    def __init__(self, npz_path: str | Path) -> None:
        data = np.load(str(npz_path), allow_pickle=True)
        self.nu_grid = np.asarray(data["nu_grid_cm1"], dtype=np.float64)
        self.t_grid = np.asarray(data["t_grid_K"], dtype=np.float64)          # (n_t,) ascending
        self.logp_grid = np.asarray(data["logp_grid_bar"], dtype=np.float64)  # (n_p,) ascending
        self.xsec = np.asarray(data["xsec_cm2"], dtype=np.float32)            # (n_t, n_p, n_nu) cm^2/molecule
        self.species = str(data["species"]) if "species" in data else "?"
        self.mol_mass_amu = float(data["mol_mass_amu"]) if "mol_mass_amu" in data else 0.0
        if self.t_grid[0] > self.t_grid[-1]:
            self.t_grid = self.t_grid[::-1]; self.xsec = self.xsec[::-1]
        if self.logp_grid[0] > self.logp_grid[-1]:
            self.logp_grid = self.logp_grid[::-1]; self.xsec = self.xsec[:, ::-1]
        self._jax_ready = False

    def _ensure_jax(self):
        if self._jax_ready:
            return
        import jax.numpy as jnp
        self._t = jnp.asarray(self.t_grid, dtype=jnp.float32)
        self._lp = jnp.asarray(self.logp_grid, dtype=jnp.float32)
        self._xs = jnp.asarray(self.xsec, dtype=jnp.float32)   # (n_t, n_p, n_nu)
        self._jax_ready = True

    def xsmatrix(self, temperatures_K, pressures_bar):
        """Return (n_layers, n_nu) cross-sections [cm^2/molecule] for the per-layer (T, P)."""
        import jax.numpy as jnp
        self._ensure_jax()
        T = jnp.asarray(temperatures_K, dtype=jnp.float32)
        logP = jnp.log10(jnp.clip(jnp.asarray(pressures_bar, dtype=jnp.float32), 1e-30, None))
        t = self._t; lp = self._lp; xs = self._xs
        nt = t.shape[0]; npp = lp.shape[0]
        # bracketing indices + clamped fractional weights (constant extrapolation outside grid)
        ti = jnp.clip(jnp.searchsorted(t, T) - 1, 0, nt - 2)
        pj = jnp.clip(jnp.searchsorted(lp, logP) - 1, 0, npp - 2)
        wt = jnp.clip((T - t[ti]) / (t[ti + 1] - t[ti]), 0.0, 1.0)[:, None]
        wp = jnp.clip((logP - lp[pj]) / (lp[pj + 1] - lp[pj]), 0.0, 1.0)[:, None]
        c00 = xs[ti, pj]; c10 = xs[ti + 1, pj]; c01 = xs[ti, pj + 1]; c11 = xs[ti + 1, pj + 1]
        out = (1 - wt) * (1 - wp) * c00 + wt * (1 - wp) * c10 + (1 - wt) * wp * c01 + wt * wp * c11
        return jnp.clip(out, 0.0, None)
