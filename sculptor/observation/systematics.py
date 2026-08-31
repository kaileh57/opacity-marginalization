"""Systematics-aware scoring for JWST NIRSpec retrievals.

The forward-model autopsy of WISE J0415-0935 (T8) established that the ExoJAX
forward model carries a localized, irreducible misfit in the NRS2 red detector
(4.0-4.8 um, peaking at 4.0-4.35 um) that survives every physical and
calibration lever tested (CO2 abundance, smooth per-detector continuum, error
floor, finer T-P, per-detector velocity lag). See
``outputs/diagnostics/fm_autopsy/AUTOPSY_SUMMARY.md``.

Because that deficiency is a *forward-model* limitation rather than a data or
NPE-training problem, population-level science (C/O, [M/H]) cannot proceed on a
vanilla chi2 likelihood without either (a) excluding the offending wavelengths
from the likelihood, or (b) inflating the per-detector noise so the residuals
are statistically consistent with the (mis-specified) model. This module
provides a production ``SystematicsModel`` implementing both strategies.

All arithmetic is done in the project's *self-normalized* score space: the
caller supplies the normalized observation ``obs_norm``, its normalized error
``obs_err_norm``, the boolean ``valid_mask`` (the 8888-px PPC mask), the model
flux already normalized to the same space, and a wavelength array (micron) on
the same 10000-length grid. This module is numpy-only and has no dependency on
the forward model or the scorer pyc; the caller does the normalization.

Detector geometry (fixed by the instrument):
  * NRS1 (bluer): 2.87-3.69 um  -> wavelength < ~3.735 um
  * NRS2 (redder): 3.79-5.14 um -> wavelength >= ~3.735 um
  * physical detector gap: 3.686-3.785 um (midpoint 3.7355 um is the split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Physical NRS1/NRS2 detector-gap midpoint (micron). Pixels strictly below this
# wavelength are NRS1, at or above are NRS2.
DETECTOR_SPLIT_UM: float = 3.7355

# Default "bad" window flagged by the autopsy as the dominant, irreducible misfit
# (4.0-4.35 um carries ~45% of the exact-fit chi2 at the highest SNR).
DEFAULT_BAD_WINDOW_UM: tuple[float, float] = (4.00, 4.35)


def detector_labels(wavelength_um: np.ndarray, split_um: float = DETECTOR_SPLIT_UM) -> np.ndarray:
    """Return an array of detector labels ('NRS1'/'NRS2') for each pixel.

    Parameters
    ----------
    wavelength_um : np.ndarray
        Per-pixel wavelength in micron (10000-length grid).
    split_um : float
        Boundary wavelength; pixels below are NRS1, at/above are NRS2.

    Returns
    -------
    np.ndarray of dtype '<U4'
        Per-pixel detector label.
    """
    wl = np.asarray(wavelength_um, dtype=np.float64)
    out = np.where(wl < split_um, "NRS1", "NRS2")
    return out.astype("<U4")


@dataclass
class SystematicsResult:
    """Outcome of applying a :class:`SystematicsModel` to one set of residuals."""

    mode: str
    chi2: float
    n_eff: int
    n_params: int
    chi2_per_dof: float
    # 'mask' mode populates dropped_pixels; 'inflate' mode populates inflation.
    dropped_pixels: int = 0
    inflation: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "chi2": float(self.chi2),
            "n_eff": int(self.n_eff),
            "n_params": int(self.n_params),
            "chi2_per_dof": float(self.chi2_per_dof),
            "dropped_pixels": int(self.dropped_pixels),
            "inflation": {k: float(v) for k, v in self.inflation.items()},
            "extra": {k: float(v) for k, v in self.extra.items()},
        }


class SystematicsModel:
    """Systematics-aware likelihood/score for a self-normalized spectrum.

    Two production modes:

    ``'mask'``
        Exclude configurable wavelength windows from the likelihood. The default
        windows are the autopsy-flagged 4.0-4.35 um band; passing
        ``mask_all_nrs2=True`` drops the entire NRS2 detector instead.

    ``'inflate'``
        Fit per-detector multiplicative error-inflation factors ``s_NRS1`` and
        ``s_NRS2`` by maximizing the standard-normal log-likelihood of the
        residuals. For a fixed detector with residuals ``r = obs - model`` and
        nominal errors ``sigma``, the per-detector MLE of ``s`` that makes
        ``r / (s * sigma)`` unit-variance is the closed form

            s_hat = sqrt( mean[ (r / sigma)^2 ] )

        i.e. the RMS of the nominal (whitened) residuals. The inflated chi2/dof
        within each detector is then ~1 by construction; the cross-detector
        chi2/dof reports how much the joint fit is dominated by the inflated arm.

    Parameters
    ----------
    mode : {'mask', 'inflate'}
    bad_windows_um : sequence of (lo, hi), optional
        Wavelength windows (micron) to exclude in 'mask' mode. Defaults to
        ``[DEFAULT_BAD_WINDOW_UM]``. Ignored if ``mask_all_nrs2`` is True.
    mask_all_nrs2 : bool
        If True (mask mode), drop every NRS2 pixel instead of the windows.
    split_um : float
        NRS1/NRS2 boundary wavelength (micron).
    min_inflation : float
        Floor for fitted inflation factors (factors below 1 are clipped to 1 so
        the model never *deflates* nominal errors; a well-fit arm keeps s=1).
    n_fit_params : int
        Number of free model parameters subtracted from the pixel count when
        forming degrees of freedom (matches the project scorer's dof = N - 9).
    """

    _VALID_MODES = ("mask", "inflate")

    def __init__(
        self,
        mode: str,
        *,
        bad_windows_um: Sequence[tuple[float, float]] | None = None,
        mask_all_nrs2: bool = False,
        split_um: float = DETECTOR_SPLIT_UM,
        min_inflation: float = 1.0,
        n_fit_params: int = 9,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.bad_windows_um = (
            [DEFAULT_BAD_WINDOW_UM] if bad_windows_um is None else [tuple(w) for w in bad_windows_um]
        )
        self.mask_all_nrs2 = bool(mask_all_nrs2)
        self.split_um = float(split_um)
        self.min_inflation = float(min_inflation)
        self.n_fit_params = int(n_fit_params)

    # -- masks -----------------------------------------------------------------

    def window_drop_mask(self, wavelength_um: np.ndarray) -> np.ndarray:
        """Boolean array, True where a pixel falls inside a configured drop region.

        In ``mask_all_nrs2`` mode this is the NRS2 detector; otherwise it is the
        union of the configured ``bad_windows_um``.
        """
        wl = np.asarray(wavelength_um, dtype=np.float64)
        if self.mask_all_nrs2:
            return wl >= self.split_um
        drop = np.zeros(wl.shape, dtype=bool)
        for lo, hi in self.bad_windows_um:
            drop |= (wl >= lo) & (wl < hi)
        return drop

    # -- inflation fit ---------------------------------------------------------

    @staticmethod
    def _fit_inflation_factor(resid: np.ndarray, sigma: np.ndarray, floor: float) -> float:
        """MLE inflation factor s = RMS of (resid / sigma), clipped at ``floor``.

        Maximizing sum_i [ -log(s*sigma_i) - 0.5*(r_i/(s*sigma_i))^2 ] over s>0
        gives s_hat^2 = mean[(r_i/sigma_i)^2]. Returns max(s_hat, floor).
        """
        if resid.size == 0:
            return floor
        whitened = resid / sigma
        s_hat = float(np.sqrt(np.mean(whitened ** 2)))
        return max(s_hat, floor)

    def fit(
        self,
        obs_norm: np.ndarray,
        model_norm: np.ndarray,
        obs_err_norm: np.ndarray,
        valid_mask: np.ndarray,
        wavelength_um: np.ndarray,
    ) -> SystematicsResult:
        """Apply the systematics model and return the rescored chi2 summary.

        Parameters
        ----------
        obs_norm, model_norm, obs_err_norm : np.ndarray
            Self-normalized observation, model flux, and observation error on the
            full 10000-length grid.
        valid_mask : np.ndarray of bool
            The PPC validity mask (True for the 8888 usable pixels).
        wavelength_um : np.ndarray
            Per-pixel wavelength (micron) on the same grid.

        Returns
        -------
        SystematicsResult
        """
        obs = np.asarray(obs_norm, dtype=np.float64)
        model = np.asarray(model_norm, dtype=np.float64)
        sigma = np.asarray(obs_err_norm, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        wl = np.asarray(wavelength_um, dtype=np.float64)

        resid = obs - model

        if self.mode == "mask":
            drop = self.window_drop_mask(wl)
            keep = valid & ~drop
            chi2 = float(np.sum((resid[keep] / sigma[keep]) ** 2))
            n_eff = int(keep.sum())
            dof = max(n_eff - self.n_fit_params, 1)
            dropped = int((valid & drop).sum())
            return SystematicsResult(
                mode="mask",
                chi2=chi2,
                n_eff=n_eff,
                n_params=self.n_fit_params,
                chi2_per_dof=chi2 / dof,
                dropped_pixels=dropped,
            )

        # inflate mode -- per-detector multiplicative error inflation
        labels = detector_labels(wl, self.split_um)
        inflation: dict[str, float] = {}
        chi2 = 0.0
        n_eff = int(valid.sum())
        for det in ("NRS1", "NRS2"):
            sel = valid & (labels == det)
            s_det = self._fit_inflation_factor(resid[sel], sigma[sel], self.min_inflation)
            inflation[det] = s_det
            if sel.any():
                chi2 += float(np.sum((resid[sel] / (s_det * sigma[sel])) ** 2))
        dof = max(n_eff - self.n_fit_params, 1)
        # per-detector chi2/dof after inflation (diagnostic; ~1 within each arm
        # up to the dof correction)
        extra: dict[str, float] = {}
        for det in ("NRS1", "NRS2"):
            sel = valid & (labels == det)
            n_det = int(sel.sum())
            if n_det:
                s_det = inflation[det]
                c = float(np.sum((resid[sel] / (s_det * sigma[sel])) ** 2))
                extra[f"chi2_per_px_{det}_inflated"] = c / n_det
        return SystematicsResult(
            mode="inflate",
            chi2=chi2,
            n_eff=n_eff,
            n_params=self.n_fit_params,
            chi2_per_dof=chi2 / dof,
            inflation=inflation,
            extra=extra,
        )

    def fit_cube(
        self,
        obs_norm: np.ndarray,
        model_cube_norm: np.ndarray,
        obs_err_norm: np.ndarray,
        valid_mask: np.ndarray,
        wavelength_um: np.ndarray,
        *,
        reduce: str = "median",
        dof_offset: int | None = None,
    ) -> SystematicsResult:
        """Apply the model to a posterior-predictive cube of normalized spectra.

        The point estimate per pixel is the **median** spectrum by default, which
        reproduces the project scorer ``score_ppc`` exactly (it forms
        ``p50 = nanpercentile(model, 50, axis=0)`` then
        ``chi2/dof = sum_px[((obs-p50)/sigma)^2] / max(N_valid - 1, 1)``).
        Pass ``reduce='mean'`` for a sample-averaged per-pixel chi2 instead.

        Parameters
        ----------
        model_cube_norm : np.ndarray, shape (n_samples, n_pixels)
            Self-normalized posterior-sample spectra.
        reduce : {'median', 'mean'}
            How to collapse the sample axis into a per-pixel chi2. 'median'
            (default) uses the median spectrum's standardized residual squared,
            matching ``score_ppc``. 'mean' averages the squared standardized
            residual over samples.
        dof_offset : int, optional
            Degrees-of-freedom subtracted from the surviving pixel count. Defaults
            to ``1`` to match ``score_ppc`` (which uses N_valid - 1); pass
            ``self.n_fit_params`` to match the per-spectrum convention.
        """
        obs = np.asarray(obs_norm, dtype=np.float64)
        cube = np.asarray(model_cube_norm, dtype=np.float64)
        sigma = np.asarray(obs_err_norm, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        wl = np.asarray(wavelength_um, dtype=np.float64)
        dof_off = 1 if dof_offset is None else int(dof_offset)

        if reduce == "median":
            p50 = np.nanpercentile(cube, 50, axis=0)
            valid = valid & np.isfinite(p50)
            px_chi2 = ((obs - p50) / sigma) ** 2  # per-pixel standardized resid^2
        elif reduce == "mean":
            whitened = (obs[None, :] - cube) / sigma[None, :]
            px_chi2 = np.mean(whitened ** 2, axis=0)
        else:
            raise ValueError(f"reduce must be 'median' or 'mean', got {reduce!r}")

        if self.mode == "mask":
            drop = self.window_drop_mask(wl)
            keep = valid & ~drop
            chi2 = float(np.nansum(px_chi2[keep]))
            n_eff = int(keep.sum())
            dof = max(n_eff - dof_off, 1)
            dropped = int((valid & drop).sum())
            return SystematicsResult(
                mode="mask",
                chi2=chi2,
                n_eff=n_eff,
                n_params=dof_off,
                chi2_per_dof=chi2 / dof,
                dropped_pixels=dropped,
            )

        labels = detector_labels(wl, self.split_um)
        inflation: dict[str, float] = {}
        chi2 = 0.0
        n_eff = int(valid.sum())
        extra: dict[str, float] = {}
        for det in ("NRS1", "NRS2"):
            sel = valid & (labels == det)
            n_det = int(sel.sum())
            if not n_det:
                inflation[det] = self.min_inflation
                continue
            # s^2 = mean over pixels of the per-pixel whitened variance
            s_hat = float(np.sqrt(np.mean(px_chi2[sel])))
            s_det = max(s_hat, self.min_inflation)
            inflation[det] = s_det
            chi2 += float(np.nansum(px_chi2[sel]) / s_det ** 2)
            extra[f"chi2_per_px_{det}_inflated"] = float(np.nansum(px_chi2[sel]) / s_det ** 2) / n_det
        dof = max(n_eff - dof_off, 1)
        return SystematicsResult(
            mode="inflate",
            chi2=chi2,
            n_eff=n_eff,
            n_params=dof_off,
            chi2_per_dof=chi2 / dof,
            inflation=inflation,
            extra=extra,
        )
