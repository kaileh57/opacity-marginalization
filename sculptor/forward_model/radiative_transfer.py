"""Emission radiative-transfer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sculptor.forward_model.chemistry import PROTON_MASS_G, ChemistryState
from sculptor.forward_model.opacity import OpacityResult


RJUP_CM = 7.1492e9
PC_CM = 3.085677581e18


@dataclass(frozen=True)
class RadiativeTransferResult:
    """Outputs from the emission solver."""

    flux_nu: np.ndarray
    total_optical_depth: np.ndarray
    layer_optical_depth: np.ndarray
    metadata: dict[str, Any]
class EmissionRadiativeTransfer:
    """Compute an emergent emission spectrum using ExoJAX ArtEmisPure."""

    def __init__(self, rtsolver: str = "ibased", nstream: int = 8) -> None:
        self.rtsolver = rtsolver
        self.nstream = nstream
        self._artemis = None
        self._artemis_key: tuple[float, float, int, int, float, float] | None = None

    def _get_artemis(self, nu_grid_cm1: np.ndarray, pressures_bar: np.ndarray):
        try:
            from exojax.rt import ArtEmisPure
        except Exception as exc:
            raise ImportError("ExoJAX is required for the emission radiative-transfer backend.") from exc

        pressure_top = float(np.min(pressures_bar))
        pressure_btm = float(np.max(pressures_bar))
        nlayer = int(len(pressures_bar))
        key = (pressure_top, pressure_btm, nlayer, int(len(nu_grid_cm1)), float(nu_grid_cm1[0]), float(nu_grid_cm1[-1]))
        if self._artemis is None or self._artemis_key != key:
            self._artemis = ArtEmisPure(
                nu_grid=nu_grid_cm1,
                pressure_btm=pressure_btm,
                pressure_top=pressure_top,
                nlayer=nlayer,
                rtsolver=self.rtsolver,
                nstream=self.nstream,
            )
            self._artemis_key = key
        return self._artemis

    def compute_flux(
        self,
        chemistry_state: ChemistryState,
        opacity_result: OpacityResult,
        gravity_cgs: float,
        radius_rjup: float,
        distance_pc: float,
        cloud_parameters: dict | None = None,
    ) -> RadiativeTransferResult:
        pressures_bar = np.asarray(chemistry_state.pressures_bar, dtype=np.float64)
        delta_pressure_bar = np.gradient(pressures_bar)
        delta_pressure_cgs = np.abs(delta_pressure_bar) * 1.0e6
        number_column = delta_pressure_cgs[:, None] / (
            gravity_cgs * chemistry_state.mean_molecular_weight[:, None] * PROTON_MASS_G
        )

        tau_layers = np.zeros((len(pressures_bar), len(opacity_result.nu_grid_cm1)), dtype=np.float64)
        for species, cross_section in opacity_result.species_cross_sections.items():
            abundance = chemistry_state.vmr_for(species)[:, None]
            tau_layers += np.asarray(cross_section, dtype=np.float64) * abundance * number_column

        if cloud_parameters is not None and cloud_parameters.get("cloud_model", "off") != "off":
            from sculptor.forward_model.clouds import compute_cloud_tau
            tau_layers += compute_cloud_tau(
                pressures_bar=pressures_bar,
                temperatures_K=np.asarray(chemistry_state.temperatures_K, dtype=np.float64),
                mean_molecular_weight=np.asarray(chemistry_state.mean_molecular_weight, dtype=np.float64),
                gravity_cgs=gravity_cgs,
                metallicity=float(cloud_parameters["metallicity"]),
                log_kzz=float(cloud_parameters["log_kzz"]),
                f_sed=float(cloud_parameters.get("f_sed", 8.0)),
                sigma_c=float(cloud_parameters.get("sigma_c", 2.0)),
                nu_grid_cm1=opacity_result.nu_grid_cm1,
                condensates=tuple(cloud_parameters.get("condensates", ("MgSiO3",))),
                backend=cloud_parameters.get("cloud_backend", "gray"),
            )

        art = self._get_artemis(opacity_result.nu_grid_cm1, pressures_bar)
        try:
            art_pressure = np.asarray(art.pressure)
        except Exception:
            art_pressure = None
        if art_pressure is not None and not np.allclose(art_pressure, pressures_bar, rtol=1.0e-3, atol=0.0):
            raise ValueError("ArtEmisPure pressure grid does not match chemistry pressure grid.")

        try:
            import jax.numpy as jnp
        except Exception as exc:
            raise ImportError("JAX is required for the ExoJAX radiative-transfer backend.") from exc

        flux_top = np.asarray(art.run(jnp.asarray(tau_layers), jnp.asarray(chemistry_state.temperatures_K)))
        scaling = (radius_rjup * RJUP_CM / (distance_pc * PC_CM)) ** 2
        flux_nu = flux_top * scaling
        total_tau = np.sum(tau_layers, axis=0)
        return RadiativeTransferResult(
            flux_nu=flux_nu,
            total_optical_depth=total_tau,
            layer_optical_depth=tau_layers,
            metadata={"gravity_cgs": gravity_cgs, "distance_pc": distance_pc, "rtsolver": self.rtsolver},
        )
