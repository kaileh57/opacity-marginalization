"""Batch spectrum generation with an optional TPU-friendly interface."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

import numpy as np

from sculptor.data_generation.chemistry_precompute import PrecomputedAtmosphere
from sculptor.data_generation.opacity_perturbation import (
    OpacityPerturbationSettings,
    extract_coefficient_matrix,
    fourier_basis,
    settings_from_env,
)
from sculptor.forward_model.chemistry import PROTON_MASS_G
from sculptor.forward_model.radiative_transfer import PC_CM, RJUP_CM
from sculptor.forward_model.spectrum import SpectrumModel


@dataclass(frozen=True)
class GeneratedBatch:
    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    spectra_clean: np.ndarray
    vmr_profiles: np.ndarray
    nu_grid_cm1: np.ndarray

    def select_rows(self, row_mask: np.ndarray) -> "GeneratedBatch":
        return GeneratedBatch(
            parameters=self.parameters[row_mask],
            parameter_names=self.parameter_names,
            spectra_clean=self.spectra_clean[row_mask],
            vmr_profiles=self.vmr_profiles[row_mask],
            nu_grid_cm1=self.nu_grid_cm1,
        )


def _rv_shift_jax(nu_grid, flux, rv_km_s):
    """JAX Doppler shift onto the fixed grid (c=299792.458 km/s)."""
    import jax.numpy as jnp
    shifted = nu_grid * (1.0 + rv_km_s / 299792.458)
    return jnp.interp(shifted, nu_grid, flux, left=flux[0], right=flux[-1])


def _vsini_kernel_jax(nu_grid, vsini_km_s, max_half, epsilon=0.3):
    """Fixed-length (static max_half) rotational kernel on an ESLOG grid; JAX/vmap-safe.
    Below one pixel -> delta (identity). Linear limb darkening (Gray 2008)."""
    import jax.numpy as jnp
    dv = 299792.458 * jnp.abs(jnp.log(nu_grid[1]) - jnp.log(nu_grid[0]))
    idx = jnp.arange(-max_half, max_half + 1)
    v = idx * dv
    x = jnp.clip(v / jnp.clip(vsini_km_s, 1e-3, None), -1.0, 1.0)
    mu = jnp.sqrt(jnp.clip(1.0 - x ** 2, 0.0, None))
    k = (2.0 * (1.0 - epsilon) * mu + 0.5 * jnp.pi * epsilon * mu ** 2)
    k = jnp.where(jnp.abs(v) <= jnp.maximum(vsini_km_s, dv), k, 0.0)
    k = jnp.where(vsini_km_s < dv, (idx == 0).astype(k.dtype), k)
    return k / jnp.clip(jnp.sum(k), 1e-12, None)


def compile_vmapped_spectrum_function(single_spectrum, n_args: int = 5):
    try:
        import jax
    except Exception as exc:
        raise ImportError("JAX is required for batched spectrum generation.") from exc
    return jax.jit(jax.vmap(single_spectrum, in_axes=tuple([0] * n_args)))


def compile_scalar_spectrum_function(single_spectrum, n_args: int = 5):
    try:
        import jax  # noqa: F401 — validate JAX is available
    except Exception as exc:
        raise ImportError("JAX is required for batched spectrum generation.") from exc
    # Return single_spectrum directly (no JIT wrapper).  The tau_layers
    # materialization inside single_spectrum breaks graph fusion before art.run,
    # so art.run compiles a small independent HLO without embedding the large
    # OpaPremodit constants.
    return single_spectrum


class SpectrumBatchGenerator:
    """Generate spectra in batches. The default implementation is CPU-safe."""

    def __init__(
        self,
        spectrum_model: SpectrumModel,
        device_batch_size: int | None = None,
        perturbation: OpacityPerturbationSettings | None = None,
    ) -> None:
        self.spectrum_model = spectrum_model
        configured_batch_size = (
            spectrum_model.track_config.tpu_batch_size if device_batch_size is None else device_batch_size
        )
        self.device_batch_size = max(int(configured_batch_size), 1)
        # Opacity-marginalization (SCULPTOR_OPACITY_PERTURB=1): per-draw smooth
        # multiplicative fields on the cached cross-sections. Disabled (the
        # default) leaves the compiled spectrum function bit-identical to the
        # unperturbed pipeline.
        if perturbation is None:
            perturbation = settings_from_env(getattr(spectrum_model.track_config, "molecules", None))
        self.perturbation = perturbation
        # Compiled-function cache keyed by the full compile key: perturbed runs
        # cycle line-list combos across shards, so a single-slot cache would
        # recompile every shard on a single-host (validation) run.
        self._compiled_batch_functions: dict[tuple[Any, ...], Callable[..., Any]] = {}

    def _perturbed_species(self) -> tuple[str, ...]:
        """Perturbed species, in settings order, restricted to this track's line absorbers."""
        if not self.perturbation.enabled:
            return ()
        database_names = {
            database.name for database in self.spectrum_model.track_config.opacity_databases
        }
        return tuple(name for name in self.perturbation.species if name in database_names)

    def _compile_batch_function(self, batch: PrecomputedAtmosphere) -> Callable[[Any, Any, Any, Any, Any], Any]:
        try:
            import jax.numpy as jnp
        except Exception as exc:
            raise ImportError("JAX is required for batched spectrum generation.") from exc

        opacity_backend = self.spectrum_model.opacity_backend
        print(
            json.dumps(
                {
                    "message": "batch compile start",
                    "device_batch_size": self.device_batch_size,
                    "n_rows": int(len(batch.parameters)),
                    "n_layers": int(len(batch.pressure_grid_bar)),
                    "n_wavenumber": int(self.spectrum_model.track_config.n_wavenumber),
                }
            ),
            flush=True,
        )
        opacity_backend.prepare()
        print(json.dumps({"message": "batch compile after opacity prepare"}), flush=True)

        radiative_transfer = self.spectrum_model.radiative_transfer
        instrument_model = self.spectrum_model.instrument_model
        pressure_grid_bar = np.asarray(batch.pressure_grid_bar, dtype=np.float32)
        pressure_grid_jax = jnp.asarray(pressure_grid_bar)
        delta_pressure_cgs = jnp.asarray(np.abs(np.gradient(pressure_grid_bar)) * 1.0e6, dtype=jnp.float32)
        convolution_kernel = jnp.asarray(instrument_model.convolution_kernel(), dtype=jnp.float32)
        variable_convolution_indices = None
        variable_convolution_weights = None
        if instrument_model.uses_variable_resolution():
            gather_indices_np, kernel_weights_np = instrument_model.variable_convolution_plan()
            variable_convolution_indices = jnp.asarray(gather_indices_np, dtype=jnp.int32)
            variable_convolution_weights = jnp.asarray(kernel_weights_np, dtype=jnp.float32)
        opacity_species = tuple(database.name for database in self.spectrum_model.track_config.opacity_databases)
        species_to_index = {name: index for index, name in enumerate(batch.species)}
        art = radiative_transfer._get_artemis(self.spectrum_model.track_config.nu_grid_cm1, pressure_grid_bar)
        print(json.dumps({"message": "batch compile after artemis init"}), flush=True)
        distance_pc = float(self.spectrum_model.track_config.default_distance_pc)
        n_wavenumber = int(self.spectrum_model.track_config.n_wavenumber)

        cloud_on = self.spectrum_model.track_config.cloud_model != "off"
        nu_grid_jax = jnp.asarray(self.spectrum_model.track_config.nu_grid_cm1, dtype=jnp.float32)
        VSINI_MAX_HALF = 8  # static kernel half-width (covers vsini<=30 km/s at G395H ESLOG dv~19.5)
        if cloud_on:
            from sculptor.forward_model.clouds import compute_cloud_tau_jax

        # Opacity-marginalization field (SCULPTOR_OPACITY_PERTURB=1): per-draw
        # smooth multiplicative field exp(coeffs @ basis) on each molecular
        # species' cross-sections, in normalized log-nu (= pixel index on the
        # ESLOG grid). Trace-time constants only; with perturb_on=False the
        # traced function is IDENTICAL to the original 14-arg path.
        perturb_on = self.perturbation.enabled
        perturbed_species = self._perturbed_species()
        perturbed_index = {name: index for index, name in enumerate(perturbed_species)}
        field_basis = (
            jnp.asarray(fourier_basis(n_wavenumber, self.perturbation.n_modes), dtype=jnp.float32)
            if perturb_on
            else None
        )

        # Capture static operators, the RT solver, and the convolution kernel outside vmap.
        # 14-arg signature (cloud + broadening); for cloud_off + rv=vsini=0 this is identical to the
        # original clear path (broadening reduces to identity), preserving backward compatibility.
        # With perturbation on, a 15th argument carries the per-draw field coefficients
        # (n_perturbed_species, 1 + 2*n_modes).
        def single_spectrum(
            vmr_profile, temperatures_K, mean_molecular_weight, gravity_cgs, radius_rjup,
            metallicity, f_sed, sigma_c, rv_km_s, vsini_km_s, cond_rho, cond_a, cond_b, cond_zmmr,
            oppert_coeffs=None,
        ) -> Any:
            cross_sections = opacity_backend.compute_cross_sections_jax(temperatures_K, pressure_grid_jax)
            number_column = delta_pressure_cgs[:, None] / (
                jnp.maximum(gravity_cgs, 1.0e-12) * mean_molecular_weight[:, None] * PROTON_MASS_G
            )
            if perturb_on:
                # (n_perturbed_species, n_wavenumber) multiplicative fields
                perturbation_fields = jnp.exp(oppert_coeffs @ field_basis)
            tau_layers = jnp.zeros((pressure_grid_jax.shape[0], n_wavenumber), dtype=jnp.float32)
            for species_name in opacity_species:
                if species_name not in species_to_index or species_name not in cross_sections:
                    continue
                abundance = vmr_profile[:, species_to_index[species_name]][:, None]
                species_cross_section = jnp.asarray(cross_sections[species_name], dtype=jnp.float32)
                if perturb_on and species_name in perturbed_index:
                    species_cross_section = (
                        species_cross_section * perturbation_fields[perturbed_index[species_name]][None, :]
                    )
                tau_layers = tau_layers + species_cross_section * abundance * number_column
            if cloud_on:
                tau_layers = tau_layers + jnp.asarray(compute_cloud_tau_jax(
                    pressure_grid_jax, temperatures_K, mean_molecular_weight, gravity_cgs,
                    metallicity, 0.0, f_sed, sigma_c, nu_grid_jax,
                    cond_rho, cond_a, cond_b, cond_zmmr), dtype=jnp.float32)
            flux_top = jnp.asarray(art.run(tau_layers, temperatures_K), dtype=jnp.float32)
            scaling = (radius_rjup * RJUP_CM / (distance_pc * PC_CM)) ** 2
            flux_scaled = flux_top * scaling
            # rotational broadening (identity at vsini<1 pixel) then radial velocity, before LSF
            kern = _vsini_kernel_jax(nu_grid_jax, vsini_km_s, VSINI_MAX_HALF)
            flux_scaled = jnp.convolve(flux_scaled, kern, mode="same")
            flux_scaled = _rv_shift_jax(nu_grid_jax, flux_scaled, rv_km_s)
            if variable_convolution_indices is not None and variable_convolution_weights is not None:
                gathered_flux = jnp.take(flux_scaled, variable_convolution_indices, mode="clip")
                return jnp.sum(gathered_flux * variable_convolution_weights, axis=1)
            return jnp.convolve(flux_scaled, convolution_kernel, mode="same")

        n_args = 15 if perturb_on else 14
        if self.device_batch_size == 1:
            compiled = compile_scalar_spectrum_function(single_spectrum, n_args=n_args)
            print(json.dumps({"message": "batch compile done", "mode": "scalar", "cloud_on": cloud_on,
                              "opacity_perturb": perturb_on}), flush=True)
            return compiled
        compiled = compile_vmapped_spectrum_function(single_spectrum, n_args=n_args)
        print(json.dumps({"message": "batch compile done", "mode": "vmapped", "cloud_on": cloud_on,
                          "opacity_perturb": perturb_on}), flush=True)
        return compiled

    def _get_batch_function(self, batch: PrecomputedAtmosphere) -> Callable[[Any, Any, Any, Any, Any], Any]:
        key = (
            type(self.spectrum_model.opacity_backend).__name__,
            tuple(batch.species),
            tuple(database.name for database in self.spectrum_model.track_config.opacity_databases),
            len(batch.pressure_grid_bar),
            int(self.spectrum_model.track_config.n_wavenumber),
            self.device_batch_size,
            self.spectrum_model.track_config.cloud_model,
            # Opacity-marginalization: the field changes the traced graph and the
            # active line-list variants are embedded as trace-time constants.
            self.perturbation.enabled,
            self.perturbation.n_modes,
            self._perturbed_species(),
            getattr(self.spectrum_model.opacity_backend, "linelist_selection_key", ()),
        )
        if key not in self._compiled_batch_functions:
            self._compiled_batch_functions[key] = self._compile_batch_function(batch)
        return self._compiled_batch_functions[key]

    def _cloud_arrays(self, batch: PrecomputedAtmosphere):
        """Per-row cloud/broadening args from the parameter table (no precompute changes needed).
        Missing params default to clear/no-broadening so old param sets reproduce clear spectra."""
        from sculptor.forward_model.clouds import select_condensate, CONDENSATE_CONSTS
        names = list(batch.parameter_names)
        n = len(batch.parameters)

        def col(name, default):
            if name in names:
                return batch.parameters[:, names.index(name)].astype(np.float32)
            return np.full(n, default, dtype=np.float32)

        metallicity = col("metallicity", 0.0)
        f_sed = col("f_sed", 8.0)
        sigma_c = col("sigma_c", 2.0)
        rv = col("rv_km_s", 0.0)
        vsini = col("vsini_km_s", 0.0)
        # condensate constants per row from a Teff proxy (deep/hot end of the T-P profile)
        if batch.temperature_profiles_K.shape[0] == n and n > 0:
            teff_proxy = np.max(batch.temperature_profiles_K, axis=1)
        else:
            teff_proxy = np.full(n, 900.0)
        rho = np.empty(n, np.float32); a = np.empty(n, np.float32)
        b = np.empty(n, np.float32); z = np.empty(n, np.float32)
        for i in range(n):
            cr, ca, cb, cz = CONDENSATE_CONSTS[select_condensate(float(teff_proxy[i]))]
            rho[i], a[i], b[i], z[i] = cr, ca, cb, cz
        return metallicity, f_sed, sigma_c, rv, vsini, rho, a, b, z

    def _generate_scalar(self, batch: PrecomputedAtmosphere) -> GeneratedBatch:
        clean_spectra = []
        for row, profile, chemistry_state in zip(batch.parameters, batch.profiles, batch.chemistry_states):
            spectrum = self.spectrum_model.generate_from_state(row, profile, chemistry_state)
            clean_spectra.append(spectrum.flux_nu)
        clean_array = np.asarray(clean_spectra, dtype=np.float32)
        return GeneratedBatch(
            parameters=batch.parameters,
            parameter_names=batch.parameter_names,
            spectra_clean=clean_array,
            vmr_profiles=batch.vmr_profiles.astype(np.float32),
            nu_grid_cm1=self.spectrum_model.track_config.nu_grid_cm1.astype(np.float32),
        )

    def generate(self, batch: PrecomputedAtmosphere) -> GeneratedBatch:
        if len(batch.parameters) == 0:
            return GeneratedBatch(
                parameters=batch.parameters,
                parameter_names=batch.parameter_names,
                spectra_clean=np.empty((0, self.spectrum_model.track_config.n_wavenumber), dtype=np.float32),
                vmr_profiles=batch.vmr_profiles.astype(np.float32),
                nu_grid_cm1=self.spectrum_model.track_config.nu_grid_cm1.astype(np.float32),
            )

        try:
            batch_function = self._get_batch_function(batch)
            mh, fsed, sigc, rv, vsini, c_rho, c_a, c_b, c_z = self._cloud_arrays(batch)
            # Opacity-marginalization: gather the per-row field coefficients from
            # the nuisance parameter columns (fails loudly if the flag is on but
            # the table was built without the oppert_* columns).
            oppert = None
            if self.perturbation.enabled:
                oppert = extract_coefficient_matrix(
                    batch.parameters,
                    batch.parameter_names,
                    self.perturbation,
                    species_subset=self._perturbed_species(),
                )
            if self.device_batch_size == 1:
                clean_rows = []
                for row_index in range(len(batch.parameters)):
                    r = row_index
                    row_args = [
                        batch.vmr_profiles[r].astype(np.float32),
                        batch.temperature_profiles_K[r].astype(np.float32),
                        batch.mean_molecular_weight_profiles[r].astype(np.float32),
                        np.float32(batch.gravity_cgs[r]),
                        np.float32(batch.radius_rjup[r]),
                        np.float32(mh[r]), np.float32(fsed[r]), np.float32(sigc[r]),
                        np.float32(rv[r]), np.float32(vsini[r]),
                        np.float32(c_rho[r]), np.float32(c_a[r]), np.float32(c_b[r]), np.float32(c_z[r]),
                    ]
                    if oppert is not None:
                        row_args.append(oppert[r])
                    clean_row = batch_function(*row_args)
                    clean_rows.append(np.asarray(clean_row, dtype=np.float32))
                clean_spectra = np.stack(clean_rows, axis=0)
            else:
                clean_chunks = []
                for start in range(0, len(batch.parameters), self.device_batch_size):
                    stop = min(start + self.device_batch_size, len(batch.parameters))
                    s = slice(start, stop)
                    chunk_args = [
                        batch.vmr_profiles[s].astype(np.float32),
                        batch.temperature_profiles_K[s].astype(np.float32),
                        batch.mean_molecular_weight_profiles[s].astype(np.float32),
                        batch.gravity_cgs[s].astype(np.float32),
                        batch.radius_rjup[s].astype(np.float32),
                        mh[s], fsed[s], sigc[s], rv[s], vsini[s],
                        c_rho[s], c_a[s], c_b[s], c_z[s],
                    ]
                    if oppert is not None:
                        chunk_args.append(oppert[s])
                    clean_chunk = batch_function(*chunk_args)
                    clean_chunks.append(np.asarray(clean_chunk, dtype=np.float32))
                clean_spectra = np.concatenate(clean_chunks, axis=0)
        except (ImportError, NotImplementedError):
            if self.perturbation.enabled:
                # The eager scalar fallback bypasses the JAX perturbation path and
                # would silently produce UNPERTURBED spectra labeled as perturbed.
                raise
            return self._generate_scalar(batch)

        return GeneratedBatch(
            parameters=batch.parameters,
            parameter_names=batch.parameter_names,
            spectra_clean=clean_spectra,
            vmr_profiles=batch.vmr_profiles.astype(np.float32),
            nu_grid_cm1=self.spectrum_model.track_config.nu_grid_cm1.astype(np.float32),
        )
