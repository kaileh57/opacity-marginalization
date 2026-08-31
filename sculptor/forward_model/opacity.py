"""Opacity backends for forward modeling."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from sculptor.config import TrackConfig
from sculptor.forward_model.chemistry import ChemistryState


BROADENING_GAMMA_SCALE_ENV = "SCULPTOR_BROADENING_GAMMA_SCALE"
BROADENING_TEXP_SHIFT_ENV = "SCULPTOR_BROADENING_TEXP_SHIFT"


def broadening_perturbation_from_env(species: str) -> tuple[float, float]:
    """Pressure-broadening perturbation ``(gamma_scale, n_Texp_shift)`` for one species.

    Reads ``SCULPTOR_BROADENING_GAMMA_SCALE[_<SPECIES>]`` (multiplicative on the
    reference Lorentz half-width) and ``SCULPTOR_BROADENING_TEXP_SHIFT[_<SPECIES>]``
    (additive on the broadening temperature exponent). The per-species variable
    wins over the global one; unset means the identity ``(1.0, 0.0)``.
    """

    def lookup(base: str, default: float) -> float:
        for key in (f"{base}_{species.upper()}", base):
            raw = os.environ.get(key)
            if raw is not None and raw.strip():
                return float(raw)
        return default

    return lookup(BROADENING_GAMMA_SCALE_ENV, 1.0), lookup(BROADENING_TEXP_SHIFT_ENV, 0.0)


def apply_broadening_perturbation(operator: Any, species: str, role: str = "base") -> bool:
    """Perturb a PreMODIT operator's pressure broadening in place (env-gated no-op).

    PreMODIT evaluates the Lorentz half-width of every line from two lookup grids
    (``exojax.opacity.premodit.premodit.unbiased_ngamma_grid``)::

        ngamma(T, P) = ngamma_ref_grid[i] * (T / Tref_broadening) ** (-n_Texp_grid[j]) * P

    where ``multi_index_uniqgrid`` assigns each line its ``(i, j)`` pair. Scaling
    ``ngamma_ref_grid`` by ``s`` therefore scales EVERY line's reference width by
    exactly ``s``, and adding ``d`` to ``n_Texp_grid`` shifts every line's
    temperature exponent by exactly ``d``, at every temperature and pressure --
    identical to rebuilding the operator from a line list whose ``gamma_air`` /
    ``alpha_ref`` were scaled and whose ``n_air`` / ``n_Texp`` were shifted, but
    with the SAME grid quantization, so the two datasets differ only in
    broadening. Nothing else in the saved operator depends on the width: the LSD
    coefficients hold line strengths on the (elower, broadpar-index) grid, and
    sculptor builds with ``memory_policy=None`` -> ``nstitch == 1`` -> closed
    aliasing mode, whose wing handling is width-independent.

    This is the ONLY available route to a broadening-perturbed G395H forward
    model: the raw g395h line lists were deleted from GCS (see
    :meth:`ExoJAXOpacityBackend._load_alt_operators`), so the build-time
    ``SCULPTOR_GAMMA_SCALE_<SPECIES>`` hook -- which edits ``mdb.gamma_air``
    before ``OpaPremodit`` construction -- can no longer be exercised for this
    track, and it never covered ExoMol providers (``alpha_ref``) or the
    temperature exponent at all.

    Returns True when a perturbation was applied.
    """
    gamma_scale, texp_shift = broadening_perturbation_from_env(species)
    if gamma_scale == 1.0 and texp_shift == 0.0:
        return False
    if gamma_scale <= 0.0:
        raise ValueError(
            f"{BROADENING_GAMMA_SCALE_ENV} for {species} must be positive, got {gamma_scale}"
        )

    info = getattr(operator, "pre_modit_info", None)
    legacy = getattr(operator, "opainfo", None)
    if info is None and legacy is None:
        raise TypeError(
            f"Cannot perturb broadening for {species}: operator "
            f"{type(operator).__name__} exposes neither pre_modit_info nor opainfo."
        )

    if info is not None:
        ngamma_ref_grid = np.asarray(info.ngamma_ref_grid)
        n_Texp_grid = np.asarray(info.n_Texp_grid)
    else:
        ngamma_ref_grid = np.asarray(legacy[2])
        n_Texp_grid = np.asarray(legacy[3])

    new_ngamma = ngamma_ref_grid * gamma_scale
    new_texp = n_Texp_grid + texp_shift

    if info is not None:
        import dataclasses

        operator.pre_modit_info = dataclasses.replace(
            info, ngamma_ref_grid=new_ngamma, n_Texp_grid=new_texp
        )
        operator.opainfo = operator.pre_modit_info.as_tuple()
    else:
        operator.opainfo = (legacy[0], legacy[1], new_ngamma, new_texp, legacy[4], legacy[5])

    # Per-line bookkeeping arrays (informational; used by plot_broadening_parameters)
    # are kept consistent with the grids that actually drive the cross section.
    if getattr(operator, "gamma_ref", None) is not None:
        operator.gamma_ref = np.asarray(operator.gamma_ref) * gamma_scale
    if getattr(operator, "n_Texp", None) is not None:
        operator.n_Texp = np.asarray(operator.n_Texp) + texp_shift

    print(
        json.dumps(
            {
                "message": "broadening_perturbation_applied",
                "database_name": species,
                "role": role,
                "gamma_scale": gamma_scale,
                "n_Texp_shift": texp_shift,
                "ngamma_ref_grid_max": float(np.max(new_ngamma)),
                "n_Texp_grid_min": float(np.min(new_texp)),
                "n_Texp_grid_max": float(np.max(new_texp)),
            }
        ),
        flush=True,
    )
    return True


@dataclass(frozen=True)
class OpacityResult:
    """Per-layer cross-sections returned by an opacity backend."""

    nu_grid_cm1: np.ndarray
    species_cross_sections: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def total_cross_section(self) -> np.ndarray:
        total = None
        for value in self.species_cross_sections.values():
            total = np.array(value, copy=True) if total is None else total + value
        if total is None:
            return np.zeros((0, 0), dtype=np.float64)
        return total


class BaseOpacityBackend:
    """Abstract opacity backend interface."""

    def __init__(self, track_config: TrackConfig) -> None:
        self.track_config = track_config

    def prepare(self) -> None:
        return None

    def compute_cross_sections(self, chemistry_state: ChemistryState) -> OpacityResult:
        raise NotImplementedError

    def compute_cross_sections_jax(self, temperatures_K: Any, pressures_bar: Any) -> dict[str, Any]:
        raise NotImplementedError


def _construct_with_supported_kwargs(factory: Any, *args: Any, **kwargs: Any) -> Any:
    """Call third-party constructors without passing unsupported keywords."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(*args, **kwargs)

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return factory(*args, **kwargs)

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return factory(*args, **filtered_kwargs)


def _register_local_hitemp_cache(molecule_path: Path) -> str | None:
    """Register staged HITEMP cache files in RADIS so no login is required."""

    molecule_name = molecule_path.stem
    local_root = molecule_path.parent
    cache_files = sorted(local_root.glob(f"{molecule_name}-*.hdf5"))
    if not cache_files:
        cache_files = sorted(local_root.glob(f"{molecule_name}-*.h5"))
    if not cache_files:
        return None

    config_path = Path.home() / "radis.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {}

    database = config.setdefault("database", {})
    database[f"HITEMP-{molecule_name}"] = {
        "info": f"Local {molecule_name} HITEMP cache",
        "path": [str(path) for path in cache_files],
        "format": "hitran",
        "parfunc": "USE_HAPI",
        "download_url": [str(path) for path in cache_files],
        "download_date": date.today().isoformat(),
    }
    config_path.write_text(json.dumps(config, indent=3))

    return "pytables" if all(path.suffix == ".h5" for path in cache_files) else "vaex"


def _collect_hitemp_cache_files(molecule_path: Path) -> tuple[list[Path], str | None]:
    molecule_name = molecule_path.stem
    local_root = molecule_path.parent
    cache_files = sorted(local_root.glob(f"{molecule_name}-*.hdf5"))
    if not cache_files:
        cache_files = sorted(local_root.glob(f"{molecule_name}-*.h5"))
    if not cache_files:
        return [], None
    engine = "pytables" if all(path.suffix == ".h5" for path in cache_files) else "vaex"
    return cache_files, engine


def _infer_hitemp_isotope(molecule_path: Path) -> int | None:
    """Always load ALL isotopologues (isotope=None) for HITEMP/HITRAN databases.

    Historical bug (found 2026-06-11, forward-model autopsy): RADIS-style cache
    filenames carry the HITRAN *molecule* number after the molecule name
    (``CO2-02_HITEMP2024.h5`` — 02 is CO2's HITRAN molecule ID; ``H2O-01_...`` —
    01 is H2O's). The previous implementation parsed that two-digit molecule ID
    with ``re.search(r"-(\\d{2})[_-]", ...)`` as an *isotopologue* number, which
    built the CO2 opacity from 13C16O2 only (isotope 2, natural-abundance
    weighted ~1.1%, band head 4.38 um) and left the main 12C16O2 nu3 band at
    4.26 um entirely out of the forward model — the dominant NRS2 misfit on the
    real T8 (outputs/diagnostics/fm_autopsy/). CH4 and H2O were already exempted
    (the same confusion patched per-molecule); the correct general behavior is
    all-isotopes for every molecule.
    """
    del molecule_path  # filename carries the HITRAN molecule ID, not an isotope
    return None


def _is_staged_database_path(path: Path, database_class: str) -> bool:
    if path.exists():
        return True
    if database_class != "MdbHitemp":
        return False
    cache_files, _ = _collect_hitemp_cache_files(path)
    return bool(cache_files)


def _write_radis_config(exojax_root: Path) -> dict[str, str]:
    """Write the local HITEMP cache config before RADIS/ExoJAX imports."""

    molecule_roots = {
        "HITEMP-CH4": exojax_root / "g395h" / "CH4_HITEMP" / "CH4",
        "HITEMP-H2O": exojax_root / "g395h" / "H2O_HITEMP" / "H2O",
        "HITEMP-CO2": exojax_root / "g395h" / "CO2_HITEMP" / "CO2",
    }
    database_entries: dict[str, dict[str, Any]] = {}
    engines: dict[str, str] = {}
    for databank_name, molecule_path in molecule_roots.items():
        cache_files, engine = _collect_hitemp_cache_files(molecule_path)
        if not cache_files or engine is None:
            continue
        molecule_name = molecule_path.stem
        database_entries[databank_name] = {
            "info": f"Local {molecule_name} HITEMP cache",
            "path": [str(path) for path in cache_files],
            "format": "hitran",
            "parfunc": "USE_HAPI",
            "download_url": [str(path) for path in cache_files],
            "download_date": date.today().isoformat(),
            # Keep the RADIS/HITEMP-specific aliases alongside the fields
            # that already worked in the standalone debug sessions.
            "parfuncfmt": "hapi",
            "levelsfmt": "radis",
        }
        engines[molecule_name] = engine

    config_path = Path(os.path.expanduser("~/radis.json"))
    config = {
        "database": database_entries,
        "databank": database_entries,
    }
    config_path.write_text(json.dumps(config, indent=2))
    os.environ["RADIS_CONFIG_PATH"] = str(config_path)
    return engines


def _patch_exomol_download_for_offline() -> None:
    try:
        import radis.api.exomolapi as exomolapi
    except Exception:
        return

    if getattr(exomolapi.MdbExomol.download, "_sculptor_offline_patch", False):
        return

    original_download = exomolapi.MdbExomol.download

    def _download_with_local_short_circuit(self, molec, extension, numtag=None):
        missing_extensions = []
        for ext in extension:
            if getattr(self, "verbose", False):
                print(
                    f"\t\t => ExoMol offline check ext={ext} path={self.path} "
                    f"defs={len(list(self.path.glob('*.def')))} pfs={len(list(self.path.glob('*.pf')))}"
                )
            if ext == ".def" and any(self.path.glob("*.def")):
                if getattr(self, "verbose", False):
                    print(f"\t\t => Using local ExoMol definition file in {self.path}")
                continue
            if ext == ".pf" and any(self.path.glob("*.pf")):
                if getattr(self, "verbose", False):
                    print(f"\t\t => Using local ExoMol partition file in {self.path}")
                continue
            normalized_ext = "__" + numtag + ext if ext == ".trans.bz2" and numtag is not None else ext
            if normalized_ext == ".broad":
                # Use locally-staged broadeners (e.g. H2/He) when present instead of reaching out
                # to exomol.com — required on egress-walled hosts (TPU pods). Newer ExoMol datasets
                # (e.g. H2O POKAZATEL) declare extra broadeners (air/self) in the .def that we don't
                # stage; with any local *.broad available, let exojax broadf use what's there.
                if any(self.path.glob("*.broad")):
                    if getattr(self, "verbose", False):
                        print(f"\t\t => Using local ExoMol broadening files in {self.path}")
                    continue
                missing_extensions.append(ext)
                continue
            pfname_arr, _ = self._construct_filenames_and_url(
                normalized_ext,
                exomolapi.exact_molname_exomol_to_simple_molname(molec.split("__")[0]),
                molec.split("__"),
                molec,
                numtag,
            )
            target_paths = [self.path / pfname for pfname in pfname_arr]
            if all(path.exists() for path in target_paths):
                if getattr(self, "verbose", False):
                    print(f"\t\t => Using local ExoMol files {[str(path) for path in target_paths]}")
                continue
            missing_extensions.append(ext)

        if missing_extensions:
            return original_download(self, molec, missing_extensions, numtag=numtag)
        return None

    _download_with_local_short_circuit._sculptor_offline_patch = True  # type: ignore[attr-defined]
    exomolapi.MdbExomol.download = _download_with_local_short_circuit


class SyntheticOpacityBackend(BaseOpacityBackend):
    """A lightweight fallback backend that approximates major molecular bands."""

    BAND_DEFINITIONS = {
        "g395h": {
            "CO": [(2140.0, 28.0, 2.0)],
            "CO2": [(2325.0, 35.0, 3.0)],
            "CH4": [(3030.0, 42.0, 3.5)],
            "H2O": [(1900.0, 90.0, 1.8), (2680.0, 80.0, 1.3), (3375.0, 70.0, 1.0)],
            "NH3": [(2060.0, 45.0, 1.4), (2970.0, 55.0, 1.2)],
            "H2S": [(2550.0, 55.0, 1.1)],
            "H2-H2": [(2600.0, 350.0, 0.08)],
            "H2-He": [(3000.0, 360.0, 0.05)],
        },
        "hband": {
            "H2O": [(5900.0, 130.0, 2.0), (6900.0, 110.0, 2.3)],
            "CH4": [(6250.0, 80.0, 2.7)],
            "NH3": [(6480.0, 60.0, 1.6)],
            "H2-H2": [(6200.0, 400.0, 0.05)],
            "H2-He": [(6200.0, 420.0, 0.03)],
        },
    }

    def compute_cross_sections_jax(self, temperatures_K: Any, pressures_bar: Any) -> dict[str, Any]:
        try:
            import jax.numpy as jnp
        except Exception as exc:
            raise ImportError("JAX is required for batched opacity generation.") from exc

        nu_grid = jnp.asarray(self.track_config.nu_grid_cm1, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        temperatures = jnp.asarray(temperatures_K, dtype=jnp.float32)
        band_library = self.BAND_DEFINITIONS[self.track_config.name]
        cross_sections: dict[str, Any] = {}

        for species in (database.name for database in self.track_config.opacity_databases):
            layer_stack = jnp.zeros((pressures.shape[0], nu_grid.shape[0]), dtype=jnp.float32)
            if species in band_library:
                for center, width, strength in band_library[species]:
                    broadened_width = width * (
                        1.0 + 0.03 * jnp.power(jnp.maximum(pressures, 1.0e-8), 0.2) + 300.0 / jnp.maximum(temperatures, 100.0)
                    )
                    gaussian = jnp.exp(-0.5 * jnp.square((nu_grid[None, :] - center) / broadened_width[:, None]))
                    layer_stack = layer_stack + strength * gaussian
                layer_stack = layer_stack * jnp.sqrt(jnp.maximum(pressures[:, None], 1.0e-8))
            cross_sections[species] = layer_stack
        return cross_sections

    def compute_cross_sections(self, chemistry_state: ChemistryState) -> OpacityResult:
        nu_grid = self.track_config.nu_grid_cm1
        band_library = self.BAND_DEFINITIONS[self.track_config.name]
        cross_sections: dict[str, np.ndarray] = {}

        for species in chemistry_state.species:
            layer_stack = np.zeros((len(chemistry_state.pressures_bar), len(nu_grid)), dtype=np.float64)
            for layer_index, (pressure_bar, temperature_K) in enumerate(
                zip(chemistry_state.pressures_bar, chemistry_state.temperatures_K)
            ):
                if species not in band_library:
                    continue
                profile = np.zeros_like(nu_grid)
                for center, width, strength in band_library[species]:
                    broadened_width = width * (1.0 + 0.03 * pressure_bar**0.2 + 300.0 / max(temperature_K, 100.0))
                    gaussian = np.exp(-0.5 * ((nu_grid - center) / broadened_width) ** 2)
                    profile += strength * gaussian
                layer_stack[layer_index] = profile * np.sqrt(np.maximum(pressure_bar, 1.0e-8))
            cross_sections[species] = layer_stack

        return OpacityResult(
            nu_grid_cm1=nu_grid,
            species_cross_sections=cross_sections,
            metadata={"backend": "synthetic"},
        )


class ExoJAXOpacityBackend(BaseOpacityBackend):
    """Optional ExoJAX-backed opacity loader."""

    AUTO_TRANGE = [200.0, 3500.0]

    def __init__(
        self,
        track_config: TrackConfig,
        exojax_root: str | Path | None = None,
        crit_override: float | None = None,
        use_exomol_broadening: bool = True,
        download_exomol_broadening: bool = True,
    ) -> None:
        super().__init__(track_config)
        self.exojax_root = Path(exojax_root) if exojax_root is not None else None
        self.crit_override = crit_override
        self.use_exomol_broadening = bool(use_exomol_broadening)
        # Egress-walled pods (TPU) cannot reach exomol.com; radis broadening DOWNLOAD hangs on a
        # SYN-SENT timeout even when local *.broad files are staged (the .download patch doesn't
        # cover radis's broadening fetch path). SCULPTOR_DOWNLOAD_BROADENING=0 disables the fetch
        # (broadf still reads the locally-staged .broad / .def defaults).
        if os.environ.get("SCULPTOR_DOWNLOAD_BROADENING", "").strip() == "0":
            download_exomol_broadening = False
        self.download_exomol_broadening = bool(download_exomol_broadening)
        self._prepared = False
        self._database_inputs: dict[str, Any] = {}
        self._database_objects: dict[str, Any] = {}
        # Opacity-marginalization (SCULPTOR_OPACITY_PERTURB=1): per-species
        # ALTERNATIVE line-list operators, loaded cache-only from
        # SCULPTOR_OPA_ALT_CACHE_DIR_<MOL> saved-operator directories, plus the
        # currently-active variant selection (0 = base, 1 = alternative).
        self._alt_database_objects: dict[str, Any] = {}
        self._linelist_selection: dict[str, int] = {}
        self.auto_trange = list(self.AUTO_TRANGE)

    @property
    def alt_species(self) -> tuple[str, ...]:
        """Species with an alternative line-list operator loaded (sorted)."""
        return tuple(sorted(self._alt_database_objects))

    @property
    def linelist_selection_key(self) -> tuple[tuple[str, int], ...]:
        """Hashable key of the active non-base variant selection (compile key)."""
        return tuple(sorted((name, variant) for name, variant in self._linelist_selection.items() if variant))

    def set_linelist_selection(self, selection: dict[str, int]) -> None:
        """Choose the active line-list variant per species (0 base, 1 alternative).

        Species omitted from ``selection`` fall back to the base operator. Any
        compiled spectrum function embeds the operators as trace-time constants,
        so callers must include :attr:`linelist_selection_key` in their compile
        cache key (SpectrumBatchGenerator does).
        """
        normalized: dict[str, int] = {}
        for name, variant in selection.items():
            variant = int(variant)
            if variant not in (0, 1):
                raise ValueError(f"line-list variant for {name!r} must be 0 or 1, got {variant}")
            if variant == 1 and name not in self._alt_database_objects and self._prepared:
                raise KeyError(
                    f"No alternative line-list operator loaded for {name!r}; set "
                    f"SCULPTOR_OPA_ALT_CACHE_DIR_{name.upper()} and SCULPTOR_OPACITY_PERTURB=1."
                )
            normalized[name] = variant
        self._linelist_selection = normalized
        print(
            json.dumps({"message": "opacity linelist selection", "selection": normalized}),
            flush=True,
        )

    def _alt_cache_paths(self) -> dict[str, Path]:
        """Configured alternative saved-operator npz paths, keyed by species."""
        paths: dict[str, Path] = {}
        for database in self.track_config.opacity_databases:
            if database.database_class not in {"MdbExomol", "MdbHitemp", "MdbHitran"}:
                continue
            env_value = os.environ.get(f"SCULPTOR_OPA_ALT_CACHE_DIR_{database.name.upper()}", "").strip()
            if not env_value:
                continue
            safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in database.name)
            paths[database.name] = Path(env_value) / f"{safe_name}_opa_premodit.npz"
        return paths

    def _load_alt_operators(self, opa_premodit_cls: Any) -> None:
        """Load alternative line-list operators (cache-only; no Mdb fallback).

        Alternatives are ONLY available as saved PreMODIT operators: the raw
        line lists were deleted from GCS, so a cache miss here is a staging
        error, not something a live rebuild can recover from.
        """
        if os.environ.get("SCULPTOR_OPACITY_PERTURB", "0").strip() != "1":
            return
        for name, cache_path in self._alt_cache_paths().items():
            if name in self._alt_database_objects:
                continue
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Alternative line-list operator for {name} not staged: {cache_path}"
                )
            print(
                json.dumps(
                    {
                        "message": "opacity alt operator cache load start",
                        "database_name": name,
                        "path": str(cache_path),
                    }
                ),
                flush=True,
            )
            self._alt_database_objects[name] = opa_premodit_cls.from_saved_opa(str(cache_path))
            print(
                json.dumps(
                    {
                        "message": "opacity alt operator cache load done",
                        "database_name": name,
                        "operator_type": type(self._alt_database_objects[name]).__name__,
                    }
                ),
                flush=True,
            )

    def _apply_broadening_perturbations(self) -> list[str]:
        """Apply the env-gated broadening perturbation to every PreMODIT operator.

        Returns the ``"<role>:<species>"`` labels that were actually perturbed
        (empty list = production behaviour, nothing touched). CIA and tabulated
        operators carry no line broadening and are skipped.
        """
        perturbed: list[str] = []
        for role, store in (("base", self._database_objects), ("alt", self._alt_database_objects)):
            for name, operator in store.items():
                if not hasattr(operator, "pre_modit_info") and not hasattr(operator, "opainfo"):
                    continue
                if apply_broadening_perturbation(operator, name, role=role):
                    perturbed.append(f"{role}:{name}")
        return perturbed

    def prime_local_resources(self) -> None:
        """Prepare local cache metadata without importing ExoJAX or JAX."""

        if self.exojax_root is not None:
            os.environ.setdefault("EXOJAX_ROOT", str(self.exojax_root))
        elif "EXOJAX_ROOT" not in os.environ:
            raise EnvironmentError("EXOJAX_ROOT must be set before importing ExoJAX.")

        root = Path(os.environ["EXOJAX_ROOT"])
        _write_radis_config(root)
        missing_paths = [
            str(root / database.relative_path)
            for database in self.track_config.opacity_databases
            if not _is_staged_database_path(root / database.relative_path, database.database_class)
        ]
        print(
            json.dumps(
                {
                    "message": "opacity local resources primed",
                    "root": str(root),
                    "missing_paths": missing_paths,
                }
            ),
            flush=True,
        )
        if missing_paths:
            raise FileNotFoundError(f"Missing staged opacity paths: {missing_paths}")

    def load_database_inputs(self, defer_activation: bool = False) -> None:
        """Load ExoMol/HITEMP/CIA database handles before JAX operator construction."""

        if self._database_inputs:
            print(json.dumps({"message": "opacity database inputs skipped", "reason": "already_loaded"}), flush=True)
            return

        if self.exojax_root is not None:
            os.environ.setdefault("EXOJAX_ROOT", str(self.exojax_root))
        elif "EXOJAX_ROOT" not in os.environ:
            raise EnvironmentError("EXOJAX_ROOT must be set before importing ExoJAX.")

        root = Path(os.environ["EXOJAX_ROOT"])
        hitemp_engines = _write_radis_config(root)
        _patch_exomol_download_for_offline()
        original_jax_platforms = os.environ.get("JAX_PLATFORMS")
        os.environ["JAX_PLATFORMS"] = "cpu"

        try:
            try:
                from exojax.database.api import MdbExomol, MdbHitemp, MdbHitran
            except Exception:
                from exojax.database import MdbExomol, MdbHitemp, MdbHitran  # type: ignore

            try:
                from exojax.database.cia.api import CdbCIA
            except Exception:
                try:
                    from exojax.database import CdbCIA  # type: ignore
                except Exception:
                    from exojax.spec.contdb import CdbCIA  # type: ignore

            nu_grid = self.track_config.nu_grid_cm1
            opa_cache_dir_env = os.environ.get("SCULPTOR_OPA_CACHE_DIR")
            opa_cache_dir = Path(opa_cache_dir_env) if opa_cache_dir_env else None
            read_opa_cache = os.environ.get("SCULPTOR_READ_OPA_CACHE", "0") == "1"
            write_opa_cache = os.environ.get("SCULPTOR_WRITE_OPA_CACHE", "0") == "1"
            if opa_cache_dir is not None and (read_opa_cache or write_opa_cache):
                opa_cache_dir.mkdir(parents=True, exist_ok=True)

            for database in self.track_config.opacity_databases:
                # Skip raw-line Mdb construction when the operator npz is already cached and we will
                # read it: avoids staging tens of GB of line lists for a cache-only run. prepare()'s
                # build loop loads such species via OpaPremodit.from_saved_opa (cache-first).
                if (
                    read_opa_cache
                    and opa_cache_dir is not None
                    and database.database_class in {"MdbExomol", "MdbHitemp", "MdbHitran"}
                ):
                    safe_name = "".join(
                        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in database.name
                    )
                    if (opa_cache_dir / f"{safe_name}_opa_premodit.npz").exists():
                        print(
                            json.dumps(
                                {
                                    "message": "opacity database input skipped",
                                    "database_name": database.name,
                                    "reason": "operator_cache_hit",
                                }
                            ),
                            flush=True,
                        )
                        continue
                path = root / database.relative_path
                print(
                    json.dumps(
                        {
                            "message": "opacity database input start",
                            "database_name": database.name,
                            "database_class": database.database_class,
                            "path": str(path),
                            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                            "defer_activation": defer_activation,
                        }
                    ),
                    flush=True,
                )
                # Probe the path so we know what the process can actually see,
                # not just what an SSH session sees.
                _probe_defs = len(list(path.glob("*.def"))) if path.is_dir() else -1
                _probe_pfs = len(list(path.glob("*.pf"))) if path.is_dir() else -1
                print(
                    json.dumps(
                        {
                            "message": "opacity path probe",
                            "database_name": database.name,
                            "database_class": database.database_class,
                            "path": str(path),
                            "path_exists": path.exists(),
                            "def_count": _probe_defs,
                            "pf_count": _probe_pfs,
                            "parent_exists": path.parent.exists(),
                        }
                    ),
                    flush=True,
                )
                if database.database_class == "MdbExomol":
                    self._database_inputs[database.name] = (
                        database.database_class,
                        _construct_with_supported_kwargs(
                            MdbExomol,
                            str(path),
                            nurange=[float(nu_grid[0]), float(nu_grid[-1])],
                            broadf=self.use_exomol_broadening,
                            broadf_download=self.download_exomol_broadening,
                            gpu_transfer=not defer_activation,
                            inherit_dataframe=defer_activation,
                            activation=not defer_activation,
                            allow_32bit=True,
                        ),
                    )
                elif database.database_class == "MdbHitemp":
                    crit = self.crit_override if self.crit_override is not None else database.crit
                    engine = hitemp_engines.get(database.name)
                    isotope = _infer_hitemp_isotope(path)
                    self._database_inputs[database.name] = (
                        database.database_class,
                        _construct_with_supported_kwargs(
                            MdbHitemp,
                            str(path),
                            nurange=[float(nu_grid[0]), float(nu_grid[-1])],
                            crit=crit,
                            isotope=isotope,
                            gpu_transfer=not defer_activation,
                            inherit_dataframe=defer_activation,
                            activation=not defer_activation,
                            allow_32bit=True,
                            engine=engine,
                        ),
                    )
                elif database.database_class == "MdbHitran":
                    crit = self.crit_override if self.crit_override is not None else database.crit
                    isotope = _infer_hitemp_isotope(path)
                    self._database_inputs[database.name] = (
                        database.database_class,
                        _construct_with_supported_kwargs(
                            MdbHitran,
                            str(path),
                            nurange=[float(nu_grid[0]), float(nu_grid[-1])],
                            crit=crit,
                            isotope=isotope,
                            gpu_transfer=not defer_activation,
                            inherit_dataframe=defer_activation,
                            activation=not defer_activation,
                            allow_32bit=True,
                        ),
                    )
                elif database.database_class == "CdbCIA":
                    self._database_inputs[database.name] = (
                        database.database_class,
                        _construct_with_supported_kwargs(
                            CdbCIA,
                            str(path),
                            nu_grid=nu_grid,
                        ),
                    )
                elif database.database_class in {"TabulatedAlkali", "TabulatedXsec"}:
                    # Pre-computed cross-section table resampled to this track's nu grid by
                    # scripts/build_alkali_opacity_table.py (from petitRADTRANS line-by-line opacities).
                    # Used for atomic alkali (Allard Na/K, which exojax cannot model) AND, for the SpeX
                    # track, the molecules (avoids staging band-split ExoMol .trans + the egress-walled
                    # download-during-build). Self-contained operator exposing xsmatrix(T,P).
                    from sculptor.forward_model.tabulated_alkali import TabulatedAlkaliOpa

                    self._database_inputs[database.name] = (
                        database.database_class,
                        TabulatedAlkaliOpa(str(path)),
                    )
                print(
                    json.dumps(
                        {
                            "message": "opacity database input done",
                            "database_name": database.name,
                            "input_type": type(self._database_inputs[database.name][1]).__name__,
                        }
                    ),
                    flush=True,
                )
        finally:
            if original_jax_platforms is None:
                os.environ.pop("JAX_PLATFORMS", None)
            else:
                os.environ["JAX_PLATFORMS"] = original_jax_platforms

    def prepare(self) -> None:  # pragma: no cover - depends on optional runtime
        if self._prepared:
            print(json.dumps({"message": "opacity prepare skipped", "reason": "already_prepared"}), flush=True)
            return

        import builtins
        import getpass

        if self.exojax_root is not None:
            os.environ.setdefault("EXOJAX_ROOT", str(self.exojax_root))
        elif "EXOJAX_ROOT" not in os.environ:
            raise EnvironmentError("EXOJAX_ROOT must be set before importing ExoJAX.")
        root = Path(os.environ["EXOJAX_ROOT"])
        _write_radis_config(root)
        original_input = builtins.input
        original_getpass = getpass.getpass

        def _patched_input(prompt: str = "") -> str:
            prompt_lower = prompt.lower()
            if "email" in prompt_lower or "password" in prompt_lower or "login" in prompt_lower:
                raise RuntimeError("RADIS tried to prompt for credentials. Check radis.json registration.")
            return original_input(prompt)

        builtins.input = _patched_input
        getpass.getpass = lambda prompt="": (_ for _ in ()).throw(
            RuntimeError("RADIS tried to prompt for credentials. Check radis.json registration.")
        )
        try:
            import caugetch

            caugetch.getch = lambda: "\r"
        except ImportError:
            pass

        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
        except Exception as exc:
            raise ImportError("JAX is required for the ExoJAX opacity backend.") from exc
        force_cpu_prepare = os.environ.get("SCULPTOR_FORCE_CPU_OPACITY_PREPARE", "0") == "1"

        try:
            print(
                json.dumps(
                    {
                        "message": "opacity prepare started",
                        "root": str(root),
                        "auto_trange": self.auto_trange,
                    }
                ),
                flush=True,
            )
            try:
                from exojax.opacity.premodit.api import OpaPremodit
            except Exception:
                try:
                    from exojax.opacity import OpaPremodit  # type: ignore
                except Exception:
                    from exojax.spec import OpaPremodit  # type: ignore

            # No memory policy: xsmatrix is called one layer at a time in
            # compute_cross_sections_jax, so nstitch is not needed.
            _memory_policy = None

            if not self._database_inputs:
                self.load_database_inputs(defer_activation=True)

            nu_grid = self.track_config.nu_grid_cm1
            auto_trange = list(self.auto_trange)

            print(
                json.dumps({"message": "opacity_opa_premodit_disable_jit_mode"}),
                flush=True,
            )

            opa_cache_dir_env = os.environ.get("SCULPTOR_OPA_CACHE_DIR")
            opa_cache_dir = Path(opa_cache_dir_env) if opa_cache_dir_env else None
            read_opa_cache = os.environ.get("SCULPTOR_READ_OPA_CACHE", "0") == "1"
            write_opa_cache = os.environ.get("SCULPTOR_WRITE_OPA_CACHE", "0") == "1"
            if opa_cache_dir is not None and (read_opa_cache or write_opa_cache):
                opa_cache_dir.mkdir(parents=True, exist_ok=True)

            for database in self.track_config.opacity_databases:
                print(
                    json.dumps(
                        {
                            "message": "opacity operator build start",
                            "database_name": database.name,
                            "database_class": database.database_class,
                        }
                    ),
                    flush=True,
                )
                # Cache-first: load the saved operator WITHOUT needing the raw Mdb, so cached
                # operators are self-contained and the bulky line lists need not be staged for a
                # cache-only run (refit/generation/inference). The Mdb is only touched on a miss.
                cache_path = None
                if opa_cache_dir is not None and database.database_class in {"MdbExomol", "MdbHitemp", "MdbHitran"}:
                    safe_name = "".join(
                        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
                        for ch in database.name
                    )
                    cache_path = opa_cache_dir / f"{safe_name}_opa_premodit.npz"
                if read_opa_cache and cache_path is not None and cache_path.exists():
                    print(
                        json.dumps(
                            {
                                "message": "opacity operator cache load start",
                                "database_name": database.name,
                                "path": str(cache_path),
                            }
                        ),
                        flush=True,
                    )
                    self._database_objects[database.name] = OpaPremodit.from_saved_opa(
                        str(cache_path)
                    )
                    print(
                        json.dumps(
                            {
                                "message": "opacity operator cache load done",
                                "database_name": database.name,
                                "operator_type": type(self._database_objects[database.name]).__name__,
                            }
                        ),
                        flush=True,
                    )
                    continue

                database_class, source = self._database_inputs[database.name]
                if database_class in {"MdbExomol", "MdbHitemp", "MdbHitran"}:
                    snapshot = None
                    if hasattr(source, "activation") and not source.activation:
                        if jax.process_count() > 1 and not force_cpu_prepare:
                            print(
                                json.dumps(
                                    {
                                        "message": "opacity database activation skipped",
                                        "database_name": database.name,
                                        "database_class": database_class,
                                        "reason": "distributed_process_uses_dataframe_path",
                                    }
                                ),
                                flush=True,
                            )
                            if hasattr(source, "df") and hasattr(source, "df_load_mask"):
                                if database_class in {"MdbHitemp", "MdbHitran"} and hasattr(source, "activate"):
                                    source.gpu_transfer = False
                                    source.activate(source.df)
                                    source.activation = True
                                else:
                                    df_masked = source.df[source.df_load_mask]
                                    if hasattr(source, "attributes_from_dataframes"):
                                        source.attributes_from_dataframes(df_masked)
                                    elif hasattr(source, "_attributes_from_dataframes"):
                                        source._attributes_from_dataframes(df_masked)
                                    if hasattr(source, "set_broadening_coef"):
                                        try:
                                            source.set_broadening_coef(
                                                df_masked,
                                                add_columns=False,
                                                species=getattr(source, "bkgdatm", None),
                                            )
                                        except TypeError:
                                            source.set_broadening_coef(df_masked, add_columns=False)
                                    elif hasattr(source, "compute_broadening"):
                                        source.compute_broadening(
                                            source.jlower.astype(int),
                                            source.jupper.astype(int),
                                        )
                                    source.activation = True
                                if hasattr(source, "to_snapshot"):
                                    print(
                                        json.dumps(
                                            {
                                                "message": "opacity database snapshot skipped",
                                                "database_name": database.name,
                                                "database_class": database_class,
                                                "reason": "distributed_from_snapshot_halts_tpu_continuator",
                                            }
                                        ),
                                        flush=True,
                                    )
                        else:
                            print(
                                json.dumps(
                                    {
                                        "message": "opacity database activation start",
                                        "database_name": database.name,
                                        "database_class": database_class,
                                    }
                                ),
                                flush=True,
                            )
                            # Activate with x64 DISABLED so partition-function arrays
                            # (gQT, T_gQT) are stored as float32.  OpaPremodit's LSD
                            # construction still runs with x64=True (set at prepare()
                            # entry) for numerical stability, but the lookup tables
                            # must be float32 so they are compatible with the float32
                            # spectrum-generation JIT without dtype conflicts.
                            jax.config.update("jax_enable_x64", False)
                            _cpu = jax.devices("cpu")[0]
                            with jax.default_device(_cpu):
                                source.gpu_transfer = True
                                source.activate(source.df)
                                source.activation = True
                            jax.config.update("jax_enable_x64", True)
                            print(
                                json.dumps(
                                    {
                                        "message": "opacity database activation done",
                                        "database_name": database.name,
                                        "database_class": database_class,
                                    }
                                ),
                                flush=True,
                            )
                    # In single-process runs, keep OpaPremodit construction on CPU
                    # to avoid using TPU memory for setup. In multi-process TPU
                    # runs, CPU JAX computations are not implemented after
                    # distributed init, so use the initialized TPU backend.
                    device_context = contextlib.nullcontext()
                    if force_cpu_prepare or jax.process_count() <= 1:
                        _cpu = jax.devices("cpu")[0]
                        device_context = jax.default_device(_cpu)
                    # FORWARD-MODEL FLOOR FIX (broadening): HITEMP defaults to AIR broadening, but a
                    # brown-dwarf atmosphere is H2/He (CO2 H2-broadening is ~1.2-1.4x air). Air-broadened
                    # (too-narrow) CO2 lines over-absorb cores / under-absorb wings -- the measured floor
                    # signature. Scale this molecule's reference broadening (env-gated; default no-op).
                    _gs = os.environ.get(f"SCULPTOR_GAMMA_SCALE_{database.name.split()[0].upper()}") \
                        or os.environ.get("SCULPTOR_GAMMA_SCALE")
                    if _gs and getattr(source, "gamma_air", None) is not None:
                        source.gamma_air = jnp.asarray(source.gamma_air) * float(_gs)
                        if getattr(source, "gamma_self", None) is not None:
                            source.gamma_self = jnp.asarray(source.gamma_self) * float(_gs)
                        print(json.dumps({"message": "broadening_gamma_scaled",
                                          "database_name": database.name, "scale": float(_gs)}), flush=True)
                    with jax.disable_jit(), device_context:
                        premodit_kwargs = dict(
                            nu_grid=nu_grid,
                            auto_trange=auto_trange,
                            conversion_dtype=jnp.float32,
                            # allow_32bit=True: LSD arrays are float32 so xsmatrix FFT
                            # works on TPU (TPU does not support float64 FFT).
                            # WARNING: conversion_dtype=float32 is safe ONLY because
                            # auto_trange is [200, 3500].  Float32 overflows during LSD
                            # construction if the temperature range is widened.  If you
                            # change AUTO_TRANGE, verify LSD tables don't contain Inf/NaN
                            # or switch conversion_dtype back to float64 (CPU-only builds).
                            allow_32bit=True,
                            memory_policy=_memory_policy,
                        )
                        if snapshot is not None:
                            snapshot_kwargs = dict(premodit_kwargs)
                            snapshot_kwargs.pop("conversion_dtype", None)
                            self._database_objects[database.name] = _construct_with_supported_kwargs(
                                OpaPremodit.from_snapshot,
                                snapshot,
                                **snapshot_kwargs,
                            )
                        else:
                            self._database_objects[database.name] = _construct_with_supported_kwargs(
                                OpaPremodit,
                                mdb=source,
                                **premodit_kwargs,
                            )
                    if write_opa_cache and cache_path is not None:
                        print(
                            json.dumps(
                                {
                                    "message": "opacity operator cache save start",
                                    "database_name": database.name,
                                    "path": str(cache_path),
                                }
                            ),
                            flush=True,
                        )
                        from exojax.opacity.premodit.ioopa import saveopa_premodit

                        # ExoJAX HITEMP all-isotope mode leaves provider.isotope
                        # as None while its own partition-function code assumes
                        # isotope=1.  The serializer expects an integer.
                        pf_provider = getattr(
                            self._database_objects[database.name], "pf_provider", None
                        )
                        if getattr(pf_provider, "isotope", 1) is None:
                            pf_provider.isotope = 1

                        saveopa_premodit(
                            self._database_objects[database.name],
                            str(cache_path),
                            format="npz",
                            extra_meta={
                                "database_name": database.name,
                                "track": self.track_config.name,
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "message": "opacity operator cache save done",
                                    "database_name": database.name,
                                    "path": str(cache_path),
                                }
                            ),
                            flush=True,
                        )
                elif database_class == "CdbCIA":
                    self._database_objects[database.name] = source
                elif database_class in {"TabulatedAlkali", "TabulatedXsec"}:
                    # Self-contained tabulated operator (xsmatrix); store directly like CIA.
                    self._database_objects[database.name] = source
                print(
                    json.dumps(
                        {
                            "message": "opacity operator build done",
                            "database_name": database.name,
                            "operator_type": type(self._database_objects[database.name]).__name__,
                        }
                    ),
                    flush=True,
                )
            # Opacity-marginalization line-list CHOICE axis: load the configured
            # alternative operators (cache-only) so set_linelist_selection can
            # switch variants per shard at runtime.
            self._load_alt_operators(OpaPremodit)
            # Pressure-broadening STRESS TEST axis (env-gated, default no-op):
            # rescale every PreMODIT operator's broadening grids AFTER both the
            # base and the alternative operators exist, so base and alternative
            # line lists carry the same width/temperature-exponent perturbation
            # and the line-list choice axis stays exactly as in production.
            perturbed = self._apply_broadening_perturbations()
            self._prepared = True
            print(
                json.dumps(
                    {
                        "message": "opacity prepare completed",
                        "database_count": len(self._database_objects),
                        "database_names": sorted(self._database_objects),
                        "alt_database_names": sorted(self._alt_database_objects),
                        "broadening_perturbed": perturbed,
                    }
                ),
                flush=True,
            )
        finally:
            # OpaPremodit initialization needs x64 for stable LSD construction,
            # but the generation path must switch back to float32 before TPU work.
            jax.config.update("jax_enable_x64", False)
            builtins.input = original_input
            getpass.getpass = original_getpass

    def compute_cross_sections_jax(self, temperatures_K: Any, pressures_bar: Any) -> dict[str, Any]:
        if not self._prepared:
            self.prepare()
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            raise ImportError("JAX is required for the ExoJAX opacity backend.") from exc

        # Disable x64: OpaPremodit LSD arrays are float32 (allow_32bit=True)
        # and TPU does not support float64 FFT used inside xsmatrix.
        jax.config.update("jax_enable_x64", False)
        try:
            cross_sections: dict[str, Any] = {}
            temperatures = jnp.asarray(temperatures_K, dtype=jnp.float32)
            pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
            if not getattr(self, "_logged_xs_device", False):
                print(
                    json.dumps({
                        "message": "xs_device_info",
                        "default_backend": jax.default_backend(),
                        "devices": [str(d) for d in jax.devices()[:4]],
                        "temperature_type": type(temperatures).__name__,
                    }),
                    flush=True,
                )
                self._logged_xs_device = True
            for species, operator in self._database_objects.items():
                if self._linelist_selection.get(species, 0) == 1:
                    # Opacity-marginalization: this species uses its ALTERNATIVE
                    # line-list operator for the current shard/combo.
                    operator = self._alt_database_objects[species]
                if hasattr(operator, "xsmatrix"):
                    cross_sections[species] = operator.xsmatrix(temperatures, pressures)
                elif hasattr(operator, "logacia_matrix"):
                    cross_sections[species] = jnp.exp(
                        operator.logacia_matrix(temperatures, pressures)
                    )
        finally:
            pass  # x64 already disabled
        return cross_sections

    def compute_cross_sections(self, chemistry_state: ChemistryState) -> OpacityResult:  # pragma: no cover - optional runtime
        if not self._prepared:
            self.prepare()
        cross_sections = {
            species: np.asarray(values)
            for species, values in self.compute_cross_sections_jax(
                chemistry_state.temperatures_K,
                chemistry_state.pressures_bar,
            ).items()
        }
        return OpacityResult(
            nu_grid_cm1=self.track_config.nu_grid_cm1,
            species_cross_sections=cross_sections,
            metadata={"backend": "exojax"},
        )
