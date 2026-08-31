"""Opacity-marginalization: per-draw perturbations of the cached cross-sections.

Every training draw sees a randomly perturbed opacity model so the trained NPE
marginalizes over line-list uncertainty (the "turn the opacity wall into an
error bar" flagship; see outputs/strategy/flashy_scan_ml.md direction A).

Two perturbation axes, both acting at RUNTIME on the cached PreMODIT operators
(the raw HITEMP line lists were deleted from GCS -- nothing here requires a
cache rebuild):

1. LINE-LIST CHOICE (discrete). Alternative saved-operator caches exist for
   H2O (HITEMP vs ExoMol POKAZATEL) and CH4 (HITEMP vs ExoMol MM). Shard k of a
   generation run uses line-list combo ``k % 2**n_alt_species``, which
   marginalizes over the choice at the DATASET level (equally valid for NPE
   training, and it means each SPMD pod worker -- which owns shards
   ``k % 8 == worker`` with 4 | 8 -- compiles exactly one combo).

2. SMOOTH MULTIPLICATIVE FIELD (continuous). Per draw, each molecular species'
   cross-section matrix is multiplied by ``exp(f_s(u))`` where ``f_s`` is a
   low-order Fourier series in normalized log-wavenumber ``u`` (the production
   grids are ESLOG, so pixel index is exactly linear in log nu):

       f_s(u) = c0 + sum_m [ a_m cos(2 pi m u) + b_m sin(2 pi m u) ]

   The coefficients are drawn i.i.d. Gaussian with per-species/per-mode sigmas
   anchored to (a) the MEASURED smooth inter-line-list xs ratios
   (scripts/measure_interlist_ratios.py writes the JSON consumed here) and
   (b) the ~7% field-wide forward-model floor (Yama et al. 2026 + our R1/R2
   measurements). CO2 gets the widest prior: its 4.0-4.35 um window carries the
   dominant chi2/dof ~66 floor, is line-shape-like, and has no alternative
   cache -- the field is the honest stand-in.

The drawn coefficients are recorded as extra parameter-table columns prefixed
``oppert_`` (NUISANCE columns: never NPE labels, always recorded for exact
reproducibility). Everything is gated by ``SCULPTOR_OPACITY_PERTURB=1``; with
the flag off the generation path is bit-identical to the unperturbed pipeline.

Environment variables
---------------------
SCULPTOR_OPACITY_PERTURB          "1" enables both axes (default off).
SCULPTOR_OPACITY_PERTURB_MODES    number of Fourier modes M (default 4).
SCULPTOR_OPACITY_PERTURB_PRIORS   path to a priors JSON (measure_interlist_ratios
                                  output); default = builtin anchored table.
SCULPTOR_OPA_ALT_CACHE_DIR_<MOL>  directory holding <MOL>_opa_premodit.npz for
                                  the ALTERNATIVE line list of that molecule
                                  (e.g. _H2O -> HITEMP-H2O operator dir when the
                                  base cache is POKAZATEL). Optional; without
                                  any, only the continuous field axis is active.
SCULPTOR_FORCE_COMBO              PIN the discrete line-list combo to this id for
                                  EVERY shard instead of the default
                                  ``combo = shard % 2**n_alt`` cycle. Default
                                  unset = production behaviour, bit-identical.
                                  Used by the discrete-vs-smooth ablation to
                                  freeze the discrete axis (SMOOTH-ONLY mode);
                                  the converse (DISCRETE-ONLY) is obtained by
                                  scaling the continuous-field priors to zero,
                                  which makes exp(f) exactly 1.0 (see
                                  scripts/scale_opmarg_priors.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Prefix marking nuisance parameter-table columns. The trainer drops every
# column with this prefix before building the target/conditioning layout.
NUISANCE_PREFIX = "oppert_"
# Column recording the discrete line-list combo used for the row's shard.
COMBO_COLUMN = "oppert_linelist_combo"
# Env var pinning the discrete line-list combo for every shard (ablation only).
FORCE_COMBO_ENV = "SCULPTOR_FORCE_COMBO"
# Salt mixed into the coefficient RNG so the perturbation stream is independent
# of every other seeded stream in the pipeline (noise, sampler, ...).
_COEFFICIENT_SALT = 0x0FA71E7D

# Builtin amplitude priors (std-dev of Gaussian coefficients, natural-log units,
# i.e. 0.05 ~ 5% multiplicative). ANCHORS:
# - ~7% field-wide line-list floor: Yama et al. 2026 (ApJ 997, 118) CO-list C/O
#   systematic on Luhman 16AB; independently matched by our G395H T8 floor
#   (outputs/paper_diseq/R1_R2_FINDINGS.md).
# - R2 A/B: CH4 HITEMP vs ExoMol MM moved its band chi2 by 4-11% -> smooth xs
#   differences at the several-percent level.
# - CO2 widest: dominant floor window (chi2/dof ~66), line-shape-like residual,
#   no alternative cache exists (killtests: unfixable at build level).
# These are DEFAULTS: production runs replace them with the measured JSON from
# scripts/measure_interlist_ratios.py via SCULPTOR_OPACITY_PERTURB_PRIORS.
DEFAULT_PRIOR_TABLE: dict[str, dict[str, object]] = {
    "H2O": {"sigma_const": 0.05, "sigma_modes": (0.05, 0.04, 0.03, 0.02)},
    "CH4": {"sigma_const": 0.05, "sigma_modes": (0.05, 0.04, 0.03, 0.02)},
    "CO": {"sigma_const": 0.04, "sigma_modes": (0.04, 0.03, 0.02, 0.015)},
    "NH3": {"sigma_const": 0.06, "sigma_modes": (0.06, 0.05, 0.035, 0.025)},
    "CO2": {"sigma_const": 0.10, "sigma_modes": (0.10, 0.08, 0.06, 0.045)},
    "H2S": {"sigma_const": 0.06, "sigma_modes": (0.06, 0.05, 0.035, 0.025)},
}
# Fallback for species without an entry anywhere (defensive; every current
# track molecule is covered above).
_FALLBACK_PRIOR = {"sigma_const": 0.05, "sigma_modes": (0.05, 0.04, 0.03, 0.02)}
# Geometric decay applied when a prior table provides fewer modes than
# requested (extra high-order modes get progressively smaller amplitude).
_MODE_EXTENSION_DECAY = 0.7


def _resize_modes(sigma_modes: tuple[float, ...], n_modes: int) -> tuple[float, ...]:
    """Adapt a sigma_modes tuple to ``n_modes`` entries (truncate or decay-extend)."""
    modes = [float(value) for value in sigma_modes]
    if not modes:
        raise ValueError("sigma_modes must contain at least one amplitude")
    while len(modes) < n_modes:
        modes.append(modes[-1] * _MODE_EXTENSION_DECAY)
    return tuple(modes[:n_modes])


@dataclass(frozen=True)
class OpacityPerturbationSettings:
    """Resolved configuration for the opacity-perturbation axes."""

    enabled: bool
    n_modes: int
    species: tuple[str, ...]
    sigma_const: dict[str, float]
    sigma_modes: dict[str, tuple[float, ...]]
    priors_source: str

    @property
    def n_coeff_per_species(self) -> int:
        return 1 + 2 * self.n_modes

    def coefficient_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for species in self.species:
            names.append(f"{NUISANCE_PREFIX}{species}_c0")
            for mode in range(1, self.n_modes + 1):
                names.append(f"{NUISANCE_PREFIX}{species}_a{mode}")
                names.append(f"{NUISANCE_PREFIX}{species}_b{mode}")
        return tuple(names)

    def column_names(self) -> tuple[str, ...]:
        """All nuisance columns appended to the parameter table."""
        return self.coefficient_names() + (COMBO_COLUMN,)

    def coefficient_sigmas(self) -> np.ndarray:
        """Per-column sigma vector matching :meth:`coefficient_names` order."""
        sigmas: list[float] = []
        for species in self.species:
            sigmas.append(self.sigma_const[species])
            for mode_sigma in self.sigma_modes[species]:
                sigmas.extend((mode_sigma, mode_sigma))
        return np.asarray(sigmas, dtype=np.float64)

    def expected_field_log_rms(self, species: str) -> float:
        """Prior-mean RMS of the log field for one species (diagnostics)."""
        const = self.sigma_const[species]
        modes = np.asarray(self.sigma_modes[species], dtype=np.float64)
        # var(f) = sigma_c0^2 + sum_m (sigma_m^2/2 + sigma_m^2/2) over cos+sin
        return float(np.sqrt(const**2 + np.sum(modes**2)))

    def as_metadata(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "n_modes": self.n_modes,
            "species": list(self.species),
            "sigma_const": {k: float(v) for k, v in self.sigma_const.items()},
            "sigma_modes": {k: [float(x) for x in v] for k, v in self.sigma_modes.items()},
            "priors_source": self.priors_source,
        }


def disabled_settings() -> OpacityPerturbationSettings:
    return OpacityPerturbationSettings(
        enabled=False,
        n_modes=0,
        species=(),
        sigma_const={},
        sigma_modes={},
        priors_source="disabled",
    )


def load_priors(path: str | Path) -> dict[str, dict[str, object]]:
    """Load a priors table from JSON (measure_interlist_ratios.py output).

    Accepts either a bare ``{species: {sigma_const, sigma_modes}}`` mapping or a
    document with a ``priors`` / ``suggested_priors`` top-level key.
    """
    document = json.loads(Path(path).read_text())
    for key in ("suggested_priors", "priors"):
        if isinstance(document, dict) and key in document:
            document = document[key]
            break
    if not isinstance(document, dict) or not document:
        raise ValueError(f"Priors JSON {path} contains no species entries")
    table: dict[str, dict[str, object]] = {}
    for species, entry in document.items():
        if "sigma_const" not in entry or "sigma_modes" not in entry:
            raise ValueError(
                f"Priors JSON {path} entry for {species!r} must define sigma_const and sigma_modes"
            )
        table[species] = {
            "sigma_const": float(entry["sigma_const"]),
            "sigma_modes": tuple(float(x) for x in entry["sigma_modes"]),
        }
    return table


def settings_from_env(molecules: tuple[str, ...] | None) -> OpacityPerturbationSettings:
    """Resolve perturbation settings from the environment for a track's molecules."""

    if os.environ.get("SCULPTOR_OPACITY_PERTURB", "0").strip() != "1":
        return disabled_settings()
    if not molecules:
        raise ValueError(
            "SCULPTOR_OPACITY_PERTURB=1 but the track config exposes no molecules; "
            "opacity perturbation requires a physical track configuration."
        )

    n_modes = int(os.environ.get("SCULPTOR_OPACITY_PERTURB_MODES", "4"))
    if not 1 <= n_modes <= 16:
        raise ValueError(f"SCULPTOR_OPACITY_PERTURB_MODES={n_modes} out of range [1, 16]")

    priors_path = os.environ.get("SCULPTOR_OPACITY_PERTURB_PRIORS", "").strip()
    if priors_path:
        override = load_priors(priors_path)
        priors_source = str(priors_path)
    else:
        override = {}
        priors_source = "builtin_defaults"

    sigma_const: dict[str, float] = {}
    sigma_modes: dict[str, tuple[float, ...]] = {}
    for species in molecules:
        entry = override.get(species) or DEFAULT_PRIOR_TABLE.get(species) or _FALLBACK_PRIOR
        sigma_const[species] = float(entry["sigma_const"])
        sigma_modes[species] = _resize_modes(tuple(entry["sigma_modes"]), n_modes)

    return OpacityPerturbationSettings(
        enabled=True,
        n_modes=n_modes,
        species=tuple(molecules),
        sigma_const=sigma_const,
        sigma_modes=sigma_modes,
        priors_source=priors_source,
    )


def alt_species_from_env(molecules: tuple[str, ...]) -> tuple[str, ...]:
    """Molecules with an alternative saved-operator cache configured (sorted).

    The sorted order defines the combo bit layout: bit k of the combo id is the
    variant index (0 = base cache, 1 = alternative cache) of alt_species[k].
    """
    return tuple(
        sorted(
            molecule
            for molecule in molecules
            if os.environ.get(f"SCULPTOR_OPA_ALT_CACHE_DIR_{molecule.upper()}", "").strip()
        )
    )


def n_linelist_combos(alt_species: tuple[str, ...]) -> int:
    return 2 ** len(alt_species)


def combo_selection(combo: int, alt_species: tuple[str, ...]) -> dict[str, int]:
    """Map a combo id to a per-species variant selection {species: 0|1}."""
    if not 0 <= combo < n_linelist_combos(alt_species):
        raise ValueError(f"combo {combo} out of range for alt species {alt_species}")
    return {species: (combo >> bit) & 1 for bit, species in enumerate(alt_species)}


def forced_combo_from_env(alt_species: tuple[str, ...]) -> int | None:
    """Combo id pinned by ``SCULPTOR_FORCE_COMBO``, or None when unset.

    Unset (the default) means the production ``combo = shard % n_combos`` cycle,
    so generation is bit-identical to a run without this feature. When set, the
    value must be a valid combo id for the configured alternative caches -- an
    out-of-range or malformed value is a hard error, never a silent fallback.
    """
    raw = os.environ.get(FORCE_COMBO_ENV, "").strip()
    if not raw:
        return None
    try:
        combo = int(raw)
    except ValueError as exc:
        raise ValueError(f"{FORCE_COMBO_ENV}={raw!r} is not an integer") from exc
    n_combos = n_linelist_combos(alt_species)
    if not 0 <= combo < n_combos:
        raise ValueError(
            f"{FORCE_COMBO_ENV}={combo} out of range [0, {n_combos}) for alt species {alt_species}"
        )
    return combo


def shard_combo(shard_index: int, alt_species: tuple[str, ...]) -> int:
    forced = forced_combo_from_env(alt_species)
    if forced is not None:
        return forced
    return int(shard_index) % n_linelist_combos(alt_species)


def sample_coefficients(
    settings: OpacityPerturbationSettings,
    n_rows: int,
    seed: int,
    salt: int = 0,
) -> np.ndarray:
    """Draw the Fourier-field coefficient block for ``n_rows`` parameter rows.

    Deterministic given (seed, salt, n_rows): the SPMD harness calls this once
    over the full replayed table (salt=0), so every pod worker reproduces the
    identical block; the orchestrator calls it per sampler batch with
    salt=<halton cursor>, which is stable across resumes.
    """
    if not settings.enabled:
        return np.empty((n_rows, 0), dtype=np.float64)
    entropy = [int(seed) & 0xFFFFFFFF, _COEFFICIENT_SALT, int(salt) & 0xFFFFFFFFFFFF]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    sigmas = settings.coefficient_sigmas()
    return rng.standard_normal((int(n_rows), sigmas.shape[0])) * sigmas[None, :]


def shard_combo_column(n_rows: int, shard_size: int, alt_species: tuple[str, ...]) -> np.ndarray:
    """Per-row combo id under the fixed row->shard mapping (SPMD harness).

    Honors ``SCULPTOR_FORCE_COMBO`` exactly as :func:`shard_combo` does, so the
    recorded nuisance column always matches the operators actually used.
    """
    forced = forced_combo_from_env(alt_species)
    if forced is not None:
        return np.full(int(n_rows), float(forced), dtype=np.float64)
    combos = n_linelist_combos(alt_species)
    return ((np.arange(int(n_rows)) // int(shard_size)) % combos).astype(np.float64)


def append_perturbation_columns(
    parameters: np.ndarray,
    parameter_names: tuple[str, ...],
    settings: OpacityPerturbationSettings,
    coefficients: np.ndarray,
    combo_column: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Append the nuisance columns to a parameter table."""
    if not settings.enabled:
        return parameters, parameter_names
    coefficient_names = settings.coefficient_names()
    if coefficients.shape != (parameters.shape[0], len(coefficient_names)):
        raise ValueError(
            f"coefficient block shape {coefficients.shape} does not match "
            f"({parameters.shape[0]}, {len(coefficient_names)})"
        )
    duplicated = set(coefficient_names + (COMBO_COLUMN,)) & set(parameter_names)
    if duplicated:
        raise ValueError(f"parameter table already contains perturbation columns: {sorted(duplicated)}")
    extended = np.concatenate(
        [np.asarray(parameters, dtype=np.float64), coefficients, np.asarray(combo_column, dtype=np.float64)[:, None]],
        axis=1,
    )
    return extended, tuple(parameter_names) + coefficient_names + (COMBO_COLUMN,)


def extract_coefficient_matrix(
    parameters: np.ndarray,
    parameter_names: tuple[str, ...],
    settings: OpacityPerturbationSettings,
    species_subset: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Gather the (n_rows, n_species, n_coeff) coefficient tensor from a table.

    ``species_subset`` (defaults to settings.species) selects/orders the species
    axis; it must match the order used when compiling the perturbed spectrum
    function.
    """
    if not settings.enabled:
        raise ValueError("extract_coefficient_matrix called with perturbation disabled")
    species_list = settings.species if species_subset is None else species_subset
    name_to_index = {name: index for index, name in enumerate(parameter_names)}
    n_coeff = settings.n_coeff_per_species
    columns = np.empty((len(species_list), n_coeff), dtype=np.int64)
    for s_index, species in enumerate(species_list):
        wanted = [f"{NUISANCE_PREFIX}{species}_c0"]
        for mode in range(1, settings.n_modes + 1):
            wanted.append(f"{NUISANCE_PREFIX}{species}_a{mode}")
            wanted.append(f"{NUISANCE_PREFIX}{species}_b{mode}")
        for c_index, name in enumerate(wanted):
            if name not in name_to_index:
                raise KeyError(
                    f"Perturbation enabled but nuisance column {name!r} is missing from the "
                    f"parameter table; regenerate the table with SCULPTOR_OPACITY_PERTURB=1."
                )
            columns[s_index, c_index] = name_to_index[name]
    values = np.asarray(parameters, dtype=np.float32)
    return values[:, columns.reshape(-1)].reshape(len(values), len(species_list), n_coeff)


def fourier_basis(n_points: int, n_modes: int) -> np.ndarray:
    """(1 + 2*n_modes, n_points) design matrix over u in [0, 1].

    Row 0 is the constant term; rows (2m-1, 2m) are cos/sin of mode m. The
    production nu grids are ESLOG (uniform in log nu), so uniform u over the
    pixel index IS normalized log-wavenumber.
    """
    u = np.linspace(0.0, 1.0, int(n_points), dtype=np.float64)
    rows = [np.ones_like(u)]
    for mode in range(1, int(n_modes) + 1):
        rows.append(np.cos(2.0 * np.pi * mode * u))
        rows.append(np.sin(2.0 * np.pi * mode * u))
    return np.stack(rows, axis=0).astype(np.float32)


def evaluate_log_field(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Numpy reference for the field: coefficients (..., n_coeff) @ basis."""
    return np.tensordot(np.asarray(coefficients, dtype=np.float64), np.asarray(basis, dtype=np.float64), axes=([-1], [0]))
