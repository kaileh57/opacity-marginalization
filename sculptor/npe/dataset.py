"""HDF5 shard dataset for NPE training."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object


def self_normalize(spectrum):
    """Per-spectrum self-normalization: log, subtract mean, divide by std.

    This removes absolute flux scale (fixing distance/unit dependence)
    while preserving the spatial pattern of absorption features.
    A shuffled spectrum has the same mean and std but completely different
    spatial structure, so the CNN must learn from feature positions.

    Applied identically to training spectra and observed spectra.
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)
    finite_positive = np.isfinite(spectrum) & (spectrum > 0.0)
    if not np.any(finite_positive):
        return np.zeros_like(spectrum, dtype=np.float32)
    filled = spectrum.copy()
    filled[~np.isfinite(filled)] = np.nanmedian(spectrum[finite_positive])
    log_spec = np.log(np.maximum(filled, 1e-30)).astype(np.float32)
    mean = np.mean(log_spec)
    std = np.std(log_spec)
    if std < 1e-10:
        return np.zeros_like(log_spec)
    return ((log_spec - mean) / std).astype(np.float32)


class ShardDataset(Dataset if torch is not None else object):
    """Load all HDF5 shards into memory and serve (spectrum, targets, conditioning) tuples.

    Parameters
    ----------
    shard_dir : str or Path
        Directory containing ``shard_*.h5`` files.
    spectrum_key : str
        HDF5 dataset name for spectra (default ``"spectra"`` = noisy).
    normalize_params : bool
        If True, normalize retrieval targets to [0, 1] using ``prior_bounds``.
    prior_bounds : dict[str, tuple[float, float]] or None
        ``{param_name: (lower, upper)}``.  Required when ``normalize_params=True``.
    conditioning_names : tuple[str, ...] or None
        Parameter names that are conditioning variables (appended to context, not targets).
    exclude_params : tuple[str, ...] or None
        Parameter names to exclude from both targets and conditioning (e.g., ``("radius_rjup",)``).
    normalization : str
        ``"log"`` for the original log-flux z-score; ``"selfnorm"`` for
        per-spectrum self-normalization (log, subtract mean, divide by std).
    downsample : int
        Spectral downsampling factor. Use ``1`` to keep full resolution.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        spectrum_key: str = "spectra",
        normalize_params: bool = True,
        prior_bounds: dict[str, tuple[float, float]] | None = None,
        conditioning_names: tuple[str, ...] | None = None,
        exclude_params: tuple[str, ...] | None = None,
        downsample: int = 1,
        normalization: str = "log",
        flux_augmentation: bool = False,
    ) -> None:
        shard_dir = Path(shard_dir)
        downsample = int(downsample)
        if downsample < 1:
            raise ValueError("downsample must be >= 1")
        shard_files = sorted(shard_dir.glob("shard_*.h5"))
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.h5 files found in {shard_dir}")

        all_spectra = []
        all_params = []
        param_names = None

        for path in shard_files:
            with h5py.File(path, "r") as f:
                all_spectra.append(f[spectrum_key][:])
                all_params.append(f["params"][:])
                if param_names is None:
                    param_names = tuple(n.decode() if isinstance(n, bytes) else str(n) for n in f["param_names"][:])

        spectra_raw = np.concatenate(all_spectra, axis=0).astype(np.float32)
        if downsample > 1:
            n_wav = spectra_raw.shape[1]
            trim = n_wav - (n_wav % downsample)
            spectra_raw = spectra_raw[:, :trim].reshape(spectra_raw.shape[0], -1, downsample).mean(axis=2)
        self.spectra = spectra_raw
        self.params = np.concatenate(all_params, axis=0).astype(np.float32)
        self.param_names = param_names
        self.n_spectra = len(self.spectra)
        self.downsample = downsample
        self.normalization = normalization

        exclude_params = set(exclude_params or ())
        conditioning_names = conditioning_names or ()
        self.conditioning_names = tuple(conditioning_names)
        self.target_names = tuple(
            n for n in self.param_names
            if n not in set(self.conditioning_names) and n not in exclude_params
        )

        self.target_indices = np.array([self.param_names.index(n) for n in self.target_names])
        self.conditioning_indices = np.array([self.param_names.index(n) for n in self.conditioning_names])

        # Normalization
        self.normalize_params = normalize_params
        if normalize_params:
            if prior_bounds is None:
                raise ValueError("prior_bounds required when normalize_params=True")
            self.prior_bounds = prior_bounds
            self.target_lower = np.array([prior_bounds[n][0] for n in self.target_names], dtype=np.float32)
            self.target_upper = np.array([prior_bounds[n][1] for n in self.target_names], dtype=np.float32)
            self.cond_lower = np.array([prior_bounds[n][0] for n in self.conditioning_names], dtype=np.float32)
            self.cond_upper = np.array([prior_bounds[n][1] for n in self.conditioning_names], dtype=np.float32)
        else:
            self.target_lower = self.target_upper = None
            self.cond_lower = self.cond_upper = None

        # Spectrum normalization
        if normalization == "selfnorm":
            # Per-spectrum self-normalization: no global stats needed
            normed = np.empty_like(self.spectra)
            for i in range(len(self.spectra)):
                normed[i] = self_normalize(self.spectra[i])
            self.spectra_normalized = normed
            # No global mean/std — each spectrum is self-contained
            self._spec_mean = None
            self._spec_std = None
        else:
            # Original log normalization (global z-score)
            eps = np.float32(1e-30)
            log_spectra = np.log(np.maximum(self.spectra, eps))
            self._spec_mean = log_spectra.mean(axis=0)
            self._spec_std = log_spectra.std(axis=0)
            self._spec_std[self._spec_std < 1e-8] = 1.0
            self.spectra_normalized = (log_spectra - self._spec_mean) / self._spec_std

    def _normalize_targets(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.target_lower) / (self.target_upper - self.target_lower)

    def _denormalize_targets(self, normed: np.ndarray) -> np.ndarray:
        return normed * (self.target_upper - self.target_lower) + self.target_lower

    def _normalize_conditioning(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.cond_lower) / (self.cond_upper - self.cond_lower)

    def __len__(self) -> int:
        return self.n_spectra

    def __getitem__(self, idx):
        spectrum = self.spectra_normalized[idx]
        raw_params = self.params[idx]

        targets = raw_params[self.target_indices]
        if self.normalize_params:
            targets = self._normalize_targets(targets)

        if len(self.conditioning_indices) > 0:
            conditioning = raw_params[self.conditioning_indices]
            if self.normalize_params:
                conditioning = self._normalize_conditioning(conditioning)
        else:
            conditioning = np.zeros(0, dtype=np.float32)

        return (
            torch.from_numpy(spectrum[None, :]),  # (1, n_wav) for Conv1d
            torch.from_numpy(targets),
            torch.from_numpy(conditioning),
        )

    def get_raw_targets(self, indices=None) -> np.ndarray:
        """Return un-normalized target parameters."""
        if indices is not None:
            return self.params[indices][:, self.target_indices]
        return self.params[:, self.target_indices]


def train_val_split(dataset: "ShardDataset", val_count: int = 1000, seed: int = 42):
    """Split a ShardDataset into train and validation subsets."""
    if torch is None:
        raise ImportError("PyTorch required")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_subset = torch.utils.data.Subset(dataset, train_indices.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_indices.tolist())
    return train_subset, val_subset, train_indices, val_indices
