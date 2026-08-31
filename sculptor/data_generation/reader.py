"""Dataset readers for generated training shards."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None


@dataclass(frozen=True)
class LoadedDataset:
    spectra: np.ndarray
    params: np.ndarray
    param_names: tuple[str, ...]
    metadata: dict[str, Any]


def _decode_param_names(values: np.ndarray) -> tuple[str, ...]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return tuple(decoded)


def _load_npz(path: Path) -> LoadedDataset:
    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        return LoadedDataset(
            spectra=np.asarray(data["spectra"], dtype=np.float32),
            params=np.asarray(data["params"], dtype=np.float32),
            param_names=_decode_param_names(np.asarray(data["param_names"])),
            metadata=metadata,
        )


def _load_hdf5(path: Path) -> LoadedDataset:
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 shards.")
    with h5py.File(path, "r") as handle:
        return LoadedDataset(
            spectra=np.asarray(handle["spectra"][...], dtype=np.float32),
            params=np.asarray(handle["params"][...], dtype=np.float32),
            param_names=_decode_param_names(np.asarray(handle["param_names"][...])),
            metadata=json.loads(str(handle.attrs["metadata"])),
        )


def load_generated_dataset(path: str | Path) -> LoadedDataset:
    directory = Path(path)
    shards = []
    for shard_path in sorted(directory.glob("*")):
        if shard_path.suffix == ".npz":
            shards.append(_load_npz(shard_path))
        elif shard_path.suffix in {".h5", ".hdf5"} and h5py is not None:
            shards.append(_load_hdf5(shard_path))
    if not shards:
        raise FileNotFoundError(f"No readable shards found in {directory}")

    param_names = shards[0].param_names
    for shard in shards[1:]:
        if shard.param_names != param_names:
            raise ValueError("Shard parameter layouts do not match.")

    merged_metadata = dict(shards[0].metadata)
    return LoadedDataset(
        spectra=np.concatenate([shard.spectra for shard in shards], axis=0),
        params=np.concatenate([shard.params for shard in shards], axis=0),
        param_names=param_names,
        metadata=merged_metadata,
    )
