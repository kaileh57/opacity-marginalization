"""Shard writing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from sculptor.utils import gcs as gcs_utils

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None


@dataclass(frozen=True)
class ShardPayload:
    spectra: np.ndarray
    spectra_clean: np.ndarray
    params: np.ndarray
    vmr_profiles: np.ndarray
    nu_grid: np.ndarray
    param_names: tuple[str, ...]
    metadata: dict[str, Any]
    error_bars: np.ndarray | None = None


class ShardWriter:
    """Write self-contained training shards to local disk or GCS."""

    def write(self, payload: ShardPayload, destination: str | Path) -> str:
        destination_str = str(destination)
        if destination_str.endswith(".h5") and h5py is not None:
            return self._write_hdf5(payload, destination_str)
        return self._write_npz(payload, destination_str)

    def _write_hdf5(self, payload: ShardPayload, destination: str) -> str:
        if gcs_utils.is_gcs_path(destination):
            temp_path = Path("_sculptor_tmp_shard.h5")
            actual_path = temp_path
        else:
            actual_path = Path(destination)
            actual_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(actual_path, "w") as handle:
            handle.create_dataset("spectra", data=payload.spectra, compression="gzip")
            handle.create_dataset("spectra_clean", data=payload.spectra_clean, compression="gzip")
            handle.create_dataset("params", data=payload.params, compression="gzip")
            handle.create_dataset("vmr_profiles", data=payload.vmr_profiles, compression="gzip")
            handle.create_dataset("nu_grid", data=payload.nu_grid)
            handle.create_dataset("param_names", data=np.asarray(payload.param_names, dtype="S"))
            handle.attrs["metadata"] = json.dumps(payload.metadata)
            if payload.error_bars is not None:
                handle.create_dataset("error_bars", data=payload.error_bars, compression="gzip")
        if gcs_utils.is_gcs_path(destination):
            gcs_utils.upload_file(actual_path, destination)
            actual_path.unlink(missing_ok=True)
        return destination

    def _write_npz(self, payload: ShardPayload, destination: str) -> str:
        actual_destination = destination if destination.endswith(".npz") else f"{destination}.npz"
        if gcs_utils.is_gcs_path(actual_destination):
            temp_path = Path("_sculptor_tmp_shard.npz")
            np.savez_compressed(
                temp_path,
                spectra=payload.spectra,
                spectra_clean=payload.spectra_clean,
                params=payload.params,
                vmr_profiles=payload.vmr_profiles,
                nu_grid=payload.nu_grid,
                param_names=np.asarray(payload.param_names),
                metadata=json.dumps(payload.metadata),
                error_bars=payload.error_bars,
            )
            gcs_utils.upload_file(temp_path, actual_destination)
            temp_path.unlink(missing_ok=True)
        else:
            destination_path = Path(actual_destination)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination_path,
                spectra=payload.spectra,
                spectra_clean=payload.spectra_clean,
                params=payload.params,
                vmr_profiles=payload.vmr_profiles,
                nu_grid=payload.nu_grid,
                param_names=np.asarray(payload.param_names),
                metadata=json.dumps(payload.metadata),
                error_bars=payload.error_bars,
            )
        return actual_destination
