"""MAST download helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


try:
    from astroquery.mast import Observations
except Exception:  # pragma: no cover - optional dependency
    Observations = None


@dataclass(frozen=True)
class MastProduct:
    target_name: str
    data_uri: str
    product_filename: str
    metadata: dict[str, Any]


class MastDownloader:
    """Small wrapper around astroquery or direct URL downloads."""

    def search(self, target_name: str) -> list[MastProduct]:  # pragma: no cover - depends on optional service
        if Observations is None:
            raise ImportError("astroquery is required for MAST target search.")
        observations = Observations.query_object(target_name)
        products = Observations.get_product_list(observations)
        results = []
        for row in products:
            results.append(
                MastProduct(
                    target_name=target_name,
                    data_uri=str(row["dataURI"]),
                    product_filename=str(row["productFilename"]),
                    metadata={key: row[key] for key in row.colnames},
                )
            )
        return results

    def download_url(self, url: str, destination: str | Path) -> Path:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination
