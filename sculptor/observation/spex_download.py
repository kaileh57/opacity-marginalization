"""SpeX Prism Library download helpers."""

from __future__ import annotations

from pathlib import Path

import requests


class SpexDownloader:
    """Download spectra from direct SpeX Prism Library URLs."""

    def download(self, url: str, destination: str | Path) -> Path:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination
