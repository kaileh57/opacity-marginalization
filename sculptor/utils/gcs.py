"""Minimal GCS helpers with local-path fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    from google.cloud import storage as gcs_storage
except Exception:  # pragma: no cover - optional dependency
    gcs_storage = None


@dataclass(frozen=True)
class GCSUri:
    bucket: str
    path: str


def is_gcs_path(path: str | Path) -> bool:
    return str(path).startswith("gs://")


def parse_gcs_uri(uri: str) -> GCSUri:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    remainder = uri[5:]
    bucket, _, path = remainder.partition("/")
    return GCSUri(bucket=bucket, path=path)


def exists(path: str | Path) -> bool:
    if is_gcs_path(path):
        if gcs_storage is None:
            return False
        uri = parse_gcs_uri(str(path))
        client = gcs_storage.Client()
        blob = client.bucket(uri.bucket).blob(uri.path)
        return blob.exists()
    return Path(path).exists()


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    if is_gcs_path(path):
        if gcs_storage is None:
            raise ImportError("google-cloud-storage is required to read GCS paths.")
        uri = parse_gcs_uri(str(path))
        client = gcs_storage.Client()
        blob = client.bucket(uri.bucket).blob(uri.path)
        return blob.download_as_text(encoding=encoding)
    return Path(path).read_text(encoding=encoding)


def write_text(path: str | Path, contents: str, encoding: str = "utf-8") -> None:
    if is_gcs_path(path):
        if gcs_storage is None:
            raise ImportError("google-cloud-storage is required to write GCS paths.")
        uri = parse_gcs_uri(str(path))
        client = gcs_storage.Client()
        blob = client.bucket(uri.bucket).blob(uri.path)
        blob.upload_from_string(contents, content_type="text/plain")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents, encoding=encoding)


def upload_file(local_path: str | Path, destination: str | Path) -> None:
    if is_gcs_path(destination):
        if gcs_storage is None:
            raise ImportError("google-cloud-storage is required to upload to GCS.")
        uri = parse_gcs_uri(str(destination))
        client = gcs_storage.Client()
        blob = client.bucket(uri.bucket).blob(uri.path)
        blob.upload_from_filename(str(local_path))
        return
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(Path(local_path).read_bytes())


def list_paths(prefix: str | Path) -> Iterable[str]:
    if is_gcs_path(prefix):
        if gcs_storage is None:
            return []
        uri = parse_gcs_uri(str(prefix))
        client = gcs_storage.Client()
        blobs = client.list_blobs(uri.bucket, prefix=uri.path)
        return [f"gs://{uri.bucket}/{blob.name}" for blob in blobs]
    prefix_path = Path(prefix)
    if not prefix_path.exists():
        return []
    return [str(path) for path in prefix_path.iterdir()]
