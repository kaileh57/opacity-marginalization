"""Helpers for optional JAX/TPU execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


try:
    import jax
except Exception:  # pragma: no cover - optional dependency
    jax = None


@dataclass(frozen=True)
class DeviceSummary:
    backend: str
    device_count: int
    platform: str


def detect_devices() -> DeviceSummary:
    if jax is None:
        return DeviceSummary(backend="python", device_count=1, platform="cpu")
    devices = jax.devices()
    platform = devices[0].platform if devices else "unknown"
    return DeviceSummary(backend="jax", device_count=len(devices), platform=platform)


def batched_map(function: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
    return [function(item) for item in items]
