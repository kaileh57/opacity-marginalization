"""CLI entry point for synthetic dataset generation."""

from __future__ import annotations

import os

_default_exojax_root = os.path.expanduser("~/exojax_db")
if not os.path.isdir(_default_exojax_root) and os.path.isdir("/tmp/exojax_db"):
    _default_exojax_root = "/tmp/exojax_db"
os.environ.setdefault("EXOJAX_ROOT", _default_exojax_root)

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sculptor.data_generation.orchestrator import GenerationConfig, GenerationOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="g395h")
    parser.add_argument("--config-path", "--config", dest="config_path")
    parser.add_argument("--samples", "--num-spectra", dest="samples", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--output", default="./artifacts/generation")
    parser.add_argument("--manifest-path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fastchem-path")
    parser.add_argument("--device-batch-size", type=int)
    parser.add_argument(
        "--parameter-file",
        help="Optional .npy file containing explicit full parameter vectors to generate.",
    )
    args = parser.parse_args()

    orchestrator = GenerationOrchestrator(
        GenerationConfig(
            track=args.track,
            config_path=args.config_path,
            samples=args.samples,
            shard_size=args.shard_size,
            output=args.output,
            manifest_path=args.manifest_path,
            seed=args.seed,
            fastchem_path=args.fastchem_path,
            device_batch_size=args.device_batch_size,
            parameter_file=args.parameter_file,
        )
    )
    result = orchestrator.run()
    print(json.dumps({"written": result}, indent=2))


if __name__ == "__main__":
    main()
