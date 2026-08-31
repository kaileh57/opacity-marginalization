"""Posterior inference helpers."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from sculptor.npe.dataset import self_normalize
from sculptor.npe.training import ConditionalNPE, TrainingConfig, torch


class PosteriorInferencer:
    """Load a trained model and draw posterior samples for observed spectra."""

    def __init__(self, model_dir: str | Path, config: TrainingConfig | None = None) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for posterior inference.")
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        if config is None:
            valid_config_keys = {field.name for field in fields(TrainingConfig)}
            training_config = {
                key: value
                for key, value in metadata["training_config"].items()
                if key in valid_config_keys
            }
            train_config = TrainingConfig(**training_config)
        else:
            train_config = config
        self.input_length = int(metadata["input_length"])
        self.target_parameter_names = tuple(metadata.get("target_parameter_names", metadata.get("parameter_names", [])))
        self.conditioning_parameter_names = tuple(metadata.get("conditioning_parameter_names", ()))
        self.normalize_targets = bool(metadata.get("normalize_targets", False))
        self.normalize_conditioning = bool(metadata.get("normalize_conditioning", False))
        self.spectrum_normalization = str(metadata.get("spectrum_normalization", "none"))
        embedding_channels = metadata.get("embedding_channels")
        if embedding_channels is not None:
            embedding_channels = tuple(int(value) for value in embedding_channels)
        embedding_kernel_size = int(metadata.get("embedding_kernel_size", 7))
        embedding_conv_blocks = metadata.get("embedding_conv_blocks")
        if embedding_conv_blocks is not None:
            embedding_conv_blocks = int(embedding_conv_blocks)
        flow_hidden_features = metadata.get("flow_hidden_features")
        if flow_hidden_features is not None:
            flow_hidden_features = tuple(int(value) for value in flow_hidden_features)
        self.target_parameter_bounds = metadata.get("target_parameter_bounds") or {}
        if "model_target_bounds" in metadata:
            model_target_bounds = metadata["model_target_bounds"]
        else:
            model_target_bounds = self.target_parameter_bounds
        lower_bounds = tuple(
            float(model_target_bounds[name][0]) for name in self.target_parameter_names
        ) if model_target_bounds else None
        upper_bounds = tuple(
            float(model_target_bounds[name][1]) for name in self.target_parameter_names
        ) if model_target_bounds else None
        self.conditioning_bounds = metadata.get("conditioning_bounds") or {}
        self.model = ConditionalNPE(
            input_length=self.input_length,
            parameter_dim=int(metadata["parameter_dim"]),
            conditioning_dim=len(self.conditioning_parameter_names),
            config=train_config,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            embedding_channels=embedding_channels,
            embedding_kernel_size=embedding_kernel_size,
            embedding_conv_blocks=embedding_conv_blocks,
            flow_hidden_features=flow_hidden_features,
        )
        checkpoint_path = model_dir / "model.pt"
        if not checkpoint_path.exists():
            for fallback_name in ("model_best.pt", "model_final.pt"):
                fallback = model_dir / fallback_name
                if fallback.exists():
                    checkpoint_path = fallback
                    break
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

    def sample(
        self,
        spectra: np.ndarray,
        conditioning: np.ndarray | None = None,
        num_samples: int = 1000,
    ) -> np.ndarray:
        spectra = np.asarray(spectra, dtype=np.float32)
        if spectra.shape[-1] != self.input_length:
            raise ValueError(f"Expected spectra with length {self.input_length}, got {spectra.shape[-1]}")
        if self.spectrum_normalization == "selfnorm":
            spectra = np.stack([self_normalize(spectrum) for spectrum in spectra], axis=0)
        if self.conditioning_parameter_names:
            if conditioning is None:
                raise ValueError(
                    f"Conditioning values are required for {self.conditioning_parameter_names}"
                )
            conditioning_array = np.asarray(conditioning, dtype=np.float32)
            if conditioning_array.ndim == 1:
                conditioning_array = conditioning_array[:, None]
            if self.normalize_conditioning:
                for index, name in enumerate(self.conditioning_parameter_names):
                    lower, upper = self.conditioning_bounds[name]
                    conditioning_array[:, index] = (conditioning_array[:, index] - lower) / (upper - lower)
        else:
            conditioning_array = np.zeros((len(spectra), 0), dtype=np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(spectra[:, None, :])
            conditioning_tensor = torch.from_numpy(conditioning_array)
            samples = self.model.sample(tensor, conditioning_tensor, num_samples=num_samples)
        sample_array = samples.permute(1, 0, 2).cpu().numpy()
        if self.normalize_targets:
            for index, name in enumerate(self.target_parameter_names):
                lower, upper = self.target_parameter_bounds[name]
                sample_array[:, :, index] = sample_array[:, :, index] * (upper - lower) + lower
        return sample_array
