"""Training loop for neural posterior estimation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from sculptor.npe.calibration import empirical_coverage
from sculptor.npe.embedding import EmbeddingConfig, SpectrumEmbeddingNet, build_embedding, torch
from sculptor.npe.flow import ConditionalFlow, FlowConfig, RobustContextAdapter
from sculptor.utils.logging import get_logger


if torch is not None:  # pragma: no branch - runtime guarded
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - optional dependency
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    validation_split: float = 0.1
    context_dim: int = 128
    flow_transforms: int = 10
    device: str = "cpu"
    robust: bool = False
    robust_lambda: float = 0.0
    robust_scale_init: float = -10.0
    embedding_arch: str = "cnn"  # "cnn" | "transformer" | "resnet" | "tcn" | "gru" | "mlp"
    # embedding hyperparams persisted here so they flow into metadata (asdict) and are restored at
    # inference -> models with non-default embeddings reload faithfully (fixes the reload mismatch).
    embed_d_model: int = 128
    embed_patch_size: int = 32
    embed_n_heads: int = 4
    embed_n_layers: int = 4
    save_best: bool = True  # restore + save the lowest-val-NLL epoch (not the final, possibly-overfit one)
    # forward-model floor / discrepancy augmentation (misspecification-robust training)
    discrepancy_template_path: str | None = None
    discrepancy_amplitude: float = 1.0
    discrepancy_prob: float = 0.6
    # PREEMPTION RESILIENCE (spot/preemptible hosts): periodically write a resumable
    # checkpoint (model + optimizer + epoch + best-val state + history) so a preempted
    # run restarts from the last checkpoint instead of epoch 0. Off unless a path is set.
    checkpoint_path: str | None = None
    checkpoint_every: int = 2  # epochs between checkpoint writes
    resume: bool = True  # load checkpoint_path at start if it exists


@dataclass(frozen=True)
class ParameterLayout:
    parameter_names: tuple[str, ...]
    target_names: tuple[str, ...]
    conditioning_names: tuple[str, ...]

    @property
    def target_indices(self) -> tuple[int, ...]:
        return tuple(self.parameter_names.index(name) for name in self.target_names)

    @property
    def conditioning_indices(self) -> tuple[int, ...]:
        return tuple(self.parameter_names.index(name) for name in self.conditioning_names)


def resolve_parameter_layout(
    parameter_names: tuple[str, ...],
    conditioning_names: tuple[str, ...] = (),
) -> ParameterLayout:
    conditioning = tuple(name for name in parameter_names if name in set(conditioning_names))
    targets = tuple(name for name in parameter_names if name not in set(conditioning))
    return ParameterLayout(
        parameter_names=tuple(parameter_names),
        target_names=targets,
        conditioning_names=conditioning,
    )


def split_parameter_array(parameters: np.ndarray, layout: ParameterLayout) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(parameters, dtype=np.float32)
    targets = values[:, layout.target_indices]
    if layout.conditioning_indices:
        conditioning = values[:, layout.conditioning_indices]
    else:
        conditioning = np.zeros((len(values), 0), dtype=np.float32)
    return targets, conditioning


def resolve_target_bounds(
    layout: ParameterLayout,
    parameter_bounds: dict[str, tuple[float, float]] | None,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    if parameter_bounds is None:
        return None, None
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for name in layout.target_names:
        if name not in parameter_bounds:
            raise KeyError(f"Missing prior bounds for target parameter {name!r}")
        lower, upper = parameter_bounds[name]
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))
    return tuple(lower_bounds), tuple(upper_bounds)


class ConditionalNPE(nn.Module if nn is not None else object):
    """Joint embedding + conditional density model."""

    def __init__(
        self,
        input_length: int,
        parameter_dim: int,
        conditioning_dim: int,
        config: TrainingConfig,
        lower_bounds: tuple[float, ...] | None = None,
        upper_bounds: tuple[float, ...] | None = None,
        embedding_channels: tuple[int, ...] | None = None,
        embedding_kernel_size: int = 7,
        embedding_conv_blocks: int | None = None,
        flow_hidden_features: tuple[int, ...] | None = None,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for NPE training.")
        super().__init__()
        self.conditioning_dim = conditioning_dim
        self.embedding = build_embedding(
            EmbeddingConfig(
                input_length=input_length,
                context_dim=config.context_dim,
                channels=embedding_channels or (32, 64, 128, 256),
                kernel_size=embedding_kernel_size,
                n_conv_blocks=embedding_conv_blocks,
                arch=config.embedding_arch,
                d_model=getattr(config, "embed_d_model", 128),
                patch_size=getattr(config, "embed_patch_size", 32),
                n_heads=getattr(config, "embed_n_heads", 4),
                n_layers=getattr(config, "embed_n_layers", 4),
            )
        )
        self.robust_adapter = (
            RobustContextAdapter(config.context_dim, init_log_scale=config.robust_scale_init)
            if config.robust
            else None
        )
        self.flow = ConditionalFlow(
            FlowConfig(
                parameter_dim=parameter_dim,
                context_dim=config.context_dim + conditioning_dim,
                transforms=config.flow_transforms,
                hidden_features=flow_hidden_features or (256, 256),
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
        )

    def _build_context(self, spectra, conditioning):
        context = self.embedding(spectra)
        if self.robust_adapter is not None:
            context = self.robust_adapter(context)
        if self.conditioning_dim == 0:
            return context
        if conditioning is None:
            raise ValueError("Conditioning inputs are required for this model.")
        if conditioning.ndim == 1:
            conditioning = conditioning.unsqueeze(-1)
        return torch.cat([context, conditioning], dim=-1)

    def log_prob(self, spectra, conditioning, parameters):
        context = self._build_context(spectra, conditioning)
        return self.flow.log_prob(parameters, context)

    def sample(self, spectra, conditioning, num_samples: int):
        context = self._build_context(spectra, conditioning)
        return self.flow.sample(context, num_samples)

    def robust_penalty(self):
        if self.robust_adapter is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.robust_adapter.penalty()


class NPETrainer:
    """Train an NPE model on generated shards or in-memory arrays."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.logger = get_logger(__name__)

    def _split(
        self,
        spectra: np.ndarray,
        parameters: np.ndarray,
        conditioning: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(spectra)
        split = max(1, int(count * (1.0 - self.config.validation_split)))
        split = min(split, count)
        return (
            spectra[:split],
            parameters[:split],
            conditioning[:split],
            spectra[split:],
            parameters[split:],
            conditioning[split:],
        )

    def train(
        self,
        spectra: np.ndarray,
        parameters: np.ndarray,
        output_dir: str | Path,
        parameter_names: tuple[str, ...],
        conditioning_names: tuple[str, ...] = (),
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch is required to train the NPE.")

        spectra = np.asarray(spectra, dtype=np.float32)
        parameters = np.asarray(parameters, dtype=np.float32)
        layout = resolve_parameter_layout(parameter_names, conditioning_names)
        target_params, conditioning = split_parameter_array(parameters, layout)
        target_lower_bounds, target_upper_bounds = resolve_target_bounds(layout, parameter_bounds)
        train_x, train_y, train_c, val_x, val_y, val_c = self._split(spectra, target_params, conditioning)

        model = ConditionalNPE(
            input_length=spectra.shape[-1],
            parameter_dim=target_params.shape[-1],
            conditioning_dim=conditioning.shape[-1],
            config=self.config,
            lower_bounds=target_lower_bounds,
            upper_bounds=target_upper_bounds,
        )
        device = torch.device(self.config.device)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_x[:, None, :]),
                torch.from_numpy(train_c),
                torch.from_numpy(train_y),
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # forward-model floor / discrepancy augmentation (injected into TRAIN batches only)
        discrepancy = None
        if self.config.discrepancy_template_path:
            from sculptor.npe.discrepancy import FloorDiscrepancy
            discrepancy = FloorDiscrepancy.from_file(
                self.config.discrepancy_template_path,
                amplitude=self.config.discrepancy_amplitude,
                prob=self.config.discrepancy_prob,
            )
            print(f"[train] floor-discrepancy augmentation ON: template={self.config.discrepancy_template_path} "
                  f"amp={self.config.discrepancy_amplitude} prob={self.config.discrepancy_prob}", flush=True)
        aug_rng = np.random.default_rng(0)

        history: list[dict[str, float]] = []
        # best-val checkpointing: keep the model at the lowest val NLL, not the (possibly overfit)
        # final epoch. Essential for a FAIR baseline-vs-robust comparison (both can overfit at high epochs).
        best_val = float("inf")
        best_state = None
        best_epoch = -1
        start_epoch = 0
        # ---- preemption resilience: resume from the last periodic checkpoint ----
        ckpt_path = Path(self.config.checkpoint_path) if self.config.checkpoint_path else None
        if ckpt_path is not None and self.config.resume and ckpt_path.exists():
            try:
                ck = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ck["model_state"])
                optimizer.load_state_dict(ck["optimizer_state"])
                start_epoch = int(ck["epoch"]) + 1
                best_val = float(ck.get("best_val", float("inf")))
                best_epoch = int(ck.get("best_epoch", -1))
                if ck.get("best_state") is not None:
                    best_state = {k: v.detach().cpu().clone() for k, v in ck["best_state"].items()}
                history = list(ck.get("history", []))
                self.logger.info(
                    "Resumed from checkpoint",
                    extra={"extra_fields": {"checkpoint": str(ckpt_path), "start_epoch": start_epoch,
                                            "best_val_nll": best_val, "best_epoch": best_epoch}},
                )
                print(f"[train] RESUMED from {ckpt_path} at epoch {start_epoch} "
                      f"(best_val={best_val:.4f} @ epoch {best_epoch})", flush=True)
            except Exception as exc:  # corrupt/partial checkpoint -> start clean rather than die
                print(f"[train] checkpoint at {ckpt_path} unusable ({exc}); starting from epoch 0", flush=True)
                start_epoch = 0

        def _write_checkpoint(epoch: int) -> None:
            if ckpt_path is None:
                return
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "best_state": best_state,
                    "history": history,
                },
                tmp,
            )
            tmp.replace(ckpt_path)  # atomic: a preemption mid-write never corrupts the checkpoint
            print(f"[train] checkpoint written at epoch {epoch} -> {ckpt_path}", flush=True)

        if start_epoch >= self.config.epochs:
            print(f"[train] checkpoint already at epoch {start_epoch} >= {self.config.epochs}; "
                  "skipping training loop", flush=True)
        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            epoch_losses = []
            for batch_spectra, batch_conditioning, batch_params in train_loader:
                batch_spectra = batch_spectra.to(device)
                if discrepancy is not None:
                    batch_spectra = discrepancy.perturb_batch(batch_spectra, aug_rng)
                batch_conditioning = batch_conditioning.to(device)
                batch_params = batch_params.to(device)
                nll = -model.log_prob(batch_spectra, batch_conditioning, batch_params).mean()
                loss = nll + self.config.robust_lambda * model.robust_penalty()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(nll.item()))

            metrics = {"epoch": epoch, "train_nll": float(np.mean(epoch_losses))}
            if len(val_x) > 0:
                model.eval()
                with torch.no_grad():
                    val_spectra = torch.from_numpy(val_x[:, None, :]).to(device)
                    val_conditioning = torch.from_numpy(val_c).to(device)
                    val_params = torch.from_numpy(val_y).to(device)
                    val_nll = -model.log_prob(val_spectra, val_conditioning, val_params).mean().item()
                    sample_count = min(64, len(val_spectra))
                    samples = model.sample(
                        val_spectra[:sample_count],
                        val_conditioning[:sample_count],
                        num_samples=256,
                    )
                    coverage = empirical_coverage(
                        samples.permute(1, 0, 2).cpu().numpy(),
                        val_y[:sample_count],
                    )
                metrics["val_nll"] = float(val_nll)
                metrics["coverage_90"] = coverage[0.9]
                if getattr(self.config, "save_best", True) and val_nll < best_val:
                    best_val = float(val_nll)
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append(metrics)
            self.logger.info("Completed training epoch", extra={"extra_fields": metrics})
            every = max(1, int(self.config.checkpoint_every))
            if ckpt_path is not None and ((epoch + 1) % every == 0 or epoch == self.config.epochs - 1):
                _write_checkpoint(epoch)

        # restore best-val weights (avoids saving an overfit final-epoch model)
        if best_state is not None:
            model.load_state_dict(best_state)
            self.logger.info("Restored best-val checkpoint",
                             extra={"extra_fields": {"best_epoch": best_epoch, "best_val_nll": best_val}})

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        metadata = {
            "training_config": asdict(self.config),
            "input_length": int(spectra.shape[-1]),
            "parameter_dim": int(target_params.shape[-1]),
            "parameter_names": list(layout.parameter_names),
            "target_parameter_names": list(layout.target_names),
            "conditioning_parameter_names": list(layout.conditioning_names),
            "embedding_channels": [32, 64, 128, 256],
            "embedding_kernel_size": 7,
            "embedding_conv_blocks": None,
            "flow_hidden_features": [256, 256],
            "spectrum_normalization": "none",
            "normalize_targets": False,
            "normalize_conditioning": False,
            "model_target_bounds": {
                name: [float(lower), float(upper)]
                for name, lower, upper in zip(layout.target_names, target_lower_bounds or (), target_upper_bounds or ())
            }
            if target_lower_bounds is not None and target_upper_bounds is not None
            else None,
            "target_parameter_bounds": {
                name: [float(lower), float(upper)]
                for name, lower, upper in zip(layout.target_names, target_lower_bounds or (), target_upper_bounds or ())
            }
            if target_lower_bounds is not None and target_upper_bounds is not None
            else None,
            "conditioning_bounds": {
                name: [float(parameter_bounds[name][0]), float(parameter_bounds[name][1])]
                for name in layout.conditioning_names
            }
            if parameter_bounds is not None and layout.conditioning_names
            else None,
            "history": history,
        }
        (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"checkpoint": str(checkpoint_path), "metadata": metadata}
