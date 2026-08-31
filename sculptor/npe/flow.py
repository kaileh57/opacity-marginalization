"""Conditional flow definitions."""

from __future__ import annotations

from dataclasses import dataclass


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

try:
    import zuko
except Exception:  # pragma: no cover - optional dependency
    zuko = None


@dataclass(frozen=True)
class FlowConfig:
    parameter_dim: int
    context_dim: int = 128
    transforms: int = 10
    hidden_features: tuple[int, ...] = (256, 256)
    lower_bounds: tuple[float, ...] | None = None
    upper_bounds: tuple[float, ...] | None = None


class RobustContextAdapter(nn.Module if nn is not None else object):
    """Learned per-feature scale adapter for robust NPE context embeddings.

    The adapter downweights embedding dimensions through a positive scale
    ``1 + softplus(raw_scale)``.  Initializing ``raw_scale`` to a large negative
    value makes the robust branch start numerically equivalent to vanilla NPE.
    """

    def __init__(self, context_dim: int, init_log_scale: float = -10.0) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to build RobustContextAdapter.")
        super().__init__()
        self.raw_scale = nn.Parameter(torch.full((context_dim,), float(init_log_scale)))

    @property
    def scale(self):
        return 1.0 + torch.nn.functional.softplus(self.raw_scale)

    def forward(self, context):
        return context / self.scale

    def penalty(self):
        delta = self.scale - 1.0
        return torch.mean(delta * delta)


class DiagonalGaussianFlow(nn.Module if nn is not None else object):
    """Fallback conditional density when spline-flow libraries are unavailable."""

    def __init__(self, parameter_dim: int, context_dim: int, hidden_features: tuple[int, ...]) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to construct the fallback conditional density.")
        super().__init__()
        layers = []
        in_features = context_dim
        for out_features in hidden_features:
            layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
            in_features = out_features
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_features, parameter_dim)
        self.log_std_head = nn.Linear(in_features, parameter_dim)

    def distribution(self, context):
        features = self.backbone(context)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-6.0, 3.0)
        return torch.distributions.Independent(torch.distributions.Normal(mean, log_std.exp()), 1)

    def log_prob(self, parameters, context):
        return self.distribution(context).log_prob(parameters)

    def sample(self, context, num_samples: int):
        distribution = self.distribution(context)
        return distribution.rsample((num_samples,))


class ConditionalFlow(nn.Module if nn is not None else object):
    """A wrapper around zuko's NSF flow with a diagonal-Gaussian fallback."""

    def __init__(self, config: FlowConfig) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to build the conditional flow.")
        super().__init__()
        self.config = config
        lower_bounds = config.lower_bounds
        upper_bounds = config.upper_bounds
        if (lower_bounds is None) != (upper_bounds is None):
            raise ValueError("lower_bounds and upper_bounds must both be provided or both be omitted.")
        if lower_bounds is not None and len(lower_bounds) != config.parameter_dim:
            raise ValueError("lower_bounds must match parameter_dim.")
        if upper_bounds is not None and len(upper_bounds) != config.parameter_dim:
            raise ValueError("upper_bounds must match parameter_dim.")
        if lower_bounds is not None and any(upper <= lower for lower, upper in zip(lower_bounds, upper_bounds)):
            raise ValueError("Each upper bound must be greater than its corresponding lower bound.")

        # Non-persistent buffers keep the bounds on-device without changing checkpoint compatibility.
        self.register_buffer(
            "lower_bounds",
            torch.tensor(lower_bounds, dtype=torch.float32) if lower_bounds is not None else None,
            persistent=False,
        )
        self.register_buffer(
            "upper_bounds",
            torch.tensor(upper_bounds, dtype=torch.float32) if upper_bounds is not None else None,
            persistent=False,
        )
        if zuko is not None:
            self.backend = zuko.flows.NSF(
                features=config.parameter_dim,
                context=config.context_dim,
                transforms=config.transforms,
                hidden_features=config.hidden_features,
            )
            self.backend_name = "zuko_nsf"
        else:
            self.backend = DiagonalGaussianFlow(
                parameter_dim=config.parameter_dim,
                context_dim=config.context_dim,
                hidden_features=config.hidden_features,
            )
            self.backend_name = "diagonal_gaussian"

    @property
    def is_bounded(self) -> bool:
        return self.lower_bounds is not None and self.upper_bounds is not None

    def _base_log_prob(self, parameters, context):
        if hasattr(self.backend, "log_prob"):
            return self.backend.log_prob(parameters, context)
        distribution = self.backend(context)
        return distribution.log_prob(parameters)

    def _base_sample(self, context, num_samples: int):
        if hasattr(self.backend, "sample"):
            return self.backend.sample(context, num_samples)
        distribution = self.backend(context)
        return distribution.sample((num_samples,))

    def _to_bounded(self, parameters):
        return self.lower_bounds + (self.upper_bounds - self.lower_bounds) * torch.sigmoid(parameters)

    def _to_unbounded(self, parameters):
        eps = torch.finfo(parameters.dtype).eps if parameters.is_floating_point() else 1.0e-6
        scaled = (parameters - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        scaled = scaled.clamp(eps, 1.0 - eps)
        return torch.log(scaled) - torch.log1p(-scaled)

    def _log_abs_det_jacobian(self, parameters):
        log_range = torch.log(self.upper_bounds - self.lower_bounds)
        log_sigmoid = -torch.nn.functional.softplus(-parameters)
        log_one_minus = -torch.nn.functional.softplus(parameters)
        return (log_range + log_sigmoid + log_one_minus).sum(dim=-1)

    def _in_bounds(self, parameters):
        return ((parameters >= self.lower_bounds) & (parameters <= self.upper_bounds)).all(dim=-1)

    def log_prob(self, parameters, context):
        if not self.is_bounded:
            return self._base_log_prob(parameters, context)
        in_bounds = self._in_bounds(parameters)
        bounded_parameters = parameters.clamp(self.lower_bounds, self.upper_bounds)
        unbounded_parameters = self._to_unbounded(bounded_parameters)
        log_prob = self._base_log_prob(unbounded_parameters, context) - self._log_abs_det_jacobian(unbounded_parameters)
        return torch.where(in_bounds, log_prob, torch.full_like(log_prob, float("-inf")))

    def sample(self, context, num_samples: int):
        samples = self._base_sample(context, num_samples)
        if not self.is_bounded:
            return samples
        return self._to_bounded(samples)


class BoundedNSF(ConditionalFlow if nn is not None else object):
    """NSF flow with sigmoid output bounding to enforce prior support.

    Wraps a ``ConditionalFlow`` and applies an element-wise sigmoid
    transform to map unbounded flow outputs into ``[lower, upper]``
    per dimension.  The ``log_prob`` method accounts for the Jacobian
    of the sigmoid change-of-variable so densities remain correct.
    """

    def __init__(self, config: FlowConfig) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to build BoundedNSF.")
        if config.lower_bounds is None or config.upper_bounds is None:
            raise ValueError("BoundedNSF requires lower_bounds and upper_bounds in FlowConfig")
        super().__init__(config)
