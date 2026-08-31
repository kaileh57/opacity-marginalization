"""Spectrum embedding network definitions."""

from __future__ import annotations

from dataclasses import dataclass


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


@dataclass(frozen=True)
class EmbeddingConfig:
    input_length: int
    context_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128, 256)
    kernel_size: int = 7
    n_conv_blocks: int | None = None  # If set, overrides len(channels); extends with doubled last channel
    # --- architecture selector + transformer (1D-ViT) hyperparameters ---
    arch: str = "cnn"            # "cnn" | "transformer"
    patch_size: int = 32         # transformer: spectrum split into patches (bigger=fewer tokens=faster attention)
    d_model: int = 128           # transformer: token/embedding width
    n_heads: int = 4             # transformer: attention heads
    n_layers: int = 4            # transformer: encoder layers
    mlp_ratio: float = 4.0       # transformer: feed-forward expansion
    dropout: float = 0.1         # transformer: dropout


class SpectrumEmbeddingNet(nn.Module if nn is not None else object):
    """1D CNN encoder for spectra."""

    def __init__(self, config: EmbeddingConfig) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to build the spectrum embedding network.")
        super().__init__()

        # Determine channel list, extending if n_conv_blocks > len(channels).
        # Very long spectra need a bit more depth so the encoder can absorb
        # local structure before pooling reduces the wavelength axis.
        channels = list(config.channels)
        n_conv_blocks = config.n_conv_blocks
        if n_conv_blocks is None:
            n_conv_blocks = len(channels)
            if config.input_length >= 8192:
                n_conv_blocks = max(n_conv_blocks, 6)
        if n_conv_blocks > len(channels):
            last_ch = channels[-1]
            max_channel = channels[-1]
            for _ in range(n_conv_blocks - len(channels)):
                last_ch = min(last_ch * 2, max_channel)
                channels.append(last_ch)

        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=config.kernel_size, padding=config.kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, config.context_dim),
            nn.ReLU(),
        )

    def forward(self, spectra):  # type: ignore[override]
        features = self.features(spectra)
        return self.projection(features)


class SpectrumTransformerEmbeddingNet(nn.Module if nn is not None else object):
    """Patch-based 1D transformer (ViT-style) encoder for spectra.

    The spectrum is split into non-overlapping patches (a strided Conv1d tokenizer), a learnable CLS
    token is prepended, learnable positional embeddings are added, and a standard transformer encoder
    attends across patches. The CLS output is projected to the context vector. Self-attention over
    patches lets the embedding relate physically-coupled but wavelength-distant features (e.g. CO2 4.3um,
    CO fundamental 4.9um, H2O bands), which a local-receptive-field CNN smears into pooled statistics.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to build the spectrum embedding network.")
        super().__init__()
        import math

        patch = max(1, int(config.patch_size))
        self.patch_size = patch
        n_patches = math.ceil(config.input_length / patch)
        self.n_patches = n_patches
        d_model = int(config.d_model)

        # Conv1d tokenizer: kernel=stride=patch -> one token per patch. padding makes length divisible.
        pad = n_patches * patch - config.input_length
        self.pad = pad
        self.tokenizer = nn.Conv1d(1, d_model, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.input_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config.n_heads),
            dim_feedforward=int(d_model * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config.n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(nn.Linear(d_model, config.context_dim), nn.ReLU())

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, spectra):  # type: ignore[override]
        # spectra: (B, 1, L). Pad the wavelength axis up to an integer number of patches.
        if self.pad > 0:
            spectra = nn.functional.pad(spectra, (0, self.pad))
        tokens = self.tokenizer(spectra)              # (B, d_model, n_patches)
        tokens = tokens.transpose(1, 2)               # (B, n_patches, d_model)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)           # (B, n_patches+1, d_model)
        x = self.input_dropout(x + self.pos_embed)
        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])                   # CLS token summary
        return self.projection(cls_out)


class _ResBlock1D(nn.Module if nn is not None else object):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=p)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=p)
        self.bn2 = nn.BatchNorm1d(cout)
        self.act = nn.ReLU()
        self.down = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride), nn.BatchNorm1d(cout))
                     if (stride != 1 or cin != cout) else None)

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + idt)


class SpectrumResNet1D(nn.Module if nn is not None else object):
    """1D ResNet encoder: residual conv blocks with strided downsampling -> global avg pool -> context."""

    def __init__(self, config: EmbeddingConfig) -> None:
        if nn is None:
            raise ImportError("PyTorch is required.")
        super().__init__()
        chans = list(config.channels) or [32, 64, 128, 256]
        k = config.kernel_size
        self.stem = nn.Sequential(nn.Conv1d(1, chans[0], k, stride=2, padding=k // 2),
                                  nn.BatchNorm1d(chans[0]), nn.ReLU())
        blocks, cin = [], chans[0]
        for cout in chans:
            blocks.append(_ResBlock1D(cin, cout, k, stride=2)); cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(cin, config.context_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class SpectrumTCN(nn.Module if nn is not None else object):
    """Dilated temporal-conv encoder: exponentially growing dilation for a large receptive field
    without attention -> global avg pool -> context."""

    def __init__(self, config: EmbeddingConfig) -> None:
        if nn is None:
            raise ImportError("PyTorch is required.")
        super().__init__()
        chans = list(config.channels) or [32, 64, 128, 256]
        k = config.kernel_size
        layers, cin = [], 1
        # stem downsample to keep memory bounded on 1e4-long spectra
        layers += [nn.Conv1d(1, chans[0], k, stride=4, padding=k // 2), nn.BatchNorm1d(chans[0]), nn.ReLU()]
        cin = chans[0]
        for i, cout in enumerate(chans):
            d = 2 ** i
            layers += [nn.Conv1d(cin, cout, k, padding=(k // 2) * d, dilation=d),
                       nn.BatchNorm1d(cout), nn.ReLU()]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(cin, config.context_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.features(x))


class SpectrumGRU(nn.Module if nn is not None else object):
    """Strided-conv stem (downsample) -> bidirectional GRU over the sequence -> last hidden -> context."""

    def __init__(self, config: EmbeddingConfig) -> None:
        if nn is None:
            raise ImportError("PyTorch is required.")
        super().__init__()
        d = config.d_model
        # downsample ~16x so the GRU sees a manageable sequence length
        self.stem = nn.Sequential(
            nn.Conv1d(1, d // 2, config.kernel_size, stride=4, padding=config.kernel_size // 2), nn.ReLU(),
            nn.Conv1d(d // 2, d, config.kernel_size, stride=4, padding=config.kernel_size // 2), nn.ReLU())
        self.gru = nn.GRU(input_size=d, hidden_size=d, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=config.dropout)
        self.head = nn.Sequential(nn.Linear(2 * d, config.context_dim), nn.ReLU())

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)         # (B, T, d)
        out, _ = self.gru(h)
        return self.head(out[:, -1, :])          # last timestep (both directions)


class SpectrumMLP(nn.Module if nn is not None else object):
    """Weak baseline: aggressive conv/pool downsample -> flatten -> MLP -> context."""

    def __init__(self, config: EmbeddingConfig) -> None:
        if nn is None:
            raise ImportError("PyTorch is required.")
        super().__init__()
        # downsample to a fixed small length, then flatten
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(16, 16, 7, stride=4, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(64))
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(16 * 64, 512), nn.ReLU(),
                                 nn.Linear(512, config.context_dim), nn.ReLU())

    def forward(self, x):
        return self.mlp(self.stem(x))


def build_embedding(config: EmbeddingConfig):
    """Factory: select the embedding architecture from config.arch."""
    arch = (config.arch or "cnn").lower()
    table = {
        "cnn": SpectrumEmbeddingNet,
        "transformer": SpectrumTransformerEmbeddingNet, "vit": SpectrumTransformerEmbeddingNet,
        "attention": SpectrumTransformerEmbeddingNet,
        "resnet": SpectrumResNet1D, "resnet1d": SpectrumResNet1D,
        "tcn": SpectrumTCN, "dilated": SpectrumTCN,
        "gru": SpectrumGRU, "rnn": SpectrumGRU,
        "mlp": SpectrumMLP,
    }
    if arch not in table:
        raise ValueError(f"Unknown embedding arch {config.arch!r} (one of {sorted(table)}).")
    return table[arch](config)
