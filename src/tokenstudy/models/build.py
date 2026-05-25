"""Factory that assembles a ForecastingModel from a spec."""
from __future__ import annotations

from ..tokenizers.base import Tokenizer
from ..tokenizers.discrete_bpe import DiscreteBpeTokenizer
from ..tokenizers.discrete_scalar import DiscreteScalarTokenizer
from ..tokenizers.patch import PatchTokenizer
from ..tokenizers.point import PointTokenizer
from ..tokenizers.variate import VariateTokenizer
from .backbone import TransformerBackbone
from .wrapper import ForecastingModel


def build_tokenizer(
    name: str, num_channels: int, lookback: int, horizon: int,
    d_model: int, vocab_size: int, bpe_base: int, bpe_final: int,
    patch_length: int, patch_stride: int,
) -> Tokenizer:
    kwargs = dict(hidden_dim=d_model, num_channels=num_channels, lookback=lookback, horizon=horizon)
    if name == "point":
        return PointTokenizer(**kwargs)
    if name == "patch":
        return PatchTokenizer(**kwargs, patch_length=patch_length, patch_stride=patch_stride)
    if name == "variate":
        return VariateTokenizer(**kwargs)
    if name == "discrete_scalar":
        return DiscreteScalarTokenizer(**kwargs, vocab_size=vocab_size)
    if name == "discrete_bpe":
        return DiscreteBpeTokenizer(**kwargs, base_vocab=bpe_base, final_vocab=bpe_final)
    raise ValueError(f"Unknown tokenizer {name!r}")


def build_model(
    tokenizer: str, num_channels: int, lookback: int, horizon: int,
    d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float,
    vocab_size: int, bpe_base: int, bpe_final: int,
    patch_length: int, patch_stride: int,
) -> ForecastingModel:
    tok = build_tokenizer(
        name=tokenizer, num_channels=num_channels, lookback=lookback, horizon=horizon,
        d_model=d_model, vocab_size=vocab_size, bpe_base=bpe_base, bpe_final=bpe_final,
        patch_length=patch_length, patch_stride=patch_stride,
    )
    bb = TransformerBackbone(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers, dropout=dropout,
    )
    return ForecastingModel(tokenizer=tok, backbone=bb)
