import torch

from tokenstudy.models.build import build_model


def _test_tokenizer_forward(name: str, C: int = 7):
    model = build_model(
        tokenizer=name, num_channels=C, lookback=336, horizon=96,
        d_model=32, n_heads=4, d_ff=64, n_layers=2, dropout=0.0,
        vocab_size=64, bpe_base=20, bpe_final=40, patch_length=16, patch_stride=8,
    )
    x = torch.randn(2, C, 336)
    if name == "discrete_bpe":
        model.tokenizer.fit(x)
    y = model(x)
    assert y.shape == (2, 96, C)


def test_point(): _test_tokenizer_forward("point")
def test_patch(): _test_tokenizer_forward("patch")
def test_variate(): _test_tokenizer_forward("variate")
def test_discrete_scalar(): _test_tokenizer_forward("discrete_scalar")
def test_discrete_bpe(): _test_tokenizer_forward("discrete_bpe")
