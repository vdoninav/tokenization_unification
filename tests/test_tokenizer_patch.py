import torch

from tokenstudy.tokenizers.patch import PatchTokenizer


def test_patch_encode_L():
    tok = PatchTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96,
        patch_length=16, patch_stride=8,
    )
    x = torch.randn(4, 7, 336)
    out = tok.encode(x)
    assert out.embeddings.shape == (28, 41, 16)


def test_patch_decode_shape():
    tok = PatchTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96,
        patch_length=16, patch_stride=8,
    )
    latent = torch.randn(28, 41, 16)
    y = tok.decode(latent)
    assert y.shape == (4, 96, 7)


def test_patch_sequence_length_matches_formula():
    tok = PatchTokenizer(
        hidden_dim=16, num_channels=21, lookback=336, horizon=96,
        patch_length=16, patch_stride=8,
    )
    assert tok.sequence_length() == (336 - 16) // 8 + 1
