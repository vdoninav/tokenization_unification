import torch

from tokenstudy.tokenizers.variate import VariateTokenizer


def test_variate_encode_L_equals_C():
    tok = VariateTokenizer(hidden_dim=16, num_channels=21, lookback=336, horizon=96)
    x = torch.randn(4, 21, 336)
    out = tok.encode(x)
    assert out.embeddings.shape == (4, 21, 16)


def test_variate_decode_shape():
    tok = VariateTokenizer(hidden_dim=16, num_channels=21, lookback=336, horizon=96)
    latent = torch.randn(4, 21, 16)
    y = tok.decode(latent)
    assert y.shape == (4, 96, 21)
