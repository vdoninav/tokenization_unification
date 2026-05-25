import torch

from tokenstudy.tokenizers.point import PointTokenizer


def test_point_encode_shape():
    tok = PointTokenizer(hidden_dim=16, num_channels=7, lookback=336, horizon=96)
    x = torch.randn(4, 7, 336)
    out = tok.encode(x)
    assert out.embeddings.shape == (28, 336, 16)
    assert out.padding_mask.shape == (28, 336)
    assert not out.padding_mask.any()


def test_point_decode_shape():
    tok = PointTokenizer(hidden_dim=16, num_channels=7, lookback=336, horizon=96)
    latent = torch.randn(28, 336, 16)
    y = tok.decode(latent)
    assert y.shape == (4, 96, 7)


def test_point_round_trip_gradient():
    tok = PointTokenizer(hidden_dim=16, num_channels=7, lookback=336, horizon=96)
    x = torch.randn(2, 7, 336, requires_grad=True)
    out = tok.encode(x)
    latent = out.embeddings
    y = tok.decode(latent)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
