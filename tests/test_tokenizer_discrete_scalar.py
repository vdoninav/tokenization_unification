import torch

from tokenstudy.tokenizers.discrete_scalar import DiscreteScalarTokenizer


def test_discrete_scalar_encode_shape():
    tok = DiscreteScalarTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96, vocab_size=64,
    )
    x = torch.randn(4, 7, 336)
    out = tok.encode(x)
    assert out.embeddings.shape == (28, 336, 16)


def test_discrete_scalar_decode_shape():
    tok = DiscreteScalarTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96, vocab_size=64,
    )
    latent = torch.randn(28, 336, 16)
    y = tok.decode(latent)
    assert y.shape == (4, 96, 7)


def test_discrete_scalar_bin_range():
    tok = DiscreteScalarTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96, vocab_size=64,
    )
    x = torch.randn(4, 7, 336) * 10
    ids = tok._binning(x)
    assert ids.min() >= 0 and ids.max() < 64


def test_discrete_scalar_decode_chunking_invariant():
    """Chunked decode must produce identical output to unchunked decode.

    Each row's decode (Linear -> softmax -> weighted sum) is independent across the B*C
    axis, so chunking is a pure memory optimization with no algorithmic effect.
    """
    torch.manual_seed(7)
    tok = DiscreteScalarTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96, vocab_size=64,
    )
    latent = torch.randn(28, 336, 16)
    tok.DECODE_CHUNK_SIZE = 1024
    y_full = tok.decode(latent.clone())
    tok.DECODE_CHUNK_SIZE = 4
    y_chunked = tok.decode(latent.clone())
    assert y_full.shape == y_chunked.shape
    assert torch.allclose(y_full, y_chunked, atol=1e-6, rtol=1e-6), (
        f"chunked decode diverged: max diff {(y_full - y_chunked).abs().max().item():.2e}"
    )


def test_discrete_scalar_decode_chunking_grad_flow():
    """Backward through chunked decode must produce gradients matching unchunked decode."""
    tok = DiscreteScalarTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96, vocab_size=64,
    )

    def run_with_chunk(cs: int) -> torch.Tensor:
        torch.manual_seed(11)
        latent = torch.randn(28, 336, 16, requires_grad=True)
        tok.DECODE_CHUNK_SIZE = cs
        y = tok.decode(latent)
        y.sum().backward()
        return latent.grad.clone()

    g_full = run_with_chunk(1024)
    g_chunked = run_with_chunk(4)
    assert torch.allclose(g_full, g_chunked, atol=1e-6, rtol=1e-6), (
        f"gradient diverged: max diff {(g_full - g_chunked).abs().max().item():.2e}"
    )
