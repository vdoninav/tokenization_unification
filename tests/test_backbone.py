import torch

from tokenstudy.models.backbone import TransformerBackbone


def test_backbone_shapes():
    b = TransformerBackbone(d_model=128, n_heads=16, d_ff=256, n_layers=3, dropout=0.2)
    x = torch.randn(2, 41, 128)
    mask = torch.zeros(2, 41, dtype=torch.bool)
    out = b(x, padding_mask=mask)
    assert out.shape == (2, 41, 128)


def test_backbone_padding_mask():
    b = TransformerBackbone(d_model=64, n_heads=8, d_ff=128, n_layers=2, dropout=0.0)
    x = torch.randn(2, 10, 64)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, 5:] = True
    out = b(x, padding_mask=mask)
    assert out.shape == (2, 10, 64)
    assert torch.isfinite(out).all()
