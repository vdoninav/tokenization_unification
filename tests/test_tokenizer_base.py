import pytest
import torch

from tokenstudy.tokenizers.base import Tokenizer, TokenBatch


def test_tokenbatch_fields():
    tb = TokenBatch(
        embeddings=torch.zeros(2, 10, 8),
        padding_mask=torch.zeros(2, 10, dtype=torch.bool),
        metadata={"lookback": 336},
    )
    assert tb.embeddings.shape == (2, 10, 8)
    assert tb.padding_mask.dtype == torch.bool


def test_tokenizer_is_abstract():
    with pytest.raises(TypeError):
        Tokenizer(hidden_dim=128, num_channels=7, lookback=336, horizon=96)
