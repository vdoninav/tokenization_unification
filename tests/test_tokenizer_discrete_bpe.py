import numpy as np
import pytest
import torch

from tokenstudy.tokenizers.discrete_bpe import DiscreteBpeTokenizer
from tokenstudy.tokenizers._bpe_train import (
    apply_merges,
    apply_merges_batch_numba,
    learn_bpe,
)


def test_learn_bpe_reaches_target():
    rng = torch.Generator().manual_seed(0)
    data = torch.randint(0, 20, (10, 200), generator=rng).tolist()
    merges, final_v = learn_bpe(data, base_vocab=20, target_vocab=40)
    assert final_v <= 40
    assert len(merges) == final_v - 20


def test_bpe_encode_shape():
    tok = DiscreteBpeTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96,
        base_vocab=20, final_vocab=40,
    )
    x_train = torch.randn(4, 7, 336)
    tok.fit(x_train)
    x = torch.randn(4, 7, 336)
    out = tok.encode(x)
    assert out.embeddings.shape[-1] == 16
    assert out.embeddings.shape[1] <= 336
    assert out.padding_mask.shape == out.embeddings.shape[:2]


def test_bpe_decode_shape():
    tok = DiscreteBpeTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96,
        base_vocab=20, final_vocab=40,
    )
    tok.fit(torch.randn(4, 7, 336))
    out = tok.encode(torch.randn(4, 7, 336))
    latent = torch.randn(*out.embeddings.shape)
    y = tok.decode(latent, padding_mask=out.padding_mask)
    assert y.shape == (4, 96, 7)


def test_learn_bpe_parallel_equals_serial():
    """Parallel pair-counting / merge-application must produce identical merges to serial.

    Uses a corpus above the _PARALLEL_MIN_TOKENS threshold so the parallel path actually
    engages (small corpora deliberately fall back to serial).
    """
    rng = torch.Generator().manual_seed(42)
    data = torch.randint(0, 30, (200, 400), generator=rng).tolist()
    serial_merges, serial_v = learn_bpe(data, base_vocab=30, target_vocab=120, n_jobs=1)
    parallel_merges, parallel_v = learn_bpe(data, base_vocab=30, target_vocab=120, n_jobs=4)
    assert serial_v == parallel_v
    assert serial_merges == parallel_merges, (
        f"Parallel diverged from serial at merge {next((i for i,(s,p) in enumerate(zip(serial_merges, parallel_merges)) if s!=p), len(serial_merges))}"
    )


def test_learn_bpe_small_corpus_skips_parallel():
    """Corpora below the threshold must use the serial path, even if n_jobs > 1.

    This guards against pool setup overhead on tiny inputs (e.g., unit tests).
    """
    rng = torch.Generator().manual_seed(0)
    data = torch.randint(0, 20, (10, 200), generator=rng).tolist()
    m1, v1 = learn_bpe(data, base_vocab=20, target_vocab=40, n_jobs=1)
    m2, v2 = learn_bpe(data, base_vocab=20, target_vocab=40, n_jobs=8)
    assert m1 == m2 and v1 == v2


def test_learn_bpe_v1_equals_v2():
    """The v1 (ship-seqs) and v2 (persistent-workers) parallel paths must produce identical merges."""
    rng = torch.Generator().manual_seed(7)
    data = torch.randint(0, 30, (200, 400), generator=rng).tolist()
    m1, v1 = learn_bpe(data, base_vocab=30, target_vocab=120, n_jobs=4, parallel_version="v1")
    m2, v2 = learn_bpe(data, base_vocab=30, target_vocab=120, n_jobs=4, parallel_version="v2")
    assert v1 == v2, f"final_vocab: v1={v1}, v2={v2}"
    assert m1 == m2, (
        f"diverged at merge {next((i for i,(a,b) in enumerate(zip(m1, m2)) if a!=b), len(m1))}"
    )


def test_bpe_encode_cache_equals_no_cache():
    """The merge cache is a pure-function memoizer; encode output must be bit-identical
    with cache ON vs cache OFF. Core-logic preservation guarantee.
    """
    import os
    torch.manual_seed(123)
    tok = DiscreteBpeTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96,
        base_vocab=20, final_vocab=40,
    )
    tok.fit(torch.randn(4, 7, 336))
    x_test = torch.randn(3, 7, 336)

    tok._merge_cache.clear()
    tok._cache_hits = tok._cache_misses = 0
    os.environ.pop("TOKENSTUDY_BPE_CACHE", None)
    out_a1 = tok.encode(x_test)
    stats_after_first = tok.cache_stats()
    out_a2 = tok.encode(x_test)
    stats_after_second = tok.cache_stats()

    assert stats_after_second["misses"] == stats_after_first["misses"], "cache should not have grown on 2nd identical encode"
    assert stats_after_second["hits"] > stats_after_first["hits"], "cache should have recorded hits on 2nd pass"

    tok._merge_cache.clear()
    os.environ["TOKENSTUDY_BPE_CACHE"] = "0"
    try:
        out_b = tok.encode(x_test)
    finally:
        os.environ.pop("TOKENSTUDY_BPE_CACHE", None)

    assert torch.equal(out_a1.padding_mask, out_b.padding_mask), "padding_mask differs cache vs no-cache"
    assert torch.equal(out_a2.padding_mask, out_b.padding_mask), "padding_mask differs cache-hit vs no-cache"
    assert torch.equal(out_a1.embeddings, out_b.embeddings), "embeddings differ cache vs no-cache"
    assert torch.equal(out_a2.embeddings, out_b.embeddings), "embeddings differ cache-hit vs no-cache"


def test_bpe_decode_chunking_invariant():
    """Chunked BPE decode must match unchunked decode bit-for-bit (per-row independence)."""
    torch.manual_seed(13)
    tok = DiscreteBpeTokenizer(
        hidden_dim=16, num_channels=7, lookback=336, horizon=96,
        base_vocab=20, final_vocab=40,
    )
    tok.fit(torch.randn(4, 7, 336))
    latent = torch.randn(28, 100, 16)
    tok.DECODE_CHUNK_SIZE = 4096
    y_full = tok.decode(latent.clone())
    tok.DECODE_CHUNK_SIZE = 4
    y_chunked = tok.decode(latent.clone())
    assert torch.allclose(y_full, y_chunked, atol=1e-6, rtol=1e-6)


def test_apply_merges_numba_equals_python():
    """The Numba JIT batch path must produce bit-identical output to pure-Python apply_merges.

    Critical correctness gate for the BPE encode hot path. If this ever fails, we have an
    algorithmic divergence and the JIT path must be disabled until it's fixed.
    """
    rng = torch.Generator().manual_seed(7)
    fit_data = torch.randint(0, 30, (200, 400), generator=rng).tolist()
    merges, _ = learn_bpe(fit_data, base_vocab=30, target_vocab=120, n_jobs=1)
    merges_arr = np.asarray(merges, dtype=np.int32) if merges else np.zeros((0, 3), dtype=np.int32)

    rng2 = torch.Generator().manual_seed(11)
    test_rows = torch.randint(0, 30, (50, 400), generator=rng2).numpy().astype(np.int32)

    py_results = [apply_merges(test_rows[r].tolist(), merges) for r in range(test_rows.shape[0])]

    out_padded, lengths = apply_merges_batch_numba(test_rows, merges_arr)

    for r in range(test_rows.shape[0]):
        L = int(lengths[r])
        nb_result = out_padded[r, :L].tolist()
        py_result = py_results[r]
        assert nb_result == py_result, (
            f"Row {r}: numba and python diverged.\n"
            f"  python (len={len(py_result)}): {py_result[:20]}...\n"
            f"  numba  (len={L}): {nb_result[:20]}..."
        )


def test_bpe_progress_callback_fires():
    """progress_callback is invoked with increasing k, and k=1 is always the first emit."""
    rng = torch.Generator().manual_seed(1)
    data = torch.randint(0, 30, (200, 400), generator=rng).tolist()
    calls: list[tuple[int, int, float, int]] = []
    def cb(k, total, elapsed, top_count):
        calls.append((k, total, elapsed, top_count))
    learn_bpe(data, base_vocab=30, target_vocab=120, n_jobs=4,
              progress_callback=cb, progress_interval=20)
    assert calls, "progress_callback was never invoked"
    assert calls[0][0] == 1, f"first call must be at k=1, got k={calls[0][0]}"
    ks = [c[0] for c in calls]
    assert ks == sorted(ks), f"k not monotonic: {ks}"
    elapsed = [c[2] for c in calls]
    assert all(a <= b for a, b in zip(elapsed, elapsed[1:])), "elapsed not monotonic"
