"""Discrete BPE tokenizer: Chronos-style quantization -> BPE merges -> embedding.

Simplification: decoder operates on **base** symbols (non-merged) so forecasting remains fixed-H
without autoregressive variable-length generation.

Known limitations (deliberate, scoped to this coursework):

1. ``self._merges`` is a plain Python list, NOT in ``state_dict``. If you ``load_state_dict`` a
   saved model, merges will be empty and encode falls back to base-symbol identity (L=T). The
   T16 training loop calls ``fit`` fresh on every run so this doesn't fire, but save/reload
   workflows would need to persist ``_merges`` separately.

2. ``encode`` uses Python loops for ``apply_merges`` per row, which is O(K*L) per sequence
   where K=|merges|, L=lookback. At the canonical ``final_vocab=2445`` this is the dominant
   tokenization cost - significant fraction of BPE wall-clock vs. the other tokenizers.
   Acceptable for the Matrix-A budget; revisit if BPE runs become the bottleneck.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .base import TokenBatch, Tokenizer
from ._bpe_train import apply_merges, apply_merges_batch_numba, learn_bpe


class DiscreteBpeTokenizer(Tokenizer):
    LO = -15.0
    HI = 15.0
    DECODE_CHUNK_SIZE = 1024

    def __init__(
        self, hidden_dim: int, num_channels: int, lookback: int, horizon: int,
        base_vocab: int = 126, final_vocab: int = 2445,
    ):
        super().__init__(hidden_dim, num_channels, lookback, horizon)
        assert base_vocab < 256, "base_vocab must fit in a byte for bytes-keyed merge cache"
        self.base_V = base_vocab
        self.final_V = final_vocab
        self.emb = nn.Embedding(final_vocab, hidden_dim)
        self.out_logits = nn.Linear(hidden_dim, base_vocab)
        bin_centers = torch.linspace(self.LO, self.HI, base_vocab)
        self.register_buffer("bin_centers", bin_centers)
        self._merges: list[tuple[int, int, int]] = []
        self._merges_arr: np.ndarray = np.zeros((0, 3), dtype=np.int32)
        self._merge_cache: dict[bytes, tuple[int, ...]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def sequence_length(self) -> int:
        return self.lookback

    def _bin(self, x: Tensor) -> Tensor:
        normed = (x - self.LO) / (self.HI - self.LO)
        ids = torch.clamp((normed * self.base_V).long(), 0, self.base_V - 1)
        return ids

    def _scale(self, x: Tensor) -> tuple[Tensor, Tensor]:
        scale = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return x / scale, scale

    def fit(self, x_train: Tensor, log_fn: "callable | None" = None) -> None:
        """Learn BPE merges from a training tensor (B, C, T).

        Parallelism: reads ``TOKENSTUDY_BPE_NJOBS`` env var (default: ``max(1, cpu_count-1)``,
        capped at 16). Parallel output is bit-identical to serial; set to ``1`` to force serial.

        ``log_fn``: optional callable(str) for progress lines. If None, progress lines print
        to stdout via ``print(..., flush=True)``. When called from the training loop, pass
        the loop's own progress emitter so BPE-fit lines go to the same place as epoch lines.
        """
        import os, time as _time, sys as _sys
        _t0 = _time.perf_counter()
        scaled, _ = self._scale(x_train)
        ids = self._bin(scaled).reshape(-1, scaled.shape[-1]).tolist()
        env = os.environ.get("TOKENSTUDY_BPE_NJOBS")
        if env is not None:
            try:
                n_jobs = max(1, int(env))
            except ValueError:
                n_jobs = 1
        else:
            cpus = os.cpu_count() or 1
            n_jobs = max(1, min(cpus - 1, 16))

        def _emit(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)
            else:
                print(msg, flush=True)

        total_tokens = sum(len(s) for s in ids)
        target_delta = self.final_V - self.base_V
        _emit(
            f"fitting BPE merges: n_seqs={len(ids)} total_tokens={total_tokens:,} "
            f"base_V={self.base_V} target_V={self.final_V} (delta={target_delta}) n_jobs={n_jobs}"
        )

        def _progress(k: int, total: int, elapsed: float, top_count: int) -> None:
            pct = 100.0 * k / max(total, 1)
            eta = (elapsed / max(k, 1)) * max(total - k, 0)
            _emit(
                f"  BPE fit: {k:>4}/{total} merges ({pct:5.1f}%)  "
                f"top_pair_count={top_count:>6}  elapsed={elapsed:>6.1f}s  eta={eta:>6.1f}s"
            )

        self._merges, final_v = learn_bpe(
            ids, base_vocab=self.base_V, target_vocab=self.final_V, n_jobs=n_jobs,
            progress_callback=_progress,
            progress_interval=max(50, target_delta // 30),
        )
        if self._merges:
            self._merges_arr = np.asarray(self._merges, dtype=np.int32)
        else:
            self._merges_arr = np.zeros((0, 3), dtype=np.int32)
        try:
            warm_rows = np.zeros((1, 4), dtype=np.int32)
            warm_merges = self._merges_arr if self._merges_arr.shape[0] > 0 else np.array([[0, 0, 1]], dtype=np.int32)
            apply_merges_batch_numba(warm_rows, warm_merges)
        except Exception:
            pass
        dt = _time.perf_counter() - _t0
        _emit(f"BPE fit complete: {len(self._merges)} merges learned, final_V={final_v}, took {dt:.1f}s")

    def encode(self, x: Tensor) -> TokenBatch:
        import os
        B, C, T = x.shape
        scaled, scale = self._scale(x)
        ids = self._bin(scaled).reshape(B * C, T)
        n = B * C
        cache_enabled = (len(self._merges) > 0) and (os.environ.get("TOKENSTUDY_BPE_CACHE", "1") != "0")
        cache = self._merge_cache
        merged: list = [None] * n
        miss_indices: list[int] = []
        miss_keys: list[bytes] = []
        miss_rows: list[list[int]] = []
        if len(self._merges) == 0:
            ids_list = ids.tolist()
            for i in range(n):
                merged[i] = tuple(ids_list[i])
        elif cache_enabled:
            ids_list = ids.tolist()
            for i in range(n):
                row_list = ids_list[i]
                key = bytes(row_list)
                cached = cache.get(key)
                if cached is not None:
                    merged[i] = cached
                    self._cache_hits += 1
                else:
                    miss_indices.append(i)
                    miss_keys.append(key)
                    miss_rows.append(row_list)
                    self._cache_misses += 1
        else:
            ids_list = ids.tolist()
            for i in range(n):
                miss_indices.append(i)
                miss_rows.append(ids_list[i])
            miss_keys = [b""] * n

        if miss_rows:
            miss_arr = np.asarray(miss_rows, dtype=np.int32)
            out_padded, lengths = apply_merges_batch_numba(miss_arr, self._merges_arr)
            for j, idx in enumerate(miss_indices):
                L = int(lengths[j])
                result = tuple(int(v) for v in out_padded[j, :L])
                merged[idx] = result
                if cache_enabled:
                    cache[miss_keys[j]] = result

        n = B * C
        max_L = max((len(m) for m in merged), default=T)
        pad_id = 0
        out_np = np.full((n, max_L), pad_id, dtype=np.int64)
        padmask_np = np.ones((n, max_L), dtype=bool)
        for i, m in enumerate(merged):
            L = len(m)
            if L:
                out_np[i, :L] = m
                padmask_np[i, :L] = False
        out = torch.from_numpy(out_np).to(x.device)
        padmask = torch.from_numpy(padmask_np).to(x.device)
        emb = self.emb(out)
        return TokenBatch(
            embeddings=emb, padding_mask=padmask,
            metadata={"batch": B, "channels": C, "scale": scale},
        )

    def cache_stats(self) -> dict[str, int | float]:
        """Returns hits/misses/hit_rate/size for the merge cache (useful for profiling)."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total else 0.0,
            "size": len(self._merge_cache),
        }

    def decode(self, latent: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        BC, L, d = latent.shape
        B = BC // self.num_channels
        tail = latent[:, -self.horizon :, :] if L >= self.horizon else \
            torch.nn.functional.pad(latent, (0, 0, self.horizon - L, 0))
        cs = max(1, self.DECODE_CHUNK_SIZE)
        if BC <= cs:
            logits = self.out_logits(tail)
            probs = torch.softmax(logits, dim=-1)
            y_scaled = (probs * self.bin_centers).sum(dim=-1)
        else:
            chunks: list[Tensor] = []
            for i in range(0, BC, cs):
                ch = tail[i : i + cs]
                logits_ch = self.out_logits(ch)
                probs_ch = torch.softmax(logits_ch, dim=-1)
                chunks.append((probs_ch * self.bin_centers).sum(dim=-1))
            y_scaled = torch.cat(chunks, dim=0)
        y_scaled = y_scaled.reshape(B, self.num_channels, self.horizon)
        return y_scaled.transpose(1, 2).contiguous()
