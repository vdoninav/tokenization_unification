"""Minimal BPE learner over integer symbol streams.

Treats each input sequence as a list of ints in [0, base_vocab). Learns merges (pair -> new id)
greedily by highest count until |V| reaches target_vocab (or no more mergeable pairs).

Three execution paths for FIT, all producing byte-identical output:
  * serial (default, n_jobs=1)
  * parallel v1 - ProcessPoolExecutor, ships seqs to workers each iteration (simpler, legacy)
  * parallel v2 - persistent worker processes with pinned chunks (default when n_jobs>1)

For ENCODE (per-batch apply_merges), there's a Numba JIT batched implementation
(``apply_merges_batch_numba``) that releases the GIL and parallelizes across rows via
``prange``, plus a pure-Python fallback if Numba isn't available. Output is bit-identical
to the serial Python ``apply_merges`` - guaranteed by ``test_apply_merges_numba_equals_python``.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np

try:
    import numba as _numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def _pair_counts(sequences: list[list[int]]) -> Counter[tuple[int, int]]:
    c: Counter[tuple[int, int]] = Counter()
    for seq in sequences:
        c.update(zip(seq, seq[1:]))
    return c


def _apply_merge(sequences: list[list[int]], a: int, b: int, new_id: int) -> None:
    for i, seq in enumerate(sequences):
        out = []
        j = 0
        n = len(seq)
        while j < n:
            if j + 1 < n and seq[j] == a and seq[j + 1] == b:
                out.append(new_id)
                j += 2
            else:
                out.append(seq[j])
                j += 1
        sequences[i] = out


def _learn_bpe_serial(
    seqs: list[list[int]],
    base_vocab: int,
    target_vocab: int,
    progress_callback: Callable[[int, int, float, int], None] | None = None,
    progress_interval: int = 100,
) -> tuple[list[tuple[int, int, int]], int]:
    """Factored-out serial loop. Shared by both paths when the parallel gate falls through."""
    merges: list[tuple[int, int, int]] = []
    next_id = base_vocab
    t0 = time.perf_counter()
    target_delta = target_vocab - base_vocab
    while next_id < target_vocab:
        pc = _pair_counts(seqs)
        if not pc:
            break
        (a, b), count = pc.most_common(1)[0]
        if count < 2:
            break
        merges.append((a, b, next_id))
        _apply_merge(seqs, a, b, next_id)
        next_id += 1
        k = len(merges)
        if progress_callback is not None and (k == 1 or k % progress_interval == 0):
            progress_callback(k, target_delta, time.perf_counter() - t0, count)
    return merges, next_id


def _pair_counts_of_chunk(sequences_chunk: list[list[int]]) -> Counter:
    c: Counter = Counter()
    for seq in sequences_chunk:
        c.update(zip(seq, seq[1:]))
    return c


def _apply_merge_to_chunk(
    args: tuple[list[list[int]], int, int, int],
) -> list[list[int]]:
    sequences_chunk, a, b, new_id = args
    out_chunk: list[list[int]] = []
    for seq in sequences_chunk:
        out: list[int] = []
        j = 0
        n = len(seq)
        while j < n:
            if j + 1 < n and seq[j] == a and seq[j + 1] == b:
                out.append(new_id)
                j += 2
            else:
                out.append(seq[j])
                j += 1
        out_chunk.append(out)
    return out_chunk


def _split_chunks(seqs: list[list[int]], n_chunks: int) -> list[tuple[int, int]]:
    n = len(seqs)
    size = (n + n_chunks - 1) // n_chunks
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def _learn_bpe_parallel_v1(
    seqs: list[list[int]],
    base_vocab: int,
    target_vocab: int,
    n_jobs: int,
    progress_callback: Callable[[int, int, float, int], None] | None = None,
    progress_interval: int = 100,
) -> tuple[list[tuple[int, int, int]], int]:
    """Legacy parallel path - ships seqs to workers each iteration. Kept as a fallback."""
    chunk_idx = _split_chunks(seqs, n_jobs)
    merges: list[tuple[int, int, int]] = []
    next_id = base_vocab
    t0 = time.perf_counter()
    target_delta = target_vocab - base_vocab

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        while next_id < target_vocab:
            chunks = [seqs[s:e] for s, e in chunk_idx]
            per_chunk = list(pool.map(_pair_counts_of_chunk, chunks))
            combined: Counter = Counter()
            for cc in per_chunk:
                combined.update(cc)
            if not combined:
                break
            (a, b), count = combined.most_common(1)[0]
            if count < 2:
                break
            merges.append((a, b, next_id))
            merge_args = [(seqs[s:e], a, b, next_id) for s, e in chunk_idx]
            merged_chunks = list(pool.map(_apply_merge_to_chunk, merge_args))
            seqs = []
            for mc in merged_chunks:
                seqs.extend(mc)
            chunk_idx = _split_chunks(seqs, n_jobs)
            next_id += 1
            k = len(merges)
            if progress_callback is not None and (k == 1 or k % progress_interval == 0):
                progress_callback(k, target_delta, time.perf_counter() - t0, count)

    return merges, next_id


def _worker_loop(chunk: list[list[int]], in_q: "mp.Queue", out_q: "mp.Queue") -> None:
    """Worker main loop. Accepts commands via ``in_q``, responds via ``out_q``.

    Commands:
      ("count", None)      -> counts pairs in local chunk, returns Counter
      ("apply", (a,b,id))  -> applies merge to local chunk in place, returns None ack
      ("stop",  None)      -> exits
    """
    while True:
        try:
            cmd, args = in_q.get()
        except (EOFError, KeyboardInterrupt):
            return
        if cmd == "stop":
            return
        if cmd == "count":
            c: Counter = Counter()
            for seq in chunk:
                c.update(zip(seq, seq[1:]))
            out_q.put(c)
        elif cmd == "apply":
            a, b, new_id = args
            for i, seq in enumerate(chunk):
                out: list[int] = []
                j = 0
                n = len(seq)
                while j < n:
                    if j + 1 < n and seq[j] == a and seq[j + 1] == b:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(seq[j])
                        j += 1
                chunk[i] = out
            out_q.put(None)
        else:
            out_q.put(("ERROR", f"unknown cmd {cmd!r}"))


def _learn_bpe_parallel_v2(
    seqs: list[list[int]],
    base_vocab: int,
    target_vocab: int,
    n_jobs: int,
    progress_callback: Callable[[int, int, float, int], None] | None = None,
    progress_interval: int = 100,
) -> tuple[list[tuple[int, int, int]], int]:
    """Persistent-worker implementation. Seqs shipped once at worker spawn."""
    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        ctx = mp.get_context()

    chunk_idx = _split_chunks(seqs, n_jobs)
    chunks = [seqs[s:e] for s, e in chunk_idx]
    actual_n = len(chunks)

    in_queues = [ctx.Queue() for _ in range(actual_n)]
    out_queues = [ctx.Queue() for _ in range(actual_n)]
    procs = [
        ctx.Process(target=_worker_loop, args=(chunks[i], in_queues[i], out_queues[i]))
        for i in range(actual_n)
    ]
    for p in procs:
        p.start()

    merges: list[tuple[int, int, int]] = []
    next_id = base_vocab
    t0 = time.perf_counter()
    target_delta = target_vocab - base_vocab

    try:
        while next_id < target_vocab:
            for q in in_queues:
                q.put(("count", None))
            combined: Counter = Counter()
            for q in out_queues:
                combined.update(q.get())

            if not combined:
                break
            (a, b), count = combined.most_common(1)[0]
            if count < 2:
                break

            for q in in_queues:
                q.put(("apply", (a, b, next_id)))
            for q in out_queues:
                q.get()

            merges.append((a, b, next_id))
            next_id += 1
            k = len(merges)
            if progress_callback is not None and (k == 1 or k % progress_interval == 0):
                progress_callback(k, target_delta, time.perf_counter() - t0, count)
    finally:
        for q in in_queues:
            q.put(("stop", None))
        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

    return merges, next_id


_PARALLEL_MIN_TOKENS = 50_000


def learn_bpe(
    sequences: list[list[int]],
    base_vocab: int,
    target_vocab: int,
    n_jobs: int = 1,
    progress_callback: Callable[[int, int, float, int], None] | None = None,
    progress_interval: int = 100,
    parallel_version: str = "v2",
) -> tuple[list[tuple[int, int, int]], int]:
    """Returns (merges, final_vocab_size). Each merge is (a, b, new_id).

    ``n_jobs``: number of worker processes. 1 = serial (default, backward compatible).
    ``progress_callback``: optional callable ``f(merges_so_far, total_merges, elapsed_sec, top_pair_count)``
      called every ``progress_interval`` merges (and on the first merge).
    ``parallel_version``: ``"v2"`` (persistent workers, recommended, default) or ``"v1"``
      (pool.map, legacy fallback). Both produce bit-identical output.

    Falls back to serial on: small corpora (<50k tokens), insufficient sequences for
    parallelism, or if the process pool cannot be created (OSError/RuntimeError).
    """
    seqs = [list(s) for s in sequences]
    total_tokens = sum(len(s) for s in seqs)
    if n_jobs <= 1 or total_tokens < _PARALLEL_MIN_TOKENS or len(seqs) < max(n_jobs * 2, 4):
        return _learn_bpe_serial(seqs, base_vocab, target_vocab, progress_callback, progress_interval)

    try:
        if parallel_version == "v1":
            return _learn_bpe_parallel_v1(
                seqs, base_vocab, target_vocab, n_jobs, progress_callback, progress_interval,
            )
        return _learn_bpe_parallel_v2(
            seqs, base_vocab, target_vocab, n_jobs, progress_callback, progress_interval,
        )
    except (OSError, RuntimeError):
        return _learn_bpe_serial(seqs, base_vocab, target_vocab, progress_callback, progress_interval)


def apply_merges(seq: list[int], merges: list[tuple[int, int, int]]) -> list[int]:
    """Apply a learned merge list to a single sequence (pure-Python reference path).

    Used as the algorithmic ground truth for the cache and as fallback when Numba isn't
    available. The Numba JIT batched version (``apply_merges_batch_numba``) is exercised
    against this function in tests to guarantee bit-identical output.
    """
    for a, b, new_id in merges:
        out: list[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return seq


if HAVE_NUMBA:
    @_numba.njit(parallel=True, cache=True, fastmath=False)
    def _apply_merges_batch_numba_jit(rows, merges):  # type: ignore[no-untyped-def]
        """JIT'd batch apply_merges. ``rows`` is (n_rows, T) int32; ``merges`` is (K, 3) int32.

        Returns (out: (n_rows, T) int32, lengths: (n_rows,) int64). The first ``lengths[r]``
        entries of ``out[r]`` are valid; the rest are zero-padded.

        Determinism: each row is independent. ``prange`` distributes rows across threads;
        the merge order *within* a row is the same as serial (loop k=0..K-1). No randomness.
        """
        n_rows = rows.shape[0]
        T = rows.shape[1]
        n_merges = merges.shape[0]
        out = np.zeros((n_rows, T), dtype=np.int32)
        lengths = np.zeros(n_rows, dtype=np.int64)
        for r in _numba.prange(n_rows):
            cur = np.empty(T, dtype=np.int32)
            nxt = np.empty(T, dtype=np.int32)
            for j in range(T):
                cur[j] = rows[r, j]
            cur_n = T
            for k in range(n_merges):
                a = merges[k, 0]
                b = merges[k, 1]
                new_id = merges[k, 2]
                nxt_idx = 0
                i = 0
                while i < cur_n:
                    if i + 1 < cur_n and cur[i] == a and cur[i + 1] == b:
                        nxt[nxt_idx] = new_id
                        nxt_idx += 1
                        i += 2
                    else:
                        nxt[nxt_idx] = cur[i]
                        nxt_idx += 1
                        i += 1
                cur, nxt = nxt, cur
                cur_n = nxt_idx
            for j in range(cur_n):
                out[r, j] = cur[j]
            lengths[r] = cur_n
        return out, lengths


def apply_merges_batch_numba(rows: np.ndarray, merges: np.ndarray):
    """Batched apply_merges with Numba JIT + prange parallelism.

    Falls back to a pure-Python loop calling ``apply_merges`` if Numba isn't available.
    Output is bit-identical to row-by-row Python ``apply_merges`` - guaranteed by
    ``test_apply_merges_numba_equals_python``.

    Args:
        rows: (n_rows, T) np.int32 array of base-vocab token IDs.
        merges: (K, 3) np.int32 array of (a, b, new_id).

    Returns:
        out: (n_rows, T) np.int32 array; ``out[r, :lengths[r]]`` is the merged sequence.
        lengths: (n_rows,) np.int64 array of valid lengths per row.
    """
    if HAVE_NUMBA and merges.shape[0] > 0:
        return _apply_merges_batch_numba_jit(rows, merges)
    n_rows = rows.shape[0]
    T = rows.shape[1]
    out = np.zeros((n_rows, T), dtype=np.int32)
    lengths = np.zeros(n_rows, dtype=np.int64)
    merges_list = [(int(m[0]), int(m[1]), int(m[2])) for m in merges]
    for r in range(n_rows):
        result = apply_merges(rows[r].tolist(), merges_list)
        L = len(result)
        out[r, :L] = result
        lengths[r] = L
    return out, lengths
