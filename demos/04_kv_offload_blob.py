"""
KV-cache offload tier: cross-replica prefix reuse via Azure Blob Storage.

A local KV cache is per-replica: a prefix computed on replica-A is invisible
to replica-B, which recomputes it from scratch on the next request. A shared
object-store tier (Azure Blob) breaks this isolation — any replica can publish
a cached prefix and any other replica can consume it, turning per-replica
waste into fleet-wide reuse.

Object storage is not fast (higher per-op latency than local cache), but it
is durable, shared across replicas, and effectively unbounded in capacity.
Those are exactly the properties needed for a cross-replica prefix-caching
tier: the value is skipping prefill (expensive compute), not minimising a
single cache read (cheap I/O in comparison).

Key design decisions encoded here:
  - Block-aligned prefix hashing: a 130-token prompt and a 192-token prompt
    sharing the first 128 tokens hash to the same key, so the shorter one
    gets a cache hit from the longer one's stored KV.
  - Idempotent writes: the first writer wins; a second replica writing the
    same prefix hash is a safe no-op — KV for a given prefix is deterministic.
  - Two-tier lookup: local replica cache first (zero cost), blob on miss
    (I/O cost), full prefill only if both miss (compute cost).

To use real Azure Blob Storage, replace BlobStore._entries with a
BlobServiceClient and swap get/put to use download_blob/upload_blob.
For local testing, Azurite (the Azure Storage emulator) works identically.

Reading guide:
  - PrefixKey, KVBlock           : cache entry shape
  - make_prefix_key()            : block-aligned prefix hashing
  - BlobStore                    : shared cross-replica namespace
  - Replica                      : local KV cache backed by BlobStore on miss
  - serve_request()              : two-tier lookup + prefill fallback + publish
  - prefill_cost(), blob_load_cost() : TTFT model (miss vs hit paths)
  - simulate_fleet()             : two replicas sharing one BlobStore; shows
                                   blob_hit on the second replica after the
                                   first has published the KV
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import NewType


PrefixKey = NewType("PrefixKey", str)

# Cost model in ticks. The ratio is the point:
# prefill is compute-bound (big GEMMs); blob load is I/O-bound — much cheaper
# per token, but not free. The miss-to-hit speedup grows with prefix length.
PREFILL_COST_PER_TOKEN = 10
BLOB_LOAD_COST_PER_TOKEN = 1    # ~10× cheaper than recomputing prefill
LOCAL_HIT_COST = 0

PREFIX_BLOCK_SIZE = 64          # hash at this token-boundary granularity


# --- Cache entry types ------------------------------------------------------

def make_prefix_key(tokens: tuple[int, ...]) -> PrefixKey:
    """Hash a token sequence aligned to PREFIX_BLOCK_SIZE.

    Alignment means two prompts sharing a long common prefix map to the same
    key even if they differ in the tail. A 130-token prompt and a 192-token
    prompt that share the first 128 tokens both produce the same key for that
    128-token aligned block, so the shorter benefits from the longer's KV.
    """
    aligned_len = (len(tokens) // PREFIX_BLOCK_SIZE) * PREFIX_BLOCK_SIZE
    digest = hashlib.sha256(repr(tokens[:aligned_len]).encode()).hexdigest()[:16]
    return PrefixKey(digest)


@dataclass(frozen=True)
class KVBlock:
    """Lightweight stand-in for the K/V tensors of a cached prefix.

    Real: paged KV tensors serialised to bytes, keyed by prefix hash in Blob.
    Here: a record carrying prefix_len (drives cost) and nominal size_bytes
    (illustrates that storage cost scales with prefix length).
    """
    prefix_len: int
    size_bytes: int     # 2 * layers * kv_heads * head_dim * prefix_len * dtype_bytes


# --- Blob store tier (shared, cross-replica) --------------------------------

@dataclass
class BlobStore:
    """Shared KV namespace across all replicas.

    Backed by a plain dict here. To use real Azure Blob Storage:

        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(conn_str)
        container = client.get_container_client("kv-cache")

    Then replace get/put with:
        get: container.download_blob(key).readall()  -> deserialise KVBlock
        put: container.upload_blob(key, data, overwrite=False)  -> idempotent

    Azurite (the local Azure Storage emulator) accepts the same SDK calls
    with a localhost connection string — no real Azure account needed.
    """
    _entries: dict[PrefixKey, KVBlock] = field(default_factory=dict)

    def get(self, key: PrefixKey) -> KVBlock | None:
        return self._entries.get(key)

    def put(self, key: PrefixKey, block: KVBlock) -> None:
        # setdefault = idempotent: if two replicas race to write the same prefix,
        # the second write is a no-op. KV for a given prefix is deterministic,
        # so last-writer-wins would also be safe; first-writer-wins is simpler.
        self._entries.setdefault(key, block)

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# --- Per-replica local cache (fast tier) ------------------------------------

@dataclass
class Replica:
    """One serving replica: a local KV cache backed by a shared BlobStore.

    The local cache is the fast tier (zero lookup cost once warm). BlobStore
    is the slow tier but has fleet-wide reach. Hits at either tier skip
    prefill entirely; only a full miss triggers compute.
    """
    replica_id: str
    blob: BlobStore
    _local: dict[PrefixKey, KVBlock] = field(default_factory=dict)

    # Counters for observing hit/miss rates across a workload.
    hits_local: int = 0
    hits_blob: int = 0
    misses: int = 0


# --- TTFT cost model --------------------------------------------------------

def prefill_cost(prompt_len: int) -> int:
    """Full prefill cost — paid only on a cache miss."""
    return prompt_len * PREFILL_COST_PER_TOKEN


def blob_load_cost(prefix_len: int) -> int:
    """Cost of loading a cached KV block from Blob Storage.

    I/O-bound and much cheaper than recomputing prefill, but grows with
    prefix length. The crossover where blob-load becomes expensive relative
    to prefill does not occur at typical prefix lengths and dtype costs.
    """
    return prefix_len * BLOB_LOAD_COST_PER_TOKEN


# --- Unified request path ---------------------------------------------------

@dataclass(frozen=True)
class ServeResult:
    replica_id: str
    cache_state: str    # "local_hit" | "blob_hit" | "miss"
    ttft_ticks: int
    prefix_len: int
    tokens_saved: int   # prefill tokens avoided (0 on a miss)


def serve_request(
    replica: Replica,
    prompt_tokens: tuple[int, ...],
) -> ServeResult:
    """Two-tier cache lookup, with full prefill and blob publish on a miss.

    Tier 1 — local cache: zero cost, per-replica, lost on pod restart.
    Tier 2 — blob store: I/O cost, shared across all replicas, durable.
    Miss: compute full prefill, then publish to blob so other replicas benefit.

    The cross-replica win only appears when a *different* replica serves a
    later request for the same prefix and finds it in blob rather than
    recomputing. See simulate_fleet() for a two-replica demonstration.
    """
    key = make_prefix_key(prompt_tokens)
    aligned_len = (len(prompt_tokens) // PREFIX_BLOCK_SIZE) * PREFIX_BLOCK_SIZE

    # Tier 1: local cache.
    block = replica._local.get(key)
    if block is not None:
        replica.hits_local += 1
        return ServeResult(replica.replica_id, "local_hit", LOCAL_HIT_COST,
                           block.prefix_len, block.prefix_len)

    # Tier 2: shared blob store.
    block = replica.blob.get(key)
    if block is not None:
        replica._local[key] = block         # warm local cache for subsequent hits
        replica.hits_blob += 1
        return ServeResult(replica.replica_id, "blob_hit",
                           blob_load_cost(block.prefix_len),
                           block.prefix_len, block.prefix_len)

    # Miss: full prefill, then publish to blob for the rest of the fleet.
    block = KVBlock(
        prefix_len=aligned_len,
        size_bytes=aligned_len * 256,       # placeholder: scales with prefix length
    )
    replica._local[key] = block
    replica.blob.put(key, block)
    replica.misses += 1
    return ServeResult(replica.replica_id, "miss",
                       prefill_cost(len(prompt_tokens)), aligned_len, 0)


def serve_request_2(replica, prompt_tokens):
    block_keys = []
    for i in range(0, len(prompt_tokens), BLOCK_SIZE):
        # 累积 hash：每个 block 的 key 包含从头到这里的所有 token
        block_keys.append(hash(prompt_tokens[:i + BLOCK_SIZE]))
    
    # 从后往前找最长命中前缀
    cached_blocks = []
    for k in block_keys:
        blk = replica._local.get(k) or replica.blob.get(k)
        if blk is None:
            break          # 第一个 miss 之后就停，后面也不会命中
        cached_blocks.append(blk)
    
    hit_len = len(cached_blocks) * BLOCK_SIZE
    miss_tokens = prompt_tokens[hit_len:]
    
    # 只对未命中的后半段跑 prefill
    new_blocks = prefill(miss_tokens, prefix_kv=cached_blocks)
    
    # 把新算出来的 block 上传 blob
    for k, blk in zip(block_keys[len(cached_blocks):], new_blocks):
        replica.blob.put(k, blk)

# --- Fleet simulation -------------------------------------------------------

def simulate_fleet(
    workload: list[tuple[str, tuple[int, ...]]],
    num_replicas: int = 2,
) -> list[ServeResult]:
    """Route requests to named replicas that share one BlobStore.

    workload is a list of (replica_id, prompt_tokens) pairs. To observe the
    cross-replica blob_hit pattern, route the same prompt to two different
    replicas in sequence:

        workload = [
            ("replica-0", prompt),   # miss  → publishes to blob
            ("replica-1", prompt),   # blob_hit → skips prefill
        ]

    The second request pays blob_load_cost instead of prefill_cost, which
    is the cross-replica reuse win this tier is designed to deliver.
    """
    blob = BlobStore()
    replicas = {
        f"replica-{i}": Replica(f"replica-{i}", blob)
        for i in range(num_replicas)
    }
    return [
        serve_request(replicas[rid], tokens)
        for rid, tokens in workload
    ]
