"""
Prefix-aware KV-cache routing with a shared offload tier.

Long shared prompt prefixes (system prompts, few-shot examples, agent loops)
let us skip prefill entirely on a cache hit. A shared blob-backed namespace
makes prefixes computed on one replica reusable by any other replica; a
prefix-aware router directs incoming requests to the replica that already
holds the relevant KV (or fetches it from blob).

Reading guide:
  - PrefixHash, KVBlock        : opaque keys + the cached payload shape
  - BlobStore                  : shared, cross-replica KV namespace (slow tier)
  - Replica                    : local in-GPU KV cache (fast tier)
  - PrefixRouter               : hash-based dispatch with hit/miss handling
  - serve_request()            : TTFT model — prefill cost vs blob-load cost
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import NewType


PrefixHash = NewType("PrefixHash", str)


# Cost model in ticks. The point of the demo is the *ratio* between
# prefill (scales with prompt length) and blob-load (scales with prefix
# length but at a much smaller constant), not the absolute numbers.
PREFILL_COST_PER_TOKEN = 10
BLOB_LOAD_COST_PER_TOKEN = 1
LOCAL_HIT_COST = 0
PREFIX_GRANULARITY = 64        # hash buckets at this token boundary


def hash_prefix(tokens: tuple[int, ...]) -> PrefixHash:
    """Stable hash over a token-prefix slice. Production systems hash at a
    fixed block granularity so partial-prefix matches still land on a key.
    """
    aligned_len = (len(tokens) // PREFIX_GRANULARITY) * PREFIX_GRANULARITY
    aligned = tokens[:aligned_len]
    digest = hashlib.sha256(repr(aligned).encode()).hexdigest()[:16]
    return PrefixHash(digest)


@dataclass(frozen=True)
class KVBlock:
    """Stand-in for the K/V tensors of a prefix. Real systems store paged
    KV blocks; the size scaling (∝ prefix_len) is what matters here.
    """
    prefix_len: int
    payload_bytes: int


# --- Shared offload tier --------------------------------------------------

@dataclass
class BlobStore:
    """Shared KV namespace across replicas. Persistent by construction;
    higher per-op latency than local cache but unbounded capacity.
    """
    entries: dict[PrefixHash, KVBlock] = field(default_factory=dict)

    def get(self, key: PrefixHash) -> KVBlock | None:
        return self.entries.get(key)

    def put(self, key: PrefixHash, block: KVBlock) -> None:
        self.entries.setdefault(key, block)


# --- Per-replica local cache ----------------------------------------------

@dataclass
class Replica:
    replica_id: int
    local_cache: dict[PrefixHash, KVBlock] = field(default_factory=dict)

    def has_local(self, key: PrefixHash) -> bool:
        return key in self.local_cache

    def install(self, key: PrefixHash, block: KVBlock) -> None:
        self.local_cache[key] = block


# --- Routing decision -----------------------------------------------------

@dataclass
class RouteDecision:
    replica: Replica
    cache_state: str           # "local_hit" | "blob_hit" | "miss"
    block: KVBlock | None      # KV to install / use, if any


class PrefixRouter:
    """Routes a request to the replica most likely to skip prefill.

    Priority: any replica with a local hit > any replica + blob hit > round-robin.
    The router's value is realized only when (a) prefixes repeat and (b) the
    namespace is shared across replicas; both conditions are required.
    """

    def __init__(self, replicas: list[Replica], blob: BlobStore):
        self.replicas = replicas
        self.blob = blob
        self._rr = 0

    def route(self, prompt_tokens: tuple[int, ...]) -> RouteDecision:
        key = hash_prefix(prompt_tokens)

        for replica in self.replicas:
            if replica.has_local(key):
                return RouteDecision(replica, "local_hit", replica.local_cache[key])

        block = self.blob.get(key)
        if block is not None:
            replica = self._round_robin()
            return RouteDecision(replica, "blob_hit", block)

        return RouteDecision(self._round_robin(), "miss", None)

    def _round_robin(self) -> Replica:
        replica = self.replicas[self._rr % len(self.replicas)]
        self._rr += 1
        return replica


# --- TTFT model: where the win actually shows up --------------------------

def serve_request(
    prompt_tokens: tuple[int, ...],
    router: PrefixRouter,
    blob: BlobStore,
) -> tuple[int, str]:
    """Returns (ttft_ticks, cache_state). On a hit, prefill is skipped and
    TTFT collapses to the cost of loading the cached KV.
    """
    decision = router.route(prompt_tokens)
    key = hash_prefix(prompt_tokens)

    if decision.cache_state == "local_hit":
        return LOCAL_HIT_COST, "local_hit"

    if decision.cache_state == "blob_hit":
        assert decision.block is not None
        decision.replica.install(key, decision.block)
        return decision.block.prefix_len * BLOB_LOAD_COST_PER_TOKEN, "blob_hit"

    # Miss: pay full prefill, then publish the result for future requests.
    ttft = len(prompt_tokens) * PREFILL_COST_PER_TOKEN
    block = KVBlock(prefix_len=len(prompt_tokens), payload_bytes=len(prompt_tokens) * 256)
    decision.replica.install(key, block)
    blob.put(key, block)
    return ttft, "miss"
