"""
KV-cache handoff between prefill and decode workers: push vs pull.

Both modes give identical ITL benefit (decode is isolated from prefill in
either case). They differ in TTFT and orchestration:

  Pull:  decode worker recvs KV after prefill finishes.
         time-to-first-decodable = prefill + transfer + decode_start
  Push:  prefill worker sends KV layer-by-layer as it computes.
         time-to-first-decodable = max(prefill + transfer, decode_queue_wait)

Reading guide:
  - HandshakeMetadata          : the one-time setup payload negotiated per pair
  - pull_handoff()             : serial handoff; simpler orchestration
  - push_handoff()             : per-layer overlap; lowest TTFT
  - TimelineCosts              : closed-form timeline inputs
  - first_token_time_pull/push : the whiteboard question in code
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


PREFILL_RANK = 0
DECODE_RANK = 1


@dataclass(frozen=True)
class HandshakeMetadata:
    """Negotiated once per (prefill_rank, decode_rank) pair, then cached.

    Holds everything the receiver needs to interpret the bulk KV bytes that
    follow without further coordination on the request critical path.
    """
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype: torch.dtype


# --- Pull mode: serial; decode waits for prefill to finish, then reads ----

def pull_handoff(kv_per_layer: list[torch.Tensor], meta: HandshakeMetadata) -> None:
    """Prefill writes the full KV after compute completes; decode reads it
    in one bulk transfer. Orchestration is straightforward: a single barrier
    plus a single recv.
    """
    if dist.get_rank() == PREFILL_RANK:
        kv_blob = torch.cat([t.flatten() for t in kv_per_layer])
        dist.send(kv_blob, dst=DECODE_RANK)
    elif dist.get_rank() == DECODE_RANK:
        size = meta.num_layers * meta.num_kv_heads * meta.head_dim * meta.block_size
        buf = torch.empty(size, dtype=meta.dtype)
        dist.recv(buf, src=PREFILL_RANK)


# --- Push mode: overlap; prefill streams each layer as it completes -------

def push_handoff(
    kv_per_layer_iter: list[torch.Tensor],
    meta: HandshakeMetadata,
) -> None:
    """Prefill streams KV layer-by-layer as compute completes; decode begins
    receiving immediately. Per-layer transfers overlap with the next layer's
    prefill compute, hiding the transport behind compute.
    """
    if dist.get_rank() == PREFILL_RANK:
        in_flight: list[dist.Work] = []
        for layer_kv in kv_per_layer_iter:
            handle = dist.isend(layer_kv.contiguous(), dst=DECODE_RANK)
            in_flight.append(handle)
        for handle in in_flight:
            handle.wait()
    elif dist.get_rank() == DECODE_RANK:
        layer_shape = (meta.num_kv_heads, meta.head_dim, meta.block_size)
        for _ in range(meta.num_layers):
            buf = torch.empty(layer_shape, dtype=meta.dtype)
            dist.recv(buf, src=PREFILL_RANK)
            # Decode can begin as soon as layer 0 lands; subsequent layers
            # arrive in the shadow of prior decode steps.


# --- Timeline model: the closed-form answer to the whiteboard question ---

@dataclass(frozen=True)
class TimelineCosts:
    prefill: float
    transfer_per_layer: float
    num_layers: int
    decode_queue_wait: float

    @property
    def transfer_total(self) -> float:
        return self.transfer_per_layer * self.num_layers


def first_token_time_pull(c: TimelineCosts) -> float:
    """Pull is fully serialized: prefill, then transfer, then decode begins."""
    return c.prefill + c.transfer_total + c.decode_queue_wait


def first_token_time_push(c: TimelineCosts) -> float:
    """Push overlaps per-layer transfer with the next layer's compute. The
    last layer still pays one transfer; the rest are hidden behind compute.
    Decode-side queueing runs concurrently, hence the max().
    """
    streamed = c.prefill + c.transfer_per_layer  # only the tail layer is exposed
    return max(streamed, c.decode_queue_wait)
