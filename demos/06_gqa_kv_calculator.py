"""
KV-cache memory under MHA / GQA / MQA across tensor-parallel degrees.

Tensor parallelism shards the KV cache along the KV-head dimension. When
TP_size > num_kv_heads, the dimension cannot be partitioned cleanly and KV
heads are *replicated* across ranks, wasting per-GPU memory. The replication
factor is ceil(TP_size / num_kv_heads).

GQA (e.g. Phi-4) suffers this only conditionally — when TP exceeds kv_heads.
MLA (e.g. DeepSeek), with effectively a single KV head, is *always* fully
duplicated under any TP > 1; do not conflate the two.

Reading guide:
  - ModelConfig                : model shape inputs
  - kv_memory_per_gpu()        : returns bytes per GPU + replication factor + waste
  - tabulate_attention_variants(): MHA vs GQA vs MQA across TP options
"""

from __future__ import annotations

from dataclasses import dataclass


# Bytes per element by dtype. KV is typically stored in the activation dtype
# (BF16/FP16) or quantized (FP8) to halve the footprint.
DTYPE_BYTES: dict[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_q_heads: int           # query heads
    num_kv_heads: int          # K/V heads (== num_q_heads for MHA, 1 for MQA, ratio for GQA)
    head_dim: int
    num_layers: int
    seq_len: int
    batch_size: int
    kv_dtype: str = "bf16"


@dataclass(frozen=True)
class KVMemoryReport:
    bytes_per_gpu: int
    replication_factor: int    # 1 means cleanly sharded; >1 means waste
    wasted_bytes: int          # bytes that exist only because of replication


def kv_cache_bytes_total(cfg: ModelConfig) -> int:
    """Total KV cache across all ranks (K and V each), pre-sharding."""
    elem_bytes = DTYPE_BYTES[cfg.kv_dtype]
    return (
        2  # K and V
        * cfg.num_layers
        * cfg.num_kv_heads
        * cfg.head_dim
        * cfg.seq_len
        * cfg.batch_size
        * elem_bytes
    )


def kv_memory_per_gpu(cfg: ModelConfig, tp_size: int) -> KVMemoryReport:
    """Per-GPU KV-cache footprint under tensor parallelism `tp_size`.

    When tp_size <= num_kv_heads, sharding is clean: per-GPU bytes = total / tp.
    When tp_size > num_kv_heads, KV heads are replicated; replication_factor
    captures the duplication and `wasted_bytes` is the cost of that choice.
    """
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")

    total = kv_cache_bytes_total(cfg)

    if tp_size <= cfg.num_kv_heads:
        return KVMemoryReport(
            bytes_per_gpu=total // tp_size,
            replication_factor=1,
            wasted_bytes=0,
        )

    # tp_size > num_kv_heads: each KV head is replicated across multiple ranks.
    # ceil division so a non-divisible split still reports an honest factor.
    replication = -(-tp_size // cfg.num_kv_heads)
    bytes_per_gpu = (total * replication) // tp_size
    ideal_bytes = total // tp_size
    return KVMemoryReport(
        bytes_per_gpu=bytes_per_gpu,
        replication_factor=replication,
        wasted_bytes=bytes_per_gpu - ideal_bytes,
    )


# --- Variant comparison ---------------------------------------------------

def make_variant(base: ModelConfig, variant: str) -> ModelConfig:
    """Return `base` reshaped to MHA / GQA / MQA, holding query heads constant.

    GQA is parameterized as a 4:1 query:kv ratio (typical for Llama/Phi-4 family).
    """
    if variant == "MHA":
        kv = base.num_q_heads
    elif variant == "GQA":
        kv = max(1, base.num_q_heads // 4)
    elif variant == "MQA":
        kv = 1
    else:
        raise ValueError(f"unknown variant: {variant}")
    return ModelConfig(
        name=f"{base.name}-{variant}",
        num_q_heads=base.num_q_heads,
        num_kv_heads=kv,
        head_dim=base.head_dim,
        num_layers=base.num_layers,
        seq_len=base.seq_len,
        batch_size=base.batch_size,
        kv_dtype=base.kv_dtype,
    )


def tabulate_attention_variants(
    base: ModelConfig,
    tp_options: tuple[int, ...] = (1, 2, 4, 8),
) -> list[tuple[str, int, KVMemoryReport]]:
    """One row per (variant, tp). The MQA rows show the all-replication regime
    that makes MQA/MLA expensive under TP > 1.
    """
    rows: list[tuple[str, int, KVMemoryReport]] = []
    for variant in ("MHA", "GQA", "MQA"):
        cfg = make_variant(base, variant)
        for tp in tp_options:
            rows.append((variant, tp, kv_memory_per_gpu(cfg, tp)))
    return rows
