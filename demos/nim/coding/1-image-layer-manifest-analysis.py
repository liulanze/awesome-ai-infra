"""
Exercise 1 - The problem (recap): images are ordered lists of layers, each layer has a digest
and size. Compute naive total bytes, deduplicated bytes (content-addressed store
keeps each unique digest once), the savings, and the first layer index where two
images diverge (the cache-invalidation point).
"""

from __future__ import annotations
from dataclasses import dataclass
# abstract base classes
from collections.abc import Iterable, Sequence

@dataclass(frozen=True)
class Layer:
    digest: str # content hash - this is the layer's identity
    size: int   # bytes - this is the layer's size in bytes

@dataclass(frozen=True)
class Image:
    name: str
    layers: tuple[Layer, ...] # ordered list of layers, from base to top

@dataclass(frozen=True)
class DedupReport:
    naive_bytes: int
    dedup_bytes: int

    @property
    def savings_bytes(self) -> int:
        return self.naive_bytes - self.dedup_bytes
    
    @property
    def savings_ratio(self) -> float:
        return self.savings_bytes / self.naive_bytes if self.naive_bytes > 0 else 0.0
    
def naive_bytes(images: Iterable[Image]) -> int:
    """Compute the total size of all images, without deduplication."""
    return sum(layer.size for image in images for layer in image.layers)

def dedup_bytes(images: Iterable[Image]) -> int:
    """Bytes a content-addressed registry actually stores: each digest once."""
    sizes: dict[str, int] = {}
    for image in images:
        for layer in image.layers:
            sizes[layer.digest] = layer.size
    return sum(sizes.values())

def dedup_report(images: Sequence[Image]) -> DedupReport:
    return DedupReport(naive_bytes(images), dedup_bytes(images))

def divergence_index(a: Image, b: Image) -> int | None:
    """First layer index where a and b differ - past this, cache can't be
    reused. None means fully identical"""
    for i, (la, lb) in enumerate(zip(a.layers, b.layers)):
        if la.digest != lb.digest:
            return i
    if len(a.layers) == len(b.layers):
        return None
    return min(len(a.layers), len(b.layers))

"""
Now the part that actually wins the round — what to say while writing it,
because this panel scores reasoning:

- Lead with the mental model: "A registry is content-addressed, so the digest is
  the layer's identity — dedup is just counting unique digests. That makes
  naive-vs-dedup a one-pass set problem." Saying this signals you actually know
  how registries work, which Xin/Derek will clock immediately.
- Call edge cases before they ask: empty manifests (zero layers), a digest
  repeated within one image (still counts once globally — my dict handles it),
  one image being a strict prefix of another (the divergence case), and the
  integrity check (same digest, different size = a bug). Naming these unprompted
  is a senior signal.
- State complexity: single pass, O(total layers), dict lookups O(1). Don't reach
  for nested loops / list.index.
- Tie it to the real job: "This is exactly why base-image strategy matters — you
  put rarely-changing layers like the CUDA runtime at the bottom and
  fast-changing layers like your code or the model manifest on top, so the
  divergence index stays high and the cache stays warm." That one sentence
  connects your code to their actual problem and will land hard with this team.
- Mention tests: "These are pure functions, so unit tests are trivial — I'd
  cover empty, shared-prefix, fully-disjoint, and the within-image-duplicate
  case." If they hand you a REPL, actually write two or three.

Common ways people lose points here: no types, using a list with in/.index
(O(n²)), forgetting within-image dedup, an off-by-one or missing prefix case in
divergence, and treating it as a toy instead of narrating the registry/caching
reality.
"""
