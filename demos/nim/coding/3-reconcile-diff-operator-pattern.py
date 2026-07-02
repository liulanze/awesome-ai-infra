"""
Exercise 3, the setup: a user declares a desired state (via a CRD / spec / Helm
values) - say, the set of NIM deployments they want, each with an image, replica
count, and GPU request. The controller reads the observed state (what's actually
in the cluster) and computes the actions to converge observed -> desired: create
what's missing, update what drifted, delete what's no longer wanted. Then it
does it again, forever.
"""

"""
# 用户提交的 CRD（Custom Resource Definition）
apiVersion: nim.nvidia.com/v1
kind: NIMDeployment
spec:
  - name: llama-7b
    image: nim/llama:v2.1
    replicas: 3
    gpus: 1
  - name: mistral-7b
    image: nim/mistral:v1.0
    replicas: 2
    gpus: 1
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NIMSpec:
    name: str
    image: str
    replicas: int
    gpus: int

@dataclass(frozen=True)
class ObservedNIM:
    name: str
    image: str
    replicas: int
    gpus: int
    ready_replicas: int = 0
    resource_version: str = ""

class Op(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

@dataclass(frozen=True)
class Action:
    op: Op
    name: str
    desired: NIMSpec | None = None
    reason: str = ""

def _owned_fields_diff(d: NIMSpec, o: ObservedNIM) -> str:
    """Compare ONLY the fields we own. Ignore status / server-populated fields
    (ready_replicas, resource_version) - comparing those causes hot-looping."""
    diffs = []
    if d.image != o.image:
        diffs.append(f"image {o.image!r}->{d.image!r}")
    if d.replicas != o.replicas:
        diffs.append(f"replicas {o.replicas}->{d.replicas}")
    if d.gpus != o.gpus:
        diffs.append(f"gpus {o.gpus}->{d.gpus}")
    return ", ".join(diffs)

def plan(desired: Iterable[NIMSpec], observed: Iterable[ObservedNIM]) -> list[Action]:
    """Level-triggered reconcile: actions to converge observed -> desired.
    Pure and idempotent - a converged input yields []."""
    desired_by = {d.name: d for d in desired}
    observed_by = {o.name: o for o in observed}
    actions: list[Action] = []

    for name, d in desired_by.items():
        o = observed_by.get(name)
        if o is None:
            actions.append(Action(Op.CREATE, name, d, reason="not present"))
        elif (why := _owned_fields_diff(d, o)):
            actions.append(Action(Op.UPDATE, name, d, reason=why))
    
    for name in observed_by:
        if name not in desired_by:
            actions.append(Action(Op.DELETE, name, reason="no longer desired"))
    
    rank = {Op.CREATE: 0, Op.UPDATE: 1, Op.DELETE: 2}
    actions.sort(key=lambda a: (rank[a.op], a.name))
    return actions

@dataclass
class ApplyResult:
    action: Action
    ok: bool
    error: str | None = None

def apply(actions: Sequence[Action], client) -> list[ApplyResult]:
    """Execute a plan. Per-action isolation so one bad resource doesn't block the rest;
        the caller requeues (with backoff) if anything failed."""
    results: list[ApplyResult] = []
    for a in actions:
        try:
            if a.op is Op.CREATE:
                client.create(a.desired)
            elif a.op is Op.UPDATE:
                client.update(a.desired)
            else:
                client.delete(a.name)
            results.append(ApplyResult(a, ok=True))
        except Exception as exc:
            logger.exception("apply failed: %s %s", a.op.value, a.name)
            results.append(ApplyResult(a, ok=False, error=repr(exc)))
    return results

