"""
RMSNorm dtype drift: why the final-multiply dtype matters.

The CUDA RMSNorm kernel was performing the final `x_normed * weight` multiply
in FP32 regardless of the weight's actual dtype. The Python reference does the
multiply in the weight dtype (e.g. BF16). Per layer the discrepancy is small;
across many norms (e.g. Q/K normalization in every attention block) it
compounds into measurable accuracy drift.

The bug is one cast. This file is two functions that differ in one line.

Reference: vLLM PR #42379.
"""

from __future__ import annotations

import torch


EPS = 1e-6


def rmsnorm_buggy(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Final multiply in FP32. Diverges from the reference whenever
    weight.dtype != float32. "More precise" silently changes outputs.
    """
    x_fp32 = x.to(torch.float32)
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    normed = x_fp32 * rms
    return (normed * weight.to(torch.float32)).to(x.dtype)


def rmsnorm_correct(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Final multiply in the weight's native dtype, matching the reference."""
    x_fp32 = x.to(torch.float32)
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    normed = (x_fp32 * rms).to(weight.dtype)
    return normed * weight


def stacked_drift(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_layers: int,
) -> list[float]:
    """Run both implementations through `num_layers` stacked norms and return
    per-layer max absolute difference. The series is monotonically non-decreasing
    in expectation; that growth is the compounding effect.
    """
    a = x.clone()
    b = x.clone()
    drift: list[float] = []
    for _ in range(num_layers):
        a = rmsnorm_buggy(a, weight)
        b = rmsnorm_correct(b, weight)
        drift.append((a - b).abs().max().item())
    return drift
