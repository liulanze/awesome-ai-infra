"""
Exercise 2 - Build orchestrator (concurrency + reliability)

Recap: run a list of build jobs with a max concurrency of N, retry transient
failures with exponential backoff, enforce a per-attempt timeout, and return a
structured result per job (succeeded / failed-after-retries / timeout, with
attempts and duration). One job failing must not sink the others.
"""

from __future__ import annotations
import time
import random
import logging
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed

logger = logging.getLogger(__name__)

class Status(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"

@dataclass(frozen=True)
class BuildJob:
    name: str
    run: Callable[[], None]

@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    timeout_s: float | None = None

    def backoff(self, attempt: int) -> float:
        capped = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return random.uniform(0, capped)

@dataclass
class JobResult:
    name: str
    status: Status
    attempts: int
    duration_s: float
    error: str | None = None

def _run_once(job: BuildJob, timeout_s: float | None) -> None:
    """One attempt, enforcing a timeout. Raises on failure or timeout"""
    if timeout_s is None:
        job.run()
        return
    
    with ThreadPoolExecutor(max_workers=1) as inner:
        fut = inner.submit(job.run)
        try:
            fut.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            raise TimeoutError(f"{job.name} exceeded {timeout_s}s") from exc

def _run_with_retries(job: BuildJob, policy: RetryPolicy) -> JobResult:
    start = time.monotonic()
    last_error: str | None = None
    final = Status.FAILED
    for attempt in range(1, policy.max_attempts + 1):
        try:
            _run_once(job, policy.timeout_s)
            return JobResult(job.name, Status.SUCCEEDED, attempt, time.monotonic() - start)
        except TimeoutError as exc:
            final, last_error = Status.TIMED_OUT, str(exc)
            logger.warning("build %s timed out (%d/%d)", job.name, attempt, policy.max_attempts)
        except Exception as exc:                       # transient build failure
            final, last_error = Status.FAILED, repr(exc)
            logger.warning("build %s failed (%d/%d): %s", job.name, attempt, policy.max_attempts, exc)
        if attempt < policy.max_attempts:
            time.sleep(policy.backoff(attempt))
    return JobResult(job.name, final, policy.max_attempts, time.monotonic() - start, last_error)


def run_builds(jobs: Sequence[BuildJob], *, max_concurrency: int,
               policy: RetryPolicy = RetryPolicy()) -> list[JobResult]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    results: list[JobResult] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {pool.submit(_run_with_retries, job, policy): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:                    # defensive: bug in the runner itself
                logger.exception("orchestrator error for %s", job.name)
                results.append(JobResult(job.name, Status.FAILED, 0, 0.0, repr(exc)))
    return results
