# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in timeline prints of the TPU trainer and rollout, for the RL timeline report.

With ``VERL_TIMELINE=1`` in the job's runtime env, the TPU components of the plugin print one line per finished span::

    VERL_TIMELINE{"name":"update_actor","group":"trainer_workers","lane":"rank0","ts":1791857954.85,"dur":13.87}

Ray forwards these lines into the job log. ``scripts/accelerators/tpu/rl_timeline/make_report.py`` turns the log into
a self-contained HTML timeline of what the trainer chips and the rollout chips do in every step. Nothing is printed
(and no hook does any extra work) when the variable is unset.

Groups, lanes and span names:

* ``trainer`` / ``loop``: ``update_weights`` (``initial_weight_sync`` before step 1) and its ``sync.*`` phases, printed
  by the driver-side weight sync (``update_raiden_weights``) with the step number. The report rebuilds the other
  phases of the trainer loop (``gen`` = waiting for rollout, ``ref``, ``adv``, ``update_actor``, ...) from the
  ``timing_s/*`` step metrics, anchored at the step's ``update_weights``.
* ``trainer_workers`` / ``rank<r>``: what trainer rank ``r`` executes: ``update_actor`` (``fwd_bwd`` and
  ``optimizer_step``), ``compute_ref_log_prob`` / ``compute_log_prob`` (forward only), and its side of the weight
  sync, ``update_weights`` (``gather``, ``bind``, ``d2h``) and ``release_raiden``. Only rank 0 prints by default; set
  ``VERL_TIMELINE_RANKS`` to a comma-separated list of ranks or to ``all``.
* ``rollout_replica_<i>`` / ``requests``: one ``request`` per ``generate`` call, with the time parked at the closed
  submission gate (``admit``), the time to the first token (``first``), the prompt / generated tokens (``in`` /
  ``out``), the finish reason (``stop``) and the weight version that generated it (``wv``).
* ``rollout_replica_<i>`` / ``server``: ``drain`` (submission gate closed for a weight sync, engine still running the
  admitted requests), ``paused`` (engine paused until the gate reopens), ``resume_engine`` and the ``rpc.*`` calls of
  the weight sync, plus a ``weights`` instant whenever a new weight version is installed.
"""

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

MARKER = "VERL_TIMELINE"
ENV_VAR = "VERL_TIMELINE"
RANKS_ENV_VAR = "VERL_TIMELINE_RANKS"

TRAINER = "trainer"
TRAINER_LOOP = "loop"
TRAINER_WORKERS = "trainer_workers"
SERVER_LANE = "server"
REQUESTS_LANE = "requests"

_FALSE_VALUES = ("", "0", "false", "no", "off")


def enabled() -> bool:
    """Whether ``VERL_TIMELINE`` is set to a true value in this process."""
    return os.environ.get(ENV_VAR, "0").strip().lower() not in _FALSE_VALUES


def rollout_group(replica_rank: Any) -> str:
    """Timeline group of a rollout replica."""
    return f"rollout_replica_{replica_rank if replica_rank is not None else 0}"


def _write(event: dict) -> None:
    # One whitespace-free token that contains digits: Ray's log deduplication ignores such words, so it never folds
    # the lines of different workers into "[repeated Nx across cluster]".
    line = MARKER + json.dumps(event, separators=(",", ":"), default=str).replace(" ", "\\u0020")
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def emit(
    name: str,
    ts: float,
    dur: float,
    group: str = TRAINER,
    lane: str = TRAINER_LOOP,
    step: Optional[int] = None,
    **args: Any,
) -> None:
    """Print one finished span that started at epoch time ``ts`` and lasted ``dur`` seconds.

    Args:
        name: Span name, e.g. ``update_actor``.
        ts: Start, in seconds since the epoch (``time.time()``).
        dur: Duration in seconds.
        group: Timeline group (one block of lanes in the report), e.g. ``trainer_workers``.
        lane: Lane within the group, e.g. ``rank0``.
        step: Trainer step the span belongs to, when known.
        **args: Extra details shown in the report's tooltip.
    """
    if not enabled():
        return
    event: dict[str, Any] = {
        "name": name,
        "group": group,
        "lane": lane,
        "ts": round(ts, 6),
        "dur": round(max(dur, 0.0), 6),
    }
    if step is not None:
        event["step"] = int(step)
    if args:
        event["args"] = args
    _write(event)


def instant(name: str, group: str, lane: str, ts: Optional[float] = None, **args: Any) -> None:
    """Print a point event (e.g. a new weight version installed on a rollout replica)."""
    if not enabled():
        return
    event: dict[str, Any] = {"ph": "i", "name": name, "group": group, "lane": lane, "ts": round(ts or time.time(), 6)}
    if args:
        event["args"] = args
    _write(event)


def meta(name: str, group: str, **args: Any) -> None:
    """Print a metadata record (e.g. a rollout replica's ``max_num_seqs``) that the report attaches to ``group``."""
    if not enabled():
        return
    event: dict[str, Any] = {"ph": "M", "name": name, "group": group, "lane": "meta", "ts": round(time.time(), 6)}
    if args:
        event["args"] = args
    _write(event)


@contextmanager
def span(
    name: str,
    group: str = TRAINER,
    lane: Optional[str] = TRAINER_LOOP,
    step: Optional[int] = None,
    **args: Any,
) -> Iterator[dict]:
    """Time the body and print it as one span when it ends (also on an exception).

    A no-op when timeline prints are disabled or ``lane`` is ``None`` (e.g. :func:`trainer_lane` on a rank that does
    not print).

    Yields:
        A dict of extra details: entries added inside the body are printed with the span.
    """
    if not enabled() or lane is None:
        yield {}
        return
    details = dict(args)
    start = time.time()
    try:
        yield details
    finally:
        emit(name, start, time.time() - start, group=group, lane=lane, step=step, **details)


def _trainer_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0") or 0)


def trainer_lane() -> Optional[str]:
    """``rank<r>`` if this trainer rank prints timeline spans, else ``None``.

    Only rank 0 prints by default (every rank runs the same SPMD program); ``VERL_TIMELINE_RANKS`` selects others,
    e.g. ``0,3`` or ``all``.
    """
    if not enabled():
        return None
    rank = _trainer_rank()
    ranks = os.environ.get(RANKS_ENV_VAR, "0").strip().lower()
    if ranks not in ("all", "*") and str(rank) not in {r.strip() for r in ranks.split(",")}:
        return None
    return f"rank{rank}"
