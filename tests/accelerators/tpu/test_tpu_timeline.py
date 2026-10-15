# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the ``VERL_TIMELINE`` prints of the TPU plugin and the RL timeline report builder.

vLLM is not installed on CI, so the rollout server timeline is tested on a stand-in for upstream ``vLLMHttpServer``
with a fake ``AsyncLLM`` engine.
"""

import asyncio
import importlib.util
import json
import os
from types import SimpleNamespace

import pytest

from verl_hardware_plugin.accelerators.tpu import timeline
from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_timeline import vLLMServerTimelineMixin

REPORT_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "accelerators", "tpu", "rl_timeline", "make_report.py"
)


def _events(text: str) -> list[dict]:
    """The timeline events printed in ``text`` (captured stdout)."""
    out = []
    for line in text.splitlines():
        if line.startswith(timeline.MARKER):
            assert " " not in line  # one token: Ray's log deduplication never folds it
            out.append(json.loads(line[len(timeline.MARKER) :]))
    return out


@pytest.fixture
def timeline_on(monkeypatch):
    monkeypatch.setenv(timeline.ENV_VAR, "1")
    monkeypatch.delenv(timeline.RANKS_ENV_VAR, raising=False)
    monkeypatch.delenv("RANK", raising=False)


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_timeline_is_off_by_default(monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv(timeline.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(timeline.ENV_VAR, value)
    assert not timeline.enabled()
    timeline.emit("update_actor", 1.0, 2.0)
    timeline.instant("weights", "rollout_replica_0", "server", wv=1)
    timeline.meta("rollout_replica", "rollout_replica_0", max_num_seqs=256)
    with timeline.span("ref") as details:
        details["x"] = 1
    assert timeline.trainer_lane() is None
    assert capsys.readouterr().out == ""


def test_emit_prints_one_token_per_span(timeline_on, capsys):
    timeline.emit("request", 10.0, 2.5, group="rollout_replica_0", lane="requests", step=3, stop="hit length", out=7)
    timeline.instant("weights", "rollout_replica_0", "server", ts=12.0, wv=3)
    timeline.meta("rollout_replica", "rollout_replica_0", max_num_seqs=256)
    request, weights, meta = _events(capsys.readouterr().out)
    assert request == {
        "name": "request",
        "group": "rollout_replica_0",
        "lane": "requests",
        "ts": 10.0,
        "dur": 2.5,
        "step": 3,
        "args": {"stop": "hit length", "out": 7},
    }
    assert weights == {
        "ph": "i",
        "name": "weights",
        "group": "rollout_replica_0",
        "lane": "server",
        "ts": 12.0,
        "args": {"wv": 3},
    }
    assert (meta["ph"], meta["name"], meta["args"]) == ("M", "rollout_replica", {"max_num_seqs": 256})


def test_span_times_the_body_and_skips_lane_none(timeline_on, capsys):
    with timeline.span("update_actor", group=timeline.TRAINER_WORKERS, lane="rank0") as details:
        details["micro_batches"] = 4
    with timeline.span("update_actor", group=timeline.TRAINER_WORKERS, lane=None):
        pass
    with pytest.raises(RuntimeError), timeline.span("ref", lane="loop"):
        raise RuntimeError("printed anyway")
    actor, ref = _events(capsys.readouterr().out)
    assert (actor["name"], actor["lane"], actor["args"], actor["dur"] >= 0) == (
        "update_actor",
        "rank0",
        {"micro_batches": 4},
        True,
    )
    assert ref["name"] == "ref"


@pytest.mark.parametrize(
    ("ranks", "rank", "lane"),
    [(None, "0", "rank0"), (None, "1", None), ("0,1", "1", "rank1"), ("all", "3", "rank3"), ("2", "0", None)],
)
def test_trainer_lane_prints_rank_zero_by_default(timeline_on, monkeypatch, ranks, rank, lane):
    if ranks is not None:
        monkeypatch.setenv(timeline.RANKS_ENV_VAR, ranks)
    monkeypatch.setenv("RANK", rank)
    assert timeline.trainer_lane() == lane


# ----------------------------------------------------------------------------------------------------------------------
# Rollout server
# ----------------------------------------------------------------------------------------------------------------------
class _FakeEngine:
    """``AsyncLLM`` stand-in: every request streams two outputs and stops at the length limit."""

    def __init__(self):
        self.pause_calls = 0

    async def generate(self, prompt=None, sampling_params=None, request_id=None, **kwargs):
        await asyncio.sleep(0)
        yield SimpleNamespace(outputs=[SimpleNamespace(finish_reason=None, token_ids=[1])])
        await asyncio.sleep(0)
        yield SimpleNamespace(outputs=[SimpleNamespace(finish_reason="length", token_ids=[1, 2, 3])])

    async def pause_generation(self, **kwargs):
        self.pause_calls += 1


class _FakeHttpServer:
    """Stands in for upstream ``vLLMHttpServer``: the hooks the timeline mixin wraps, reduced to their contract."""

    def __init__(self):
        self.replica_rank, self.node_rank, self.nnodes, self.global_steps = 1, 0, 1, None
        self.config = SimpleNamespace(max_num_seqs=256, tensor_model_parallel_size=4)
        self._server_address = "10.0.0.2"
        self.engine = None
        self.calls = []

    async def launch_server(self, master_address=None, master_port=None, dp_rpc_port=None):
        self.engine = _FakeEngine()
        return "launched"

    async def _park_until_admitted(self, request_id):
        return None

    async def generate(self, prompt_ids, sampling_params, request_id, priority=0):
        rejected = await self._park_until_admitted(request_id)
        if rejected is not None:
            return rejected
        final = None
        async for output in self.engine.generate(
            prompt=prompt_ids, sampling_params=sampling_params, request_id=request_id
        ):
            final = output
        return SimpleNamespace(token_ids=final.outputs[0].token_ids, stop_reason="completed")

    async def abort_all_requests(self, reset_prefix_cache=True, reject_request=False, abort_only=False):
        self.calls.append(("abort", abort_only))
        if not abort_only:
            await self.engine.pause_generation(wait_for_inflight_requests=False, clear_cache=reset_prefix_cache)
        return {"aborted_count": 2, "request_ids": ["a", "b"]}

    async def resume_engine_generation(self):
        self.calls.append("resume_engine")

    async def open_submission_gate(self):
        self.calls.append("open_gate")

    async def resume_generation(self):
        await self.resume_engine_generation()
        await self.open_submission_gate()

    async def set_global_steps(self, global_steps):
        self.global_steps = global_steps

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return [method]


class _Server(vLLMServerTimelineMixin, _FakeHttpServer):
    pass


async def _serve(server):
    """Launch, generate one request, run a weight sync, generate again."""
    assert await server.launch_server() == "launched"
    await server.set_global_steps(0)
    first = await server.generate([5, 6, 7, 8], {}, "req-0")
    assert await server.abort_all_requests() == {"aborted_count": 2, "request_ids": ["a", "b"]}
    assert await server.collective_rpc("install_raiden_weights") == ["install_raiden_weights"]
    await server.set_global_steps(1)
    await server.resume_generation()
    second = await server.generate([5], {}, request_id="req-1")
    await server.abort_all_requests(abort_only=True)  # admission stays open: no drain / paused
    return first, second


def test_server_timeline_prints_requests_drain_pause_and_rpcs(timeline_on, capsys):
    server = _Server()
    first, second = asyncio.run(_serve(server))
    assert first.token_ids == [1, 2, 3] and second.stop_reason == "completed"
    assert server.calls == [("abort", False), "resume_engine", "open_gate", ("abort", True)]
    events = _events(capsys.readouterr().out)
    assert all(e["group"] == "rollout_replica_1" for e in events)
    names = [e["name"] for e in events]
    assert names == [
        "rollout_replica",
        "weights",
        "request",
        "drain",
        "abort",
        "rpc.install_raiden_weights",
        "weights",
        "resume_engine",
        "paused",
        "request",
    ]
    meta = events[0]
    assert (meta["ph"], meta["args"]) == ("M", {"max_num_seqs": 256, "tp": 4, "nnodes": 1, "host": "10.0.0.2"})
    req0, req1 = (e for e in events if e["name"] == "request")
    assert req0["lane"] == "requests"
    assert {k: req0["args"][k] for k in ("in", "out", "stop", "wv")} == {"in": 4, "out": 3, "stop": "length", "wv": 0}
    assert 0 <= req0["args"]["admit"] <= req0["args"]["first"] <= req0["dur"] + 1e-3
    assert {k: req1["args"][k] for k in ("in", "wv")} == {"in": 1, "wv": 1}
    drain, abort, paused = (next(e for e in events if e["name"] == n) for n in ("drain", "abort", "paused"))
    assert abort["args"] == {"aborted": 2}
    assert drain["ts"] == abort["ts"] and drain["dur"] <= abort["dur"]
    assert paused["ts"] == pytest.approx(drain["ts"] + drain["dur"], abs=1e-5)  # paused from the engine pause on
    assert [e["args"]["wv"] for e in events if e["name"] == "weights"] == [0, 1]
    assert not server._tl_admit and not server._tl_first and not server._tl_finish  # nothing leaks per request


def test_server_timeline_is_a_pass_through_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv(timeline.ENV_VAR, raising=False)
    server = _Server()
    first, second = asyncio.run(_serve(server))
    assert first.token_ids == [1, 2, 3] and second.token_ids == [1, 2, 3]
    assert not server._tl_hooked and "generate" not in vars(server.engine)  # the engine is left untouched
    assert capsys.readouterr().out == ""


def test_server_timeline_marks_aborted_requests(timeline_on, capsys):
    server = _Server()

    async def aborted(*args, **kwargs):
        yield SimpleNamespace(outputs=[SimpleNamespace(finish_reason="abort", token_ids=[1, 2])])

    async def scenario():
        await server.launch_server()
        # An engine whose requests are aborted by a weight sync (partial output, finish_reason "abort").
        server.engine = SimpleNamespace(generate=aborted, pause_generation=_FakeEngine().pause_generation)
        server._tl_install_engine_hooks()
        return await server.generate([1, 2], {}, "req-a")

    output = asyncio.run(scenario())
    assert output.token_ids == [1, 2]
    (request,) = [e for e in _events(capsys.readouterr().out) if e["name"] == "request"]
    assert (request["args"]["stop"], request["args"]["out"]) == ("abort", 2)


# ----------------------------------------------------------------------------------------------------------------------
# Report builder
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def make_report():
    spec = importlib.util.spec_from_file_location("make_report", REPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _line(prefix: str, **event) -> str:
    return f"\x1b[36m({prefix} pid=1, ip=10.0.0.1)\x1b[0m {timeline.MARKER}" + json.dumps(event, separators=(",", ":"))


SYNTHETIC_LOG = [
    _line("TaskRunnerV1", name="initial_weight_sync", group="trainer", lane="loop", ts=100.0, dur=2.0, step=0),
    _line(
        "TPUvLLMHttpServer",
        name="request",
        group="rollout_replica_0",
        lane="requests",
        ts=103.0,
        dur=10.0,
        args={"admit": 0.0, "first": 1.0, "out": 120, "in": 50, "stop": "stop", "wv": 0},
    ),
    _line("WorkerDict", name="compute_ref_log_prob", group="trainer_workers", lane="rank0", ts=114.0, dur=2.0),
    _line("WorkerDict", name="update_actor", group="trainer_workers", lane="rank0", ts=116.5, dur=8.0),
    _line("WorkerDict", name="fwd_bwd", group="trainer_workers", lane="rank0", ts=116.5, dur=7.0),
    _line("TPUvLLMHttpServer", name="drain", group="rollout_replica_0", lane="server", ts=125.0, dur=0.5),
    _line("TPUvLLMHttpServer", name="paused", group="rollout_replica_0", lane="server", ts=125.5, dur=1.5),
    _line("TaskRunnerV1", name="update_weights", group="trainer", lane="loop", ts=125.0, dur=2.0, step=1),
    "(TaskRunnerV1 pid=1) step:1 - timing_s/gen:11.0 - timing_s/ref:2.2 - timing_s/adv:0.2 - timing_s/update_actor:8.2"
    " - timing_s/update_weights:2.0 - timing_s/step:24.0 - training/off_policy/trajectory_spans/mean:np.float64(1.5)",
    "(TaskRunnerV1 pid=1) step:0 - val-core/openai/gsm8k/acc/mean@1:0.5",
    "garbage VERL_TIMELINE{not json}",
]


def test_report_parses_events_and_metrics(make_report):
    events, metrics, bad = make_report.parse_log_lines(SYNTHETIC_LOG)
    assert (len(events), bad) == (8, 1)
    assert list(metrics) == [1]  # the step-0 validation line has no timing_s/* metrics
    assert metrics[1]["timing_s/step"] == 24.0
    assert metrics[1]["training/off_policy/trajectory_spans/mean"] == 1.5


def test_report_summary_splits_each_step(make_report):
    events, metrics, _ = make_report.parse_log_lines(SYNTHETIC_LOG)
    (step,) = make_report.step_windows(events, metrics)
    assert (step["step"], step["t0"], step["t1"], step["t_next"]) == (1, 103.0, 127.0, 127.0)
    summary = make_report.text_summary(events, metrics)
    row = next(line for line in summary.splitlines() if line.split()[0] == "1").split()
    # step, window, timing_s/step, gen, ref, update_actor, update_weights | chips: actor, ref, sync, idle |
    # rollout: generating, gate closed, paused, idle, tokens/s
    assert [float(v) for v in row[1:]] == pytest.approx(
        [24.0, 24.0, 11.0, 2.2, 8.2, 2.0, 8.0, 2.0, 0.0, 14.0, 10.0, 0.0, 2.0, 12.0, 5.0], abs=0.01
    )
    assert "1 rollout requests (0 aborted" in summary


def test_report_embeds_payload_into_the_page(make_report, tmp_path):
    log = tmp_path / "job.log"
    log.write_text("\n".join(SYNTHETIC_LOG) + "\n")
    out = tmp_path / "report.html"
    assert make_report.main([str(log), "-o", str(out), "--save-events", str(tmp_path / "events.json")]) == 0
    html = out.read_text()
    start = html.index('<script id="tl-embedded" type="application/json">') + len(
        '<script id="tl-embedded" type="application/json">'
    )
    payload = json.loads(html[start : html.index("</script>", start)])
    assert payload["source"] == "job.log" and len(payload["events"]) == 8 and "1" in payload["metrics"]
    assert json.loads((tmp_path / "events.json").read_text()) == payload
    with pytest.raises(ValueError):
        make_report.build_html("<html></html>", payload)
