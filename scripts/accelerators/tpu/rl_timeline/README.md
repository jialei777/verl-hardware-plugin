# RL Timeline & 1 Hz HBM Memory Profiler for TPU Runs

Shows over wall-clock time what the trainer and every rollout replica (sampler slice) of a `verl` GRPO/PPO run on TPU
are doing, step by step — including **1 Hz per-chip TPU HBM memory traces** correlated with each RL phase:

* **trainer loop** (driver): `step`, then `gen` = waiting for rollout, `old_log_prob`, `ref`, `adv`, `update_actor`,
  and `update_weights` with its `sync.*` phases (quiesce, trainer send, P2P transfer, sampler install, resume).
* **trainer ranks & HBM**:
  * What trainer rank 0 runs on its TPU chips: `update_actor` (`fwd_bwd` + `optimizer_step`),
    `compute_ref_log_prob` (reference forward), `update_weights` (`gather` / `bind` / `d2h` of the weight sync), and
    `release_raiden`. Gaps are idle chips.
  * **1 Hz Trainer HBM row**: Per-chip HBM usage (`GiB` / `GB`) and capacity (`31.25 GiB` on TPU v6e) sampled every
    second from `libtpu` (`tpu-info`), showing the idle floor, `compute_ref_log_prob` peak, and `update_actor` peak.
* **rollout replica N & HBM**:
  * State (generating / still generating behind a closed gate for a weight sync / paused for the weight transfer /
    idle), the weight version it serves (`vK` = weights after step `K`), requests decoding and queued, the weight-sync
    RPCs, and one bar per request.
  * **1 Hz Rollout HBM row**: Per-chip HBM usage on the vLLM rollout slice.

Step labels (`S1`, `S2`, ...) and alternating stripes mark the steps on every lane. The summary cards and the per-step
table report how many seconds of every step the trainer chips spend in actor update, reference forward, weight sync
and idle, how many the rollout replicas spend generating, paused and idle, and the **per-step Trainer HBM peak and
idle floor (GiB)**.

No change to `verl` is needed: the TPU components of `verl-hardware-plugin` print the spans and 1 Hz HBM counter events
(see `verl_hardware_plugin/accelerators/tpu/timeline.py`), and the trainer-loop phases are rebuilt from the
`timing_s/*` step metrics, anchored at each step's weight sync.

---

## 1. Record (Built-In Timeline + Automatic 1 Hz HBM Sampling)

Add `VERL_TIMELINE=1` to the job's runtime env:

* `VERL_TIMELINE_RANKS=0,1` or `all` prints more trainer ranks (`0` by default).
* `VERL_TIMELINE_HBM_PERIOD=1.0` sets the HBM polling period in seconds (`1.0` by default; set `0` to disable HBM
  polling). When enabled, `TorchTitanTPUEngine` (on rank 0) and `TPUvLLMHttpServer` automatically spawn a lightweight
  daemon thread (`timeline.start_hbm_sampler`) that queries `tpu_info` every second and emits
  `VERL_TIMELINE{"ph":"C","name":"hbm",...}` lines into the Ray job log.

```bash
ray job submit --working-dir . \
  --runtime-env-json '{"py_modules": ["../verl-hardware-plugin/verl_hardware_plugin"],
                       "env_vars": {"VERL_TIMELINE": "1",
                                    "VERL_TIMELINE_HBM_PERIOD": "1.0",
                                    "VERL_PLATFORM": "tpu",
                                    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin"}}' \
  -- env TOTAL_TRAINING_STEPS=100 VAL_BEFORE_TRAIN=False TEST_FREQ=-1 \
     bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
```

---

## 2. Optional: Sidecar 1 Hz HBM Monitor (Captures Pre-Init Model Load from `0 GiB`)

The built-in `timeline.start_hbm_sampler` starts as soon as `TorchTitanTPUEngine.initialize()` and
`TPUvLLMHttpServer.launch_server()` finish. If you also want to capture the **early startup HBM ramp from `0 GiB`**
(Actor checkpoint load $\to$ colocated Reference checkpoint load $\to$ initial weight sync) before the engine finishes
initializing, start `hbm_monitor.py` on each TPU worker pod before submitting the Ray job:

```bash
# 1. Copy and start the 1 Hz HBM sidecar on each TPU worker pod:
for POD in $(kubectl get pods -l ray.io/node-type=worker -o jsonpath='{.items[*].metadata.name}'); do
  kubectl cp scripts/accelerators/tpu/rl_timeline/hbm_monitor.py "$POD":/tmp/hbm_monitor.py -c ray-worker
  kubectl exec "$POD" -c ray-worker -- bash -c \
    "pkill -f '[h]bm_monitor.py' || true; rm -f /tmp/hbm_monitor.csv; nohup python3 /tmp/hbm_monitor.py /tmp/hbm_monitor.csv 1.0 >/tmp/hbm_monitor.log 2>&1 &"
done

# 2. After the job finishes, copy the CSVs back:
kubectl cp <trainer-worker-pod>:/tmp/hbm_monitor.csv trainer_hbm.csv -c ray-worker
kubectl cp <rollout-worker-pod>:/tmp/hbm_monitor.csv rollout_hbm.csv -c ray-worker
```

---

## 3. Build & View the Timeline + HBM Dashboards

### A. Unified Interactive Timeline + HBM Report (`make_report.py`)

`make_report.py` extracts both the timeline spans and the 1 Hz HBM counter events from the job log (and can also merge
sidecar CSVs via `--hbm-csv`) into a single self-contained HTML file:

```bash
# Directly from a live Ray cluster (includes built-in 1 Hz HBM events from VERL_TIMELINE=1):
python3 scripts/accelerators/tpu/rl_timeline/make_report.py \
  --ray-address http://localhost:8265 --job <submission-id> \
  --save-log job.log --save-events events.json -o report.html --summary

# Or from a saved log + optional sidecar HBM CSVs (to include pre-init model load from 0 GiB):
python3 scripts/accelerators/tpu/rl_timeline/make_report.py job.log \
  --hbm-csv trainer_workers=trainer_hbm.csv \
  --hbm-csv rollout_replica_0=rollout_hbm.csv \
  --save-events events.json -o report.html --summary

# Keep only a specific step window (e.g. steady-state steps 10-20):
python3 scripts/accelerators/tpu/rl_timeline/make_report.py job.log \
  --steps 10-20 -o report_steps_10_20.html --summary
```

### B. Dedicated Multi-Panel HBM Build-Up & Phase-Shaded Report (`hbm_report.py`)

To generate the standalone multi-chart SVG HBM report (showing Chart 1: Startup Build-Up `Init -> Step 3`, Chart 2:
Steady-State Zoom, Chart 3: Full-Run Trainer HBM with RL phase shading, Chart 4: Full-Run Rollout HBM, plus the
per-step HBM table in `GiB` / `GB`):

```bash
# Using events.json (and optional <worker_ip>=<csv> sidecar files):
python3 scripts/accelerators/tpu/rl_timeline/hbm_report.py \
  events.json job.log hbm_report \
  10.80.9.7=trainer_hbm.csv 10.80.8.7=rollout_hbm.csv \
  --steps 2-5
```

This writes `hbm_report.html`.

### C. Interactive Controls in `report.html`

* **Zoom & Pan**: Mouse wheel zooms, drag pans, drag on the time axis zooms to a range, double-click fits, click a step
  label (or a row of the step table) to zoom to one step, arrow keys move between steps; `report.html#steps=3-5` (or
  `#step=4`) opens zoomed to those steps.
* **HBM Inspection**: Hover over the **trainer HBM** or **rollout HBM** lane in the timeline to inspect per-chip HBM
  usage (`GiB` and `GB`), % of `31.25 GiB` capacity, window peak/floor, and TPU duty cycle at any second.
* **Compare Two Runs**: Click **Compare…** to load a second log or report and compare steady-state phase times and
  mean/peak HBM side by side.
* **Export Perfetto**: Writes a Chrome trace (`.perfetto.json`) for `ui.perfetto.dev`, including per-chip HBM counter
  tracks.
