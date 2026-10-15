# RL timeline report for TPU runs

Shows over wall-clock time what the trainer and every rollout replica (sampler slice) of a verl GRPO/PPO run on TPU
are doing, step by step:

* **trainer loop** (driver): `step`, then `gen` = waiting for rollout, `old_log_prob`, `ref`, `adv`, `update_actor`,
  and `update_weights` with its `sync.*` phases (quiesce, trainer send, P2P transfer, sampler install, resume).
* **trainer ranks**: what trainer rank 0 runs on its TPU chips: `update_actor` (`fwd_bwd` + `optimizer_step`),
  `compute_ref_log_prob` (reference forward), `update_weights` (`gather` / `bind` / `d2h` of the weight sync) and
  `release_raiden`. Gaps are idle chips.
* **rollout replica N**: state (generating / still generating behind a closed gate for a weight sync / paused for the
  weight transfer / idle), the weight version it serves (`vK` = weights after step `K`), requests decoding and queued,
  the weight-sync RPCs, and one bar per request.

Step labels (`S1`, `S2`, ...) and alternating stripes mark the steps on every lane. The summary panel and the per-step
table report how many seconds of every step the trainer chips spend in actor update, reference forward, weight sync
and idle, and how many the rollout replicas spend generating, paused and idle.

No change to verl is needed: the TPU components of verl-hardware-plugin print the spans (see
`verl_hardware_plugin/accelerators/tpu/timeline.py`), and the trainer-loop phases are rebuilt from the `timing_s/*`
step metrics, anchored at each step's weight sync.

## Record

Add `VERL_TIMELINE=1` to the job's runtime env (`VERL_TIMELINE_RANKS=0,1` or `all` prints more trainer ranks; rank 0
by default):

```bash
ray job submit --working-dir . \
  --runtime-env-json '{"py_modules": ["../verl-hardware-plugin/verl_hardware_plugin"],
                       "env_vars": {"VERL_TIMELINE": "1", "VERL_PLATFORM": "tpu",
                                    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin", ...}}' \
  -- env TOTAL_TRAINING_STEPS=10 VAL_BEFORE_TRAIN=False TEST_FREQ=-1 bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
```

Ten steps are enough to see the overlap of training and rollout. The prints cost one short stdout line per rollout
request plus a few per step.

## View

```bash
python3 scripts/accelerators/tpu/rl_timeline/make_report.py --ray-address http://localhost:8265 --job <submission-id> \
  --save-log job.log -o report.html --summary
# or from a saved log:
python3 scripts/accelerators/tpu/rl_timeline/make_report.py job.log -o report.html --summary
```

`report.html` is self-contained (open it from disk or share it). `index.html` alone also works: open it and drop a job
log on it. `--summary` prints the per-step table as text.

Mouse wheel zooms, drag pans, drag on the time axis zooms to a range, double-click fits, click a step label (or a row
of the step table) to zoom to one step, arrow keys move between steps; `report.html#steps=3-5` (or `#step=4`) opens
zoomed to those steps. **color by weights** colors the request bars by the weight version that generated them.
**Compare…** loads a second log or report and shows the steady-state phase means of both runs; **Export Perfetto**
writes a Chrome trace for ui.perfetto.dev.
