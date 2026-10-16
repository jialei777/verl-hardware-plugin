# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Sample TPU HBM usage of every local chip from libtpu's runtime metrics service (``tpu-info``) once per period.

When ``VERL_TIMELINE=1`` is set on a training job, ``verl_hardware_plugin.accelerators.tpu.timeline.start_hbm_sampler``
already polls ``tpu-info`` inside the trainer and rollout processes and prints 1 Hz ``ph="C"`` HBM counter events
directly into the Ray job log.

Use ``hbm_monitor.py`` as a standalone sidecar inside a TPU worker pod when you also want to capture HBM build-up
*before* ``TorchTitanTPUEngine.initialize()`` or ``TPUvLLMHttpServer.launch_server()`` finishes (e.g. during initial
checkpoint loading), or on a run where ``VERL_TIMELINE`` was not enabled:

    python3 hbm_monitor.py [/tmp/hbm_monitor.csv] [period_seconds] [comma_separated_ports]

Each process that owns TPU chips serves runtime metrics on its own local gRPC port (``8431, 8432, ...``), so a few
ports are polled and appended as CSV rows:

    unix_time,port,device_id,memory_usage_bytes,total_memory_bytes,duty_cycle_pct
"""

import sys
import time

from tpu_info import device, metrics


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hbm_monitor.csv"
    period = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    ports = [int(p) for p in (sys.argv[3] if len(sys.argv) > 3 else "8431,8432,8433,8434").split(",")]

    chip_type, count = device.get_local_chips()
    with open(out_path, "a", buffering=1) as fh:
        fh.write(f"# chip_type={chip_type} count={count} ports={ports}\n")
        fh.write("unix_time,port,device_id,memory_usage_bytes,total_memory_bytes,duty_cycle_pct\n")
        while True:
            t = time.time()
            for port in ports:
                try:
                    usages = metrics.get_chip_usage(chip_type, addr=f"localhost:{port}")
                except Exception:  # no TPU workload on this port (yet)
                    continue
                for u in usages:
                    fh.write(f"{t:.3f},{port},{u.device_id},{u.memory_usage},{u.total_memory},{u.duty_cycle_pct:.1f}\n")
            time.sleep(max(0.0, period - (time.time() - t)))


if __name__ == "__main__":
    main()
