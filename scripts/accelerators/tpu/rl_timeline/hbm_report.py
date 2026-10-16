# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Plot TPU HBM usage per chip (from ``hbm_monitor.py`` CSVs or ``events.json``) against the RL timeline of a run.

Usage::

    python3 scripts/accelerators/tpu/rl_timeline/hbm_report.py <events.json> <job.log> <out_prefix> \\
        [<host_ip>=<csv> ...] [--steps A-B]

Produces ``<out_prefix>.html``, a standalone SVG HBM memory report showing:
1. Initial HBM build-up from 0 GiB through Actor/Ref model load, initial weight sync, Step 1 AdamW state init, and
   steady-state training.
2. Optional zoomed steady-state step window (`--steps A-B`).
3. Full-run Trainer and Rollout per-chip HBM traces with RL phase shading (`compute_ref_log_prob`, `update_actor`,
   `update_weights`) and a per-step HBM floor/peak table.
"""

import json
import re
import statistics as st
import sys
from collections import defaultdict

GIB = float(1024**3)
GB = 1e9
PHASE_COLORS = {
    "compute_ref_log_prob": "#9ccc65",
    "update_actor": "#ef5350",
    "update_weights": "#8e24aa",
    "initial_weight_sync": "#ab47bc",
    "release_raiden": "#ce93d8",
}
CHIP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load_csv(path, t_min=None, t_max=None):
    rows = defaultdict(list)  # (port, device_id) -> [(t, used, total, duty)]
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("unix_time"):
                continue
            t, port, dev, used, total, duty = line.strip().split(",")
            tf = float(t)
            if t_min is not None and tf < t_min:
                continue
            if t_max is not None and tf > t_max:
                continue
            rows[(int(port), int(dev))].append((tf, int(used), int(total), float(duty)))
    by_dev = {}
    for (port, dev), series in sorted(rows.items(), key=lambda kv: (kv[0][0] != 8431, -len(kv[1]))):
        by_dev.setdefault(dev, series)
    return {dev: sorted(s) for dev, s in sorted(by_dev.items())}


def load_events_hbm(events, group):
    """Extract per-chip series ``{dev: [(t, used_bytes, total_bytes, duty)]}`` from ``ph='C'`` events."""
    by_dev = defaultdict(list)
    for e in events:
        if e.get("ph") != "C" or e.get("name") != "hbm" or e.get("group") != group:
            continue
        ts = e.get("ts")
        args = e.get("args") or {}
        used = args.get("used") or []
        cap_b = int((args.get("cap") or 31.25) * GIB)
        duty = args.get("duty") or []
        for idx, u_gib in enumerate(used):
            d = duty[idx] if idx < len(duty) else 0.0
            by_dev[idx].append((float(ts), int(float(u_gib) * GIB), cap_b, float(d)))
    return {dev: sorted(s) for dev, s in sorted(by_dev.items())}


def host_roles(log_path):
    trainer, rollout = set(), set()
    pat = re.compile(r"\((\w+) pid=\d+, ip=([\d.]+)\)")
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            if m.group(1) == "WorkerDict":
                trainer.add(m.group(2))
            elif m.group(1) == "TPUvLLMHttpServer":
                rollout.add(m.group(2))
    return trainer, rollout


def spans(events, names):
    out = []
    for e in events:
        if "dur" not in e or e.get("name") not in names:
            continue
        if (e.get("group") == "trainer_workers" and e.get("lane") == "rank0") or (
            e.get("lane") == "loop" and e.get("name") == "initial_weight_sync"
        ):
            out.append((e["ts"], e["ts"] + e["dur"], e["name"]))
    return sorted(out)


def step_starts(events, metrics):
    syncs = sorted(
        (e["ts"], e["ts"] + e["dur"], e.get("step", (e.get("args") or {}).get("step")))
        for e in events
        if e.get("lane") == "loop" and e.get("name") in ("update_weights", "initial_weight_sync")
    )
    starts = {}
    for t0, t1, step in syncs:
        if step is not None:
            starts[int(step) + 1] = t1
    return starts


def window_max(series, a, b):
    vals = [u for t, u, _, _ in series if a <= t <= b]
    return max(vals) if vals else None


def window_min(series, a, b):
    vals = [u for t, u, _, _ in series if a <= t <= b]
    return min(vals) if vals else None


def svg_chart(title, series_by_chip, t0, t1, ymax, shades=(), steps=(), t_origin=None, width=1560, height=340):
    if t_origin is None:
        t_origin = t0
    pad_l, pad_r, pad_t, pad_b = 72, 24, 32, 34
    w, h = width - pad_l - pad_r, height - pad_t - pad_b

    def x(t):
        return pad_l + (t - t0) / max(t1 - t0, 1e-9) * w

    def y(v):
        return pad_t + h - v / ymax * h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;margin-bottom:14px" '
        f'font-family="Inter,system-ui,sans-serif" font-size="11">'
    ]
    out.append(f'<text x="{pad_l}" y="20" font-size="13" font-weight="600" fill="#111">{title}</text>')
    for a, b, name in shades:
        if b < t0 or a > t1:
            continue
        x0 = x(max(a, t0))
        rw = max(x(min(b, t1)) - x0, 0.5)
        out.append(
            f'<rect x="{x0:.1f}" y="{pad_t}" width="{rw:.1f}" height="{h}" '
            f'fill="{PHASE_COLORS.get(name, "#ccc")}" opacity="0.24"><title>{name} ({b - a:.2f}s)</title></rect>'
        )
    for g in range(0, int(ymax / GIB) + 1, 4):
        out.append(f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{y(g * GIB):.1f}" y2="{y(g * GIB):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{pad_l - 6}" y="{y(g * GIB) + 4:.1f}" text-anchor="end" fill="#555">{g} GiB</text>')
    for step, ts in steps:
        if t0 <= ts <= t1:
            xs = x(ts)
            out.append(
                f'<line x1="{xs:.1f}" x2="{xs:.1f}" y1="{pad_t}" y2="{pad_t + h}" '
                f'stroke="#aaa" stroke-dasharray="2,3"/>'
            )
            out.append(f'<text x="{xs + 2:.1f}" y="{pad_t + 11}" fill="#444" font-size="10">S{step}</text>')
    for i, (chip, series) in enumerate(sorted(series_by_chip.items())):
        pts = " ".join(f"{x(t):.1f},{y(u):.1f}" for t, u, _, _ in series if t0 <= t <= t1)
        col = CHIP_COLORS[i % 4]
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.35" points="{pts}"/>')
        out.append(
            f'<text x="{pad_l + w - 70}" y="{pad_t + 14 + 14 * i}" fill="{col}" font-weight="600">chip {chip}</text>'
        )
    total = max((s[0][2] for s in series_by_chip.values() if s), default=0)
    if total:
        yt = y(total)
        out.append(
            f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{yt:.1f}" y2="{yt:.1f}" stroke="#b71c1c" stroke-dasharray="6,3"/>'
        )
        out.append(
            f'<text x="{pad_l + 6}" y="{yt - 5:.1f}" fill="#b71c1c" font-weight="600">'
            f"HBM capacity {total / GIB:.2f} GiB ({total / GB:.2f} GB)</text>"
        )
    n_ticks = 10
    for k in range(n_ticks + 1):
        tk = t0 + (t1 - t0) * k / n_ticks
        out.append(f'<line x1="{x(tk):.1f}" x2="{x(tk):.1f}" y1="{pad_t + h}" y2="{pad_t + h + 4}" stroke="#888"/>')
        out.append(
            f'<text x="{x(tk):.1f}" y="{height - 10}" text-anchor="middle" fill="#555">'
            f"+{(tk - t_origin) / 60:.1f}m ({tk - t_origin:.0f}s)</text>"
        )
    out.append("</svg>")
    return "\n".join(out)


def _fmt_cell(row, key):
    if key not in row:
        return "—"
    return f"{row[key] / GIB:6.2f} GiB ({row[key] / GB:6.2f} GB)"


def main():
    events_path, log_path, out_prefix, *rest = sys.argv[1:]
    step_range = None
    if "--steps" in rest:
        i = rest.index("--steps")
        step_range = tuple(int(v) for v in rest[i + 1].split("-"))
        rest = rest[:i] + rest[i + 2 :]
    hosts = dict(arg.split("=", 1) for arg in rest)
    d = json.load(open(events_path))
    events, metrics = d["events"], {int(k): v for k, v in d["metrics"].items()}
    span_ts = [e["ts"] + (e.get("dur") or 0.0) for e in events if "ts" in e and e.get("ph") not in ("M", "C")] + [
        e["ts"] for e in events if "ts" in e and e.get("ph") not in ("M", "C")
    ]
    t_min = (min(span_ts) - 240.0) if span_ts else None
    t_max = (max(span_ts) + 30.0) if span_ts else None
    trainer_ips, rollout_ips = host_roles(log_path)
    data = {ip: load_csv(path, t_min=t_min, t_max=t_max) for ip, path in hosts.items()}
    trainer_ip = next((ip for ip in data if ip in trainer_ips), "trainer_workers")
    rollout_ip = next((ip for ip in data if ip in rollout_ips), "rollout_replica_0")
    tr = data.get(trainer_ip) or load_events_hbm(events, "trainer_workers")
    ro = data.get(rollout_ip) or load_events_hbm(events, "rollout_replica_0")
    print(
        f"trainer source {trainer_ip} (log: {sorted(trainer_ips)}), "
        f"rollout source {rollout_ip} (log: {sorted(rollout_ips)})"
    )

    starts = step_starts(events, metrics)
    ph = spans(events, set(PHASE_COLORS))
    all_t = [t for s in list(tr.values()) + list(ro.values()) for t, *_ in s]
    t0, t1 = min(all_t), max(all_t)
    ymax = 33 * GIB

    for label, dd in (("trainer", tr), ("rollout", ro)):
        for chip, s in sorted(dd.items()):
            used = [u for _, u, _, _ in s]
            print(
                f"{label} chip {chip}: {len(s)} samples, capacity {s[0][2] / GIB:.2f} GiB, "
                f"used min {min(used) / GIB:.2f} median {st.median(used) / GIB:.2f} "
                f"max {max(used) / GIB:.2f} GiB ({max(used) / GB:.2f} GB)"
            )

    by_step = defaultdict(dict)
    for a, b, name in ph:
        step = max((k for k, v in starts.items() if v <= a), default=0)
        vals = [window_max(s, a, b) for s in tr.values()]
        vals = [v for v in vals if v is not None]
        if vals:
            by_step[step][name] = max(max(vals), by_step[step].get(name, 0))
    for step in sorted(starts):
        a = starts[step]
        b = starts.get(step + 1, t1)
        lows = [window_min(s, a, b) for s in tr.values()]
        lows = [v for v in lows if v is not None]
        if lows:
            by_step[step]["min"] = min(lows)

    steps_list = sorted(starts.items())
    t_step4 = starts.get(4, min(t0 + 420, t1))
    tr_max_gib = max((max(u for _, u, _, _ in s) for s in tr.values() if s), default=0) / GIB
    ro_max_gib = max((max(u for _, u, _, _ in s) for s in ro.values() if s), default=0) / GIB
    charts = [
        svg_chart(
            f"1. Startup & Initial Build-Up (t = 0 to +{(t_step4 - t0) / 60:.1f}m, Init → Step 3) — "
            f"Peak {tr_max_gib:.2f} GiB / 31.25 GiB",
            tr,
            t0,
            t_step4,
            ymax,
            ph,
            steps_list,
            t_origin=t0,
        ),
    ]
    if step_range:
        a, b = starts.get(step_range[0], t0), starts.get(step_range[1] + 1, t1)
        charts.append(
            svg_chart(
                f"2. Steady-State Zoom (Steps {step_range[0]}–{step_range[1]}): "
                f"compute_ref_log_prob (green) → update_actor (red) → update_weights (purple)",
                tr,
                a,
                b,
                ymax,
                ph,
                [s for s in steps_list if a <= s[1] <= b],
                t_origin=t0,
            )
        )
    charts.extend(
        [
            svg_chart(
                f"3. Full Run — Trainer 4× v6e Chips ({trainer_ip}): HBM Used (GiB) with RL Phase Shading "
                f"(Peak {tr_max_gib:.2f} GiB / 31.25 GiB)",
                tr,
                t0,
                t1,
                ymax,
                ph,
                steps_list,
                t_origin=t0,
            ),
            svg_chart(
                f"4. Full Run — Rollout 4× v6e Chips ({rollout_ip}): HBM Used (GiB) "
                f"(Peak {ro_max_gib:.2f} GiB / 31.25 GiB)",
                ro,
                t0,
                t1,
                ymax,
                (),
                steps_list,
                t_origin=t0,
            ),
        ]
    )

    rows_html = []
    for step in sorted(by_step):
        r = by_step[step]
        sync_key = "update_weights" if "update_weights" in r else "initial_weight_sync"
        rows_html.append(
            f"<tr><td style='padding:4px 8px'>Step {step}</td>"
            f"<td style='padding:4px 8px'>{_fmt_cell(r, 'min')}</td>"
            f"<td style='padding:4px 8px'>{_fmt_cell(r, 'compute_ref_log_prob')}</td>"
            f"<td style='padding:4px 8px'><b>{_fmt_cell(r, 'update_actor')}</b></td>"
            f"<td style='padding:4px 8px'>{_fmt_cell(r, sync_key)}</td></tr>"
        )

    src_title = d.get("source", out_prefix)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TPU HBM Memory Dashboard — {src_title}</title></head>
<body style="margin:16px;background:#f8f9fa;font-family:Inter,system-ui,sans-serif;color:#222">
  <h2 style="margin:0 0 6px 0">TPU v6e HBM Memory Build-Up &amp; RL Phase Correlation — {src_title}</h2>
  <div style="font-size:13px;color:#555;margin-bottom:14px">
    Source: <code>{src_title}</code> |
    Trainer Peak: <b>{tr_max_gib:.2f} GiB / 31.25 GiB</b> ({tr_max_gib / 31.25 * 100:.1f}%) |
    Rollout Peak: <b>{ro_max_gib:.2f} GiB / 31.25 GiB</b> ({ro_max_gib / 31.25 * 100:.1f}%) |
    Shading: <span style="background:#ab47bc44;padding:2px 6px;border-radius:3px">initial_weight_sync</span>
    <span style="background:#9ccc6544;padding:2px 6px;border-radius:3px">compute_ref_log_prob</span>
    <span style="background:#ef535044;padding:2px 6px;border-radius:3px">update_actor (fwd_bwd + optimizer_step)</span>
    <span style="background:#8e24aa44;padding:2px 6px;border-radius:3px">update_weights</span>
  </div>
  {"".join(charts)}
  <div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:14px;margin-top:8px">
    <h3 style="margin:0 0 8px 0;font-size:15px">Per-Step Trainer HBM Summary (Max Across 4 Chips, GiB / GB)</h3>
    <table style="border-collapse:collapse;font-size:12px;width:760px">
      <thead><tr style="background:#f5f5f5;text-align:left">
        <th style="padding:6px 8px">Step</th><th style="padding:6px 8px">Step Min (Idle Floor)</th>
        <th style="padding:6px 8px"><code>compute_ref_log_prob</code> Max</th>
        <th style="padding:6px 8px"><code>update_actor</code> Peak</th>
        <th style="padding:6px 8px"><code>update_weights</code> Max</th>
      </tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>
</body></html>"""
    open(out_prefix + ".html", "w").write(html)
    print(f"\nwrote {out_prefix}.html")


if __name__ == "__main__":
    main()
