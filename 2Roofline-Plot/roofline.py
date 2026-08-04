#!/usr/bin/env python3
"""Interactive 3D energy-roofline with ONE point per phase per model.

Same axes as roofline3d.py (x = AI log, y = TFLOP/s log, z = power W),
but instead of plotting every decode step it plots exactly TWO points
per model: the prefill step and the AGGREGATE of all valid decode steps.

Two modes:

  MEASURED (--measured) - the publication path. F/B per step are ncu
                       COUNTER measurements: pass each model's per-phase
                       kernel JSONs (run_step.sh writes them as
                       <name>_<phase>_all.json). Bytes are L2-level
                       (Thor exposes no DRAM counters), so the roof and
                       gates switch to the L2 ceiling (--peak-l2-gbs,
                       or peak_l2_gbs in ceilings.json). Times, power,
                       energy still come from the clean run - ncu's
                       replay timing is never used; TFLOP/s here is
                       counted FLOPs / stopwatch seconds.
  ANALYTIC (no flag)   F/B per step come from step_metric5's formulas
                       (columns already in <base>.csv); roof = DRAM
                       ceiling. This is a PREDICTION, kept as the
                       cross-check against the counters - the plot and
                       stdout are labeled so it cannot be mistaken for
                       measured data.

Correctness rules baked in (same in both modes):

  * The decode point is the TIME-WEIGHTED aggregate, not a mean of
    per-step ratios:
        TFLOP/s = sum(F_i) / sum(t_i)      (not mean of F_i/t_i)
        AI      = sum(F_i) / sum(B_i)
        tok/s   = sum(tok_i) / sum(t_i)
        power   = sum(E_i)  / sum(t_i)     (E_i = P_avg_i * t_i)
    A mean of ratios over-weights fast steps - exactly how the phantom
    drain step inflated earlier decode averages by ~50%.
  * vLLM v1's terminal output-drain step (a ~1 ms bookkeeping iteration
    with no model forward, billed a full step of analytic work by older
    step_metric5 versions) is excluded from the AVERAGE - but never
    hidden: each excluded row is plotted as a red X at its real
    (impossible) coordinates, and a full forensic record (latency vs
    median, implied TFLOP/s / GB/s / tok/s, power-window quality) is
    printed so the artifact stays investigable. --hide-excluded removes
    the X markers for final figures; --no-exclude averages everything
    and demotes the impossibility aborts to warnings (inspection only).
    The criterion is physical (latency < --min-latency-frac x median =
    no forward pass in the window), not statistical outlier trimming;
    CSVs from the fixed step_metric5 pass through unchanged.
  * The decode marker shows the spread of the steps behind it: a +/-std
    error bar (per-step TFLOP/s over the KEPT steps). A drain-like spike
    that somehow survived filtering is caught by the roof gate below (it
    aborts if any single kept step's TFLOP/s exceeds the roof), not by a
    visual whisker.
  * Fails closed: aborts if any KEPT row implies memory traffic above
    the active ceiling, if an aggregated point lands above the roof by
    more than --roof-tolerance, or if even a SINGLE kept step peaks
    above the roof (a spike can hide inside a healthy-looking average).
    Points above a correctly measured roofline are physically
    impossible; refusing to plot them beats silently publishing them.

Usage (wherever the .csv/_summary.json files are - Jetson or laptop):

  measured (after run_step.sh; it writes the *_all.json inputs):
    python3 roofline_3d_avg.py model1 model2 model3 model4 \
        --measured model1=llama_prefill_all.json,llama_decode_all.json \
        --measured model2=gemma_prefill_all.json,gemma_decode_all.json \
        --measured model3=gpt_prefill_all.json,gpt_decode_all.json \
        --measured model4=qwen_prefill_all.json,qwen_decode_all.json \
        --peak-tflops 137.9 --peak-memory-gbs 239.7 \
        --out all_models_3d_avg_measured.html
    (append ,PER_STEP_LAUNCHES[,DECODE_STEPS] to a spec if the prefill
    capture hit the ncu launch cap and one step's launch count is
    ambiguous - same rule as measured_roofline.py)

  analytic cross-check (no captures needed):
    python3 roofline_3d_avg.py model1 model2 model3 model4 \
        --peak-tflops 137.9 --peak-memory-gbs 239.7 \
        --out all_models_3d_avg_analytic.html

  merged vLLM + Transformers, same machine (run from the dir above the
  two engine folders; engine is auto-detected per base from the
  summary's config.engine, falling back to the path):
    python3 roofline.py vllm/b/llama_b001 ... tf/b/llama_b001 ... \
        --peak-tflops 717.2 --peak-memory-gbs 4231.9 --fold-drain \
        --topic "Roofline H200" --out all_models_h200_merged.html
    Transformers points get a big surrounding ring (legend explains);
    vLLM points are drawn as before. Same model = same color in both
    engines.
"""
import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>html,body{{margin:0;height:100%;font-family:sans-serif}}#plot{{height:100vh}}</style>
</head><body>
<div id="plot"></div>
<script>
var traces = {traces};
var layout = {layout};
Plotly.newPlot('plot', traces, layout, {{responsive: true}});
</script>
</body></html>
"""

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Engine handling for merged vLLM + Transformers figures: every base is
# tagged 'vllm' or 'tf'; Transformers points get a big surrounding ring,
# vLLM points stay plain. Same model = same color in both engines.
ENGINE_LABEL = {"vllm": "vLLM", "tf": "TF"}

# Fixed default 2D y-axis so EVERY figure (Thor / A100 / H200, vLLM /
# TF, every batch) shares the same axis. 0.1 is the floor used on the
# earlier JT/A100 figures (their yrange was [0.1:400]); the ceiling is
# raised to 2200 because the H200 data outgrew 400 (llama vLLM b032
# prefill = 493 TFLOP/s, and the 717.2 TFLOP/s H200 roof must stay
# visible; 2200 ~ 3x that roof). --y-min/--y-max still override.
Y_AXIS_DEFAULT = (0.1, 2200.0)

# CSV columns that must parse as numbers for a row to be usable at all.
REQUIRED = ("latency_s", "flops", "bytes", "tokens")

LAUNCH_CAP = 700  # matches run_step.sh --launch-count


def fnum(s):
    """'' / None / non-numeric -> None, else float."""
    if s is None or s == "":
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(base):
    """Return (summary dict, list of per-step row dicts) for one model."""
    with open(base + "_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    rows = []
    with open(base + ".csv", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {"phase": raw.get("phase", ""), "step": raw.get("step", "?")}
            for k in ("tokens", "latency_s", "flops", "bytes",
                      "ai_flops_per_byte", "achieved_tflops", "achieved_gbs",
                      "gpu_power_w_avg", "gpu_power_w_peak", "gpu_power_w_dyn",
                      "gpu_util_pct_avg", "gpu_util_pct_peak",
                      "gpu_energy_j", "gpu_energy_dyn_j"):
                row[k] = fnum(raw.get(k))
            row["gpu_samples"] = fnum(raw.get("gpu_samples"))
            row["gpu_quality"] = raw.get("gpu_quality") or "?"
            if any(row[k] is None for k in REQUIRED):
                print(f"[warn] {base} step {row['step']}: missing "
                      f"{[k for k in REQUIRED if row[k] is None]}; row skipped")
                continue
            rows.append(row)
    return summary, rows


def detect_engine(base, summary):
    """Tag one base 'tf' or 'vllm', plus where the tag came from.

    The transformers twin (step_metric_tf.py) writes
    config.engine = "transformers" into the summary; the vLLM twin
    writes no engine key. When the key is missing, fall back to the
    base's path (…/tf/… vs …/vllm/…, or *_hf_* basenames), nearest
    path component first. Default: vllm (the original pipeline).
    """
    engine_cfg = str(summary.get("config", {}).get("engine", "")).lower()
    if engine_cfg:
        tf_like = engine_cfg in ("transformers", "tf", "hf")
        return ("tf" if tf_like else "vllm"), "summary config.engine"
    for part in reversed(re.split(r"[\\/]", base.lower())):
        if "vllm" in part:
            return "vllm", "path"
        if (part in ("tf", "hf") or "transformer" in part
                or "_tf" in part or "tf_" in part or "_hf" in part):
            return "tf", "path"
    return "vllm", "default (no engine key, no path hint)"


def check_row_consistency(base, rows):
    """Recompute the CSV's derived columns from its primary columns.

    Catches a truncated/hand-edited CSV before it poisons the averages.
    """
    for r in rows:
        checks = (
            ("ai_flops_per_byte", r["flops"] / r["bytes"]),
            ("achieved_tflops", r["flops"] / r["latency_s"] / 1e12),
            ("achieved_gbs", r["bytes"] / r["latency_s"] / 1e9),
        )
        for name, expect in checks:
            got = r[name]
            if got is None:
                continue
            if abs(got - expect) > 1e-6 * max(abs(expect), 1.0):
                raise SystemExit(
                    f"[error] {base} step {r['step']}: column {name}={got:g} "
                    f"but primary columns imply {expect:g}; CSV is "
                    "inconsistent - regenerate it")


def split_drain(decode_rows, min_frac):
    """Split decode rows into (kept, excluded-as-drain).

    The vLLM v1 drain step is ~100x shorter than a real decode step, so
    a factor-of-min_frac cut on the median latency separates them with a
    huge margin. Real steps vary by a few percent; the drain by ~10_000%.
    """
    if not decode_rows:
        return [], []
    med = statistics.median(r["latency_s"] for r in decode_rows)
    kept = [r for r in decode_rows if r["latency_s"] >= min_frac * med]
    dropped = [r for r in decode_rows if r["latency_s"] < min_frac * med]
    return kept, dropped


# ---------------------------------------------------------------------------
# measured mode: per-step F/B from kernel_roofline --top 500 JSONs
# (same steps-captured logic as measured_roofline.py)
# ---------------------------------------------------------------------------
def parse_measured_specs(specs):
    """--measured base=pre.json,dec.json[,per_step[,dec_steps]] -> dict."""
    out = {}
    for spec in specs:
        try:
            base, rest = spec.split("=", 1)
            parts = rest.split(",")
            pre_json, dec_json = parts[0], parts[1]
            per_step = float(parts[2]) if len(parts) > 2 else None
            dec_steps = float(parts[3]) if len(parts) > 3 else None
        except (ValueError, IndexError):
            raise SystemExit(f"[error] bad --measured spec {spec!r}; expected "
                             "BASE=PREFILL_JSON,DECODE_JSON"
                             "[,PER_STEP_LAUNCHES[,DECODE_STEPS]]")
        out[base] = (pre_json, dec_json, per_step, dec_steps)
    return out


def phase_totals(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    kernels = data["kernels"]
    return {
        "flops": sum(k["flops"] for k in kernels),
        "bytes": sum(k["bytes"] for k in kernels),
        "launches": sum(k["launches"] for k in kernels),
        "byte_source": data.get("byte_source", "?"),
        "coverage": data.get("coverage_pct"),
    }


def measured_per_step(base, pre_json, dec_json, per_step, dec_steps):
    """Return {'prefill': {...}, 'decode': {...}} with per-step F/B."""
    pre, dec = phase_totals(pre_json), phase_totals(dec_json)
    for name, tot in (("prefill", pre), ("decode", dec)):
        if tot["coverage"] is not None and tot["coverage"] < 95.0:
            print(f"[warn] {base} {name}: kernel JSON covers only "
                  f"{tot['coverage']:.1f}% of profiled time - regenerate "
                  "with kernel_roofline.py --top 500 so F/B are complete")
    if per_step is None:
        if pre["launches"] >= LAUNCH_CAP:
            raise SystemExit(
                f"[error] {base}: prefill capture hit the {LAUNCH_CAP}-"
                f"launch cap ({pre['launches']}), so one step's launch "
                "count is unknown; append ,PER_STEP_LAUNCHES to the "
                "--measured spec (prefill F/B sums are then partial too)")
        per_step = float(pre["launches"])
    dec_steps = dec_steps or dec["launches"] / per_step
    pre_steps = pre["launches"] / per_step
    print(f"[info] {base}: per-step launches {per_step:.0f}; decode capture "
          f"= {dec_steps:.2f} steps, prefill = {pre_steps:.2f} "
          f"(bytes: {dec['byte_source']})")
    return {
        "prefill": {"flops": pre["flops"] / pre_steps,
                    "bytes": pre["bytes"] / pre_steps,
                    "byte_source": pre["byte_source"],
                    "coverage": pre["coverage"]},
        "decode": {"flops": dec["flops"] / dec_steps,
                   "bytes": dec["bytes"] / dec_steps,
                   "byte_source": dec["byte_source"],
                   "coverage": dec["coverage"]},
    }


def apply_measured(rows, m):
    """Replace analytic F/B on each row; keep the analytic values aside.

    Returns (rows, mean analytic F, mean analytic B) for the validation
    ratio. Derived per-row rates are recomputed so the physics gate and
    the hover spread reflect the measured numbers.
    """
    fa = statistics.fmean(r["flops"] for r in rows)
    ba = statistics.fmean(r["bytes"] for r in rows)
    for r in rows:
        r["flops"], r["bytes"] = m["flops"], m["bytes"]
        r["ai_flops_per_byte"] = m["flops"] / m["bytes"]
        r["achieved_tflops"] = m["flops"] / r["latency_s"] / 1e12
        r["achieved_gbs"] = m["bytes"] / r["latency_s"] / 1e9
    return rows, fa, ba


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def aggregate(rows, label, fold_rows=(), fold_fb=None):
    """Time-weighted aggregate of per-step rows -> one plottable point.

    Rates use full totals. Telemetry (power/util) is averaged only over
    the wall time of rows where the sampler produced a value, so a step
    with a missing NVML sample cannot drag the average toward zero.

    fold_rows (--fold-drain): drain steps whose WORK and TIME are added
    to the totals but which get no per-step rate. Valid because every
    forward finished inside some sync-bracketed window, so sums are
    immune to the boundary misattribution that wrecks the per-step
    value. fold_fb=(F,B) overrides their per-row F/B in measured mode.
    """
    fold_rows = list(fold_rows)
    t = (sum(r["latency_s"] for r in rows)
         + sum(r["latency_s"] for r in fold_rows))
    F = sum(r["flops"] for r in rows)
    B = sum(r["bytes"] for r in rows)
    if fold_fb is not None:
        F += fold_fb[0] * len(fold_rows)
        B += fold_fb[1] * len(fold_rows)
    else:
        F += sum(r["flops"] for r in fold_rows)
        B += sum(r["bytes"] for r in fold_rows)
    tok = (sum(r["tokens"] for r in rows)
           + sum(r["tokens"] for r in fold_rows))
    if t <= 0 or F <= 0 or B <= 0:
        raise SystemExit(f"[error] {label}: non-positive totals "
                         f"(t={t:g}, F={F:g}, B={B:g})")

    def weighted(value_key):
        num = den = 0.0
        for r in list(rows) + fold_rows:
            v = r[value_key]
            if v is not None:
                num += v * r["latency_s"]
                den += r["latency_s"]
        return (num / den) if den > 0 else None

    def peak(key):
        vals = [r[key] for r in list(rows) + fold_rows
                if r[key] is not None]
        return max(vals) if vals else None

    energy = sum(r["gpu_energy_j"] for r in list(rows) + fold_rows
                 if r["gpu_energy_j"] is not None)
    per_step_tf = [r["achieved_tflops"] for r in rows]
    return {
        "n": len(rows),
        "n_fold": len(fold_rows),
        "time_s": t,
        "ai": F / B,
        "tflops": F / t / 1e12,
        "gbs": B / t / 1e9,
        "tps": tok / t,
        "pw": weighted("gpu_power_w_avg"),
        "pw_dyn": weighted("gpu_power_w_dyn"),
        "pwpk": peak("gpu_power_w_peak"),
        "util": weighted("gpu_util_pct_avg"),
        "utilpk": peak("gpu_util_pct_peak"),
        "energy_j": energy if energy > 0 else None,
        "j_per_tok": (energy / tok) if energy > 0 and tok else None,
        "tf_min": min(per_step_tf),
        "tf_max": max(per_step_tf),
        "tf_std": (statistics.stdev(per_step_tf)
                   if len(per_step_tf) > 1 else 0.0),
    }


def roof_at(ai, peak_t, bw):
    """Roofline ceiling (TFLOP/s) at arithmetic intensity ai."""
    return min(peak_t, ai * bw / 1000.0)


# ---------------------------------------------------------------------------
# plot pieces
# ---------------------------------------------------------------------------
def hover(model, phase, a, peak_t, bw, n_dropped, provenance):
    lines = [f"<b>{model}</b> {phase}"
             + (f" average of {a['n']} steps" if a["n"] > 1 else "")
             + (f" ({n_dropped} drain step excluded)" if n_dropped else "")]
    lines.append(f"AI {a['ai']:.3g} FLOP/byte")
    lines.append(f"{a['tflops']:.4g} TFLOP/s = {a['tps']:.4g} tok/s")
    if a["n"] > 1:
        lines.append(f"per-step std {a['tf_std']:.2g} TFLOP/s (error bar); "
                     f"min {a['tf_min']:.3g} / max {a['tf_max']:.3g}")
    pct = 100.0 * a["tflops"] / roof_at(a["ai"], peak_t, bw)
    lines.append(f"{a['gbs']:.4g} GB/s ({pct:.0f}% of roof at this AI)")
    if a["pw"] is not None:
        s = f"power {a['pw']:.3g} W avg"
        if a["pwpk"] is not None:
            s += f" / {a['pwpk']:.3g} W peak"
        if a["pw_dyn"] is not None:
            s += f" ({a['pw_dyn']:.3g} W dyn)"
        lines.append(s)
    if a["util"] is not None:
        s = f"util {a['util']:.0f}% avg"
        if a["utilpk"] is not None:
            s += f" / {a['utilpk']:.0f}% peak"
        lines.append(s)
    if a["energy_j"] is not None:
        s = f"{a['energy_j']:.3g} J over {a['time_s']:.3g} s"
        if a["j_per_tok"] is not None:
            s += f"; {a['j_per_tok']:.3g} J/token"
        if a["pw"]:  # same time-weighted power as the z axis: consistent
            s += f"; {a['tflops'] * 1e12 / a['pw'] / 1e9:.3g} GFLOP/s per W"
        lines.append(s)
    if provenance:
        lines.append(provenance)
    return "<br>".join(lines)


def point_trace(model, phase, a, color, peak_t, bw, n_dropped, provenance):
    trace = {
        "type": "scatter3d", "mode": "markers",
        "name": f"{model} {phase}" + (" avg" if a["n"] > 1 else ""),
        "legendgroup": f"{model} {phase}",
        "x": [a["ai"]], "y": [a["tflops"]],
        "z": [a["pw"] if a["pw"] is not None else 0.0],
        "text": [hover(model, phase, a, peak_t, bw, n_dropped, provenance)],
        "hoverinfo": "text",
        "marker": {
            "symbol": "square" if phase == "prefill" else "circle",
            "size": max(6.0, min(16.0, (a["util"] or 50) / 100 * 10 + 6)),
            "color": [a["util"] if a["util"] is not None else 50],
            "colorscale": "Viridis", "cmin": 0, "cmax": 100,
            "colorbar": {"title": "util %", "x": 1.02, "len": 0.5},
            "showscale": phase == "decode",
            "line": {"color": color, "width": 2},
        },
    }
    if a["n"] > 1:  # +/- std of per-step TFLOP/s, centered on sumF/sumT
        trace["error_y"] = {"type": "data", "array": [a["tf_std"]],
                            "visible": True, "color": color,
                            "thickness": 3, "width": 4}
    return trace


def ring_trace_3d(model, phase, a, color):
    """Big open ring around a Transformers point (3D counterpart of the
    2D pt-6 ring). Same legendgroup as the point, so one legend click
    toggles marker + ring together."""
    size = max(6.0, min(16.0, (a["util"] or 50) / 100 * 10 + 6))
    return {
        "type": "scatter3d", "mode": "markers",
        "name": f"{model} {phase} ring",
        "legendgroup": f"{model} {phase}",
        "showlegend": False, "hoverinfo": "skip",
        "x": [a["ai"]], "y": [a["tflops"]],
        "z": [a["pw"] if a["pw"] is not None else 0.0],
        "marker": {"symbol": "circle-open", "size": size + 8,
                   "color": color, "line": {"color": color, "width": 3}},
    }


def excluded_trace(model, rows, bw, bw_label):
    """Excluded drain steps, plotted as red X at their real coordinates.

    Nothing is removed invisibly: the step stays in the CSV, is drawn
    here with its impossible implied numbers, and only the AVERAGE
    omits it. --hide-excluded turns the markers off for final figures;
    --no-exclude averages them anyway (inspection only).
    """
    text = []
    for r in rows:
        n_samp = ("?" if r["gpu_samples"] is None
                  else f"{r['gpu_samples']:.0f}")
        text.append(
            f"<b>{model}</b> decode step {r['step']} - EXCLUDED from the "
            "average<br>"
            f"vLLM v1 output-drain step: {r['latency_s'] * 1e3:.3g} ms wall, "
            "no model forward<br>"
            f"billed work would imply {r['achieved_tflops']:.4g} TFLOP/s and "
            f"{r['achieved_gbs']:.0f} GB/s = "
            f"{r['achieved_gbs'] / bw:.0f}x the {bw_label} ceiling<br>"
            f"power window: {n_samp} sample(s), quality "
            f"'{r['gpu_quality']}'<br>"
            "row kept in the CSV for investigation")
    return {
        "type": "scatter3d", "mode": "markers",
        "name": f"{model} excluded (drain)",
        "x": [r["ai_flops_per_byte"] for r in rows],
        "y": [r["achieved_tflops"] for r in rows],
        "z": [r["gpu_power_w_avg"] if r["gpu_power_w_avg"] is not None
              else 0.0 for r in rows],
        "text": text, "hoverinfo": "text",
        "marker": {"symbol": "x", "size": 6, "color": "#b0342c"},
    }


def roof_surface(peak_t, bw, x_min, x_max, z_max, name):
    n = 60
    lx0, lx1 = math.log10(x_min), math.log10(x_max)
    xs = [10 ** (lx0 + (lx1 - lx0) * i / (n - 1)) for i in range(n)]
    ys = [roof_at(x, peak_t, bw) for x in xs]
    return {
        "type": "surface",
        "x": [xs, xs], "y": [ys, ys],
        "z": [[0.0] * n, [z_max] * n],
        "opacity": 0.25, "showscale": False,
        "colorscale": [[0, "#666666"], [1, "#666666"]],
        "name": name, "hoverinfo": "skip",
    }


def ridge_line(ridge, peak_t, z_max):
    return {
        "type": "scatter3d", "mode": "lines",
        "x": [ridge, ridge], "y": [peak_t, peak_t], "z": [0.0, z_max],
        "line": {"color": "#111111", "width": 5, "dash": "dash"},
        "name": f"ridge {ridge:.3g} F/B", "hoverinfo": "name",
    }


# ---------------------------------------------------------------------------
# companion 2D figure: ALL models on one log-log axes, 2 points per model
# (classic paper roofline; same aggregates, gnuplot .dat/.gnuplot/PNG)
# ---------------------------------------------------------------------------
def _label_text(a):
    """Compact per-point annotation: raw W, GPU util %, tok/s.

    The idle-subtracted (dyn) wattage is deliberately NOT shown here any
    more - it survives in the 3D hover and the console table.
    """
    pw = "?" if a["pw"] is None else f"{a['pw']:.0f}W"
    util = "" if a["util"] is None else f" {a['util']:.0f}%"
    tps = (f" {a['tps'] / 1000:.1f}k t/s" if a["tps"] >= 1000
           else f" {a['tps']:.0f} t/s")
    return pw + util + tps


# Candidate label placements, tried in this order; offsets are in
# character units of the terminal font. The first collision-free
# candidate wins, so labels prefer the right side, then the left, then
# the diagonals, then straight above/below.
_LABEL_PLACEMENTS = (
    ("E",  "left",   1.4,  0.0),
    ("W",  "right", -1.4,  0.0),
    ("NE", "left",   1.0,  0.95),
    ("SE", "left",   1.0, -0.95),
    ("NW", "right", -1.0,  0.95),
    ("SW", "right", -1.0, -0.95),
    ("N",  "center", 0.0,  1.25),
    ("S",  "center", 0.0, -1.25),
)


def _place_labels(label_pts, obstacle_pts, x_lo, x_hi, y_lo, y_hi):
    """One placement string per label so labels overlap neither each
    other, nor any plotted marker, nor the plot border.

    label_pts:    [(ai, tflops, text), ...] labelled points, plot order
    obstacle_pts: [(ai, tflops, half_w_px, half_h_px), ...] marker boxes
                  every label must clear (ringed / error-bar points get
                  a bigger box)
    Greedy: each label tries _LABEL_PLACEMENTS in order and takes the
    first candidate that collides with nothing placed so far; if all
    candidates collide it takes the least-overlapping one. Geometry is
    approximated in output pixels (1500x950 pngcairo, 14pt labels) -
    plenty to steer neighbouring labels to different sides.
    """
    ax_x0, ax_x1 = 100.0, 1452.0         # plot area in the 1500px canvas
    ax_y0, ax_y1 = 128.0, 872.0          # below 3-line title, above tics
    char_w, char_h = 9.6, 20.0           # gnuplot 'offset char' units
    lbl_char_w, lbl_h = 8.4, 19.0        # 14pt label text metrics
    lx0, lx1 = math.log10(x_lo), math.log10(x_hi)
    ly0, ly1 = math.log10(y_lo), math.log10(y_hi)

    def px(ai):
        return ax_x0 + (math.log10(ai) - lx0) / (lx1 - lx0) * (ax_x1 - ax_x0)

    def py(tf):
        return ax_y1 - (math.log10(tf) - ly0) / (ly1 - ly0) * (ax_y1 - ax_y0)

    def overlap(b1, b2):
        w = min(b1[2], b2[2]) - max(b1[0], b2[0])
        h = min(b1[3], b2[3]) - max(b1[1], b2[1])
        return w * h if w > 0 and h > 0 else 0.0

    obstacles = []
    for ai, tf, half_w, half_h in obstacle_pts:
        x, y = px(ai), py(tf)
        obstacles.append((x - half_w, y - half_h, x + half_w, y + half_h))

    placed, out = [], []
    for ai, tf, text in label_pts:
        x, y = px(ai), py(tf)
        w = len(text) * lbl_char_w
        best = None
        for _name, justify, dx, dy in _LABEL_PLACEMENTS:
            ax_, ay_ = x + dx * char_w, y - dy * char_h
            if justify == "left":
                bx0, bx1 = ax_, ax_ + w
            elif justify == "right":
                bx0, bx1 = ax_ - w, ax_
            else:
                bx0, bx1 = ax_ - w / 2, ax_ + w / 2
            box = (bx0 - 2, ay_ - lbl_h / 2 - 2, bx1 + 2, ay_ + lbl_h / 2 + 2)
            score = sum(overlap(box, o) for o in obstacles)
            score += sum(overlap(box, p) for p in placed)
            if box[0] < ax_x0:                  # keep text inside the plot
                score += (ax_x0 - box[0]) * lbl_h * 3
            if box[2] > ax_x1:
                score += (box[2] - ax_x1) * lbl_h * 3
            if box[1] < ax_y0:
                score += (ax_y0 - box[1]) * w * 3
            if box[3] > ax_y1:
                score += (box[3] - ax_y1) * w * 3
            if best is None or score < best[0]:
                best = (score, justify, dx, dy, box)
            if best[0] == 0.0:
                break
        _score, justify, dx, dy, box = best
        placed.append(box)
        out.append(f"{justify} offset char {dx:g},{dy:g}")
    return out


def write_2d(stem, table, excluded, peak_t, bw, measured_mode, inspection,
             point_labels=True, y_min=None, y_max=None, topic=None):
    """table rows: (model, engine, phase, aggregate, roof, naive,
    n_drop, ratio); excluded: (model, ai, tflops) of vLLM drain steps,
    drawn as red X. (inspection only changes console output upstream.)

    Merged-engine drawing: Transformers ('tf') rows get a big open ring
    around their marker plus one black key entry that explains it; vLLM
    rows are drawn exactly as before. Same model = same color in both
    engines, one legend entry per model.
    """
    dat, statements = [], []
    for model, engine, phase, a, *_rest in table:
        dat.append("\t".join((model, engine, phase, f"{a['ai']:.6g}",
                              f"{a['tflops']:.6g}", f"{a['tf_std']:.6g}",
                              f"{a['tf_min']:.6g}", f"{a['tf_max']:.6g}",
                              _label_text(a))))
    for model, ai, tf in excluded:
        dat.append("\t".join((model, "vllm", "excluded", f"{ai:.6g}",
                              f"{tf:.6g}", "0", f"{tf:.6g}", f"{tf:.6g}",
                              "excluded")))
    with open(stem + ".dat", "w", encoding="utf-8") as f:
        f.write("# model\tengine\tphase\tai\ttflops\tstd\tmin\tmax\tlabel\n")
        f.write("\n".join(dat) + "\n")

    # fixed default y-axis (Y_AXIS_DEFAULT) so EVERY figure shares one
    # scale across machines, engines and batch sizes; flags override.
    y_lo = Y_AXIS_DEFAULT[0] if y_min is None else y_min
    y_hi = Y_AXIS_DEFAULT[1] if y_max is None else y_max
    for model, engine, phase, a, *_rest in table:  # never clip silently
        if not y_lo <= a["tflops"] <= y_hi:
            print(f"[warn] {model} ({ENGINE_LABEL[engine]}) {phase} at "
                  f"{a['tflops']:.3g} TFLOP/s lies outside the fixed "
                  f"y-range [{y_lo:g}:{y_hi:g}] and will NOT be visible; "
                  "override with --y-min/--y-max")
    for _m, ai, tf in excluded:
        if not y_lo <= tf <= y_hi:
            print(f"[warn] excluded drain X at {tf:.3g} TFLOP/s lies "
                  f"outside [{y_lo:g}:{y_hi:g}] and will not be visible "
                  "(its forensic record is in the console output above)")

    ridge = 1000.0 * peak_t / bw
    ais = [r[3]["ai"] for r in table] + [e[1] for e in excluded]
    x_lo, x_hi = min(1.0, min(ais) / 2), max(max(ais) * 3, ridge * 3)

    models = []   # first-appearance order fixes each model's color
    for model, engine, phase, a, *_rest in table:
        if model not in models:
            models.append(model)
    any_tf = any(r[1] == "tf" for r in table)

    # collision-aware label placement: with two engines merged, points
    # can sit almost on top of each other (decode shares one AI), so
    # each label picks the first free side (E, W, NE, SE, NW, SW, N, S).
    # The roof line itself is an obstacle too, so labels don't sit on it.
    roof_obstacles = []
    for i in range(80):
        x = 10 ** (math.log10(x_lo)
                   + (math.log10(x_hi) - math.log10(x_lo)) * i / 79)
        y = roof_at(x, peak_t, bw)
        if y_lo <= y <= y_hi:
            roof_obstacles.append((x, y, 14.0, 5.0))
    placements = _place_labels(
        [(r[3]["ai"], r[3]["tflops"], _label_text(r[3])) for r in table],
        [(r[3]["ai"], r[3]["tflops"],
          16.0 if r[1] == "tf" else 12.0,
          26.0 if (r[2] == "decode" and r[3]["n"] > 1)
          else (16.0 if r[1] == "tf" else 12.0)) for r in table]
        + [(ai, tf, 10.0, 10.0) for _m, ai, tf in excluded]
        + roof_obstacles,
        x_lo, x_hi, y_lo, y_hi) if point_labels else []

    titled = set()   # one legend entry per model, on its first statement
    for mi, model in enumerate(models):
        color = COLORS[mi % len(COLORS)]
        for ri, (m, engine, phase, a, *_rest) in enumerate(table):
            if m != model:
                continue
            title = f"title '{model}'" if model not in titled else "notitle"
            titled.add(model)
            if phase == "decode" and a["n"] > 1:
                statements.append(
                    f"'{stem}.dat' every ::{ri}::{ri} using 4:5:6 "
                    f"with yerrorbars lc rgb '{color}' lw 3 pt 7 ps 1.6 "
                    f"{title}")
            else:
                pt = 5 if phase == "prefill" else 7
                statements.append(
                    f"'{stem}.dat' every ::{ri}::{ri} using 4:5 "
                    f"with points lc rgb '{color}' pt {pt} ps 1.8 {title}")
            if engine == "tf":  # big surrounding ring = Transformers
                statements.append(
                    f"'{stem}.dat' every ::{ri}::{ri} using 4:5 "
                    f"with points lc rgb '{color}' pt 6 ps 3.4 lw 2 "
                    "notitle")
            if point_labels:
                statements.append(
                    f"'{stem}.dat' every ::{ri}::{ri} using 4:5:9 "
                    f"with labels {placements[ri]} font ',14' "
                    f"tc rgb '{color}' notitle")
    if any_tf:  # explain the ring once, in the key
        statements.append("keyentry with points pt 6 ps 2 lw 2 "
                          "lc rgb 'black' title 'Transformers = ringed'")
    if excluded:  # contiguous block appended after the aggregate rows
        lo, hi = len(table), len(table) + len(excluded) - 1
        statements.append(
            f"'{stem}.dat' every ::{lo}::{hi} using 4:5 with points "
            "lc rgb '#b0342c' pt 2 ps 2 lw 2 "
            "title 'excluded (drain, not averaged)'")

    bw_word = "L2" if measured_mode else "DRAM"
    # three-line title: topic / marker guide / label guide. The
    # ANALYTIC-vs-MEASURED mode tag was removed from the figure at
    # Sean's request (2026-08-04); it still prints to the console.
    head = topic.replace('"', "'") if topic else "Roofline"
    title_txt = (head + "\\nPrefill (square) + Decode (circle)"
                 + "\\nlabels: raw W | GPU util % | tok/s")
    script = f"""set terminal pngcairo size 1500,950 font 'DejaVu Sans,16'
set output '{stem}.png'
set datafile separator '\\t'
set logscale xy
set grid back lc rgb '#dddddd'
set xlabel 'Arithmetic intensity (FLOP/byte)'
set ylabel 'TFLOP/s'
set title "{title_txt}"
set xrange [{x_lo:.6g}:{x_hi:.6g}]
set yrange [{y_lo:.6g}:{y_hi:.6g}]
peak_t = {peak_t:.6g}
peak_b = {bw:.6g}
ridge = 1000.0 * peak_t / peak_b
roof(x) = (x * peak_b / 1000 < peak_t) ? x * peak_b / 1000 : peak_t
set object 1 rectangle from graph 0, graph 0 to first ridge, graph 1 \\
    fc rgb '#eef4fb' fillstyle solid noborder behind
set object 2 rectangle from first ridge, graph 0 to graph 1, graph 1 \\
    fc rgb '#fdf1ee' fillstyle solid noborder behind
set arrow 1 from first ridge, graph 0 to first ridge, graph 1 nohead \\
    dashtype 2 lc rgb '#333333'
set label 1 sprintf('ridge %.0f F/B', ridge) at first ridge * 1.15, \\
    first peak_t * 0.4 tc rgb '#333333'
set label 2 'memory-bound' at graph 0.02, graph 0.05 tc rgb '#4a6fa5'
set label 3 'compute-bound' at graph 0.86, graph 0.94 tc rgb '#a55b4a'
set key left top box opaque
plot roof(x) with lines lw 3 lc rgb '#555555' \\
        title sprintf('roof: %.1f TF/s | %.1f {bw_word} GB/s', peak_t, peak_b), \\
    {", ".join(statements)}
"""
    with open(stem + ".gnuplot", "w", encoding="utf-8") as f:
        f.write(script)
    try:
        subprocess.run(["gnuplot", stem + ".gnuplot"], check=True)
        print(f"[done] {stem}.png (2D, all models on one axes)")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[warn] 2D PNG not rendered ({exc}); run: "
              f"gnuplot {stem}.gnuplot")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bases", nargs="+",
                   help="run basenames; reads <base>.csv + <base>_summary.json")
    p.add_argument("--peak-tflops", type=float, required=True,
                   help="measured compute ceiling (calibrate.py)")
    p.add_argument("--peak-memory-gbs", type=float, required=True,
                   help="measured DRAM bandwidth ceiling (calibrate.py)")
    p.add_argument("--measured", action="append", default=[],
                   metavar="BASE=PRE_JSON,DEC_JSON[,PER_STEP[,DEC_STEPS]]",
                   help="use ncu-counted F/B for this base (kernel JSONs "
                        "from kernel_roofline.py --top 500). If used, "
                        "EVERY base needs a spec, and the roof switches "
                        "to the L2 ceiling (bytes are L2-level on Thor)")
    p.add_argument("--peak-l2-gbs", type=float, default=None,
                   help="measured L2 bandwidth ceiling; required in "
                        "--measured mode (default: read peak_l2_gbs from "
                        "--ceilings)")
    p.add_argument("--ceilings", default="ceilings.json",
                   help="ceilings.json to read peak_l2_gbs from in "
                        "--measured mode (default ./ceilings.json)")
    p.add_argument("--min-latency-frac", type=float, default=0.5,
                   help="decode rows shorter than this fraction of the "
                        "median latency are excluded as vLLM v1 drain "
                        "steps (default 0.5; drain is ~0.01x). Applies "
                        "to vLLM bases ONLY - the transformers twin has "
                        "no drain step, and its fast-but-real steps "
                        "must not be cut")
    p.add_argument("--roof-tolerance", type=float, default=0.02,
                   help="abort if a plotted point exceeds the roof by more "
                        "than this relative margin (default 2%%)")
    p.add_argument("--fold-drain", action="store_true",
                   help="count each drain step's work AND time in the "
                        "phase totals instead of excluding them (its "
                        "tokens/FLOPs are real; only its per-step rate is "
                        "meaningless). Sums are immune to the boundary "
                        "misattribution, so this is the most complete "
                        "aggregate; it shifts decode by well under 1%%. "
                        "Folded steps get no red X and no per-step rate")
    p.add_argument("--no-point-labels", action="store_true",
                   help="omit the per-point power/util/tok-s annotations "
                        "on the 2D figure")
    p.add_argument("--hide-excluded", action="store_true",
                   help="do not draw excluded drain steps as red X markers "
                        "(default: they ARE drawn at their real coordinates "
                        "so no data point disappears invisibly)")
    p.add_argument("--no-exclude", action="store_true",
                   help="INSPECTION ONLY: average every decode row "
                        "including drain steps, and demote roof/physics "
                        "violations from aborts to warnings, so the "
                        "artifact can be studied in place. Never for "
                        "publication figures")
    p.add_argument("--y-min", type=float, default=None,
                   help="override the 2D y-axis (TFLOP/s) lower bound "
                        f"(default {Y_AXIS_DEFAULT[0]:g}, FIXED so every "
                        "figure across machines/engines/batches shares "
                        "one scale). Points outside the range are "
                        "reported, never clipped silently")
    p.add_argument("--y-max", type=float, default=None,
                   help="override the 2D y-axis (TFLOP/s) upper bound "
                        f"(default {Y_AXIS_DEFAULT[1]:g}, sized so the "
                        "717 TFLOP/s H200 roof and its b032 prefill "
                        "points stay on-chart)")
    p.add_argument("--topic", default=None,
                   help="headline for BOTH figures, e.g. 'roofline a100 vllm' "
                        "or 'roofline jetson thor transformers' (quote "
                        "multi-word topics on the command line). Replaces the "
                        "default 'Energy roofline ...' heading; the mode tag "
                        "(ANALYTIC/MEASURED) and the marker/label guide stay")
    p.add_argument("--out", default="roofline_3d_avg.html")
    args = p.parse_args()
    peak_t = args.peak_tflops

    specs = parse_measured_specs(args.measured)
    measured_mode = bool(specs)
    if measured_mode:
        missing = [b for b in args.bases if b not in specs]
        if missing:
            raise SystemExit(f"[error] --measured mode: no spec for "
                             f"{missing}; give every base a spec or none "
                             "(one plot uses one byte convention)")
        bw = args.peak_l2_gbs
        if bw is None:
            try:
                with open(args.ceilings, encoding="utf-8") as f:
                    bw = json.load(f).get("peak_l2_gbs")
            except OSError:
                bw = None
            if bw is None:
                raise SystemExit(
                    "[error] --measured mode needs the L2 ceiling: pass "
                    "--peak-l2-gbs, or run calibrate.py --l2-validate "
                    "--update-ceilings so ceilings.json has peak_l2_gbs "
                    "(kernel bytes are L2 traffic; a DRAM roof would be "
                    "the wrong ceiling)")
        bw_label = f"L2 {bw:g} GB/s"
        print(f"[info] measured mode: F/B from ncu counters; roof uses the "
              f"{bw_label} ceiling (L2 bytes are an upper bound on DRAM)")
    else:
        bw = args.peak_memory_gbs
        bw_label = f"DRAM {bw:g} GB/s"
        print("[note] ANALYTIC mode: F/B are formulas, not measurements. "
              "For counted FLOPs (the publication numbers) pass --measured "
              "with the *_all.json files run_step.sh produced.")

    def violation(msg):
        """Physically impossible result: abort, or warn in inspection mode."""
        if args.no_exclude:
            print(f"[warn] {msg} (continuing: --no-exclude inspection mode)")
        else:
            raise SystemExit(f"[error] {msg}")

    traces, points, table, excluded_pts = [], [], [], []
    model_colors = {}   # same model -> same color in BOTH engines
    for base in args.bases:
        try:
            summary, rows = load(base)
        except OSError as exc:
            print(f"[error] {base}: {exc}; skipping")
            continue
        check_row_consistency(base, rows)
        model = summary.get("config", {}).get("model", base).split("/")[-1]
        engine, engine_src = detect_engine(base, summary)
        eng_disp = ENGINE_LABEL[engine]
        disp = f"{model} ({eng_disp})"   # 3D trace/hover name
        print(f"[info] {base}: engine {eng_disp} (from {engine_src})")
        if model not in model_colors:
            model_colors[model] = COLORS[len(model_colors) % len(COLORS)]
        color = model_colors[model]

        prefill = [r for r in rows if r["phase"] == "prefill"]
        decode = [r for r in rows if r["phase"] == "decode"]
        if engine == "tf":
            # Transformers twin: exactly gen_len real forwards - the
            # vLLM v1 output-drain iteration does not exist there, so
            # the latency cut must not run. It would misfire on fast-
            # but-real steps (H200 decode steps 1-7 run ~9 ms vs the
            # ~27 ms median - real forwards, ~1/3 x median, nothing
            # like the ~0.04 x median bookkeeping drain step).
            kept, dropped = decode, []
        else:
            kept, dropped = split_drain(decode, args.min_latency_frac)
        if args.no_exclude:
            kept, dropped = decode, []
        for r in dropped:  # full forensic record, so this is investigable
            med = statistics.median(x["latency_s"] for x in decode)
            print(f"[excluded] {model} decode step {r['step']}: "
                  f"{r['latency_s'] * 1e3:.3g} ms vs {med * 1e3:.3g} ms "
                  f"median ({med / r['latency_s']:.0f}x shorter) - vLLM v1 "
                  "output-drain step, no model forward.")
            print(f"[excluded]   billed work would imply "
                  f"{r['achieved_tflops']:.4g} TFLOP/s, "
                  f"{r['achieved_gbs']:.0f} GB/s "
                  f"({r['achieved_gbs'] / bw:.0f}x the {bw_label} ceiling), "
                  f"{r['tokens'] / r['latency_s']:.0f} tok/s from batch "
                  f"{r['tokens']:.0f}; power window "
                  f"{('?' if r['gpu_samples'] is None else format(r['gpu_samples'], '.0f'))}"
                  f" sample(s), quality '{r['gpu_quality']}'.")
            if args.fold_drain:
                print("[excluded]   --fold-drain: its work and time are "
                      "counted in the phase totals; it just gets no "
                      "per-step rate (none exists).")
            else:
                print("[excluded]   the row STAYS in the CSV and is "
                      "plotted as a red X (--hide-excluded to hide; "
                      "--fold-drain to count its work in the totals; "
                      "--no-exclude to average it anyway).")
        if len(dropped) > max(2, 0.02 * len(decode)):
            raise SystemExit(f"[error] {model}: {len(dropped)} of "
                             f"{len(decode)} decode steps look like drain "
                             "steps; that is not one drain bug, the run "
                             "itself is suspect - not plotting it")

        # naive means over UNFILTERED analytic rows: what the old plot
        # effectively averaged; shown so the drain distortion is visible.
        naive = {ph: statistics.fmean(r["achieved_tflops"] for r in rws)
                 for ph, rws in (("prefill", prefill), ("decode", decode))
                 if rws}

        prov = {"prefill": "", "decode": ""}
        ratio_note = {}
        if measured_mode:
            m = measured_per_step(base, *specs[base])
            for ph, rws in (("prefill", prefill), ("decode", kept)):
                if not rws:
                    continue
                rws, fa, ba = apply_measured(rws, m[ph])
                ratio_note[ph] = (m[ph]["flops"] / fa, m[ph]["bytes"] / ba)
                cov = m[ph]["coverage"]
                prov[ph] = (f"F/B: ncu counters ({m[ph]['byte_source']}"
                            + (f", {cov:.0f}% coverage" if cov else "") + ")")

        # Physics gate on KEPT rows only: implied traffic above the
        # active ceiling means a timing/accounting bug, not fast hardware.
        for r in kept + prefill:
            if r["achieved_gbs"] > 1.05 * bw:
                violation(
                    f"{model} step {r['step']}: implies "
                    f"{r['achieved_gbs']:.0f} GB/s > measured {bw_label} "
                    "ceiling - timing artifact still present, fix the "
                    "run instead of plotting it")

        fold = dropped if args.fold_drain else []
        for phase, phase_rows, n_drop in (("prefill", prefill, 0),
                                          ("decode", kept,
                                           0 if fold else len(dropped))):
            if not phase_rows:
                print(f"[warn] {model}: no {phase} rows; skipping phase")
                continue
            if phase == "decode" and fold:
                fb = ((m["decode"]["flops"], m["decode"]["bytes"])
                      if measured_mode else None)
                a = aggregate(phase_rows, f"{model} {phase}", fold, fb)
                note = (f"{len(fold)} drain step folded into totals "
                        "(work+time counted; spread from real steps only)")
                prov[phase] = (prov[phase] + "<br>" + note
                               if prov[phase] else note)
            else:
                a = aggregate(phase_rows, f"{model} {phase}")
            roof = roof_at(a["ai"], peak_t, bw)
            if a["tflops"] > (1.0 + args.roof_tolerance) * roof:
                violation(
                    f"{model} {phase} aggregate {a['tflops']:.3g} "
                    f"TFLOP/s exceeds the {roof:.3g} TFLOP/s roof at "
                    f"AI={a['ai']:.3g} - physically impossible")
            # Spike alarm: the AVERAGE can look fine while one step is
            # impossible (a drain-like near-zero-latency division). No
            # single kept step may exceed the roof either.
            if a["n"] > 1 and a["tf_max"] > (1.0 + args.roof_tolerance) * roof:
                violation(
                    f"{model} {phase}: one step peaks at "
                    f"{a['tf_max']:.3g} TFLOP/s, above the {roof:.3g} roof "
                    "- a drain-like spike survived filtering; investigate "
                    "before publishing")
            table.append((model, engine, phase, a, roof, naive.get(phase),
                          n_drop, ratio_note.get(phase)))
            traces.append(point_trace(disp, phase, a, color, peak_t, bw,
                                      n_drop, prov[phase]))
            if engine == "tf":  # ring = Transformers, in 3D too
                traces.append(ring_trace_3d(disp, phase, a, color))
            points.append(a)
        if dropped and not args.hide_excluded and not args.fold_drain:
            traces.append(excluded_trace(disp, dropped, bw, bw_label))
            excluded_pts += [(model, r["ai_flops_per_byte"],
                              r["achieved_tflops"]) for r in dropped]

    if not traces:
        raise SystemExit("[error] no plottable runs")

    # --- validation table ---------------------------------------------
    print()
    print(f"{'model':<28}{'eng':<6}{'phase':<9}{'n':>4}{'AI':>7}{'TF/s':>8}"
          f"{'std':>8}{'min':>8}{'max':>8}"
          f"{'roof':>8}{'%roof':>7}{'GB/s':>8}{'tok/s':>8}{'W':>7}{'drop':>5}")
    for model, engine, phase, a, roof, naive_tf, n_drop, ratio in table:
        print(f"{model:<28}{ENGINE_LABEL[engine]:<6}{phase:<9}"
              f"{a['n']:>4}{a['ai']:>7.3g}"
              f"{a['tflops']:>8.3g}"
              f"{a['tf_std']:>8.2g}{a['tf_min']:>8.3g}{a['tf_max']:>8.3g}"
              f"{roof:>8.3g}"
              f"{100 * a['tflops'] / roof:>6.0f}%{a['gbs']:>8.3g}"
              f"{a['tps']:>8.4g}"
              f"{(a['pw'] if a['pw'] is not None else float('nan')):>7.3g}"
              f"{n_drop:>5}")
        if ratio:
            print(f"{'':<34}[check] measured vs analytical: "
                  f"F {ratio[0]:.2f}x  B {ratio[1]:.2f}x "
                  "(F should be ~1; B >1 means L2 reuse traffic)")
        if n_drop and naive_tf and naive_tf > 1.1 * a["tflops"]:
            print(f"{'':<34}[info] naive mean over ALL {phase} rows would "
                  f"be {naive_tf:.3g} TF/s ({naive_tf / a['tflops']:.2f}x); "
                  "the excluded drain step caused that inflation")
    print()

    ridge = 1000.0 * peak_t / bw
    all_ai = [a["ai"] for a in points]
    all_pw = [a["pw"] for a in points if a["pw"] is not None]
    x_min = min(1.0, min(all_ai) / 2)
    x_max = max(max(all_ai) * 2, ridge * 3)
    z_max = (max(all_pw) if all_pw else 30.0) * 1.25
    roof_name = "roof (L2 bytes)" if measured_mode else "roof"
    traces.insert(0, roof_surface(peak_t, bw, x_min, x_max, z_max, roof_name))
    traces.insert(1, ridge_line(ridge, peak_t, z_max))

    # the ANALYTIC/MEASURED mode tag is console-only now (removed from
    # both figure titles at Sean's request, 2026-08-04)
    if args.topic:
        title3d = (f"{args.topic} (hover for metrics; click legend "
                   "to toggle)")
    else:
        title3d = ("Energy roofline - prefill + decode average per "
                   "model (hover for metrics; click legend to toggle)")
    layout = {
        "title": {"text": title3d},
        "scene": {
            "xaxis": {"title": "AI (FLOP/byte)", "type": "log"},
            "yaxis": {"title": "TFLOP/s", "type": "log"},
            "zaxis": {"title": "power (W avg)", "range": [0, z_max]},
            "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 0.7}},
        },
        "legend": {"x": 0, "y": 1},
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
    }
    html = HTML.format(title=(args.topic or "energy roofline 3d (averaged)"),
                       traces=json.dumps(traces),
                       layout=json.dumps(layout))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    # companion 2D: all models on ONE axes, 2 points per model
    write_2d(re.sub(r"\.html?$", "", args.out) + "_2d",
             table, excluded_pts, peak_t, bw, measured_mode,
             args.no_exclude, point_labels=not args.no_point_labels,
             y_min=args.y_min, y_max=args.y_max, topic=args.topic)
    print(f"[done] {args.out} ({len(points)} points, std error bars "
          "on multi-step aggregates); open it in a browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())