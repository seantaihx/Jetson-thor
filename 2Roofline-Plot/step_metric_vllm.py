#!/usr/bin/env python3
#!/usr/bin/env python3
"""Step-resolved vLLM profiler with an analytical roofline (dense models only).

Measures every scheduler step (one prefill + N decode steps) of a fixed
batch: synchronized wall latency, analytical FLOPs/bytes, throughput,
achieved TFLOP/s and GB/s, and NVML power/utilization sampled on a
background thread.

New in this version (over step_metric4):
  * Steps are identified by the tokens they actually produce, read from
    engine.step()'s outputs. vLLM v1 runs a terminal output-drain step
    with no model forward (sub-ms); it previously got billed a full
    decode step's FLOPs, inflating decode averages ~30-50%. Zero-token
    steps are now excluded (prefill chunks fold their time into the
    prefill record instead).
  * Per-phase totals (sum FLOPs / sum time, etc.) are reported alongside
    the per-step means: mean-of-rates != aggregate rate.
  * NVTX ranges ("warmup"/"prefill"/"decode") wrap every engine.step so
    Nsight Compute can gate capture per phase (--nvtx --nvtx-include).
    Negligible overhead when no profiler is attached.
  * --idle-baseline-s samples NVML power on the quiescent GPU before
    engine init and reports dynamic power/energy (measured minus idle)
    per step alongside the raw values - the paper's idle-baseline
    subtraction. Pin clocks BEFORE launching: pinned clocks raise idle
    power, so the baseline must be taken in the run's clock state.
  * Counter-measured per-kernel roofline: ncu recipe below, then feed the
    report to kernel_roofline.py (separate characterization run).

Fixes over step_metric4's predecessor:
  * VLLM_ENABLE_V1_MULTIPROCESSING=0 is set before vLLM is imported, so the
    EngineCore runs in this process: engine.step() executes exactly one
    scheduler step and torch.cuda.synchronize() bounds the timed GPU work.
  * enable_prefix_caching=False, so prefill FLOPs/bytes match what the GPU
    actually computes (identical prompts otherwise hit the block cache).
  * One untimed warmup generation covers Triton JIT and lazy init.
  * Prompts are passed as token IDs, not raw strings (deprecated in vLLM).
  * Prefill attention FLOPs are causal-masked (2*N^2*d, not 4*N^2*d) and
    prefill logits are counted once per sequence, not per prompt token.
  * Reports total vs. streamed parameters (untied lm_head handled via
    tie_word_embeddings); streamed params drive the FLOP/byte model.
  * MoE models use active params per token (attention + top-k experts +
    lm_head) for FLOPs and bytes. With batch > 1 the true per-step weight
    traffic is the union of routed experts, somewhere between active and
    total, so MoE byte counts are a lower bound. gpt-oss additionally ships
    MXFP4 expert weights, which --dtype-bytes 2 overstates: treat its
    bytes/roofline as approximate.
  * Unified-memory boards (Jetson): --kv-cache-gb (default 4) passes
    kv_cache_memory_bytes so vLLM's flaky memory profiling is skipped.

Remaining limits:
  * NVML is device-wide; Jetson power sensors update every few tens of ms,
    slower than one decode step, so per-step power is smoothed. Jetson NVML
    has no energy counter: energy = avg power x latency.
  * FLOP/byte formulas ignore activations and non-matmul work.

Setup (on the target NVIDIA system):
    pip install vllm pynvml        # torch/transformers ship with vllm
    sudo apt-get install gnuplot

Run (use ceilings measured by calibrate.py, not datasheet numbers):
    python3 step_metric5.py --choice 1 --batch-size 4 \
        --gen-len 256 --peak-memory-gbs <measured> --peak-tflops <measured> \
        --out llama_metrics

Model choices: 1=llama  2=gemma  3=gpt-oss (MoE)  4=qwen

Combine finished runs into one paper-style multi-panel figure
(arXiv:2512.01644 Fig. 9 layout, plus tok/s, power, util annotations):
    python3 step_metric5.py \
        --combine llama_metrics gemma_metrics gpt_metrics qwen_metrics \
        --peak-memory-gbs <measured> --peak-tflops <measured> --out all_models

Paper-exact per-kernel roofline (separate run; root needed for GPU perf
counters; expect tens of minutes - counter replay is slow):
    sudo $(which ncu) --target-processes all --nvtx \
        --nvtx-include "decode/" --launch-count 700 -f -o llama_decode \
        --metrics "regex:sm__ops_path_tensor_src_.*,dram__bytes.sum,\
gpu__time_duration.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum" \
        $(which python3) step_metric5.py --choice 1 --gen-len 4 \
        --idle-baseline-s 0 --out ncu_scratch
    sudo chown $USER llama_decode.ncu-rep
    python3 kernel_roofline.py llama_decode.ncu-rep --phase decode \
        --peak-tflops <measured> --peak-memory-gbs <measured> \
        --out llama_decode_kernels
For prefill: --nvtx-include "prefill/" and --gen-len 2. The timed metrics
printed by the profiler itself are meaningless under ncu replay - use the
ncu report only.
"""
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import threading
import time

# Must be set before vllm is imported anywhere.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# Load models from the local HF cache without hitting the network (these nodes
# often have no reliable internet). Export HF_HUB_OFFLINE=0 to allow downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SYSTEM_PROMPT = "You are a high performance computing scientific coding assistant"
USER_PROMPT = (
    "I have a GPU code written in CUDA. I need to convert this to a portable "
    "programming model. How can I convert a CUDA kernel to RAJA. "
    "Give two working examples."
)
MODEL_SPECS = [
    {"choice": "1", "name": "llama",
     "model_id": "/hpcgpfs01/scratch/stai/models/Llama-3.1-8B-Instruct",
     "hub": "meta-llama/Meta-Llama-3.1-8B-Instruct", "moe": False},
    {"choice": "2", "name": "gemma",
     "model_id": "/hpcgpfs01/scratch/stai/models/gemma-4-E4B-it",
     "hub": "google/gemma-4-E4B-it", "moe": False,
     "note": "multimodal 'effective-4B' architecture; the dense formulas "
             "are approximate for this model"},
    {"choice": "3", "name": "gpt",
     "model_id": "/hpcgpfs01/scratch/stai/models/gpt-oss-20b",
     "hub": "openai/gpt-oss-20b", "moe": True,
     "note": "MXFP4 expert weights; --dtype-bytes 2 overstates weight "
             "bytes, treat bytes/roofline as approximate"},
    {"choice": "4", "name": "qwen",
     "model_id": "/hpcgpfs01/scratch/stai/models/Qwen2.5-7B-Instruct",
     "hub": "Qwen/Qwen2.5-7B-Instruct", "moe": False},
]


# ---------------------------------------------------------------------------
# Analytical model (dense decoder)
# ---------------------------------------------------------------------------
def load_arch(model_id: str, dtype_bytes: int) -> dict:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cfg = getattr(cfg, "text_config", None) or cfg

    def first(*names):
        for name in names:
            value = getattr(cfg, name, None)
            if value:
                return value
        return None

    sliding = getattr(cfg, "use_sliding_window", None)
    if sliding is None:
        sliding = getattr(cfg, "sliding_window", None) is not None
    if sliding:
        print("[warn] config uses sliding-window attention; the attention "
              "FLOP/KV-byte terms assume full attention and will overcount")
    heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    return {
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "q_dim": heads * head_dim,
        "kv_dim": getattr(cfg, "num_key_value_heads", heads) * head_dim,
        "intermediate": cfg.intermediate_size,
        "vocab": cfg.vocab_size,
        "tied": bool(getattr(cfg, "tie_word_embeddings", False)),
        "dtype_bytes": dtype_bytes,
        "experts": first("num_local_experts", "num_experts",
                         "n_routed_experts"),
        "experts_per_tok": first("num_experts_per_tok", "top_k",
                                 "num_selected_experts"),
    }


def param_counts(a: dict, moe: bool) -> tuple:
    """Return (total, active) parameters.

    active = weights involved per token (attention + MLP-or-top-k-experts +
    lm_head); this drives per-step FLOPs and weight bytes. The embedding
    table is gathered row-wise, so it counts toward total (twice if untied)
    but not toward per-step traffic. For MoE with batch > 1 the true weight
    traffic is the union of routed experts (between active and total), so
    bytes are a lower bound.
    """
    attn = 2 * a["hidden"] * a["q_dim"] + 2 * a["hidden"] * a["kv_dim"]
    mlp = 3 * a["hidden"] * a["intermediate"]  # dense MLP / one expert
    if moe:
        if not (a["experts"] and a["experts_per_tok"]):
            raise SystemExit("[error] MoE selected but the config does not "
                             "expose num_experts and experts_per_tok")
        layer_total = attn + a["experts"] * mlp
        layer_active = attn + a["experts_per_tok"] * mlp
    else:
        layer_total = layer_active = attn + mlp
    lm_head = a["vocab"] * a["hidden"]
    active = a["layers"] * layer_active + lm_head
    total = a["layers"] * layer_total + lm_head * (1 if a["tied"] else 2)
    # active params that are routed-expert weights (quantized separately for
    # MXFP4 MoEs); 0 for dense so the byte split below is a no-op there.
    a["expert_params"] = a["layers"] * a["experts_per_tok"] * mlp if moe else 0
    return total, active


def _weight_bytes(a: dict, streamed: int) -> float:
    """Per-step weight DRAM bytes, billing quantized experts separately.

    Non-expert weights (attention, lm_head) move at dtype_bytes; routed-expert
    weights move at expert_dtype_bytes (e.g. ~0.53 B for MXFP4 = 4-bit + block
    scale), set via --expert-dtype-bytes. For dense models expert_params == 0
    and expert_dtype_bytes defaults to dtype_bytes, so this reduces exactly to
    streamed * dtype_bytes (unchanged)."""
    ew = a.get("expert_params", 0)
    eb = a.get("expert_dtype_bytes", a["dtype_bytes"])
    return (streamed - ew) * a["dtype_bytes"] + ew * eb


def prefill_cost(a: dict, streamed: int, n_tokens: int, batch: int) -> tuple:
    """FLOPs and bytes for one full-batch prefill step."""
    layer_params = streamed - a["vocab"] * a["hidden"]
    flops = batch * (
        2 * layer_params * n_tokens                    # dense matmuls
        + 2 * a["vocab"] * a["hidden"]                 # logits, last token only
        + a["layers"] * 2 * n_tokens**2 * a["q_dim"]   # causal attention
    )
    kv_token_bytes = a["layers"] * a["kv_dim"] * 2 * a["dtype_bytes"]
    bytes_moved = (_weight_bytes(a, streamed)          # weights read once
                   + batch * kv_token_bytes * n_tokens)  # KV cache writes
    return flops, bytes_moved


def decode_cost(a: dict, streamed: int, kv_len: int, batch: int) -> tuple:
    """FLOPs and bytes for one decode step (one new token per sequence)."""
    flops = batch * (
        2 * streamed                                   # dense matmuls + logits
        + a["layers"] * 4 * (kv_len + 1) * a["q_dim"]  # attention over context
    )
    kv_token_bytes = a["layers"] * a["kv_dim"] * 2 * a["dtype_bytes"]
    bytes_moved = (_weight_bytes(a, streamed)            # weights read once
                   + batch * kv_token_bytes * (kv_len + 1))  # KV read + write
    return flops, bytes_moved


# ---------------------------------------------------------------------------
# NVML power/utilization sampling
# ---------------------------------------------------------------------------
def _resolve_sampled_handles(nv, all_handles, selector):
    """Pick which NVML device handle(s) to sample for power/utilization.

    Many SLURM clusters honour --gres=gpu:1 for CUDA (CUDA_VISIBLE_DEVICES) but
    NOT for NVML: nvmlDeviceGetCount() still returns every physical GPU on the
    node, so summing over all handles adds the idle siblings' power and averages
    utilization down by the GPU count. This resolves the single GPU the job is
    actually using so power/util are correct even then.

      selector 'auto'/None : the compute GPU, from CUDA_VISIBLE_DEVICES (or the
                             only GPU when just one is visible).
      selector 'all'       : every visible GPU (old behaviour; power summed).
      selector '<int>'     : that NVML index.
      selector 'GPU-<uuid>': the device with that UUID.
    Returns (handles, human_description).
    """
    n = len(all_handles)

    def uuid_of(h):
        try:
            u = nv.nvmlDeviceGetUUID(h)
            return u.decode() if isinstance(u, bytes) else u
        except Exception:
            return "?"

    def by_index(i):
        if 0 <= i < n:
            return [all_handles[i]], f"NVML index {i} (uuid {uuid_of(all_handles[i])})"
        return None

    def by_uuid(u):
        u = u.strip()
        for i, h in enumerate(all_handles):
            hu = uuid_of(h)
            if hu == u or hu.endswith(u):
                return [h], f"uuid {hu} (NVML index {i})"
        return None

    sel = (selector or "auto").strip()
    if sel == "all":
        return all_handles, f"all {n} visible GPU(s) (power SUMMED, util averaged)"
    if sel != "auto":
        r = (by_uuid(sel) if sel.startswith(("GPU-", "MIG-"))
             else by_index(int(sel)) if sel.lstrip("-").isdigit() else None)
        if r:
            return r
        print(f"[warn] --nvml-device {sel!r} matched none of {n} visible "
              "GPU(s); sampling all (may be polluted)")
        return all_handles, f"all {n} visible GPU(s) (fallback)"
    # auto: identify the compute GPU
    cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    if cvd:
        first = cvd[0].strip()
        r = (by_uuid(first) if first.startswith(("GPU-", "MIG-"))
             else by_index(int(first)) if first.lstrip("-").isdigit() else None)
        if r:
            return r[0], r[1] + " [CUDA_VISIBLE_DEVICES]"
    if n == 1:
        return all_handles, f"the only visible GPU (uuid {uuid_of(all_handles[0])})"
    print(f"[warn] {n} GPUs visible to NVML but the compute GPU could not be "
          "identified (CUDA_VISIBLE_DEVICES unset?); sampling ALL - power will "
          "be summed. Pass --nvml-device <index|GPU-uuid>.")
    return all_handles, f"all {n} visible GPU(s) (could not auto-pick)"


class GpuSampler:
    """Background NVML power/util sampling of the SELECTED GPU(s).

    Defaults to only the compute GPU (see _resolve_sampled_handles); pass
    device_selector='all' to sum every visible GPU (the old behaviour)."""

    def __init__(self, interval_s: float, device_selector=None):
        self.interval_s = interval_s
        self.samples = []  # (ts_ns, power_w, util_pct); read only after stop()
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            all_handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                           for i in range(pynvml.nvmlDeviceGetCount())]
            self._handles, desc = _resolve_sampled_handles(
                pynvml, all_handles, device_selector)
            print(f"[info] NVML power/util sampling: {desc}")
        except Exception as exc:
            print(f"[warn] NVML unavailable; power/util will be null ({exc})")
            self._nv, self._handles = None, []

    def _sample(self):
        power = util = None
        try:
            power = sum(self._nv.nvmlDeviceGetPowerUsage(h)
                        for h in self._handles) / 1000.0
        except Exception:
            pass
        try:
            util = statistics.fmean(
                self._nv.nvmlDeviceGetUtilizationRates(h).gpu
                for h in self._handles)
        except Exception:
            pass
        self.samples.append((time.perf_counter_ns(), power, util))

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self):
        if self._handles:
            self._sample()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._sample()
            self._thread = None

    def window(self, start_ns: int, end_ns: int, latency_s: float) -> dict:
        def spread(vals: list) -> dict:
            return {"avg": statistics.fmean(vals) if vals else None,
                    "std": statistics.stdev(vals) if len(vals) > 1 else None,
                    "peak": max(vals) if vals else None}

        if not self.samples:
            rows, quality = [], "unavailable"
        else:
            rows = [s for s in self.samples if start_ns <= s[0] <= end_ns]
            quality = ("sampled" if len(rows) >= 2
                       else "single_sample" if rows else "nearest")
            if not rows:  # step shorter than sampling period: nearest sample
                mid = (start_ns + end_ns) // 2
                rows = [min(self.samples, key=lambda s: abs(s[0] - mid))]
        power = spread([p for _, p, _ in rows if p is not None])
        util = spread([u for _, _, u in rows if u is not None])
        return {
            "gpu_samples": len(rows),
            "gpu_quality": quality,
            "gpu_power_w_avg": power["avg"],
            "gpu_power_w_std": power["std"],
            "gpu_power_w_peak": power["peak"],
            "gpu_util_pct_avg": util["avg"],
            "gpu_util_pct_std": util["std"],
            "gpu_util_pct_peak": util["peak"],
            "gpu_energy_j": (power["avg"] * latency_s
                             if power["avg"] is not None else None),
        }


def measure_idle_power(seconds: float, interval_s: float, device_selector=None):
    """Mean NVML power over a quiescent window before the engine exists."""
    sampler = GpuSampler(interval_s, device_selector)
    if not sampler._handles:
        return None
    print(f"[info] sampling idle power for {seconds:g}s "
          "(GPU must be quiescent; pin clocks before launching, not after)")
    sampler.start()
    time.sleep(seconds)
    sampler.stop()
    powers = [p for _, p, _ in sampler.samples if p is not None]
    if not powers:
        return None
    idle = statistics.fmean(powers)
    print(f"[info] idle baseline {idle:.2f} W over {len(powers)} samples "
          f"(std {statistics.stdev(powers):.3f} W)" if len(powers) > 1
          else f"[info] idle baseline {idle:.2f} W (single sample)")
    return idle


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def stats(values) -> dict | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return {"n": len(vals), "avg": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}


def summarize(records: list) -> dict:
    metrics = ("latency_s", "tokens_per_second", "achieved_tflops",
               "achieved_gbs", "gpu_power_w_avg", "gpu_power_w_peak",
               "gpu_power_w_dyn", "gpu_util_pct_avg", "gpu_util_pct_peak",
               "gpu_energy_j", "gpu_energy_dyn_j")
    out = {}
    for phase in ("prefill", "decode"):
        rows = [r for r in records if r["phase"] == phase]
        out[phase] = {m: stats(r.get(m) for r in rows) for m in metrics}
        out[phase]["steps"] = len(rows)
        # Aggregate rates: mean of per-step rates is NOT the phase rate.
        f_sum = sum(r["flops"] for r in rows)
        b_sum = sum(r["bytes"] for r in rows)
        t_sum = sum(r["latency_s"] for r in rows)
        tok_sum = sum(r["tokens"] for r in rows)
        out[phase]["phase_totals"] = {
            "flops": f_sum, "bytes": b_sum, "time_s": t_sum,
            "tokens": tok_sum,
            "tflops": f_sum / t_sum / 1e12 if t_sum else None,
            "gbs": b_sum / t_sum / 1e9 if t_sum else None,
            "ai": f_sum / b_sum if b_sum else None,
            "tokens_per_second": tok_sum / t_sum if t_sum else None,
        }
    return out


def mark_drain_like(records: list) -> list:
    """Flag token-CARRYING drain steps by wall time.

    The measured loop already excludes vLLM v1's zero-token drain
    iteration, but the engine can also deliver the final batch of tokens
    on a near-instant last iteration (the forward that produced them ran
    inside an earlier step's window). new_tokens == batch, so the token
    check passes - yet no model forward fits in the wall time, and
    billing a full step's F/B against ~1 ms fabricates an impossible
    above-roof point. Rows are only FLAGGED (drain_like=True): they stay
    in the CSV for investigation; summary stats and the roofline .dat
    leave them out."""
    for r in records:
        r["drain_like"] = False
    decode = [r for r in records if r["phase"] == "decode"]
    if len(decode) < 4:
        return records
    med = statistics.median(r["latency_s"] for r in decode)
    for r in decode:
        if r["latency_s"] < 0.5 * med:
            r["drain_like"] = True
            print(f"[warn] decode step {r['step']}: "
                  f"{r['latency_s'] * 1e3:.3g} ms vs {med * 1e3:.3g} ms "
                  f"median ({med / max(r['latency_s'], 1e-9):.0f}x shorter)"
                  " - token-carrying drain step, no model forward fits in "
                  "this window. Billed work would imply "
                  f"{r['achieved_tflops']:.4g} TFLOP/s / "
                  f"{r['achieved_gbs']:.0f} GB/s (impossible). Row KEPT in "
                  "the CSV (drain_like=True); excluded from summary stats "
                  "and the roofline .dat.")
    return records


def write_outputs(base: str, records: list, summary: dict):
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with open(base + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] CSV: {base}.csv")
    print(f"[done] summary: {base}_summary.json")


# Paper-style panel (arXiv:2512.01644 Fig. 9): sloped memory roof meeting a
# flat compute ceiling at the ridge point, shaded bound regions, and a stats
# box with throughput / power / utilization.
PANEL_TEMPLATE = """unset label
unset object
unset arrow
set title "{title}" font ',13'
set xlabel "Arithmetic Intensity (FLOP/byte)"
set ylabel "Performance (TFLOP/s)"
set logscale xy
set xrange [{x_min:.6g}:{x_max:.6g}]
set yrange [{y_min:.6g}:{y_max:.6g}]
set grid xtics ytics
set key top left
set object 1 rectangle from graph 0, graph 0 to first {ridge:.6g}, graph 1 \\
    fillcolor rgb '#dce9f7' fillstyle solid 0.45 noborder behind
set object 2 rectangle from first {ridge:.6g}, graph 0 to graph 1, graph 1 \\
    fillcolor rgb '#fbdcdc' fillstyle solid 0.45 noborder behind
set label 1 "Memory Bound" at graph 0.04, graph 0.55 left \\
    textcolor rgb '#2563eb' font ',11'
set label 2 "Compute Bound" at graph 0.96, graph 0.55 right \\
    textcolor rgb '#dc2626' font ',11'
set label 3 "Ridge({ridge:.4g}, {tflops:.4g})" \\
    at first {ridge:.6g}, first {tflops:.6g} right \\
    point pointtype 5 pointsize 1.2 \\
    offset character -1.5, character 1 font ',10'
set label 4 "{stats}" at graph 0.96, graph 0.16 right font ',10'
peak_t = {tflops:.6g}
peak_b = {gbs:.6g}
roof(x) = (x * peak_b / 1000.0 < peak_t) ? x * peak_b / 1000.0 : peak_t
plot roof(x) with lines linewidth 3 linecolor rgb '#111111' title 'Roof', \\
     '{dat}' using 3:(strcol(2) eq 'prefill' ? $4 : 1/0) with points \\
         pointtype 5 pointsize 1.6 linecolor rgb '#1f77b4' title 'Prefill', \\
     '{dat}' using 3:(strcol(2) eq 'decode' ? $4 : 1/0) with points \\
         pointtype 7 pointsize 1.0 linecolor rgb '#ff7f0e' title 'Decode'
"""


# Time-series panel (decode steps): throughput / power / utilization.
TS_TEMPLATE = """unset label
unset object
unset arrow
unset logscale
set xrange [*:*]
set yrange [*:*]
set title "{title}" font ',12'
set xlabel "step"
set ylabel "{ylabel}"
set grid xtics ytics
set key top right font ',9'
plot {series}
"""


def _series(dat: str, col: int, color: str, label: str) -> str:
    return (f"'{dat}' using 1:(strcol(2) eq 'decode' ? ${col} : 1/0) "
            f"with points pointtype 7 pointsize 0.4 "
            f"linecolor rgb '{color}' title '{label}'")


def _ts(dat: str, title: str, ylabel: str, series: list) -> str:
    return TS_TEMPLATE.format(title=title, ylabel=ylabel,
                              series=", \\\n     ".join(series))


def _dashboard(dat: str, name: str, ais: list, perfs: list,
               peak_tflops, peak_gbs, stats: str) -> list:
    """Roofline + tok/s + power + util panels for one model run."""
    return [
        _panel(dat, name, ais, perfs, peak_tflops, peak_gbs, stats),
        _ts(dat, f"{name}: decode throughput", "tokens/s",
            [_series(dat, 5, "#ff7f0e", "tok/s")]),
        _ts(dat, f"{name}: GPU power", "watts",
            [_series(dat, 6, "#d62728", "avg"),
             _series(dat, 8, "#f4a6a6", "peak")]),
        _ts(dat, f"{name}: GPU utilization", "%",
            [_series(dat, 7, "#2ca02c", "avg"),
             _series(dat, 9, "#a9d7a9", "peak")]),
    ]


def _num(value) -> str:
    return f"{value:.6g}" if isinstance(value, (int, float)) else "NaN"


def stats_text(summary: dict) -> str:
    """Decode-phase annotation lines for the plot (\\n-joined for gnuplot)."""
    d = summary["decode"]

    def get(metric, field):
        s = d.get(metric)
        return s[field] if s else None

    lines = []
    totals = d.get("phase_totals") or {}
    tps = totals.get("tokens_per_second") or get("tokens_per_second", "avg")
    gbs = totals.get("gbs") or get("achieved_gbs", "avg")
    if tps is not None and gbs is not None:
        lines.append(f"decode {tps:.1f} tok/s, {gbs:.0f} GB/s (phase totals)")
    p_avg, p_pk = get("gpu_power_w_avg", "avg"), get("gpu_power_w_peak", "max")
    if p_avg is not None and p_pk is not None:
        lines.append(f"power {p_avg:.1f} W avg / {p_pk:.1f} W peak")
    u_avg, u_pk = get("gpu_util_pct_avg", "avg"), get("gpu_util_pct_peak", "max")
    if u_avg is not None and u_pk is not None:
        lines.append(f"util {u_avg:.0f}% avg / {u_pk:.0f}% peak")
    e_dyn = get("gpu_energy_dyn_j", "avg")
    e_raw = get("gpu_energy_j", "avg")
    batch = summary.get("config", {}).get("batch_size")
    if batch and e_dyn is not None and e_raw is not None:
        lines.append(f"{e_dyn / batch:.2f} J/tok dyn "
                     f"({e_raw / batch:.2f} raw)")
    elif batch and e_raw is not None:
        lines.append(f"{e_raw / batch:.2f} J/token")
    return "\\n".join(lines)


def _panel(dat: str, title: str, ais: list, perfs: list,
           peak_tflops, peak_gbs, stats: str) -> str:
    if peak_tflops is None:  # empirical fallback: observed ceilings
        peak_tflops = max(perfs)
        peak_gbs = max(1000.0 * p / a for a, p in zip(ais, perfs) if a > 0)
        title += " (empirical ceilings)"
    ridge = 1000.0 * peak_tflops / peak_gbs
    return PANEL_TEMPLATE.format(
        dat=dat, title=title, stats=stats,
        tflops=peak_tflops, gbs=peak_gbs, ridge=ridge,
        x_min=min(1.0, min(ais) / 2),
        x_max=max(max(ais) * 2, ridge * 3),  # keep the knee in view
        y_min=min(perfs) / 3,
        y_max=peak_tflops * 2.5)


def _render(script_path: str, script: str, png: str):
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"[done] gnuplot script: {script_path}")
    try:
        subprocess.run(["gnuplot", script_path], check=True,
                       capture_output=True, text=True)
        print(f"[done] roofline PNG: {png}")
    except FileNotFoundError:
        print(f"[warn] gnuplot not installed; render later with: "
              f"gnuplot {script_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[warn] gnuplot failed: {(exc.stderr or '').strip()}")


def write_roofline(base: str, records: list, peak_tflops, peak_gbs,
                   title: str, stats: str):
    dat, script, png = (base + "_roofline.dat", base + "_roofline.gnuplot",
                        base + "_roofline.png")
    with open(dat, "w", encoding="utf-8") as f:
        for r in records:
            f.write("\t".join((
                str(r["step"]), r["phase"],
                f"{r['ai_flops_per_byte']:.6g}",
                f"{r['achieved_tflops']:.6g}",
                f"{r['tokens_per_second']:.6g}",
                _num(r.get("gpu_power_w_avg")),
                _num(r.get("gpu_util_pct_avg")),
                _num(r.get("gpu_power_w_peak")),
                _num(r.get("gpu_util_pct_peak")))) + "\n")
    print(f"[done] roofline data: {dat}")
    ais = [r["ai_flops_per_byte"] for r in records]
    perfs = [r["achieved_tflops"] for r in records]
    header = (f"set terminal pngcairo size 1500,1100 enhanced font 'Sans,11'\n"
              f"set output '{png}'\nset datafile separator \"\\t\"\n"
              f"set datafile missing 'NaN'\n"
              f"set multiplot layout 2,2\n")
    panels = _dashboard(dat, title, ais, perfs, peak_tflops, peak_gbs, stats)
    _render(script, header + "\n".join(panels) + "\nunset multiplot\n", png)


def reprocess(base: str, peak_tflops, peak_gbs) -> int:
    """Rebuild <base>_summary.json + roofline .dat/PNG from <base>.csv.

    No GPU, no engine: applies mark_drain_like to a run recorded before
    the fix existed, so the token-carrying drain step drops out of the
    stats and plots. The CSV itself is never rewritten - raw data stays
    exactly as measured."""
    records = []
    with open(base + ".csv", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            r = {}
            for k, v in raw.items():
                if k in ("phase", "gpu_quality", "step"):
                    r[k] = v
                elif v in ("", None):
                    r[k] = None
                else:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        r[k] = v
            records.append(r)
    if not records:
        raise SystemExit(f"[error] {base}.csv is empty")
    mark_drain_like(records)
    clean = [r for r in records if not r["drain_like"]]
    try:
        with open(base + "_summary.json", encoding="utf-8") as f:
            config = json.load(f).get("config", {})
    except OSError:
        config = {}
    summary = summarize(clean)
    summary["config"] = config
    with open(base + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    n_drain = len(records) - len(clean)
    print(f"[done] summary rebuilt from {base}.csv "
          f"({n_drain} drain-like row(s) excluded): {base}_summary.json")
    title = (str(config.get("model", base)).split("/")[-1]
             + f" (batch {config.get('batch_size', '?')})")
    write_roofline(base, clean, peak_tflops, peak_gbs, title,
                   stats_text(summary))
    return 0


def combine_plots(bases: list, out: str, peak_tflops, peak_gbs) -> int:
    """Merge finished runs (one per model) into a multi-panel figure."""
    panels = []
    for base in bases:
        try:
            with open(base + "_summary.json", encoding="utf-8") as f:
                summary = json.load(f)
            with open(base + "_roofline.dat", encoding="utf-8") as f:
                rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
        except OSError as exc:
            raise SystemExit(f"[error] missing outputs for '{base}' "
                             f"(run the profiler first): {exc}")
        ais = [float(r[2]) for r in rows]
        perfs = [float(r[3]) for r in rows]
        title = summary.get("config", {}).get("model", base).split("/")[-1]
        panels.extend(_dashboard(base + "_roofline.dat", title, ais, perfs,
                                 peak_tflops, peak_gbs, stats_text(summary)))
    rows_n = len(bases)
    script, png = out + "_roofline.gnuplot", out + "_roofline.png"
    header = (f"set terminal pngcairo size 2400,{520 * rows_n} "
              f"enhanced font 'Sans,10'\nset output '{png}'\n"
              f"set datafile separator \"\\t\"\n"
              f"set datafile missing 'NaN'\n"
              f"set multiplot layout {rows_n},4 rowsfirst\n")
    _render(script, header + "\n".join(panels) + "\nunset multiplot\n", png)
    return 0


# ---------------------------------------------------------------------------
# CLI and profiling loop
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--choice",
                   choices=[s["choice"] for s in MODEL_SPECS], default="1",
                   help="1=llama 2=gemma 3=gpt-oss(MoE) 4=qwen")
    p.add_argument("--model", default=None,
                   help="any dense HF model id; overrides --choice")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dtype-bytes", type=int, default=2,
                   help="weight bytes/param; must match --dtype")
    p.add_argument("--expert-dtype-bytes", type=float, default=None,
                   help="bytes/param for routed MoE expert weights when they "
                        "are quantized below --dtype (e.g. 0.53 for gpt-oss "
                        "MXFP4 = 4-bit + block scale). Attention, KV cache and "
                        "lm_head stay at --dtype-bytes. Default = --dtype-bytes "
                        "(no split); dense models ignore it.")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gen-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gpu-mem-util", type=float, default=0.4,
                   help="only used when profiling runs (--kv-cache-gb 0)")
    p.add_argument("--kv-cache-gb", type=float, default=4.0,
                   help="fixed KV-cache budget in GB; skips vLLM's flaky "
                        "memory profiling on unified-memory boards "
                        "(0 = profile with --gpu-mem-util instead)")
    p.add_argument("--no-enforce-eager", action="store_true",
                   help="allow CUDA graph capture (production-like timing)")
    p.add_argument("--gpu-sample-interval-ms", type=float, default=10.0)
    p.add_argument("--nvml-device", default="auto",
                   help="which GPU NVML samples for power/util: 'auto' "
                        "(default; the compute GPU from CUDA_VISIBLE_DEVICES - "
                        "correct even when a SLURM gpu:1 alloc still lets NVML "
                        "see every GPU on the node), 'all' (sum every visible "
                        "GPU - the old behaviour), or an explicit NVML index / "
                        "'GPU-<uuid>'.")
    p.add_argument("--idle-baseline-s", type=float, default=10.0,
                   help="seconds of NVML idle-power sampling before engine "
                        "init; subtracted to report dynamic power/energy "
                        "(0 = skip). Pin clocks before launching so the "
                        "baseline matches the run's clock state.")
    p.add_argument("--peak-tflops", type=float, default=None)
    p.add_argument("--peak-memory-gbs", type=float, default=None)
    p.add_argument("--combine", nargs="+", metavar="OUT_BASE", default=None,
                   help="skip profiling; merge finished runs "
                        "(<base>_roofline.dat + <base>_summary.json) into "
                        "one multi-panel paper-style figure")
    p.add_argument("--reprocess", action="store_true",
                   help="skip profiling; re-derive <--out>_summary.json + "
                        "roofline .dat/PNG from the existing <--out>.csv, "
                        "applying the drain-step marking - use on runs "
                        "recorded before that fix. The CSV is not rewritten")
    p.add_argument("--out", default="step_metrics")
    args = p.parse_args()
    if (args.peak_tflops is None) != (args.peak_memory_gbs is None):
        p.error("--peak-tflops and --peak-memory-gbs must be given together")
    return args


def main() -> int:
    args = parse_args()
    if args.combine:
        return combine_plots(args.combine, args.out, args.peak_tflops,
                             args.peak_memory_gbs)
    if args.reprocess:
        return reprocess(args.out, args.peak_tflops, args.peak_memory_gbs)
    if args.model:
        model_id, is_moe = args.model, False
        print("[warn] --model override assumes a dense model; use a "
              "MODEL_SPECS choice for MoE")
    else:
        spec = next(s for s in MODEL_SPECS if s["choice"] == args.choice)
        # One file, both machines: local path if it exists (A100: /hpcgpfs01
        # models), else the HF Hub id (Jetson: HF cache).
        model_id = (spec["model_id"] if os.path.isdir(spec["model_id"])
                    else spec.get("hub", spec["model_id"]))
        is_moe = spec["moe"]
        print(f"[info] choice {spec['choice']} = {spec['name']}: {model_id} "
              f"(moe={is_moe})")
        if spec.get("note"):
            print(f"[note] {spec['note']}")

    arch = load_arch(model_id, args.dtype_bytes)
    arch["expert_dtype_bytes"] = (args.expert_dtype_bytes
                                  if args.expert_dtype_bytes is not None
                                  else args.dtype_bytes)
    config_moe = bool(arch["experts"] and arch["experts_per_tok"])
    if config_moe != is_moe:
        print(f"[warn] selected moe={is_moe} but config detection says "
              f"moe={config_moe}; using the selected setting")
    total, streamed = param_counts(arch, is_moe)
    print(f"[info] {model_id}: total {total / 1e9:.2f}B params, "
          f"active per step {streamed / 1e9:.2f}B "
          f"({streamed * args.dtype_bytes / 1e9:.2f} GB)")
    if is_moe:
        print("[warn] MoE: FLOPs/bytes assume top-k experts per token; "
              "with batch > 1 real weight traffic is the expert union, so "
              "bytes are a lower bound")
    if is_moe and arch["expert_dtype_bytes"] != args.dtype_bytes:
        ew = arch["expert_params"]
        split_gb = ((streamed - ew) * args.dtype_bytes
                    + ew * arch["expert_dtype_bytes"]) / 1e9
        print(f"[info] expert byte split: {ew / 1e9:.2f}B expert params @ "
              f"{arch['expert_dtype_bytes']:g} B + {(streamed - ew) / 1e9:.2f}B "
              f"other @ {args.dtype_bytes} B -> {split_gb:.2f} GB weight "
              f"traffic (vs {streamed * args.dtype_bytes / 1e9:.2f} GB flat)")

    # Idle-power baseline: before the engine exists, so the GPU is quiescent.
    idle_power_w = None
    if args.idle_baseline_s > 0:
        idle_power_w = measure_idle_power(
            args.idle_baseline_s, args.gpu_sample_interval_ms / 1000.0,
            device_selector=args.nvml_device)
        if idle_power_w is None:
            print("[warn] idle baseline unavailable; dynamic power/energy "
                  "will be null")

    import torch
    from torch.cuda import nvtx
    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # tokenize=False + encode is stable across transformers versions (newer
    # releases changed apply_chat_template(tokenize=True) to return a dict).
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": USER_PROMPT}],
        add_generation_prompt=True, tokenize=False)
    # the rendered template already contains BOS; don't add it twice
    ids = list(tokenizer(text, add_special_tokens=False).input_ids)
    if not ids or not all(isinstance(t, int) for t in ids):
        raise SystemExit(f"[error] tokenization returned "
                         f"{type(ids).__name__}, expected list[int]")
    prompt = {"prompt_token_ids": ids}  # TokensPrompt: no raw-string path
    n_prompt = len(ids)
    print(f"[info] prompt is {n_prompt} tokens, batch {args.batch_size}")

    extra = {}
    if args.kv_cache_gb > 0:
        if "kv_cache_memory_bytes" in getattr(EngineArgs,
                                              "__dataclass_fields__", {}):
            extra["kv_cache_memory_bytes"] = int(args.kv_cache_gb * 1e9)
            print(f"[info] fixed KV budget {args.kv_cache_gb:g} GB; skipping "
                  "vLLM memory profiling (unified-memory safe)")
            kv_token_bytes = (arch["layers"] * arch["kv_dim"] * 2
                              * arch["dtype_bytes"])
            if extra["kv_cache_memory_bytes"] < kv_token_bytes * args.max_model_len:
                print("[warn] --kv-cache-gb holds less than one "
                      "max_model_len sequence; raise it or lower "
                      "--max-model-len")
        else:
            print("[warn] this vLLM lacks kv_cache_memory_bytes; falling "
                  "back to memory profiling, which is flaky on unified "
                  "memory - rerun if the init assertion fires")
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=model_id,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=not args.no_enforce_eager,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=False,  # keep analytics == actual compute
        # large enough that the whole batched prefill lands in one step
        max_num_batched_tokens=max(4096, args.batch_size * n_prompt),
        trust_remote_code=True,
        **extra,
    ))

    # Untimed warmup: triggers JIT compilation and lazy allocations. The
    # NVTX range keeps these kernels out of --nvtx-include captures.
    engine.add_request("warmup", prompt,
                       SamplingParams(max_tokens=8, temperature=0.0,
                                      ignore_eos=True))
    nvtx.range_push("warmup")
    while engine.has_unfinished_requests():
        engine.step()
    nvtx.range_pop()
    sync()
    print("[info] warmup complete; starting measured run")

    sampling = SamplingParams(max_tokens=args.gen_len, temperature=0.0,
                              ignore_eos=True)
    for i in range(args.batch_size):
        engine.add_request(str(i), prompt, sampling)

    sampler = GpuSampler(args.gpu_sample_interval_ms / 1000.0, args.nvml_device)
    sampler.start()
    records, kv_len, step = [], n_prompt, 0
    seen_tokens, pending_ns, pending_start = {}, 0, None
    try:
        while engine.has_unfinished_requests():
            phase = "prefill" if step == 0 else "decode"
            sync()
            t0 = time.perf_counter_ns()
            nvtx.range_push(phase)  # gates ncu --nvtx-include "<phase>/"
            outputs = engine.step()
            nvtx.range_pop()
            sync()
            t1 = time.perf_counter_ns()
            # A step only counts if it produced tokens. vLLM v1 issues a
            # terminal drain/bookkeeping step with no model forward, and
            # chunked prefill can run token-less chunk steps.
            new_tokens = 0
            for out in outputs or []:
                total = sum(len(c.token_ids) for c in (out.outputs or []))
                prev = seen_tokens.get(out.request_id, 0)
                if total > prev:
                    new_tokens += total - prev
                    seen_tokens[out.request_id] = total
            if new_tokens == 0:
                if step == 0:  # prefill chunk: its GPU time is prefill time
                    pending_ns += t1 - t0
                    if pending_start is None:
                        pending_start = t0  # power window must cover chunks
                    print(f"[info] token-less prefill chunk "
                          f"({(t1 - t0) / 1e6:.1f} ms) folded into the "
                          "prefill record")
                else:
                    print(f"[info] excluded zero-token scheduler step "
                          f"({(t1 - t0) / 1e6:.2f} ms; output drain)")
                continue
            if new_tokens != args.batch_size:
                # Fixed batch, no speculative decoding: every producing
                # step must emit exactly one token per sequence, or the
                # analytical FLOP/byte accounting is wrong for this run.
                raise SystemExit(
                    f"[error] step {step} produced {new_tokens} tokens "
                    f"for batch {args.batch_size}; unexpected scheduling "
                    "(speculative decoding? request dropped?) - aborting "
                    "instead of mis-billing analytical work")
            latency = max(((t1 - t0) + pending_ns) / 1e9, 1e-9)
            if pending_start is not None:
                t0 = pending_start  # start_ns spans all folded chunks
            pending_ns, pending_start = 0, None
            if phase == "prefill":
                tokens = n_prompt * args.batch_size
                ctx = n_prompt
                flops, bytes_moved = prefill_cost(
                    arch, streamed, n_prompt, args.batch_size)
            else:
                tokens = args.batch_size
                ctx = kv_len
                flops, bytes_moved = decode_cost(
                    arch, streamed, kv_len, args.batch_size)
                kv_len += 1
            records.append({
                "step": step, "phase": phase, "tokens": tokens,
                "kv_len": ctx, "start_ns": t0, "end_ns": t1,
                "latency_s": latency,
                "flops": int(flops), "bytes": int(bytes_moved),
                "tokens_per_second": tokens / latency,
                "ai_flops_per_byte": flops / bytes_moved,
                "achieved_tflops": flops / latency / 1e12,
                "achieved_gbs": bytes_moved / latency / 1e9,
            })
            step += 1
    finally:
        sampler.stop()

    # Prefill emits the first token, so gen_len token-producing steps are
    # expected (drain/chunk steps are already excluded above).
    if len(records) != args.gen_len:
        print(f"[warn] {len(records)} token-producing steps but expected "
              f"{args.gen_len}; the scheduler merged/split steps and phase "
              "accounting for this run may be off")
    mark_drain_like(records)

    for record in records:
        record.update(sampler.window(record["start_ns"], record["end_ns"],
                                     record["latency_s"]))
        if (idle_power_w is not None
                and record.get("gpu_power_w_avg") is not None):
            dyn = max(record["gpu_power_w_avg"] - idle_power_w, 0.0)
            record["gpu_power_w_dyn"] = dyn
            record["gpu_energy_dyn_j"] = dyn * record["latency_s"]
        else:
            record["gpu_power_w_dyn"] = None
            record["gpu_energy_dyn_j"] = None
        print(json.dumps(record))

    clean = [r for r in records if not r["drain_like"]]
    summary = summarize(clean)
    summary["config"] = {
        "model": model_id, "moe": is_moe, "dtype": args.dtype,
        "dtype_bytes": args.dtype_bytes,
        "expert_dtype_bytes": arch["expert_dtype_bytes"],
        "expert_params": arch["expert_params"],
        "batch_size": args.batch_size, "gen_len": args.gen_len,
        "prompt_tokens": n_prompt, "enforce_eager": not args.no_enforce_eager,
        "total_params": total, "active_params": streamed,
        "idle_power_w": idle_power_w,
        "idle_baseline_s": args.idle_baseline_s,
    }
    decode = summary["decode"]
    if decode["steps"] and decode["tokens_per_second"]:
        tot = decode["phase_totals"]
        line = (f"[info] decode phase totals: {tot['tokens_per_second']:.1f} "
                f"tok/s, {tot['gbs']:.0f} GB/s, {tot['tflops']:.2f} TFLOP/s "
                f"over {decode['steps']} steps "
                f"(per-step means: {decode['tokens_per_second']['avg']:.1f} "
                f"tok/s, {decode['achieved_tflops']['avg']:.2f} TFLOP/s)")
        if (decode["gpu_power_w_avg"] and decode["gpu_power_w_peak"]
                and decode["gpu_util_pct_avg"]):
            line += (f"; power {decode['gpu_power_w_avg']['avg']:.1f} W avg / "
                     f"{decode['gpu_power_w_peak']['max']:.1f} W peak, "
                     f"util {decode['gpu_util_pct_avg']['avg']:.0f}% avg / "
                     f"{decode['gpu_util_pct_peak']['max']:.0f}% peak")
        if decode.get("gpu_power_w_dyn"):
            line += (f"; dynamic {decode['gpu_power_w_dyn']['avg']:.1f} W "
                     f"(idle {idle_power_w:.1f} W subtracted)")
        print(line)
    write_outputs(args.out, records, summary)  # CSV keeps ALL rows
    write_roofline(args.out, clean, args.peak_tflops, args.peak_memory_gbs,
                   f"{model_id.split('/')[-1]} (batch {args.batch_size})",
                   stats_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())