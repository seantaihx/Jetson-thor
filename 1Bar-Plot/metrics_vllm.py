#!/usr/bin/env python3
"""
vLLM-based benchmark — mirrors get_transformers_metrics.py metrics exactly.

Install:
  pip install vllm pynvml psutil

Hardware sampling uses pynvml exclusively (power, GPU util, memory).
On Jetson boards, pynvml memory readings are often unsupported and will
be recorded as None — this is expected and identical in both scripts.
"""

import argparse                          # parses command-line arguments like --runs 5
import gc                                # Python garbage collector, used to force memory cleanup
import json                              # read/write JSON files
import statistics as stats               # mean/stdev — same module used in transformers script
import threading                         # run hardware sampler in parallel with inference
import time                              # perf_counter for timing, sleep for sampler interval
from datetime import datetime, timezone  # for UTC timestamps on each run
from pathlib import Path                 # modern file path handling

import psutil                            # cross-platform CPU/RAM usage
import pynvml                            # NVIDIA Management Library — GPU power, util, memory
import torch                             # needed only for torch.cuda.empty_cache() to free GPU VRAM
from vllm import LLM, SamplingParams     # LLM: loads and runs the model
                                         # SamplingParams: controls generation (temp, tokens, etc.)


SYSTEM_PROMPT = "You are a high performance computing scientific coding assistant"
USER_PROMPT = (
    "I have a GPU code written in CUDA. I need to convert this to a portable programming model. "
    "How can I convert a CUDA kernel to RAJA. Give two working examples."
)

MODEL_SPECS = [
    {"choice": "1", "name": "llama", "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    {"choice": "2", "name": "gemma", "model_id": "google/gemma-4-E4B-it"},
    {"choice": "3", "name": "gpt",   "model_id": "openai/gpt-oss-20b"},
]


# -------------
# Utilities
# -------------

def now_utc():
    return datetime.now(timezone.utc).isoformat()  # Returns the current time as an ISO 8601 string e.g. "2026-06-19T14:32:00+00:00" for timestamping in JSON

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None  # Safe average, returns None if the list is empty or all-None

def std(values):
    """Sample standard deviation (matches statistics.stdev — same formula used in transformers script).
    Returns None if fewer than 2 valid values."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return stats.stdev(values)

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    # Serialises the entire results dict to a pretty-printed JSON file.
    # sort_keys=True keeps the output consistent/diff-friendly across saves.
    # Called after every single run so a crash loses at most one run of data.


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def dtype_size_bytes(dtype_value):
    """Return the byte size for a given dtype string or torch.dtype."""
    if dtype_value is None:
        return None
    name = str(dtype_value).lower().replace("torch.", "")
    if "float32" in name or "fp32" in name:
        return 4
    if "float16" in name or "half" in name or "fp16" in name or "bfloat16" in name or "bf16" in name:
        return 2
    if "float8" in name or "fp8" in name or "uint8" in name or "int8" in name:
        return 1
    return None


def _get_attr(obj, names, *args):
    """Try several attribute/method names for vLLM version compatibility."""
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if callable(value):
            try:
                return value(*args)
            except TypeError:
                try:
                    return value()
                except TypeError:
                    continue
        return value
    return None


def get_kv_params(llm):
    """
    Extract the four architectural values needed for the corrected KV cache formula:
      num_layers    — number of transformer layers
      num_kv_heads  — number of KV attention heads (per rank)
      head_dim      — dimension of each head
      dtype_bytes   — bytes per element for the KV cache dtype

    These are model architecture properties, independent of hardware/settings.
    Returns (num_layers, num_kv_heads, head_dim, dtype_bytes) — any may be None.
    """
    try:
        vllm_config     = llm.llm_engine.vllm_config
        model_config    = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        cache_config    = vllm_config.cache_config

        # --- num_layers ---
        num_layers = _get_attr(model_config, ["get_num_layers", "num_layers"], parallel_config)
        if num_layers is None and hasattr(model_config, "hf_config"):
            num_layers = getattr(model_config.hf_config, "num_hidden_layers", None)

        # --- num_kv_heads ---
        num_kv_heads = _get_attr(
            model_config,
            ["get_num_kv_heads", "get_total_num_kv_heads", "num_kv_heads"],
            parallel_config,
        )
        if num_kv_heads is None and hasattr(model_config, "hf_config"):
            num_kv_heads = getattr(model_config.hf_config, "num_key_value_heads", None)

        # --- head_dim ---
        head_dim = _get_attr(model_config, ["get_head_size", "head_size"])
        if head_dim is None and hasattr(model_config, "hf_config"):
            hidden = getattr(model_config.hf_config, "hidden_size", None)
            heads  = getattr(model_config.hf_config, "num_attention_heads", None)
            if hidden and heads:
                head_dim = hidden // heads

        # --- dtype_bytes ---
        cache_dtype = getattr(cache_config, "cache_dtype", None)
        if cache_dtype in (None, "auto"):
            cache_dtype = getattr(model_config, "dtype", None)
        dtype_bytes = dtype_size_bytes(cache_dtype)

        return num_layers, num_kv_heads, head_dim, dtype_bytes

    except Exception as e:
        print(f"WARNING: Could not extract KV params ({e}). KV cache size will be None.")
        return None, None, None, None


def calc_kv_cache_used_mb(prompt_tokens, generated_tokens, num_layers, num_kv_heads, head_dim, dtype_bytes):
    """
    KV Cache Size (bytes) = batch_size × seq_len × num_kv_heads × head_dim × 2 × element_size
    batch_size = 1 (one request at a time in this benchmark)
    seq_len    = token_count (tokens generated in this run)
    × 2        = K and V matrices

    Returns size in MB, or None if any parameter is missing.
    """
    seq_len = (prompt_tokens or 0) + (generated_tokens or 0)
    if any(v is None for v in [seq_len, num_layers, num_kv_heads, head_dim, dtype_bytes]) or seq_len == 0:
        return None
    total_bytes = 1 * seq_len * num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# pynvml init / support probe
# Identical logic to get_transformers_metrics.py — both scripts use pynvml
# exclusively for hardware sampling (power, GPU util, memory).
# ---------------------------------------------------------------------------

def init_nvml():
    pynvml.nvmlInit()                                   # starts the NVML session, must be called before anything else
    handles = []                                        # will hold one handle object per GPU
    info = []                                           # will hold metadata dicts (index, name) per GPU
    count = pynvml.nvmlDeviceGetCount()                 # asks the driver how many NVIDIA GPUs are visible
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)   # get a handle to GPU #i
        handles.append(handle)                          # save it for later sampling
        name = pynvml.nvmlDeviceGetName(handle)         # get the GPU model string e.g. "Orin"
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")      # older pynvml returns bytes, newer returns str
        info.append({"index": i, "name": name})         # save index + name into metadata list
    return handles, info                                # handles used for sampling; info saved into JSON metadata


def probe_nvml_support(handles):
    """
    Test which pynvml calls actually work on this hardware.
    Jetson unified-memory boards often return NVMLError_NotSupported
    for memory (and sometimes power/util). We check once at startup so we
    can warn the user and avoid silently recording all-None metrics.
    """
    support = {"power": True, "util": True, "memory": True}  # Assume everything works; flip to False if a test call fails

    if not handles:                                           # No GPUs found at all — disable everything and warn
        support = {"power": False, "util": False, "memory": False}
        print("WARNING: No NVML devices found. All hardware metrics will be None.")
        return support

    h = handles[0]                                            # test against the first GPU — if it fails there, it'll fail on all

    try:
        pynvml.nvmlDeviceGetPowerUsage(h)                     # test call — result is thrown away
    except pynvml.NVMLError as e:
        support["power"] = False                              # mark power as unsupported
        print(f"WARNING: pynvml power readings not supported on this hardware ({e}).")
        print("         Power metrics will be recorded as None.")

    try:
        pynvml.nvmlDeviceGetUtilizationRates(h)
    except pynvml.NVMLError as e:
        support["util"] = False
        print(f"WARNING: pynvml utilization readings not supported on this hardware ({e}).")
        print("         Utilization metrics will be recorded as None.")

    try:
        pynvml.nvmlDeviceGetMemoryInfo(h)
    except pynvml.NVMLError as e:
        support["memory"] = False
        print(f"WARNING: pynvml memory readings not supported on this hardware ({e}).")
        print("         On Jetson, unified memory means GPU/CPU share RAM.")
        print("         Consider reading /proc/meminfo for total memory pressure.")
    # Jetson uses unified memory — CPU and GPU share the same physical RAM pool
    # so nvml's GPU memory carve-out number doesn't tell the full story.

    return support


def sample_hardware(handles, nvml_support):
    """Takes a single hardware snapshot across all GPUs right now. Called repeatedly
    by the sampler thread during inference."""

    gpu_power = []                                                                  # one entry per GPU
    gpu_util = []
    gpu_mem_used = []
    gpu_mem_total = []

    for handle in handles:                                                          # loop over each GPU
        if nvml_support["power"]:                                                   # only attempt if we know it works
            try:
                gpu_power.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)   # nvmlDeviceGetPowerUsage returns milliwatts — divide by 1000 to get Watts
            except Exception:
                gpu_power.append(None)                                              # unexpected runtime failure — record None
        else:
            gpu_power.append(None)                                                  # known unsupported — skip the call entirely

        if nvml_support["util"]:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util.append(float(util.gpu))                                    # util.gpu is 0-100 integer — GPU compute %
            except Exception:
                gpu_util.append(None)
        else:
            gpu_util.append(None)

        if nvml_support["memory"]:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used.append(mem.used / 1024 / 1024)                         # bytes → MB
                gpu_mem_total.append(mem.total / 1024 / 1024)                       # bytes → MB
            except Exception:
                gpu_mem_used.append(None)
                gpu_mem_total.append(None)
        else:
            gpu_mem_used.append(None)
            gpu_mem_total.append(None)

    return {
        "gpu_power_w_total": (
            sum(v for v in gpu_power if v is not None)                                  # Sum watts across all GPUs (total board power draw)
            if any(v is not None for v in gpu_power) else None                          # Returns None only if every GPU returned None
        ),
        "gpu_util_pct_avg": mean(gpu_util),                                             # Average utilisation % across all GPUs
        "gpu_mem_used_mb_total": (
            sum(v for v in gpu_mem_used if v is not None)                               # Total MB used across all GPUs
            if any(v is not None for v in gpu_mem_used) else None
        ),
        "gpu_mem_total_mb_total": (
            sum(v for v in gpu_mem_total if v is not None)                              # Total MB capacity across all GPUs
            if any(v is not None for v in gpu_mem_total) else None
        ),
        "cpu_pct": psutil.cpu_percent(interval=None),                                   # CPU usage % — interval=None returns since last call (non-blocking)
        "mem_pct": psutil.virtual_memory().percent,                                     # System RAM usage %
    }


def sampler_thread(handles, nvml_support, stop_event, samples, interval_s):
    """Runs on a background thread. Keeps calling sample_hardware() every interval_s
    seconds until the main thread sets stop_event."""
    psutil.cpu_percent(interval=None)                                                   # discard first (always 0.0)
    while not stop_event.is_set():                                                      # keep looping until main thread says stop
        samples.append({"ts": now_utc(), **sample_hardware(handles, nvml_support)})
        # Append a timestamped snapshot to the shared list.
        # **sample_hardware(...) unpacks the dict into the outer dict.
        time.sleep(interval_s)                                                          # wait before next sample (default 0.25s = 4 Hz)


# -----------------------
# Summarisation helpers
# -----------------------

def summarize_samples(samples):
    return {
        "sample_count": len(samples),
        "gpu_power_w_avg":      mean(s["gpu_power_w_total"]   for s in samples),
        "gpu_power_w_peak":     max((s["gpu_power_w_total"]   for s in samples if s["gpu_power_w_total"]   is not None), default=None),
        "gpu_util_pct_avg":     mean(s["gpu_util_pct_avg"]    for s in samples),
        "gpu_util_pct_peak":    max((s["gpu_util_pct_avg"]    for s in samples if s["gpu_util_pct_avg"]    is not None), default=None),
        "cpu_pct_avg":  mean(s["cpu_pct"] for s in samples),
        "cpu_pct_peak": max((s["cpu_pct"] for s in samples if s["cpu_pct"] is not None), default=None),
        "mem_pct_avg":  mean(s["mem_pct"] for s in samples),
        "mem_pct_peak": max((s["mem_pct"] for s in samples if s["mem_pct"] is not None), default=None),
        "raw_samples": samples,
    }


def model_summary(model_spec, runs, kv_params):
    sw = [r["software"] for r in runs]
    hw = [r["hardware"] for r in runs]
    num_layers, num_kv_heads, head_dim, dtype_bytes = kv_params
    return {
        "choice":   model_spec["choice"],
        "name":     model_spec["name"],
        "model_id": model_spec["model_id"],
        "run_count": len(runs),
        "kv_architecture": {
            "num_layers":   num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim":     head_dim,
            "dtype_bytes":  dtype_bytes,
        },
        "software": {
            "mean_generation_time_s":      mean(r["generation_time_s"]     for r in sw),
            "std_generation_time_s":        std(r["generation_time_s"]      for r in sw),
            "mean_generated_token_count":  mean(r["generated_token_count"]  for r in sw),
            "std_generated_token_count":    std(r["generated_token_count"]   for r in sw),
            "mean_tokens_per_second":      mean(r["tokens_per_second"]      for r in sw),
            "std_tokens_per_second":        std(r["tokens_per_second"]       for r in sw),
            "mean_kv_cache_used_mb":       mean(r["kv_cache_used_mb"]       for r in sw),
            "std_kv_cache_used_mb":         std(r["kv_cache_used_mb"]        for r in sw),
        },
        "hardware": {
            "mean_gpu_power_w_avg":    mean(r["gpu_power_w_avg"]   for r in hw),
            "std_gpu_power_w_avg":      std(r["gpu_power_w_avg"]    for r in hw),
            "mean_gpu_util_pct_avg":   mean(r["gpu_util_pct_avg"]  for r in hw),
            "std_gpu_util_pct_avg":     std(r["gpu_util_pct_avg"]   for r in hw),
            "mean_cpu_pct_avg":        mean(r["cpu_pct_avg"]        for r in hw),
            "std_cpu_pct_avg":          std(r["cpu_pct_avg"]         for r in hw),
            "mean_mem_pct_avg":        mean(r["mem_pct_avg"]        for r in hw),
            "std_mem_pct_avg":          std(r["mem_pct_avg"]         for r in hw),
        },
        "raw_runs": runs,
    }


# ---------------------------------------------------------------------------
# Output helpers — TSV summary, same columns as transformers script
# ---------------------------------------------------------------------------

def write_summary_tsv(path, summaries):
    lines = [
        "model\trun_count"
        "\tmean_generation_time_s\tstd_generation_time_s"
        "\tmean_generated_token_count\tstd_generated_token_count"
        "\tmean_tokens_per_second\tstd_tokens_per_second"
        "\tmean_kv_cache_used_mb\tstd_kv_cache_used_mb"
        "\tnum_layers\tnum_kv_heads\thead_dim\tdtype_bytes"
        "\tmean_gpu_power_w_avg\tstd_gpu_power_w_avg"
        "\tmean_gpu_util_pct_avg\tstd_gpu_util_pct_avg"
        "\tmean_cpu_pct_avg\tstd_cpu_pct_avg"
        "\tmean_mem_pct_avg\tstd_mem_pct_avg"
    ]
    for model_name, s in summaries.items():
        sw  = s["software"]
        hw  = s["hardware"]
        kva = s.get("kv_architecture", {})
        lines.append("\t".join([
            model_name,
            str(s["run_count"]),
            str(sw["mean_generation_time_s"]),
            str(sw["std_generation_time_s"]),
            str(sw["mean_generated_token_count"]),
            str(sw["std_generated_token_count"]),
            str(sw["mean_tokens_per_second"]),
            str(sw["std_tokens_per_second"]),
            str(sw["mean_kv_cache_used_mb"]),
            str(sw["std_kv_cache_used_mb"]),
            str(kva.get("num_layers")),
            str(kva.get("num_kv_heads")),
            str(kva.get("head_dim")),
            str(kva.get("dtype_bytes")),
            str(hw["mean_gpu_power_w_avg"]),
            str(hw["std_gpu_power_w_avg"]),
            str(hw["mean_gpu_util_pct_avg"]),
            str(hw["std_gpu_util_pct_avg"]),
            str(hw["mean_cpu_pct_avg"]),
            str(hw["std_cpu_pct_avg"]),
            str(hw["mean_mem_pct_avg"]),
            str(hw["std_mem_pct_avg"]),
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(v, d=3):
    """Format a numeric value with d decimals, or 'N/A' if None.
    Prevents crashes in the final summary print when std() returns None
    (e.g. only 1 successful run)."""
    return f"{v:.{d}f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Defines all the CLI flags so you can tweak the benchmark without editing the file."""
    p = argparse.ArgumentParser()
    p.add_argument("--runs",                  type=int,   default=10)
    p.add_argument("--warmup-runs",           type=int,   default=2)
    p.add_argument("--max-tokens",            type=int,   default=4096)
    p.add_argument("--temperature",           type=float, default=0.0)
    p.add_argument("--top-p",                 type=float, default=1.0)
    p.add_argument("--seed",                  type=int,   default=1234)
    p.add_argument("--sample-interval-s",     type=float, default=0.25)
    p.add_argument("--tensor-parallel-size",  type=int,   default=1)
    p.add_argument("--gpu-memory-utilization",type=float, default=0.3)
    p.add_argument("--max-model-len",         type=int,   default=None)
    p.add_argument("--trust-remote-code",     action="store_true")
    p.add_argument("--enforce-eager",         action="store_true")
    p.add_argument("--out-dir",               type=Path,  default=Path("benchmark_out_vllm"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """For each model: load → warmup → timed runs → save → unload."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    handles, gpu_info = init_nvml()

    # Check once at startup which pynvml calls this board actually supports.
    # Jetson boards use unified memory and often block the memory query.
    nvml_support = probe_nvml_support(handles)

    results = {
        "meta": {
            "created_utc":       now_utc(),
            "system_prompt":     SYSTEM_PROMPT,
            "user_prompt":       USER_PROMPT,
            "model_specs":       MODEL_SPECS,
            "runs_per_model":    args.runs,
            "warmup_runs":       args.warmup_runs,
            "max_tokens":        args.max_tokens,
            "sample_interval_s": args.sample_interval_s,
            "hardware":          gpu_info,
            "nvml_support":      nvml_support,
            "inference_engine":  "vllm",
        },
        "runs":    [],
        "models":  {},
        "summary": {},
    }

    json_path = args.out_dir / "vllm_benchmark_results.json"

    try:
        for spec in MODEL_SPECS:
            print(f"\n{'='*60}")
            print(f"Loading model: {spec['model_id']}")
            print(f"{'='*60}")

            llm_kwargs = {
                "model":                    spec["model_id"],
                "tensor_parallel_size":     args.tensor_parallel_size,
                "gpu_memory_utilization":   args.gpu_memory_utilization,
                "trust_remote_code":        args.trust_remote_code,
                "enforce_eager":            args.enforce_eager,
            }

            # Only pass max_model_len if the user explicitly set it;
            # passing None causes vLLM to reject the kwarg.
            if args.max_model_len is not None:
                llm_kwargs["max_model_len"] = args.max_model_len

            llm = LLM(**llm_kwargs)

            # ------------------------------------------------------------------
            # Extract KV architecture params once after the model is loaded.
            # These are model properties (not hardware/settings dependent).
            # Used in the per-run KV cache size formula:
            #   KV cache (bytes) = batch_size × seq_len × num_kv_heads × head_dim × 2 × dtype_bytes
            # ------------------------------------------------------------------
            num_layers, num_kv_heads, head_dim, dtype_bytes = get_kv_params(llm)
            print(f"KV params — num_kv_heads: {num_kv_heads}, head_dim: {head_dim}, dtype_bytes: {dtype_bytes}")

            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT},
            ]

            params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )

            # Warmup — results discarded
            if args.warmup_runs > 0:
                print(f"Running {args.warmup_runs} warmup run(s)...")
                for _ in range(args.warmup_runs):
                    llm.chat(prompt, params)

            model_runs = []
            results["models"][spec["name"]] = {
                "choice":    spec["choice"],
                "model_id":  spec["model_id"],
                "kv_architecture": {
                    "num_layers":   num_layers,
                    "num_kv_heads": num_kv_heads,
                    "head_dim":     head_dim,
                    "dtype_bytes":  dtype_bytes,
                },
                "raw_runs":  model_runs,
                "summary":   None,
            }

            for run_index in range(1, args.runs + 1):
                samples    = []
                stop_event = threading.Event()
                t = threading.Thread(
                    target=sampler_thread,
                    args=(handles, nvml_support, stop_event, samples, args.sample_interval_s),
                    daemon=True,
                )

                start = time.perf_counter()
                t.start()
                outputs = llm.chat(prompt, params)
                stop_event.set()
                t.join()
                end = time.perf_counter()

                result        = outputs[0]
                output        = result.outputs[0]
                token_count   = len(output.token_ids or [])
                prompt_tokens = len(result.prompt_token_ids or [])
                gen_time_s    = end - start
                tps           = token_count / gen_time_s if gen_time_s > 0 else None

                # --------------------------------------------------------------
                # KV Cache used this run (MB)
                # Formula: batch_size × seq_len × num_kv_heads × head_dim × 2 × dtype_bytes
                #   batch_size = 1  (one request at a time)
                #   seq_len    = prompt_tokens + generated tokens
                #   × 2        = K matrix + V matrix
                # --------------------------------------------------------------
                kv_cache_used_mb = calc_kv_cache_used_mb(
                    prompt_tokens, token_count, num_layers, num_kv_heads, head_dim, dtype_bytes
                )

                hw_summary = summarize_samples(samples)

                run_record = {
                    "model_choice": spec["choice"],
                    "model_name":   spec["name"],
                    "model_id":     spec["model_id"],
                    "run_index":    run_index,
                    "timestamp_utc": now_utc(),
                    "prompt": {
                        "system": SYSTEM_PROMPT,
                        "user":   USER_PROMPT,
                    },
                    "sampling": {
                        "temperature": args.temperature,
                        "top_p":       args.top_p,
                        "max_tokens":  args.max_tokens,
                        "seed":        args.seed,
                    },
                    "software": {
                        "generation_time_s":     gen_time_s,
                        "generated_token_count": token_count,
                        "tokens_per_second":     tps,
                        "kv_cache_used_mb":      kv_cache_used_mb,
                        "prompt_token_count":    prompt_tokens,
                        "finish_reason":         getattr(output, "finish_reason", None),
                        "output_text":           output.text,
                    },
                    "hardware": hw_summary,
                }

                results["runs"].append(run_record)
                model_runs.append(run_record)

                print(json.dumps({
                    "model_name":         spec["name"],
                    "run_index":          run_index,
                    "generation_time_s":  gen_time_s,
                    "generated_tokens":   token_count,
                    "tokens_per_second":  tps,
                    "kv_cache_used_mb":   kv_cache_used_mb,
                    "finish_reason":      getattr(output, "finish_reason", None),
                    "gpu_power_w_avg":    hw_summary["gpu_power_w_avg"],
                    "gpu_util_pct_avg":   hw_summary["gpu_util_pct_avg"],
                    "cpu_pct_avg":        hw_summary["cpu_pct_avg"],
                    "mem_pct_avg":        hw_summary["mem_pct_avg"],
                }, indent=2))

                save_json(json_path, results)

            # Summarise this model
            summary = model_summary(spec, model_runs, (num_layers, num_kv_heads, head_dim, dtype_bytes))
            results["models"][spec["name"]]["summary"] = summary
            results["summary"][spec["name"]] = summary
            save_json(json_path, results)

            # ------------------------------------------------------------------
            # FREE GPU MEMORY before loading the next model.
            # Without this, on Jetson (limited VRAM) the next LLM() call will
            # OOM because the previous model is still resident.
            # ------------------------------------------------------------------
            print(f"\nUnloading {spec['name']} from GPU memory...")

            del llm
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(20)   # let Jetson unified memory fully reclaim pages
            print("GPU memory freed.\n")

        # -----------------------------------------------------------------------
        # All models done — write outputs
        # -----------------------------------------------------------------------
        save_json(json_path, results)

        tsv_path = args.out_dir / "vllm_summary.tsv"
        write_summary_tsv(tsv_path, results["summary"])

        print("\nFinal summary")
        print("=" * 60)
        for model_name, s in results["summary"].items():
            sw  = s["software"]
            hw  = s["hardware"]
            kva = s.get("kv_architecture", {})
            print(f"\nModel: {model_name}")
            print(f"  Model ID:               {s['model_id']}")
            print(f"  Runs:                   {s['run_count']}")
            print(f"  Mean generation time:   {fmt(sw['mean_generation_time_s'])} ± {fmt(sw['std_generation_time_s'])} s")
            print(f"  Mean generated tokens:  {fmt(sw['mean_generated_token_count'], 1)} ± {fmt(sw['std_generated_token_count'], 1)}")
            print(f"  Mean tokens/sec:        {fmt(sw['mean_tokens_per_second'], 2)} ± {fmt(sw['std_tokens_per_second'], 2)}")
            print(f"  Mean KV cache used:     {fmt(sw['mean_kv_cache_used_mb'])} ± {fmt(sw['std_kv_cache_used_mb'])} MB")
            print(f"  KV heads:               {kva.get('num_kv_heads')}")
            print(f"  Head dim:               {kva.get('head_dim')}")
            print(f"  Dtype bytes:            {kva.get('dtype_bytes')}")
            print(f"  Mean GPU power (W):     {fmt(hw['mean_gpu_power_w_avg'])} ± {fmt(hw['std_gpu_power_w_avg'])}")
            print(f"  Mean GPU util (%):      {fmt(hw['mean_gpu_util_pct_avg'])} ± {fmt(hw['std_gpu_util_pct_avg'])}")
            print(f"  Mean CPU util (%):      {fmt(hw['mean_cpu_pct_avg'])} ± {fmt(hw['std_cpu_pct_avg'])}")
            print(f"  Mean memory util (%):   {fmt(hw['mean_mem_pct_avg'])} ± {fmt(hw['std_mem_pct_avg'])}")

        print(f"\nJSON log:  {json_path}")
        print(f"TSV summary: {tsv_path}")

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
