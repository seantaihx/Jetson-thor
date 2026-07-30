#!/usr/bin/env python3
"""
Transformers-based benchmark — mirrors get_vllm_metrics.py metrics exactly.

Install:
  pip install transformers torch pynvml psutil

Hardware sampling uses pynvml exclusively (power, GPU util, memory) —
identical mechanism and field names to get_vllm_metrics.py, so the two
JSON outputs are directly comparable.
"""

import argparse                          # CLI flags
import gc                                # garbage collector for memory cleanup
import json                              # read/write JSON
import statistics as stats               # mean/stdev
import threading                         # sampler thread
import time                              # timing
from datetime import datetime, timezone  # UTC timestamps
from pathlib import Path                 # file paths

import psutil                            # CPU/RAM usage
import pynvml                            # NVIDIA Management Library — GPU power, util, memory
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor


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

TRANSFORMERS_LOAD_CONFIG = {
    "llama": {"use_processor": False, "torch_dtype": torch.bfloat16},
    "gemma": {"use_processor": True,  "torch_dtype": torch.bfloat16},
    "gpt":   {"use_processor": False, "torch_dtype": torch.bfloat16},
}


# -------------
# Utilities
# -------------

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None

def std(values):
    """Sample standard deviation. Returns None if fewer than 2 valid values."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return stats.stdev(values)

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# KV cache size — same formula as vLLM script
# KV Cache Size (MB) = batch_size × seq_len × num_kv_heads × head_dim × 2 × dtype_bytes / 1024²
# ---------------------------------------------------------------------------

def get_kv_params_from_model(model):
    """
    Extract num_kv_heads, head_dim, dtype_bytes directly from a loaded
    transformers model. Mirrors get_kv_params() in the vLLM script.
    Returns (num_layers, num_kv_heads, head_dim, dtype_bytes) — any may be None.
    """
    try:
        cfg = model.config
        # some models wrap config inside text_config (e.g. Gemma multimodal)
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        num_kv_heads         = getattr(cfg, "num_key_value_heads", None)
        num_attention_heads  = getattr(cfg, "num_attention_heads", None)
        hidden_size          = getattr(cfg, "hidden_size", None)
        head_dim             = getattr(cfg, "head_dim", None)

        # fall back: if no explicit kv heads, all heads are KV heads (MHA)
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads

        # fall back: compute head_dim from hidden_size / num_attention_heads
        if head_dim is None and hidden_size and num_attention_heads:
            head_dim = hidden_size // num_attention_heads

        # dtype bytes — model.dtype is the weight dtype (bfloat16 → 2 bytes)
        dtype_bytes = torch.tensor([], dtype=model.dtype).element_size()
        num_layers  = getattr(cfg, "num_hidden_layers", None)
        return num_layers, num_kv_heads, head_dim, dtype_bytes

    except Exception as e:
        print(f"WARNING: Could not extract KV params ({e}). KV cache size will be None.")
        return None, None, None, None


def calc_kv_cache_used_mb(prompt_tokens, generated_tokens, num_layers, num_kv_heads, head_dim, dtype_bytes):
    """
    KV Cache Size (bytes) = batch_size × seq_len × num_kv_heads × head_dim × 2 × dtype_bytes
    batch_size = 1  (one request at a time)
    seq_len    = prompt_tokens + generated_tokens
    × 2        = K matrix + V matrix
    Returns MB, or None if any param is missing.
    """
    seq_len = (prompt_tokens or 0) + (generated_tokens or 0)
    if any(v is None for v in [seq_len, num_layers, num_kv_heads, head_dim, dtype_bytes]) or seq_len == 0:
        return None
    total_bytes = 1 * seq_len * num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# pynvml init / support probe
# Identical logic to get_vllm_metrics.py — both scripts use pynvml
# exclusively for hardware sampling (power, GPU util, memory).
# ---------------------------------------------------------------------------

def init_nvml():
    pynvml.nvmlInit()
    handles, info = [], []
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        handles.append(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        info.append({"index": i, "name": name})
    return handles, info


def probe_nvml_support(handles):
    """
    Test which pynvml calls actually work on this hardware.
    Jetson unified-memory boards often return NVMLError_NotSupported
    for memory (and sometimes power/util). We check once at startup so we
    can warn the user and avoid silently recording all-None metrics.
    """
    support = {"power": True, "util": True, "memory": True}
    if not handles:
        print("WARNING: No NVML devices found. All hardware metrics will be None.")
        return {"power": False, "util": False, "memory": False}

    h = handles[0]
    try:
        pynvml.nvmlDeviceGetPowerUsage(h)
    except pynvml.NVMLError as e:
        support["power"] = False
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

    return support


def sample_hardware(handles, nvml_support):
    """Takes a single hardware snapshot across all GPUs right now. Called repeatedly
    by the sampler thread during inference. Identical logic to get_vllm_metrics.py."""
    gpu_power = []
    gpu_util = []
    gpu_mem_used = []
    gpu_mem_total = []

    for handle in handles:
        if nvml_support["power"]:
            try:
                gpu_power.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            except Exception:
                gpu_power.append(None)
        else:
            gpu_power.append(None)

        if nvml_support["util"]:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util.append(float(util.gpu))
            except Exception:
                gpu_util.append(None)
        else:
            gpu_util.append(None)

        if nvml_support["memory"]:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used.append(mem.used / 1024 / 1024)
                gpu_mem_total.append(mem.total / 1024 / 1024)
            except Exception:
                gpu_mem_used.append(None)
                gpu_mem_total.append(None)
        else:
            gpu_mem_used.append(None)
            gpu_mem_total.append(None)

    return {
        "gpu_power_w_total": (
            sum(v for v in gpu_power if v is not None)
            if any(v is not None for v in gpu_power) else None
        ),
        "gpu_util_pct_avg": mean(gpu_util),
        "gpu_mem_used_mb_total": (
            sum(v for v in gpu_mem_used if v is not None)
            if any(v is not None for v in gpu_mem_used) else None
        ),
        "gpu_mem_total_mb_total": (
            sum(v for v in gpu_mem_total if v is not None)
            if any(v is not None for v in gpu_mem_total) else None
        ),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "mem_pct": psutil.virtual_memory().percent,
    }


def sampler_thread(handles, nvml_support, stop_event, samples, interval_s):
    """Runs on a background thread. Keeps calling sample_hardware() every interval_s
    seconds until the main thread sets stop_event. Identical logic to get_vllm_metrics.py."""
    psutil.cpu_percent(interval=None)  # discard first (always 0.0)
    while not stop_event.is_set():
        samples.append({"ts": now_utc(), **sample_hardware(handles, nvml_support)})
        time.sleep(interval_s)


def summarize_samples(samples):
    """Same field names as get_vllm_metrics.py's summarize_samples()."""
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


# ---------------------------------------------------------------------------
# Single inference run
# ---------------------------------------------------------------------------

def run_once(model, tokenizer, messages, run_index, max_new_tokens,
             handles, nvml_support, sample_interval_s):
    """
    Run one inference, collect hardware metrics via a pynvml background-thread
    sampler (identical mechanism to vLLM's sampler_thread), return a run
    record matching the vLLM script's run_record structure.
    """
    # Apply chat template — same approach as vLLM llm.chat()
    chat_template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if hasattr(tokenizer, "parse_response"):  # Gemma processor only
        chat_template_kwargs["enable_thinking"] = False

    prompt = tokenizer.apply_chat_template(messages, **chat_template_kwargs)

    inputs       = tokenizer(text=prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[-1]

    # Start pynvml sampler thread — same pattern as vLLM script
    samples    = []
    stop_event = threading.Event()
    t = threading.Thread(
        target=sampler_thread,
        args=(handles, nvml_support, stop_event, samples, sample_interval_s),
        daemon=True,
    )

    # --- Inference ---
    start = time.perf_counter()
    t.start()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # greedy — matches vLLM temperature=0.0
            use_cache=True,        # KV cache enabled
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()   # ensure GPU ops complete before stopping timer

    end = time.perf_counter()
    stop_event.set()
    t.join()

    gen_time_s    = end - start
    output_tokens = output.shape[-1]
    token_count   = output_tokens - input_tokens   # generated tokens only
    tps           = token_count / gen_time_s if gen_time_s > 0 else None

    # Decode only the newly generated tokens (matches vLLM's output.text,
    # which contains only the completion, not the prompt)
    generated_ids = output[0][input_tokens:]
    output_text   = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # finish_reason: "stop" if generation ended on EOS before hitting the
    # token budget, "length" if it was cut off at max_new_tokens — same
    # semantics as vLLM's output.finish_reason
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and token_count < max_new_tokens and len(generated_ids) > 0:
        finish_reason = "stop"
    elif token_count >= max_new_tokens:
        finish_reason = "length"
    else:
        finish_reason = None

    hw_summary = summarize_samples(samples)

    return {
        "run_index":             run_index,
        "generation_time_s":     gen_time_s,
        "generated_token_count": token_count,
        "tokens_per_second":     tps,
        "prompt_token_count":    input_tokens,
        "finish_reason":         finish_reason,
        "output_text":           output_text,
        "hardware":              hw_summary,
    }


# ---------------------------------------------------------------------------
# Model-level summary — same structure as vLLM model_summary()
# ---------------------------------------------------------------------------

def model_summary(model_spec, runs, kv_params):
    """Average and std across all runs — identical field names to vLLM script."""
    sw = [r["software"] for r in runs]
    hw = [r["hardware"] for r in runs]
    num_layers, num_kv_heads, head_dim, dtype_bytes = kv_params

    return {
        "choice":    model_spec["choice"],
        "name":      model_spec["name"],
        "model_id":  model_spec["model_id"],
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
            "mean_gpu_power_w_avg":   mean(r["gpu_power_w_avg"]   for r in hw),
            "std_gpu_power_w_avg":     std(r["gpu_power_w_avg"]    for r in hw),
            "mean_gpu_util_pct_avg":  mean(r["gpu_util_pct_avg"]  for r in hw),
            "std_gpu_util_pct_avg":    std(r["gpu_util_pct_avg"]   for r in hw),
            "mean_cpu_pct_avg":       mean(r["cpu_pct_avg"]        for r in hw),
            "std_cpu_pct_avg":         std(r["cpu_pct_avg"]         for r in hw),
            "mean_mem_pct_avg":       mean(r["mem_pct_avg"]        for r in hw),
            "std_mem_pct_avg":         std(r["mem_pct_avg"]         for r in hw),
        },
        "raw_runs": runs,
    }


# ---------------------------------------------------------------------------
# TSV output — same columns as vLLM script for direct comparison
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
    p = argparse.ArgumentParser()
    p.add_argument("--runs",              type=int,   default=10)
    p.add_argument("--warmup-runs",       type=int,   default=1)
    p.add_argument("--max-tokens",        type=int,   default=4096)
    p.add_argument("--seed",              type=int,   default=1234)
    p.add_argument("--sample-interval-s", type=float, default=0.25)
    p.add_argument("--out-dir",           type=Path,  default=Path("benchmark_out_transformers"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """For each model: load → extract KV params → warmup → timed runs → save → unload."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    handles, gpu_info = init_nvml()
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
            "inference_engine":  "transformers",
        },
        "runs":    [],
        "models":  {},
        "summary": {},
    }

    json_path = args.out_dir / "transformers_benchmark_results.json"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT},
    ]

    try:
        for spec in MODEL_SPECS:
            print(f"\n{'='*60}")
            print(f"Loading model: {spec['model_id']}")
            print(f"{'='*60}")

            load_cfg = TRANSFORMERS_LOAD_CONFIG[spec["name"]]

            if load_cfg["use_processor"]:
                tokenizer = AutoProcessor.from_pretrained(spec["model_id"])
            else:
                tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])

            model = AutoModelForCausalLM.from_pretrained(
                spec["model_id"],
                torch_dtype=load_cfg["torch_dtype"],
                device_map="auto",
            )
            model.eval()

            underlying = getattr(tokenizer, "tokenizer", tokenizer)
            if underlying.pad_token is None:
                underlying.pad_token = underlying.eos_token

            # ------------------------------------------------------------------
            # Extract KV architecture params — same formula as vLLM script
            # ------------------------------------------------------------------
            num_layers, num_kv_heads, head_dim, dtype_bytes = get_kv_params_from_model(model)
            print(f"KV params — num_kv_heads: {num_kv_heads}, head_dim: {head_dim}, dtype_bytes: {dtype_bytes}")

            # Warmup — results discarded, warms up CUDA kernels and KV cache
            if args.warmup_runs > 0:
                print(f"Running {args.warmup_runs} warmup run(s)...")
                for _ in range(args.warmup_runs):
                    run_once(model, tokenizer, messages, 0, args.max_tokens,
                             handles, nvml_support, args.sample_interval_s)

            model_runs = []
            results["models"][spec["name"]] = {
                "choice":   spec["choice"],
                "model_id": spec["model_id"],
                "kv_architecture": {
                    "num_layers":   num_layers,
                    "num_kv_heads": num_kv_heads,
                    "head_dim":     head_dim,
                    "dtype_bytes":  dtype_bytes,
                },
                "raw_runs": model_runs,
                "summary":  None,
            }

            for run_index in range(1, args.runs + 1):
                r = run_once(model, tokenizer, messages, run_index, args.max_tokens,
                             handles, nvml_support, args.sample_interval_s)

                token_count   = r["generated_token_count"]
                gen_time_s    = r["generation_time_s"]
                tps           = r["tokens_per_second"]
                prompt_tokens = r["prompt_token_count"]
                hw_summary    = r["hardware"]

                # KV cache used this run — same formula as vLLM script
                kv_cache_used_mb = calc_kv_cache_used_mb(
                    prompt_tokens, token_count, num_layers, num_kv_heads, head_dim, dtype_bytes
                )

                run_record = {
                    "model_choice":  spec["choice"],
                    "model_name":    spec["name"],
                    "model_id":      spec["model_id"],
                    "run_index":     run_index,
                    "timestamp_utc": now_utc(),
                    "prompt": {
                        "system": SYSTEM_PROMPT,
                        "user":   USER_PROMPT,
                    },
                    "sampling": {
                        "max_tokens": args.max_tokens,
                        "do_sample":  False,   # greedy
                        "use_cache":  True,
                    },
                    "software": {
                        "generation_time_s":     gen_time_s,
                        "generated_token_count": token_count,
                        "tokens_per_second":     tps,
                        "kv_cache_used_mb":      kv_cache_used_mb,
                        "prompt_token_count":    prompt_tokens,
                        "finish_reason":         r["finish_reason"],
                        "output_text":           r["output_text"],
                    },
                    "hardware": hw_summary,
                }

                results["runs"].append(run_record)
                model_runs.append(run_record)

                print(json.dumps({
                    "model_name":        spec["name"],
                    "run_index":         run_index,
                    "generation_time_s": gen_time_s,
                    "generated_tokens":  token_count,
                    "tokens_per_second": tps,
                    "kv_cache_used_mb":  kv_cache_used_mb,
                    "finish_reason":     r["finish_reason"],
                    "gpu_power_w_avg":   hw_summary["gpu_power_w_avg"],
                    "gpu_util_pct_avg":  hw_summary["gpu_util_pct_avg"],
                    "cpu_pct_avg":       hw_summary["cpu_pct_avg"],
                    "mem_pct_avg":       hw_summary["mem_pct_avg"],
                }, indent=2))

                save_json(json_path, results)

            # Summarise this model
            summary = model_summary(spec, model_runs, (num_layers, num_kv_heads, head_dim, dtype_bytes))
            results["models"][spec["name"]]["summary"] = summary
            results["summary"][spec["name"]] = summary
            save_json(json_path, results)

            # ------------------------------------------------------------------
            # Free model memory before loading next model
            # transformers doesn't have vLLM's engine process so we just
            # delete the model object and clear CUDA cache
            # ------------------------------------------------------------------
            print(f"\nUnloading {spec['name']} from memory...")
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(20)   # let Jetson unified memory fully reclaim pages
            print("Memory freed.\n")

        # -----------------------------------------------------------------------
        # All models done
        # -----------------------------------------------------------------------
        save_json(json_path, results)

        tsv_path = args.out_dir / "transformers_summary.tsv"
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

        print(f"\nJSON log:    {json_path}")
        print(f"TSV summary: {tsv_path}")

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
