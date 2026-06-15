#!/usr/bin/env python3
"""
Install:
  pip install vllm pynvml psutil

Optional:
  gnuplot on PATH
"""


import argparse                          # parses command-line arguments like --runs 5
import gc                                # Python garbage collector, used to force memory cleanup
import json                              # read/write JSON files
import subprocess                        # run shell commands (used to call gnuplot)
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
    return datetime.now(timezone.utc).isoformat()  # Returns the current time as an ISO 8601 string e.g. "2026-06-11T14:32:00+00:00" for timestamping in JSON

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None # Safe average, returns None if the list is empty or all-None

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8") # Serialises the entire results dict to a pretty-printed JSON file
                                                                                  # sort_keys=True keeps the output consistent/diff-friendly across saves
                                                                                  # Called after every single run so a crash loses at most one run of data


# --------------------------
# NVML / hardware sampling:
# 
# initialises the NVIDIA Management Library and returns a handle for each GPU, plus their names. Called once at startup.
# --------------------------


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
    for power and sometimes memory.  We check once at startup so we
    can warn the user and avoid silently recording all-None metrics.
    """
    support = {"power": True, "util": True, "memory": True}                         # Assume everything works; flip to False if a test call fails

    if not handles:                                                                 # No GPUs found at all — disable everything and warn
        support = {"power": False, "util": False, "memory": False}                  
        print("WARNING: No NVML devices found. All hardware metrics will be None.")
        return support

    h = handles[0]                                                                  # test against the first GPU — if it fails there, it'll fail on all

    try:
        pynvml.nvmlDeviceGetPowerUsage(h)                                           # test call — result is thrown away
    except pynvml.NVMLError as e:
        support["power"] = False                                                    # mark power as unsupported
        print(f"WARNING: pynvml power readings not supported on this hardware ({e}).")
        print("         Power metrics will be recorded as None.")
        print("         On Jetson boards this is normal — use tegrastats for power.") # tegrastats is NVIDIA's Jetson-specific tool that can read SoC power rails

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
    # so nvml's GPU memory carve-out number doesn't tell the full story

    return support


def sample_hardware(handles, nvml_support):
    #takes a single hardware snapshot across all GPUs right now. Called repeatedly by the sampler thread during inference.
    
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
    '''
    runs on a background thread. Keeps calling sample_hardware() every interval_s seconds until the main thread sets stop_event.
    '''
    psutil.cpu_percent(interval=None)                                                   # discard first (always 0.0)
    while not stop_event.is_set():                                                      # keep looping until main thread says stop
        samples.append({"ts": now_utc(), **sample_hardware(handles, nvml_support)})
        # Append a timestamped snapshot to the shared list
        # **sample_hardware(...) unpacks the dict into the outer dict
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


def model_summary(model_spec, runs):
    '''
    after all runs for one model are done, averages across runs to produce the final per-model summary that goes into the JSON and TSV.
    '''
    sw = [r["software"] for r in runs]                  # list of software dicts, one per run
    hw = [r["hardware"] for r in runs]                  # list of hardware summary dicts, one per run
    return {
        "choice":   model_spec["choice"],
        "name":     model_spec["name"],
        "model_id": model_spec["model_id"],
        "run_count": len(runs),
        "software": {
            "mean_generation_time_s":    mean(r["generation_time_s"]    for r in sw),
            "mean_generated_token_count": mean(r["generated_token_count"] for r in sw),
            "mean_tokens_per_second":    mean(r["tokens_per_second"]    for r in sw),
        },
        "hardware": {
            "mean_gpu_power_w_avg":    mean(r["gpu_power_w_avg"]    for r in hw),
            "mean_gpu_util_pct_avg":   mean(r["gpu_util_pct_avg"]   for r in hw),
            "mean_cpu_pct_avg":        mean(r["cpu_pct_avg"]        for r in hw),
            "mean_mem_pct_avg":        mean(r["mem_pct_avg"]        for r in hw),
        },
        "raw_runs": runs,
    }


# ---------------------------------------------------------------------------
# Output helpers
#
# write the two files gnuplot needs — the data table and the plot script.
# ---------------------------------------------------------------------------

def write_summary_tsv(path, summaries):
    lines = [
        "model\trun_count\tmean_generation_time_s\tmean_generated_token_count"
        "\tmean_tokens_per_second\tmean_gpu_power_w_avg\tmean_gpu_util_pct_avg"
        "\tmean_cpu_pct_avg\tmean_mem_pct_avg"
    ]
    for model_name, s in summaries.items():
        sw = s["software"]
        hw = s["hardware"]
        lines.append(
            "\t".join([
                model_name,
                str(s["run_count"]),
                str(sw["mean_generation_time_s"]),
                str(sw["mean_generated_token_count"]),
                str(sw["mean_tokens_per_second"]),
                str(hw["mean_gpu_power_w_avg"]),
                str(hw["mean_gpu_util_pct_avg"]),
                str(hw["mean_cpu_pct_avg"]),
                str(hw["mean_mem_pct_avg"]),
            ])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""
def write_gnuplot_script(plot_dir, tsv_path):
    script = plot_dir / "plot_metrics.gp"
    script.write_text(
        f'''set datafile separator "\\t"
set terminal pngcairo size 1400,700 enhanced font ",12"
set style data histograms
set style histogram clustered gap 1
set style fill solid 1.0 border -1
set boxwidth 0.8
set xtics rotate by -30
set grid ytics

set output "{(plot_dir / "mean_tokens_per_second.png").as_posix()}"
set title "Mean Tokens per Second"
plot "{tsv_path.as_posix()}" skip 1 using 5:xtic(1) title "tokens/s"

set output "{(plot_dir / "mean_generation_time.png").as_posix()}"
set title "Mean Generation Time (s)"
plot "{tsv_path.as_posix()}" skip 1 using 3:xtic(1) title "seconds"

set output "{(plot_dir / "gpu_utilization.png").as_posix()}"
set title "Mean GPU Utilization (%)"
plot "{tsv_path.as_posix()}" skip 1 using 7:xtic(1) title "GPU util %"

set output "{(plot_dir / "gpu_power.png").as_posix()}"
set title "Mean GPU Power (W)"
plot "{tsv_path.as_posix()}" skip 1 using 6:xtic(1) title "Power (W)"

set output
''',
        encoding="utf-8",
    )
    return script           # path returned so main() can pass it to gnuplot subprocess
"""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    '''
    defines all the CLI flags so you can tweak the benchmark without editing the file.
    '''
    p = argparse.ArgumentParser()
    p.add_argument("--runs",                  type=int,   default=3)
    p.add_argument("--warmup-runs",           type=int,   default=1)
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
    p.add_argument("--out-dir",               type=Path,  default=Path("benchmark_out"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    '''
    For each model: load → warmup → timed runs → save → unload.
    '''
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    #plot_dir = args.out_dir / "plots"
    #plot_dir.mkdir(parents=True, exist_ok=True)

    handles, gpu_info = init_nvml()

    # Check once at startup which pynvml calls this board actually supports.
    # Jetson Orin uses unified memory and often blocks power/memory queries.
    nvml_support = probe_nvml_support(handles)

    results = {
        "meta": {
            "created_utc":       now_utc(),
            "system_prompt":     SYSTEM_PROMPT,
            "user_prompt":       USER_PROMPT,
            "model_specs":       MODEL_SPECS,
            "runs_per_model":    args.runs,
            "warmup_runs":       args.warmup_runs,
            "sample_interval_s": args.sample_interval_s,
            "hardware":          gpu_info,
            "nvml_support":      nvml_support,
        },
        "runs":    [],
        "models":  {},
        "summary": {},
    }

    json_path = args.out_dir / "benchmark_results.json"

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

                result      = outputs[0]
                output      = result.outputs[0]
                token_count = len(output.token_ids or [])
                gen_time_s  = end - start
                tps         = token_count / gen_time_s if gen_time_s > 0 else None

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
                        "generation_time_s":    gen_time_s,
                        "generated_token_count": token_count,
                        "tokens_per_second":     tps,
                        "prompt_token_count":   len(result.prompt_token_ids or []),
                        "finish_reason":        getattr(output, "finish_reason", None),
                        "output_text":          output.text,
                    },
                    "hardware": hw_summary,
                }

                results["runs"].append(run_record)
                model_runs.append(run_record)

                print(json.dumps({
                    "model_name":         spec["name"],
                    "run_index":          run_index,
                    "generation_time_s":  gen_time_s,
                    "generated_tokens":    token_count,
                    "tokens_per_second":   tps,
                    "gpu_power_w_avg":    hw_summary["gpu_power_w_avg"],
                    "gpu_util_pct_avg":   hw_summary["gpu_util_pct_avg"],
                    "cpu_pct_avg":        hw_summary["cpu_pct_avg"],
                    "mem_pct_avg":        hw_summary["mem_pct_avg"],
                }, indent=2))

                save_json(json_path, results)

            # Summarise this model
            summary = model_summary(spec, model_runs)
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
            time.sleep(5)
            print("GPU memory freed.\n")
            
        # -----------------------------------------------------------------------
        # All models done — write outputs
        # -----------------------------------------------------------------------
        save_json(json_path, results)

        tsv_path   = args.out_dir / "summary.tsv"
        write_summary_tsv(tsv_path, results["summary"])
        #gp_script = write_gnuplot_script(plot_dir, tsv_path)
        '''
        try:
            subprocess.run(["gnuplot", str(gp_script)], check=True)
            print(f"Graphs written to: {plot_dir}")
        except FileNotFoundError:
            print("gnuplot not found on PATH; skipping graph generation.")
        except subprocess.CalledProcessError as exc:
            print(f"gnuplot failed (exit {exc.returncode}); skipping graphs.")
        '''
        print("\nFinal summary")
        print("=" * 60)
        for model_name, s in results["summary"].items():
            sw = s["software"]
            hw = s["hardware"]
            print(f"\nModel: {model_name}")
            print(f"  Model ID:               {s['model_id']}")
            print(f"  Runs:                   {s['run_count']}")
            print(f"  Mean generation time:   {sw['mean_generation_time_s']:.3f} s")
            print(f"  Mean generated tokens:  {sw['mean_generated_token_count']:.1f}")
            print(f"  Mean tokens/sec:        {sw['mean_tokens_per_second']:.2f}")
            print(f"  Mean GPU power (W):     {hw['mean_gpu_power_w_avg']}")
            print(f"  Mean GPU util (%):      {hw['mean_gpu_util_pct_avg']}")
            print(f"  Mean CPU util (%):      {hw['mean_cpu_pct_avg']}")
            print(f"  Mean memory util (%):   {hw['mean_mem_pct_avg']}")

        print(f"\nJSON log:  {json_path}")
        #print(f"Graphs:    {plot_dir}")

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
