# 1Bar-Plot — whole-generation benchmark

Measures what a single inference request actually costs end to end: tokens/second,
GPU power (raw and idle-subtracted), GPU utilization, and energy per token, for four
models under two engines. One request at a time, up to 4096 generated tokens,
10 timed runs per model after 2 warmups.

This is a *different workload* from `2Roofline-Plot`, which profiles batch-4 /
256-token runs step by step. Numbers from the two folders are not interchangeable —
in particular the power figures differ because the workloads differ. If you want
bar charts whose workload matches the roofline data, use `2Roofline-Plot/bars_from_step.py`
instead of this folder.

Models: llama (Llama-3.1-8B-Instruct), gemma (gemma-4-E4B-it), gpt (gpt-oss-20b),
qwen (Qwen2.5-7B-Instruct).

## Requirements

Python 3.10–3.12, Linux, an NVIDIA GPU.

```bash
python3 -m venv benchenv && source benchenv/bin/activate
pip install torch pynvml psutil
pip install vllm            # for metrics_vllm.py
pip install transformers    # for metrics_transformers.py
pip install matplotlib numpy   # only for compare_enforce_eager_vllm.py
```

Charts are rendered by **gnuplot ≥ 5.2**, which is a system package, not a pip one
(`apt install gnuplot-nox`). Without it the scripts still write `.gp` files you can
render anywhere.

## What each script does

`metrics_vllm.py` runs the benchmark under vLLM. `metrics_transformers.py` runs the
identical benchmark under HF Transformers — same prompt, same greedy decoding, same
sampling of hardware counters, same output schema — so the two JSONs are directly
comparable.

`plot_transformers_vs_vllm.py` takes one machine's pair of JSONs and emits throughput,
utilization and power bar charts.

`combine.py` is the multi-machine version: each `--series` adds one bar per model
group, so you can put Jetson / A100 / H200 and both engines on a single chart.

`compare_enforce_eager_vllm.py` compares two vLLM runs, CUDA graphs versus
`--enforce-eager`. It is the only script here that uses matplotlib rather than gnuplot.

## Run sequence

Measure first (both engines), then plot. Nothing here reads the `.tsv` files — they
are a human-readable export only; every plot script reads the JSONs.

### 1. Point the script at your models

`MODEL_SPECS` near the top of both metrics scripts is hardcoded. On a cluster with a
local model tree use filesystem paths; on a machine that resolves from the HF cache
use Hub ids:

```python
# SDCC / A100
{"choice": "1", "name": "llama", "model_id": "/hpcgpfs01/scratch/stai/models/Llama-3.1-8B-Instruct"},
# Thor / H200
{"choice": "1", "name": "llama", "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
```

Getting this wrong is the most common failure: a Hub id on a machine with no network
raises `LocalEntryNotFoundError` at the first tokenizer load.

The output filenames are hardcoded too (`vllm_benchmark_results_ic2_2.json`,
`transformers_benchmark_results_ic2.json`). Edit them, or keep them and separate runs
with `--out-dir`.

### 2. Run the benchmark

On a shared multi-GPU node, allocate a single GPU:

```bash
salloc --exclusive -p csi -A csivisitors --nodes=1 --gres=gpu:1
source /path/to/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # offline compute nodes

python3 metrics_vllm.py         --out-dir benchmark_out_vllm
python3 metrics_transformers.py --out-dir benchmark_out_transformers
```

Defaults already match the published configuration: `--runs 10 --warmup-runs 2
--max-tokens 4096 --sample-interval-s 0.25 --idle-baseline-s 10`.

**enforce_eager.** `--enforce-eager` is opt-in here, so the default is CUDA graphs
enabled — production-like timing. Do not pass it for a normal run; it exists to produce
the eager half of the pair that `compare_enforce_eager_vllm.py` consumes, and those runs
belong in their own `--out-dir` under an `_EnforceEager` name.

Two related gotchas. `2Roofline-Plot/step_metric_vllm.py` has the **inverted** default —
it runs eager unless you pass `--no-enforce-eager` — because per-step attribution needs
eager execution. So vLLM throughput from that folder is not comparable to throughput
from this one, on execution mode as well as workload. And the setting is currently *not*
recorded in this folder's JSON `meta`, so a file's execution mode can only be inferred
from its filename; the roofline summaries do record it as `config.enforce_eager`. Worth
adding here if you touch the script.

Budget roughly 60 minutes for the vLLM sweep and 80 for Transformers; gemma is the
slowest model by a wide margin.

### 3. Plot

```bash
# one machine, both engines
python3 plot_transformers_vs_vllm.py \
    --vllm-json benchmark_out_vllm/vllm_benchmark_results_ic2_2.json \
    --transformers-json benchmark_out_transformers/transformers_benchmark_results_ic2.json \
    --out-dir plots_ic2 --title "A100" --run-gnuplot

# several machines on one chart
python3 combine.py \
    --series "JT vLLM=data/jetson/vllm_benchmark_results_jt.json" \
    --series "JT Transformers=data/jetson/transformers_benchmark_results_jt.json" \
    --series "A100 vLLM=data/a100/vllm_benchmark_results_ic2_2.json" \
    --series "A100 Transformers=data/a100/transformers_benchmark_results_ic2.json" \
    --series "H200 vLLM=data/h200/vllm_benchmark_results_h200.json" \
    --series "H200 Transformers=data/h200/transformers_benchmark_results_h200.json" \
    --out-dir plots_combined --title "Jetson Thor vs A100 vs H200" --run-gnuplot
```

`--power-key` selects raw versus dynamic power. For cross-machine comparison prefer
the dynamic key, which subtracts each platform's idle floor.

## The multi-GPU measurement trap

NVML reports the whole physical node. It does **not** honour `CUDA_VISIBLE_DEVICES`,
so on a 4-GPU node a `--gres=gpu:1` job still sees four devices: power gets summed
across all of them and utilization gets averaged down by the device count. On an A100
node that adds roughly 192 W of idle-sibling draw to every reading and divides
utilization by about four.

Both metrics scripts therefore take `--nvml-device`, defaulting to `auto`, which
resolves the compute GPU from `CUDA_VISIBLE_DEVICES` and samples only that one. You
can force a choice with an index (`--nvml-device 0`) or a UUID, or restore the old
behaviour with `--nvml-device all`.

Check two things at startup:

```
[info] NVML sampling: NVML index 2 (uuid GPU-…) [CUDA_VISIBLE_DEVICES]
Idle power baseline: 63.10 W over 40 samples
```

An idle baseline near 256 W on an A100, or a line reading "all 4 visible GPU(s)",
means the run is polluted — stop and pass `--nvml-device 0`.

Afterwards the JSON records what was actually measured:

```bash
python3 -c "import json;m=json.load(open('benchmark_out_vllm/vllm_benchmark_results_ic2_2.json'))['meta'];\
print(m['gpu_count'], m['gpu_count_visible'], m['nvml_device'], round(m['idle_power_w'],1))"
# 1 4 NVML index 2 (uuid GPU-…) [CUDA_VISIBLE_DEVICES] 63.1
```

`gpu_count: 1` alongside `gpu_count_visible: 4` is the signature you want. Files
lacking the `nvml_device` key predate this fix and were measured across every GPU on
the node — in those, `gpu_power_w_dyn_avg` and the dynamic energy fields are still
correct (the idle baseline cancels the siblings) and `tokens_per_second` is unaffected,
but raw power, utilization and raw J/token are not comparable across machines.

## Data layout

```
data/<machine>/<engine>_benchmark_results_<host>.json   full record: meta + every run
data/<machine>/<engine>_summary_<host>.tsv              flat per-model summary
```

`meta` carries the provenance that matters: `gpu_count`, `gpu_count_visible`,
`nvml_device`, `idle_power_w`, `runs_per_model`, `max_tokens`, `model_specs`.