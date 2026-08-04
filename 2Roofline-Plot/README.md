# 2Roofline-Plot — step-resolved roofline profiling

Profiles inference one decoding step at a time and places each phase on a roofline:
arithmetic intensity (FLOP/byte) against achieved TFLOP/s, under empirically measured
compute and memory ceilings. Prefill and decode are always kept separate — they sit at
opposite ends of the roofline and pooling them is meaningless.

Workload is batch 4, exactly 256 generated tokens, 80-token prompt. That is *not* the
`1Bar-Plot` workload (batch 1, up to 4096 tokens), so power and throughput numbers from
the two folders are not comparable. `bars_from_step.py` exists so you can get bar charts
whose workload does match this data.

Models: llama, gemma, gpt (gpt-oss-20b), qwen — selected with `--choice 1..4`.

## Requirements

```bash
python3 -m venv seanvenv && source seanvenv/bin/activate
pip install torch pynvml
pip install vllm                      # step_metric_vllm.py
pip install transformers accelerate   # step_metric_tf.py
```

`roofline.py` and `bars_from_step.py` are pure standard library. Both render through
**gnuplot ≥ 5.2** (`apt install gnuplot-nox`); the 5.2 minimum is real, because the
merged plot uses `keyentry`. `calibrate.py` needs only torch.

## What each script does

`calibrate.py` measures the machine's ceilings: peak BF16 GEMM TFLOP/s by sweeping
dense square matmuls, and DRAM GB/s by STREAM-style kernels on buffers far larger than
L2. Writes `ceilings.json`. This runs once per machine and everything downstream
depends on it.

`step_metric_vllm.py` and `step_metric_tf.py` are the profilers — the same measurement
under the two engines. Analytical FLOP/byte model, NVTX ranges, per-step NVML power and
utilization sampling. They emit `<out>.csv` (one row per step), `<out>_summary.json`,
and a per-run roofline `.dat`/`.gnuplot`/`.png`.

`roofline.py` is the figure generator. It merges both engines onto one axes, colouring
by model and ringing the Transformers points, and writes a 2D gnuplot PNG plus an
interactive 3D Plotly HTML.

`bars_from_step.py` builds throughput / utilization / power bars from the step_metric
summaries, so the bar comparison and the roofline describe the same runs.

## Run sequence

Calibrate once per machine, profile every model you care about, then plot.

### 1. Ceilings

```bash
python3 calibrate.py --update-ceilings --out ceilings.json
```

`ceilings.json` is not committed — it is machine-specific and must be produced on the
machine itself. Measured values for the three machines in this study:

| machine | `--peak-tflops` | `--peak-memory-gbs` | ridge (FLOP/byte) |
|---|---|---|---|
| Jetson AGX Thor | 137.9 | 239.7 | 575 |
| A100-SXM4-80GB | 291.7 | 1794.5 | 163 |
| H200 NVL | 717.2 | 4231.9 | 169 |

The ridge point is where a machine stops being memory-bound. Prefill lands
compute-bound on the A100 and H200 but memory-bound on Thor, which is the central
comparison this folder exists to make.

### 2. Profile

One run per model per batch size. Edit `MODEL_SPECS` first if your models live at
filesystem paths rather than Hub ids.

```bash
salloc --exclusive -p csi -A csivisitors --nodes=1 --gres=gpu:1   # shared nodes only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for c in 1 2 3 4; do
  for b in 1 2 4 8 16 32; do
    out=$(printf "%s_b%03d" "$(sed -n "${c}p" <<< $'llama\ngemma\ngpt\nqwen')" "$b")
    python3 step_metric_vllm.py --choice $c --batch-size $b --out "$out" \
        --peak-tflops 717.2 --peak-memory-gbs 4231.9
  done
done
```

Swap `step_metric_vllm.py` for `step_metric_tf.py` and use distinct `--out` names
(the tf default is `hf_step_metrics`, the vLLM default `step_metrics`) so the two
engines never overwrite each other.

`--reprocess` re-derives the outputs from an existing CSV without re-running the model,
which is how you regenerate figures after changing the analytical model. `--nvml-device`
behaves exactly as in `1Bar-Plot` and matters for the same reason — see the note at the
end.

One flag needs care: `--expert-dtype-bytes 0.53` should be passed **only** on the
gpt-oss vLLM runs, where the experts are MXFP4 4-bit while attention and KV stay at
2 bytes. Leaving it off overstates gpt-oss memory traffic roughly fourfold and pushes
the point off the roof. Dense models and all Transformers runs need nothing.

### enforce_eager — note the inverted default

`step_metric_vllm.py` runs vLLM with `enforce_eager=True` **by default**, and
`--no-enforce-eager` turns it off. That is the opposite of `1Bar-Plot/metrics_vllm.py`,
where `--enforce-eager` is an opt-in flag and CUDA graphs are on by default.

The default here is deliberate, not an oversight. This profiler times and attributes
work one decoding step at a time; CUDA graph capture fuses launches and moves work
across the step boundaries the measurement depends on, so eager execution is what makes
per-step FLOP and byte attribution meaningful. The cost is that absolute latency runs
slightly slower than a production deployment would.

Every committed vLLM run in `data/` records `"enforce_eager": true` in its summary
`config`, so the published roofline figures are all eager. If you add runs, keep it that
way unless you are specifically studying graph capture — and if you do use
`--no-enforce-eager`, keep those runs in a separate output directory, because mixing the
two on one figure compares different execution modes.

This is also why roofline throughput should not be read against `1Bar-Plot` throughput:
different workload *and* different execution mode.

### 3. Plot

```bash
python3 roofline.py \
    data/h200_vllm/llama_metrics data/h200_vllm/gemma_metrics \
    data/h200_vllm/gpt_metrics   data/h200_vllm/qwen_metrics \
    data/h200_transformers/llama_metrics data/h200_transformers/gemma_metrics \
    data/h200_transformers/gpt_metrics   data/h200_transformers/qwen_metrics \
    --peak-tflops 717.2 --peak-memory-gbs 4231.9 --fold-drain \
    --topic "Roofline H200 vLLM + Transformers" \
    --out all_model_roofline/h200_merged.html
```

Arguments are run basenames — the script appends `.csv` and `_summary.json` itself. The
engine of each run is detected automatically from `config.engine` in the summary, with
the directory name as fallback, and printed per file so you can check it.

Reading the figure: squares are prefill, circles are decode, and a ring around a marker
means Transformers. Point labels are raw watts, GPU utilization, tokens/second. The
y-axis is fixed across every machine and batch size so figures can be compared side by
side — `Y_AXIS_DEFAULT` near the top of the script, currently 0.1–2200 TFLOP/s.
`--y-min`/`--y-max` override it, and any point falling outside the range is reported on
stderr rather than silently dropped.

One caveat on that floor: the slowest point in the full sweep is Thor / Transformers /
gpt at batch 1, whose decode aggregate is 0.0975 TFLOP/s, so a 0.1 floor clips it off
the bottom of that one figure. Lowering `Y_AXIS_DEFAULT` to `(0.05, 2200.0)` keeps every
point on-chart and still aligns all figures.

`--fold-drain` folds vLLM's terminal bookkeeping iteration into the phase totals instead
of excluding it. That step does about 1 ms of no real work, and left in the per-step
average it inflates decode throughput by roughly half. The drain step does not exist
under Transformers, and the script knows not to apply the filter there.

The script fails closed. If any kept step implies memory traffic above the measured
ceiling, or an aggregate lands above the roof, it aborts rather than publishing a
physically impossible point. `--no-exclude` demotes those aborts to warnings for
inspection — never for a publication figure.

```bash
python3 bars_from_step.py --series "H200 vLLM=data/h200_vllm" \
                          --series "H200 TF=data/h200_transformers" \
                          --phase decode --out-dir bars_h200 --run-gnuplot
```

## Not included in this folder

`--l2-validate` on `calibrate.py` and `--measured` on `roofline.py` both need
`kernel_roofline.py`, which is not in this repo. Without it, use the analytic path —
FLOP and byte counts from the model formulas rather than ncu counters. The analytic
path is the one all committed figures use.

`ceilings.json` is likewise absent by design; produce it with `calibrate.py`.

## The multi-GPU measurement trap

NVML reports the whole physical node and ignores `CUDA_VISIBLE_DEVICES`, so on a 4-GPU
node a single-GPU job still sees four devices: power is summed across all of them and
utilization is divided by the device count. Both step_metric scripts default to
`--nvml-device auto`, which resolves the compute GPU and samples only that one.

Confirm at startup that the idle baseline is near 63 W on an A100, not 256 W, and that
the NVML line names one device rather than "all 4 visible GPU(s)".

## Data layout

```
data/<machine>_<engine>/<model>_metrics.csv            per-step rows
data/<machine>_<engine>/<model>_metrics_summary.json   config + aggregates
all_model_roofline/<machine>_merged_2d.png             merged figure
```