# 3Correlation — does GPU utilization predict throughput?

Correlates NVIDIA's reported GPU utilization, and idle-subtracted (dynamic) power,
against measured throughput, using the per-step CSVs produced by `2Roofline-Plot`.
Reports Pearson, Spearman and Kendall coefficients and draws heatmaps, scatter plots
and pair grids.

The question is practical: utilization is the number everyone reaches for when asked
"is the GPU busy?", and this folder tests how much it actually tells you. The short
answer from this data is that it depends on saturation — see the findings below.

## Requirements

```bash
pip install numpy scipy matplotlib
```

No GPU and no inference engine needed; this stage only reads CSVs. It is the one part
of the pipeline you can run on a laptop.

## What it reads

`correlation.py` consumes the per-step CSVs from `2Roofline-Plot` — the same
`<model>_b<batch>.csv` files, copied here under `data/`. It reads columns *by name*, so
it handles both schemas transparently: the vLLM CSVs have 24 columns and no
`drain_like` column, the Transformers ones do.

```
data/<machine>_<engine>/<model>_b<batch>.csv
```

Six directories, one per machine-engine pair, each holding a full 4-model × 6-batch
sweep (24 CSVs). Grouping is by parent directory, so each directory becomes one cell in
the results.

## Run sequence

This stage depends on `2Roofline-Plot` having been run — there is nothing to measure
here. Point it at the data directories and it does the rest.

```bash
python3 correlation.py data/* --outdir corr_out --out-prefix combined
```

That reproduces every published number and figure. To look at a single cell:

```bash
python3 correlation.py data/a100_vllm --outdir corr_a100_vllm --out-prefix a100_vllm
```

Useful flags: `--phases` restricts to prefill or decode (they are never pooled by
default, and pooling them is a mistake — see below); `--group-by` and `--group` control
how rows are bucketed; `--per-group-heatmaps` emits one heatmap per cell instead of a
combined one; `--pairs` draws the pair grid; `--keep-drain` retains vLLM's terminal
bookkeeping step, which is excluded by default; `--no-plot`, `--no-heatmap` and
`--no-scatter` skip figure generation when you only want the coefficients.

Drain rows are dropped using the `drain_like` column when present, and otherwise by the
physical criterion of latency below half the median — the same rule `roofline.py`
applies.

## Interpreting the output — two traps

**Prefill and decode must never be pooled.** They differ by orders of magnitude in both
arithmetic intensity and duration, so a pooled correlation describes the split between
two clusters rather than any relationship within them. The script keeps them separate by
default; keep it that way.

**Pooling across machines reverses the sign.** Pooled prefill utilization against
throughput comes out at about −0.45, which reads as "more utilization, less throughput".
It is a Simpson's-paradox artifact: Thor runs at high utilization and low absolute
throughput, the A100 at lower utilization and much higher throughput, so the
cross-machine trend inverts the within-machine one. Never read that pooled number
causally.

## What the data shows

Utilization tracks throughput only where the GPU is *not* saturated. On the A100,
decode utilization against tokens/second is clearly positive — Pearson +0.56 under
Transformers, +0.37 under vLLM. On Thor it collapses to nothing, +0.05 and +0.01, because
utilization there is pinned near its ceiling: once the duty cycle is effectively 100%,
the metric carries no information about how much work is getting done.

Dynamic power is the robust correlate. It is positive in every cell — +0.59 and +0.39 on
the A100, +0.80 and +0.76 on Thor — and +0.81 even pooled across machines for prefill.

The mechanism is that NVML's utilization is a *duty cycle*: the fraction of time at
least one kernel was resident, not a measure of how much of the machine is being used.
A kernel occupying one SM and a kernel saturating all of them both report 100%. So the
working rule is: if utilization has small variance or sits near a ceiling, ignore the
sign of its coefficient entirely, and reach for dynamic power instead.

An earlier revision of this analysis reported a strong *negative* utilization correlation
on Thor (about −0.8). That came from single-batch data where four models formed four
separate blobs, and the coefficient described the between-model spread rather than
anything within a model. The full batch sweep, which supplies real within-model
variation, replaces it with the ~0 reported above.

## Data provenance

The CSVs here must come from single-GPU runs. NVML sums power across every visible GPU
and averages utilization across them, so a multi-GPU allocation divides utilization by
the device count — which would corrupt exactly the variable this analysis studies. The
committed data was collected with `--nvml-device auto` (or on genuinely single-GPU
machines); verify with `idle_power_w` in the matching `_summary.json`, which should be
about 63 W on an A100 rather than 256 W.