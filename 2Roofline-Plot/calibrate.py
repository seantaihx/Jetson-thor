#!/usr/bin/env python3
"""Empirical roofline ceilings: peak BF16 GEMM TFLOP/s and DRAM GB/s.

Method (matches arXiv:2512.01644 / LBNL Empirical Roofline Toolkit):
  * Compute roof: dense square GEMMs swept over sizes, CUDA-event timed,
    best sustained rate = ceiling. BF16 inputs, FP32 accumulate (Tensor
    Cores), i.e. the same numeric path vLLM uses.
  * Memory roof: STREAM-style kernels on buffers far larger than L2.
    triad c = a + s*b counts 3 arrays (2 read + 1 write, STREAM
    convention); copy counts 2. Best sustained rate = ceiling.
  * min-time over iterations (STREAM convention: best of N), median also
    reported so you can see run-to-run noise.

Before running, pin clocks or DVFS makes the numbers unrepeatable:
    sudo nvpmodel -m 0 && sudo jetson_clocks

Run:
    python3 calibrate.py                      # defaults
    python3 calibrate.py --sizes 2048 4096 8192 16384 --mem-gib 2

Paste the two printed values into the profiler as
    --peak-tflops <gemm> --peak-memory-gbs <mem>
and report both measured and datasheet (Thor: 273 GB/s) in the paper.
"""
import argparse
import json
import statistics

import torch


def bench(fn, iters: int, warmup: int, repeat: int = 1) -> tuple:
    """Return per-call (min_s, median_s) over iters CUDA-event-timed
    measurements; each measurement times `repeat` back-to-back calls so
    launch overhead does not dominate very short kernels (L2 test)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True))
    times = []
    for _ in range(iters):
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3 / repeat)
    return min(times), statistics.median(times)


def gemm_roof(sizes, dtype, iters, warmup):
    rows = []
    for n in sizes:
        try:
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            c = torch.empty(n, n, device="cuda", dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"[skip] GEMM {n}x{n}: out of memory")
            continue
        tmin, tmed = bench(lambda: torch.matmul(a, b, out=c), iters, warmup)
        flops = 2.0 * n**3
        rows.append({"n": n, "tflops_best": flops / tmin / 1e12,
                     "tflops_median": flops / tmed / 1e12})
        print(f"  GEMM {n:>6}^3: {rows[-1]['tflops_best']:8.2f} TFLOP/s best"
              f"  ({rows[-1]['tflops_median']:.2f} median)")
        del a, b, c
        torch.cuda.empty_cache()
    if not rows:
        raise SystemExit("[error] every GEMM size OOMed; pass smaller --sizes")
    return max(r["tflops_best"] for r in rows), rows


def mem_roof(gib, iters, warmup):
    n = int(gib * 2**30 // 4)  # fp32 elements; bandwidth is dtype-agnostic
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    out = {}
    tmin, tmed = bench(lambda: torch.add(a, b, alpha=2.5, out=c),
                       iters, warmup)
    out["triad_gbs"] = 3 * n * 4 / tmin / 1e9          # 2 read + 1 write
    print(f"  triad ({gib:g} GiB/buf): {out['triad_gbs']:7.1f} GB/s best"
          f"  ({3 * n * 4 / tmed / 1e9:.1f} median)")
    tmin, tmed = bench(lambda: c.copy_(a), iters, warmup)
    out["copy_gbs"] = 2 * n * 4 / tmin / 1e9           # 1 read + 1 write
    print(f"  copy  ({gib:g} GiB/buf): {out['copy_gbs']:7.1f} GB/s best"
          f"  ({2 * n * 4 / tmed / 1e9:.1f} median)")
    return max(out.values()), out


def l2_roof(iters, warmup, l2_bytes):
    """L2-resident bandwidth ceiling: same triad/copy kernels on a buffer
    that fits in L2, so traffic is served from cache, not DRAM. Needed to
    roof-match kernel bytes measured via lts__t_sectors (L2 traffic).

    Provisional: counts algorithmic bytes and amortizes but does not
    remove inter-kernel launch gaps, so it likely UNDERSTATES the true L2
    ceiling - and an understated ceiling CAN push valid kernels above the
    plotted roof. Before publication, validate with the same counters the
    workload uses (one sudo run + one analyze):
        sudo $NCU --nvtx --nvtx-include "l2roof/" -f -o l2val --metrics \\
            "gpu__time_duration.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum" \\
            $(which python3) calibrate.py --l2-only
        python3 calibrate.py --l2-validate l2val.ncu-rep --ncu-binary $NCU \\
            --update-ceilings
    That computes BW_L2 = 32*(S_read+S_write)/t from the report - the
    exact byte definition kernel_roofline.py uses - and (with
    --update-ceilings) replaces peak_l2_gbs in ceilings.json."""
    if not l2_bytes:
        print("[warn] L2 size unknown (old torch); pass --l2-mib explicitly")
        return None, {}
    # triad touches 3 fp32 buffers; keep their sum ~60% of L2 capacity.
    n = int(l2_bytes * 0.6 // (3 * 4))
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    out = {"buffer_bytes_each": n * 4, "l2_bytes": l2_bytes}
    # NVTX range so ncu can capture exactly these kernels (--l2-only mode
    # + the validation recipe in the docstring above).
    from torch.cuda import nvtx
    nvtx.range_push("l2roof")
    tmin, tmed = bench(lambda: torch.add(a, b, alpha=2.5, out=c),
                       iters, warmup, repeat=200)
    out["triad_gbs"] = 3 * n * 4 / tmin / 1e9
    print(f"  L2 triad ({3 * n * 4 / 2**20:.1f} MiB working set): "
          f"{out['triad_gbs']:7.1f} GB/s best "
          f"({3 * n * 4 / tmed / 1e9:.1f} median)")
    tmin, tmed = bench(lambda: c.copy_(a), iters, warmup, repeat=200)
    nvtx.range_pop()
    out["copy_gbs"] = 2 * n * 4 / tmin / 1e9
    print(f"  L2 copy : {out['copy_gbs']:7.1f} GB/s best "
          f"({2 * n * 4 / tmed / 1e9:.1f} median)")
    print("[note] algorithmic-byte L2 ceiling is provisional; validate "
          "with counters (see docstring: --l2-only / --l2-validate)")
    del a, b, c
    torch.cuda.empty_cache()
    return max(out["triad_gbs"], out["copy_gbs"]), out


def l2_validate(report, ncu_binary, ceilings_path, update):
    """Counter-validated L2 ceiling: BW = 32*(S_read+S_write)/t from an
    ncu capture of the l2roof NVTX range - byte-definition-identical to
    kernel_roofline.py's workload measurement."""
    from kernel_roofline import (export_csv, metric_cols, parse_rows,
                                 to_float)
    rows, tscale = parse_rows(export_csv(report, ncu_binary))
    if not rows:
        raise SystemExit("[error] no launches in the report; was "
                         "--nvtx --nvtx-include \"l2roof/\" used?")
    heads = rows[0].keys()
    scols = metric_cols(heads, r"lts__t_sectors_op_(read|write)\.sum$")
    tcol = next((h for h in heads
                 if h.startswith("gpu__time_duration")), None)
    if not scols or tcol is None:
        raise SystemExit("[error] report lacks lts sector / duration "
                         "metrics; use the --metrics list from the "
                         "docstring")
    sectors = sum(to_float(r[c]) or 0.0 for r in rows for c in scols)
    t = sum((to_float(r[tcol]) or 0.0) * tscale for r in rows)
    if t <= 0 or sectors <= 0:
        raise SystemExit("[error] zero time or sectors in the report")
    bw = 32.0 * sectors / t / 1e9
    print(f"[done] counter-validated L2 bandwidth: {bw:.1f} GB/s "
          f"({len(rows)} launches, {32.0 * sectors / 1e9:.2f} GB in "
          f"{t * 1e3:.1f} ms)")
    try:
        with open(ceilings_path, encoding="utf-8") as f:
            data = json.load(f)
        old = data.get("peak_l2_gbs")
        if old:
            print(f"[info] algorithmic estimate was {old:.1f} GB/s "
                  f"(ratio {bw / old:.3f})")
        if update:
            data["peak_l2_gbs"] = bw
            data["peak_l2_gbs_source"] = "lts_counter_validated"
            with open(ceilings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[done] {ceilings_path} updated with the "
                  "counter-validated L2 ceiling")
    except OSError:
        print(f"[warn] {ceilings_path} not found; use "
              f"--peak-memory-gbs {bw:.1f} --roof-level l2 manually")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 12288, 16384])
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--mem-gib", type=float, default=1.0,
                   help="per-buffer size; must dwarf L2 (3 buffers live)")
    p.add_argument("--l2-mib", type=float, default=None,
                   help="L2 cache size override in MiB (default: query "
                        "torch device properties)")
    p.add_argument("--l2-only", action="store_true",
                   help="run ONLY the L2 kernels (for profiling the "
                        "'l2roof' NVTX range under ncu); does not write "
                        "ceilings.json")
    p.add_argument("--l2-validate", metavar="NCU_REP", default=None,
                   help="compute the counter-validated L2 ceiling from an "
                        "ncu report of --l2-only (no GPU needed)")
    p.add_argument("--update-ceilings", action="store_true",
                   help="with --l2-validate: write the validated value "
                        "into --out")
    p.add_argument("--ncu-binary", default="ncu")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default="ceilings.json")
    args = p.parse_args()

    if args.l2_validate:
        return l2_validate(args.l2_validate, args.ncu_binary, args.out,
                           args.update_ceilings)

    if not torch.cuda.is_available():
        raise SystemExit("[error] CUDA not available")
    dev = torch.cuda.get_device_name(0)
    print(f"[info] {dev}, torch {torch.__version__}, dtype {args.dtype}")
    print("[info] pin clocks first: sudo nvpmodel -m 0 && sudo jetson_clocks")

    if args.l2_only:
        props = torch.cuda.get_device_properties(0)
        l2_bytes = (int(args.l2_mib * 2**20) if args.l2_mib
                    else getattr(props, "L2_cache_size", None)
                    or getattr(props, "l2_cache_size", None))
        print("[run] L2 roof only (NVTX range 'l2roof' active)")
        l2_roof(args.iters, args.warmup, l2_bytes)
        return 0

    print("[run] compute roof (dense GEMM sweep)")
    dtype = getattr(torch, args.dtype)
    peak_tflops, gemm_rows = gemm_roof(args.sizes, dtype,
                                       args.iters, args.warmup)
    print("[run] memory roof (STREAM triad / copy, DRAM)")
    peak_gbs, mem_rows = mem_roof(args.mem_gib, args.iters, args.warmup)

    props = torch.cuda.get_device_properties(0)
    l2_bytes = (int(args.l2_mib * 2**20) if args.l2_mib
                else getattr(props, "L2_cache_size", None)
                or getattr(props, "l2_cache_size", None))
    print("[run] L2 roof (cache-resident triad / copy)")
    peak_l2_gbs, l2_rows = l2_roof(args.iters, args.warmup, l2_bytes)

    result = {"device": dev, "dtype": args.dtype,
              "peak_tflops": peak_tflops, "peak_memory_gbs": peak_gbs,
              "peak_l2_gbs": peak_l2_gbs,
              "gemm": gemm_rows, "memory": mem_rows, "l2": l2_rows,
              "ridge_flop_per_byte": 1000.0 * peak_tflops / peak_gbs}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[done] measured ceilings -> {args.out}")
    print(f"[done] step roofline (analytical DRAM bytes): "
          f"--peak-tflops {peak_tflops:.1f} --peak-memory-gbs {peak_gbs:.1f} "
          f"(ridge {result['ridge_flop_per_byte']:.1f} FLOP/byte)")
    if peak_l2_gbs:
        print(f"[done] kernel roofline (lts/L2-counted bytes): "
              f"--peak-tflops {peak_tflops:.1f} "
              f"--peak-memory-gbs {peak_l2_gbs:.1f} --roof-level l2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
