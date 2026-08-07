#!/usr/bin/env python3
"""
Three-way comparison bar charts for a SINGLE machine:

  - vLLM, CUDA graphs      (enforce_eager = False)
  - vLLM, enforce_eager    (enforce_eager = True)
  - Transformers

Same chart style / same gnuplot pipeline as combine.py, but with three
correctly-labelled series instead of combine.py's four hardcoded
JT/IC2 series.

Example (H200):
  python3 combine3.py \
    --vllm-graphs-json  "no_eager/h200_no_eager.json" \
    --vllm-eager-json   "Jetson-thor/h200/benchmark_out_vllm_h200/vllm_benchmark_results_ic2_2.json" \
    --transformers-json "Jetson-thor/h200/benchmark_out_transformers/transformers_benchmark_results_ic2.json" \
    --out-dir plots_h200_eager3 \
    --title "H200 NVL - vLLM CUDA graphs vs enforce_eager vs Transformers" \
    --run-gnuplot

Any input file that is missing or empty plots as NaN (gnuplot just skips
that bar) rather than aborting the run.

NOTE ON POWER: --power-metric defaults to "dyn" (gpu_power_w_dyn_avg =
measured minus idle baseline), which is what combine.py plotted. Pass
--power-metric raw for gpu_power_w_avg. The y-label states which one is
being shown, so the figure is never ambiguous.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Series definition: internal key -> legend label. Order == column order.
# ---------------------------------------------------------------------------
SERIES: list[tuple[str, str]] = [
    ("vllm_graphs", "vLLM (CUDA graphs)"),
    ("vllm_eager", "vLLM (enforce_eager)"),
    ("transformers", "Transformers"),
]
SERIES_KEYS = [k for k, _ in SERIES]
N_SERIES = len(SERIES)


def load_json_or_none(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def avg(values: Iterable[Any]) -> float | None:
    nums = [float(v) for v in values if valid_number(v)]
    return mean(nums) if nums else None


def model_metric_from_summary(data: dict[str, Any], model_name: str, section: str, key: str) -> float | None:
    summary = data.get("summary", {}).get(model_name, {})
    section_obj = summary.get(section, {})
    summary_key = f"mean_{key}"
    if summary_key in section_obj and valid_number(section_obj[summary_key]):
        return float(section_obj[summary_key])
    if key in section_obj and valid_number(section_obj[key]):
        return float(section_obj[key])
    return None


def model_metric_from_raw_runs(data: dict[str, Any], model_name: str, section: str, key: str) -> float | None:
    values = []
    for run in data.get("runs", []):
        if run.get("model_name") != model_name:
            continue
        value = run.get(section, {}).get(key)
        if valid_number(value):
            values.append(value)
    return avg(values)


def get_model_names(*datasets: dict[str, Any] | None) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for data in datasets:
        if not data:
            continue
        for spec in data.get("meta", {}).get("model_specs", []):
            name = spec.get("name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        for name in data.get("summary", {}).keys():
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        for run in data.get("runs", []):
            name = run.get("model_name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def get_model_label(*datasets: dict[str, Any] | None, model_name: str) -> str:
    for data in datasets:
        if not data:
            continue
        for spec in data.get("meta", {}).get("model_specs", []):
            if spec.get("name") == model_name and spec.get("model_id"):
                return spec["model_id"]
        model_entry = data.get("models", {}).get(model_name)
        if model_entry and model_entry.get("model_id"):
            return model_entry["model_id"]
        for run in data.get("runs", []):
            if run.get("model_name") == model_name and run.get("model_id"):
                return run["model_id"]
    return model_name


def get_metric(data: dict[str, Any] | None, model_name: str, section: str, key: str) -> float | None:
    if not data:
        return None
    value = model_metric_from_summary(data, model_name, section, key)
    if value is not None:
        return value
    return model_metric_from_raw_runs(data, model_name, section, key)


def gp_value(value: float | None) -> str:
    return "NaN" if value is None else f"{value:.6f}"


def describe_dataset(name: str, data: dict[str, Any] | None) -> str:
    """One-line provenance/sanity line printed to the console per input."""
    if not data:
        return f"  {name:22} MISSING/EMPTY -> plotted as NaN"
    meta = data.get("meta", {})
    hw = meta.get("hardware", []) or []
    gpu = hw[0].get("name", "?") if hw else "?"
    return (
        f"  {name:22} {meta.get('inference_engine','?'):12} "
        f"{gpu:22} gpus_seen={len(hw)} gpu_count={meta.get('gpu_count')} "
        f"idle_W={meta.get('idle_power_w')} created={meta.get('created_utc','?')[:19]}"
    )


def write_data_file(out_dir: Path, datasets: dict[str, dict[str, Any] | None], power_key: str) -> Path:
    models = get_model_names(*datasets.values())
    path = out_dir / "combined_eager3.dat"

    header_cols = []
    for suffix in ("tps", "gpu_util", "gpu_power"):
        header_cols.extend(f"{k}_{suffix}" for k in SERIES_KEYS)

    with path.open("w", encoding="utf-8") as f:
        f.write("# model " + " ".join(header_cols) + "\n")
        for model in models:
            row = [f'"{model}"']
            for section, key in [
                ("software", "tokens_per_second"),
                ("hardware", "gpu_util_pct_avg"),
                ("hardware", power_key),
            ]:
                for dataset_name in SERIES_KEYS:
                    row.append(gp_value(get_metric(datasets.get(dataset_name), model, section, key)))
            f.write(" ".join(row) + "\n")

    return path


def write_gnuplot_script(
    out_dir: Path,
    metric_name: str,
    ylabel: str,
    data_file: str,
    output_png: str,
    title: str,
    first_column: int,
    yrange: str | None,
    legend_lines: list[str] | None = None,
) -> Path:
    gp_path = out_dir / f"{metric_name}_bar.gp"
    legend_lines = legend_lines or []
    canvas_height = 650
    canvas_width = 1460 if legend_lines else 1000

    lines = [
        f"set terminal pngcairo size {canvas_width},{canvas_height} enhanced font 'Verdana,16'",
        "set style data histogram",
        "set style histogram cluster gap 1",
        "set style fill solid border -1",
        "set boxwidth 0.9",
    ]
    if yrange:
        lines.append(f"set yrange {yrange}")
    lines.extend(
        [
            f'set ylabel "{ylabel}"',
            'set xlabel "Model"',
            "set xtics rotate by -30",
            # noenhanced: without it gnuplot's enhanced text turns the "_" in
            # "enforce_eager" into a subscript ("enforce_eager" -> "enforce eager").
            "set key outside right noenhanced",
            f'set title "{title}"',
        ]
    )
    for i, legend_line in enumerate(legend_lines):
        y_offset = 0.42 - 0.05 * i
        escaped = legend_line.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'set label "{escaped}" at screen 0.725, screen {y_offset:.3f} left font \'Verdana,9\' front noenhanced'
        )

    plot_parts = []
    for i, (_, label) in enumerate(SERIES):
        col = first_column + i
        source = f"'{data_file}'" if i == 0 else "''"
        plot_parts.append(f"     {source} using {col}:xtic(1) title '{label}'")
    plot_stmt = "plot \\\n" + ", \\\n".join(plot_parts)

    lines.extend([f"set output '{output_png}'", plot_stmt, ""])
    gp_path.write_text("\n".join(lines), encoding="utf-8")
    return gp_path


def run_gnuplot(gp_paths: list[Path], cwd: Path) -> None:
    if shutil.which("gnuplot") is None:
        raise RuntimeError("gnuplot was not found in PATH. Install gnuplot or run the .gp files manually later.")
    for gp in gp_paths:
        subprocess.run(["gnuplot", gp.name], cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-way vLLM-graphs / vLLM-eager / Transformers bar charts.")
    p.add_argument("--vllm-graphs-json", type=Path, required=True,
                   help="vLLM run with enforce_eager=False (CUDA graphs), i.e. the *_no_eager.json")
    p.add_argument("--vllm-eager-json", type=Path, required=True,
                   help="vLLM run with enforce_eager=True")
    p.add_argument("--transformers-json", type=Path, required=True,
                   help="Transformers run")
    p.add_argument("--out-dir", type=Path, default=Path("plots_eager3"))
    p.add_argument("--title", default="vLLM CUDA graphs vs enforce_eager vs Transformers")
    p.add_argument("--power-metric", choices=["dyn", "raw"], default="dyn",
                   help="dyn = gpu_power_w_dyn_avg (measured minus idle, combine.py's behaviour); "
                        "raw = gpu_power_w_avg. Default: dyn")
    p.add_argument("--throughput-yrange", default=None, help="Example: [0:400]. Default: auto")
    p.add_argument("--power-yrange", default=None, help="Example: [0:500]. Default: auto")
    p.add_argument("--run-gnuplot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "vllm_graphs": load_json_or_none(args.vllm_graphs_json),
        "vllm_eager": load_json_or_none(args.vllm_eager_json),
        "transformers": load_json_or_none(args.transformers_json),
    }
    paths = {
        "vllm_graphs": args.vllm_graphs_json,
        "vllm_eager": args.vllm_eager_json,
        "transformers": args.transformers_json,
    }

    print("Inputs:")
    for key, label in SERIES:
        print(describe_dataset(label, datasets[key]))
        print(f"  {'':22} <- {paths[key]}")

    # Sanity warnings that matter for this particular comparison.
    for key, label in SERIES:
        data = datasets[key]
        if not data:
            continue
        meta = data.get("meta", {})
        gpus_seen = len(meta.get("hardware", []) or [])
        if meta.get("gpu_count") not in (None, 1) or (
            gpus_seen > 1 and meta.get("gpu_count") is None
        ):
            print(f"  [warn] {label}: gpu_count={meta.get('gpu_count')} over {gpus_seen} visible GPUs "
                  f"- power/util may be a multi-GPU sum, not comparable with single-GPU runs.")
        for model, summ in data.get("summary", {}).items():
            util = summ.get("hardware", {}).get("mean_gpu_util_pct_avg")
            if valid_number(util) and util < 20:
                print(f"  [warn] {label}/{model}: mean GPU util {util:.1f}% - suspiciously low "
                      f"(work may be sharded onto GPUs that are not the monitored one).")

    engines = {k: (datasets[k] or {}).get("meta", {}).get("inference_engine") for k in SERIES_KEYS}
    if engines["vllm_graphs"] not in (None, "vllm") or engines["vllm_eager"] not in (None, "vllm"):
        print(f"  [warn] expected inference_engine 'vllm' for both vLLM slots, got {engines}")
    if engines["transformers"] not in (None, "transformers"):
        print(f"  [warn] expected inference_engine 'transformers' for the Transformers slot, got {engines}")
    print("  [note] enforce_eager is NOT recorded in these JSON files - which run is eager is taken "
          "from the file you pass, not verified from the data.")

    power_key = "gpu_power_w_dyn_avg" if args.power_metric == "dyn" else "gpu_power_w_avg"
    power_label = ("Average Dynamic Power (W, measured - idle)"
                   if args.power_metric == "dyn" else "Average Power Consumption (W)")

    data_path = write_data_file(args.out_dir, datasets, power_key)

    legend_lines = []
    models = get_model_names(*datasets.values())
    for model in models:
        model_id = get_model_label(*datasets.values(), model_name=model)
        if model_id != model:
            legend_lines.append(f"{model} = {model_id}")

    gp_paths: list[Path] = []
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir, metric_name="throughput", ylabel="Throughput (tokens/s)",
            data_file=data_path.name, output_png="throughput_eager3.png", title=args.title,
            first_column=2, yrange=args.throughput_yrange or "[0:*]", legend_lines=legend_lines,
        )
    )
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir, metric_name="utilization", ylabel="Average Utilization (%)",
            data_file=data_path.name, output_png="utilization_eager3.png", title=args.title,
            first_column=2 + N_SERIES, yrange="[0:100]", legend_lines=legend_lines,
        )
    )
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir, metric_name="power", ylabel=power_label,
            data_file=data_path.name, output_png="power_eager3.png", title=args.title,
            first_column=2 + 2 * N_SERIES, yrange=args.power_yrange or "[0:*]", legend_lines=legend_lines,
        )
    )

    print(f"\nWrote data file: {data_path}")
    for gp in gp_paths:
        print(f"Wrote gnuplot script: {gp}")

    if args.run_gnuplot:
        run_gnuplot(gp_paths, args.out_dir)
        print("Generated PNG files:")
        for name in ["throughput_eager3.png", "utilization_eager3.png", "power_eager3.png"]:
            print(f"  {args.out_dir / name}")
    else:
        print(f"\nTo render: cd {args.out_dir} && gnuplot throughput_bar.gp utilization_bar.gp power_bar.gp")


if __name__ == "__main__":
    main()