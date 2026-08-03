#!/usr/bin/env python3
"""
Create combined comparison bar charts from any number of benchmark JSONs.

Each --series adds one bar per model group, in the order given:

  python3 combine.py \
    --series "JT vLLM=data/jetson/vllm_benchmark_results_jt.json" \
    --series "JT Transformers=data/jetson/transformers_benchmark_results_jt.json" \
    --series "IC2 vLLM=data/a100/vllm_benchmark_results_ic2_2.json" \
    --series "IC2 Transformers=data/a100/transformers_benchmark_results_ic2.json" \
    --series "H200 vLLM=data/h200/vllm_benchmark_results.json" \
    --series "H200 Transformers=data/h200/transformers_benchmark_results.json" \
    --out-dir plots_combined \
    --title "Jetson-Thor vs IC2 vs H200" \
    --run-gnuplot

The old fixed flags (--jt-vllm-json, --jt-transformers-json, --ic2-vllm-json,
--ic2-transformers-json) still work and are equivalent to four --series
entries with the original labels, so existing invocations are unchanged.

If one input file is missing or empty, the script still runs and plots NaN
for that series.  Power bars use gpu_power_w_dyn_avg (idle-subtracted), the
same key the previous version used; pass --power-key raw for the raw
gpu_power_w_avg instead.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


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


def slug(label: str) -> str:
    """Column-header-safe version of a series label."""
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def write_data_file(
    out_dir: Path,
    series: list[tuple[str, dict[str, Any] | None]],
    power_key: str,
) -> Path:
    """One row per model; column blocks are tps, util, power, each with one
    column per series (same order as the bars)."""
    models = get_model_names(*(d for _, d in series))
    path = out_dir / "combined_transformers_vs_vllm.dat"

    metric_cols = [
        ("tps", "software", "tokens_per_second"),
        ("gpu_util", "hardware", "gpu_util_pct_avg"),
        ("gpu_power", "hardware", power_key),
    ]

    with path.open("w", encoding="utf-8") as f:
        header = ["# model"]
        for suffix, _, _ in metric_cols:
            header += [f"{slug(label)}_{suffix}" for label, _ in series]
        f.write(" ".join(header) + "\n")
        for model in models:
            row = [f'"{model}"']
            for _, section, key in metric_cols:
                for _, data in series:
                    row.append(gp_value(get_metric(data, model, section, key)))
            f.write(" ".join(row) + "\n")

    return path


def write_gnuplot_script(
    out_dir: Path,
    metric_name: str,
    ylabel: str,
    data_file: str,
    output_png: str,
    title: str,
    columns: list[int],
    series_labels: list[str],
    yrange: str | None,
    legend_lines: list[str] | None = None,
) -> Path:
    gp_path = out_dir / f"{metric_name}_bar.gp"
    legend_lines = legend_lines or []
    canvas_height = 650
    # same base sizes as before; widen a bit when more than 4 series
    canvas_width = (1320 if legend_lines else 1000) + max(0, len(columns) - 4) * 70

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
            "set key outside right #above #fixed top horizontal Right noreverse noenhanced autotitle nobox",
            f'set title "{title}"',
        ]
    )
    for i, legend_line in enumerate(legend_lines):
        y_offset = 0.42 - 0.05 * i
        escaped = legend_line.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'set label "{escaped}" at screen 0.745, screen {y_offset:.3f} left font \'Verdana,8\' front'
        )
    plot_parts = []
    for i, (col, label) in enumerate(zip(columns, series_labels)):
        src = f"'{data_file}'" if i == 0 else "''"
        safe_label = label.replace("'", "''")
        plot_parts.append(f"{src} using {col}:xtic(1) title '{safe_label}'")
    lines.extend(
        [
            f"set output '{output_png}'",
            "plot " + ", \\\n     ".join(plot_parts),
            "",
        ]
    )
    gp_path.write_text("\n".join(lines), encoding="utf-8")
    return gp_path


def run_gnuplot(gp_paths: list[Path], cwd: Path) -> None:
    if shutil.which("gnuplot") is None:
        raise RuntimeError("gnuplot was not found in PATH. Install gnuplot or run the .gp files manually later.")
    for gp in gp_paths:
        result = subprocess.run(["gnuplot", gp.name], cwd=cwd)
        if result.returncode != 0:
            print(f"[warn] gnuplot failed for {gp.name} (often an all-NaN "
                  f"series, e.g. missing gpu_power_w_dyn_avg in an old "
                  f"JSON - try --power-key raw); other charts still render")


def yrange_arg(text: str) -> str:
    """Accept '[MIN:MAX]' or 'MIN:MAX' and return gnuplot '[MIN:MAX]'.

    MIN/MAX may be numbers or '*' (auto). A fixed MIN and MAX pins the
    axis so plots from different runs are directly comparable.
    """
    inner = text.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    parts = inner.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"y-range must look like MIN:MAX or [MIN:MAX], got: {text!r}")
    for part in (p.strip() for p in parts):
        if part == "*" or part == "":
            continue
        try:
            float(part)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"y-range bounds must be numbers or '*', got: {text!r}")
    return f"[{parts[0].strip() or '*'}:{parts[1].strip() or '*'}]"


def parse_series_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"--series needs LABEL=path.json, got: {text!r}"
        )
    label, _, path = text.partition("=")
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"--series needs LABEL=path.json, got: {text!r}"
        )
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--series",
        action="append",
        type=parse_series_arg,
        default=[],
        metavar="LABEL=JSON",
        help="add one series (bar) per model group; repeatable, order = bar order",
    )
    # legacy fixed flags, kept so old invocations behave identically
    p.add_argument("--jt-vllm-json", type=Path, default=None)
    p.add_argument("--jt-transformers-json", type=Path, default=None)
    p.add_argument("--ic2-vllm-json", type=Path, default=None)
    p.add_argument("--ic2-transformers-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("plots_combined"))
    p.add_argument("--title", default="Transformers vs vLLM")
    p.add_argument("--throughput-yrange", type=yrange_arg, default=None,
                   help="fixed y-axis MIN:MAX (e.g. 0:250 or [0:250]). Default: auto")
    p.add_argument("--util-yrange", type=yrange_arg, default="[0:100]",
                   help="fixed y-axis MIN:MAX for the utilization chart. Default: 0:100")
    p.add_argument("--power-yrange", type=yrange_arg, default=None,
                   help="fixed y-axis MIN:MAX (e.g. 0:600). Default: auto")
    p.add_argument(
        "--power-key",
        choices=["dyn", "raw"],
        default="dyn",
        help="dyn = gpu_power_w_dyn_avg (idle-subtracted, previous behaviour); raw = gpu_power_w_avg",
    )
    p.add_argument("--run-gnuplot", action="store_true")
    args = p.parse_args()

    legacy = [
        ("JT vLLM", args.jt_vllm_json),
        ("JT Transformers", args.jt_transformers_json),
        ("IC2 vLLM", args.ic2_vllm_json),
        ("IC2 Transformers", args.ic2_transformers_json),
    ]
    series: list[tuple[str, Path]] = [(l, p_) for l, p_ in legacy if p_ is not None]
    series += args.series
    if not series:
        p.error("no inputs: pass at least one --series LABEL=path.json")
    labels = [l for l, _ in series]
    if len(set(labels)) != len(labels):
        p.error(f"duplicate series labels: {labels}")
    args.series_resolved = series
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    series: list[tuple[str, dict[str, Any] | None]] = []
    for label, path in args.series_resolved:
        data = load_json_or_none(path)
        if data is None:
            print(f"[warn] {label}: {path} missing/empty/unparsable - plotted as NaN")
        series.append((label, data))

    power_key = "gpu_power_w_dyn_avg" if args.power_key == "dyn" else "gpu_power_w_avg"
    data_path = write_data_file(args.out_dir, series, power_key)

    # flag any series that will be all-NaN for a metric (kills that chart)
    models = get_model_names(*(d for _, d in series))
    for metric_label, section, key in [
        ("throughput", "software", "tokens_per_second"),
        ("utilization", "hardware", "gpu_util_pct_avg"),
        ("power", "hardware", power_key),
    ]:
        for label, data in series:
            if data and all(get_metric(data, m, section, key) is None for m in models):
                print(f"[warn] {label}: no '{key}' anywhere - the {metric_label} "
                      f"chart will fail for this series"
                      + (" (old JSON without dyn power? --power-key raw, or re-run "
                         "the benchmark with the harmonized metrics script)"
                         if key == "gpu_power_w_dyn_avg" else ""))

    legend_lines = []
    models = get_model_names(*(d for _, d in series))
    for model in models:
        model_id = get_model_label(*(d for _, d in series), model_name=model)
        if model_id != model:
            legend_lines.append(f"{model} = {model_id}")

    n = len(series)
    labels = [label for label, _ in series]
    # data-file column blocks: 1 = model, then tps, util, power (n cols each)
    tps_cols = list(range(2, 2 + n))
    util_cols = list(range(2 + n, 2 + 2 * n))
    power_cols = list(range(2 + 2 * n, 2 + 3 * n))

    gp_paths: list[Path] = []
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir,
            metric_name="throughput",
            ylabel="Throughput (tokens/s)",
            data_file=data_path.name,
            output_png="throughput_combined.png",
            title=args.title,
            columns=tps_cols,
            series_labels=labels,
            yrange=args.throughput_yrange or "[0:*]",
            legend_lines=legend_lines,
        )
    )
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir,
            metric_name="utilization",
            ylabel="Average Utilization (%)",
            data_file=data_path.name,
            output_png="utilization_combined.png",
            title=args.title,
            columns=util_cols,
            series_labels=labels,
            yrange=args.util_yrange,
            legend_lines=legend_lines,
        )
    )
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir,
            metric_name="power",
            ylabel="Average Power Consumption (W)",
            data_file=data_path.name,
            output_png="power_combined.png",
            title=args.title,
            columns=power_cols,
            series_labels=labels,
            yrange=args.power_yrange or "[0:*]",
            legend_lines=legend_lines,
        )
    )

    print(f"Wrote data file: {data_path}")
    for gp in gp_paths:
        print(f"Wrote gnuplot script: {gp}")

    if args.run_gnuplot:
        run_gnuplot(gp_paths, args.out_dir)
        print("Generated PNG files:")
        for name in ["throughput_combined.png", "utilization_combined.png", "power_combined.png"]:
            print(f"  {args.out_dir / name}")


if __name__ == "__main__":
    main()