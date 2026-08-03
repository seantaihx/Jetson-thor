#!/usr/bin/env python3
"""
Bar charts (throughput / utilization / power) built from step_metric
summaries instead of the metrics_vllm/metrics_transformers benchmark.

Why: the 1Bar-Plot benchmark and the step_metric profiler run DIFFERENT
workloads (batch 1 x up-to-4096 tokens vs batch 4 x exactly 256 tokens),
so their power numbers are not comparable. The step_metric sweeps were
run with IDENTICAL settings for vLLM and Transformers on every device,
so bars built from them are matched by construction - no new GPU runs.

Each --series points at a directory of step_metric outputs
(<model>_metrics_summary.json files, e.g. 2Roofline-Plot/data/h200_vllm):

  python3 bars_from_step.py \
    --series "JT vLLM=data/jetson_vllm" \
    --series "JT Transformers=data/jetson_transformers" \
    --series "IC2 vLLM=data/a100_vllm" \
    --series "IC2 Transformers=data/a100_transformers" \
    --series "H200 vLLM=data/h200_vllm" \
    --series "H200 Transformers=data/h200_transformers" \
    --out-dir plots_step_bars --title "Decode (batch 4, 256 tokens)" \
    --run-gnuplot

Metrics plotted (decode phase by default, --phase prefill for prefill):
  throughput  = phase_totals tokens_per_second (time-weighted aggregate)
  utilization = gpu_util_pct_avg.avg
  power       = gpu_power_w_dyn.avg (idle-subtracted; --power-key raw
                for gpu_power_w_avg)

A missing directory/file plots as NaN, same as combine.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def valid_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_summaries(directory: Path) -> dict[str, dict[str, Any]]:
    """Map short model name (file prefix before '_') -> summary dict."""
    out: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*_summary.json")):
        short = path.name.split("_")[0]
        try:
            with path.open("r", encoding="utf-8") as f:
                out[short] = json.load(f)
        except json.JSONDecodeError:
            print(f"[warn] unparsable summary skipped: {path}")
    return out


def phase_metric(summary: dict[str, Any] | None, phase: str, metric: str) -> float | None:
    """metric is 'tps' | 'util' | 'power_dyn' | 'power_raw'."""
    if not summary or phase not in summary:
        return None
    ph = summary[phase]
    try:
        if metric == "tps":
            v = ph["phase_totals"]["tokens_per_second"]
        elif metric == "util":
            v = ph["gpu_util_pct_avg"]["avg"]
        elif metric == "power_dyn":
            v = ph["gpu_power_w_dyn"]["avg"]
        elif metric == "power_raw":
            v = ph["gpu_power_w_avg"]["avg"]
        else:
            return None
    except (KeyError, TypeError):
        return None
    return float(v) if valid_number(v) else None


def model_id_of(summary: dict[str, Any] | None) -> str | None:
    if not summary:
        return None
    return summary.get("config", {}).get("model")


def gp_value(v: float | None) -> str:
    return "NaN" if v is None else f"{v:.6f}"


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def yrange_arg(text: str) -> str:
    inner = text.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    parts = inner.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"y-range must look like MIN:MAX or [MIN:MAX], got: {text!r}")
    for part in (p.strip() for p in parts):
        if part in ("*", ""):
            continue
        try:
            float(part)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"y-range bounds must be numbers or '*', got: {text!r}")
    return f"[{parts[0].strip() or '*'}:{parts[1].strip() or '*'}]"


def parse_series_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--series needs LABEL=dir, got: {text!r}")
    label, _, path = text.partition("=")
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(f"--series needs LABEL=dir, got: {text!r}")
    return label, Path(path)


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
            print(f"[warn] gnuplot failed for {gp.name} (often an all-NaN series); "
                  f"other charts still render")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--series", action="append", type=parse_series_arg, default=[],
                   metavar="LABEL=DIR", required=True,
                   help="directory of <model>_metrics_summary.json files; "
                        "repeatable, order = bar order")
    p.add_argument("--phase", choices=["decode", "prefill"], default="decode",
                   help="which step_metric phase to plot (default decode)")
    p.add_argument("--out-dir", type=Path, default=Path("plots_step_bars"))
    p.add_argument("--title", default=None,
                   help="default: '<Phase> phase (step_metric, batch 4 x 256 tok)'")
    p.add_argument("--throughput-yrange", type=yrange_arg, default=None,
                   help="fixed y-axis MIN:MAX (e.g. 0:250). Default: auto")
    p.add_argument("--util-yrange", type=yrange_arg, default="[0:100]",
                   help="fixed y-axis MIN:MAX for utilization. Default: 0:100")
    p.add_argument("--power-yrange", type=yrange_arg, default=None,
                   help="fixed y-axis MIN:MAX (e.g. 0:600). Default: auto")
    p.add_argument("--power-key", choices=["dyn", "raw"], default="dyn",
                   help="dyn = idle-subtracted gpu_power_w_dyn (default); raw = gpu_power_w_avg")
    p.add_argument("--run-gnuplot", action="store_true")
    args = p.parse_args()
    labels = [l for l, _ in args.series]
    if len(set(labels)) != len(labels):
        p.error(f"duplicate series labels: {labels}")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    series: list[tuple[str, dict[str, dict[str, Any]]]] = []
    for label, directory in args.series:
        summaries = load_summaries(directory)
        if not summaries:
            print(f"[warn] {label}: no *_summary.json in {directory} - plotted as NaN")
        series.append((label, summaries))

    # model order: canonical pipeline order first, then anything else as seen
    canonical = ["llama", "gemma", "gpt", "qwen"]
    models: list[str] = []
    seen: set[str] = set()
    for _, summaries in series:
        seen.update(summaries)
    models = [m for m in canonical if m in seen]
    for _, summaries in series:
        for m in summaries:
            if m not in models:
                models.append(m)
    if not models:
        raise SystemExit("[error] no summaries found in any series directory")

    power_metric = "power_dyn" if args.power_key == "dyn" else "power_raw"
    metric_cols = [("tps", "tps"), ("gpu_util", "util"), ("gpu_power", power_metric)]

    data_path = args.out_dir / "step_bars.dat"
    with data_path.open("w", encoding="utf-8") as f:
        header = ["# model"]
        for suffix, _ in metric_cols:
            header += [f"{slug(label)}_{suffix}" for label, _ in series]
        f.write(" ".join(header) + "\n")
        for model in models:
            row = [f'"{model}"']
            for _, metric in metric_cols:
                for _, summaries in series:
                    row.append(gp_value(
                        phase_metric(summaries.get(model), args.phase, metric)))
            f.write(" ".join(row) + "\n")

    # all-NaN early warning, one line per bad series x metric
    for (suffix, metric) in metric_cols:
        for label, summaries in series:
            if summaries and all(
                phase_metric(summaries.get(m), args.phase, metric) is None
                for m in models
            ):
                print(f"[warn] {label}: no '{metric}' data - the {suffix} chart "
                      f"will fail for this series")

    legend_lines = []
    for model in models:
        for _, summaries in series:
            mid = model_id_of(summaries.get(model))
            if mid and mid != model:
                legend_lines.append(f"{model} = {mid}")
                break

    # no underscores in the default title: gnuplot 'enhanced' mode would
    # typeset them as subscripts
    title = args.title or (
        f"{args.phase.capitalize()} phase (step-metric runs, batch 4 x 256 tok)"
    )
    power_label = ("Average Dynamic Power (W)" if args.power_key == "dyn"
                   else "Average Power Consumption (W)")

    n = len(series)
    labels = [label for label, _ in series]
    tps_cols = list(range(2, 2 + n))
    util_cols = list(range(2 + n, 2 + 2 * n))
    power_cols = list(range(2 + 2 * n, 2 + 3 * n))

    gp_paths = [
        write_gnuplot_script(args.out_dir, "throughput", "Throughput (tokens/s)",
                             data_path.name, "throughput_step.png", title,
                             tps_cols, labels,
                             args.throughput_yrange or "[0:*]", legend_lines),
        write_gnuplot_script(args.out_dir, "utilization", "Average Utilization (%)",
                             data_path.name, "utilization_step.png", title,
                             util_cols, labels, args.util_yrange, legend_lines),
        write_gnuplot_script(args.out_dir, "power", power_label,
                             data_path.name, "power_step.png", title,
                             power_cols, labels,
                             args.power_yrange or "[0:*]", legend_lines),
    ]

    print(f"Wrote data file: {data_path}")
    for gp in gp_paths:
        print(f"Wrote gnuplot script: {gp}")
    if args.run_gnuplot:
        run_gnuplot(gp_paths, args.out_dir)
        print("Generated PNG files:")
        for name in ["throughput_step.png", "utilization_step.png", "power_step.png"]:
            print(f"  {args.out_dir / name}")


if __name__ == "__main__":
    main()