#!/usr/bin/env python3
"""
Read vLLM and Transformers benchmark JSON files, calculate model-level averages,
and generate gnuplot bar charts using the same visual settings as the example
throughput_bar.gp and utilization_bar.gp files.

Usage:
  python3 plot_transformers_vs_vllm.py \
    --vllm-json benchmark_out_vllm/vllm_benchmark_results.json \
    --transformers-json benchmark_out_transformers/transformers_benchmark_results.json \
    --out-dir plots \
    --title "Nvidia Jetson-thor"

Then run the generated gnuplot scripts:
  gnuplot plots/throughput_bar.gp
  gnuplot plots/utilization_bar.gp
  gnuplot plots/power_bar.gp

Or, if --run-gnuplot is used and gnuplot is installed, the PNGs are generated automatically.
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


MetricSpec = tuple[str, str, str, str, str | None]
# (short_name, json_section, json_key, ylabel, yrange)
METRICS: list[MetricSpec] = [
    (
        "throughput",
        "software",
        "tokens_per_second",
        "Throughput (tokens/s)",
        None,  # auto unless --throughput-yrange is given
    ),
    (
        "utilization",
        "hardware",
        "gpu_util_pct_avg",
        "Average Utilization (%)",
        "[0:100]",
    ),
    (
        "power",
        "hardware",
        "gpu_power_w_avg",
        "Average Power Consumption (W)",
        None,
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def avg(values: Iterable[Any]) -> float | None:
    nums = [float(v) for v in values if valid_number(v)]
    return mean(nums) if nums else None


def model_metric_from_summary(data: dict[str, Any], model_name: str, section: str, key: str) -> float | None:
    """Prefer the already-computed summary value when it exists."""
    summary = data.get("summary", {}).get(model_name, {})
    section_obj = summary.get(section, {})

    summary_key = f"mean_{key}"
    if summary_key in section_obj and valid_number(section_obj[summary_key]):
        return float(section_obj[summary_key])

    # Fallback for names that are already summary-style.
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


def get_model_names(*datasets: dict[str, Any]) -> list[str]:
    """Use the union of model names while preserving the order in MODEL_SPECS / summary / runs."""
    seen: set[str] = set()
    names: list[str] = []

    for data in datasets:
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


def get_model_label(*datasets: dict[str, Any], model_name: str) -> str:
    """The model_id for a given short model_name, else the raw model_name."""
    for data in datasets:
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


def build_legend_lines(models: list[str], vllm: dict[str, Any], transformers: dict[str, Any]) -> list[str]:
    """One 'short_name = model_id' line per model, for display below the chart."""
    lines = []
    for model in models:
        model_id = get_model_label(vllm, transformers, model_name=model)
        if model_id != model:
            lines.append(f"{model} = {model_id}")
    return lines


def get_metric(data: dict[str, Any], model_name: str, section: str, key: str) -> float | None:
    value = model_metric_from_summary(data, model_name, section, key)
    if value is not None:
        return value
    return model_metric_from_raw_runs(data, model_name, section, key)


def gp_value(value: float | None) -> str:
    # gnuplot understands NaN and skips the missing bar.
    return "NaN" if value is None else f"{value:.6f}"


def write_data_file(out_dir: Path, vllm: dict[str, Any], transformers: dict[str, Any]) -> Path:
    models = get_model_names(vllm, transformers)

    path = out_dir / "transformers_vs_vllm.dat"

    with path.open("w", encoding="utf-8") as f:
        f.write(
            "# model "
            "vllm_tps transformers_tps "
            "vllm_gpu_util transformers_gpu_util "
            "vllm_gpu_power transformers_gpu_power\n"
        )
        for model in models:
            v_tps = get_metric(vllm, model, "software", "tokens_per_second")
            t_tps = get_metric(transformers, model, "software", "tokens_per_second")
            v_util = get_metric(vllm, model, "hardware", "gpu_util_pct_avg")
            t_util = get_metric(transformers, model, "hardware", "gpu_util_pct_avg")
            v_power = get_metric(vllm, model, "hardware", "gpu_power_w_avg")
            t_power = get_metric(transformers, model, "hardware", "gpu_power_w_avg")

            f.write(
                f'"{model}" '
                f"{gp_value(v_tps)} {gp_value(t_tps)} "
                f"{gp_value(v_util)} {gp_value(t_util)} "
                f"{gp_value(v_power)} {gp_value(t_power)}\n"
            )

    return path


def write_gnuplot_script(
    out_dir: Path,
    metric_name: str,
    ylabel: str,
    data_file: str,
    output_png: str,
    title: str,
    columns: tuple[int, int],
    yrange: str | None,
    legend_lines: list[str] | None = None,
) -> Path:
    gp_path = out_dir / f"{metric_name}_bar.gp"
    legend_lines = legend_lines or []

    canvas_height = 600
    canvas_width = 1200 if legend_lines else 900

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
    # Place the model_id legend just below the vLLM/Transformers key box,
    # which sits in the right margin around screen y ~0.55-0.70.
    for i, legend_line in enumerate(legend_lines):
        y_offset = 0.42 - 0.05 * i
        escaped = legend_line.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'set label "{escaped}" at screen 0.745, screen {y_offset:.3f} '
            "left font 'Verdana,8' front"
        )
    lines.extend(
        [
            f"set output '{output_png}'",
            (
                f"plot '{data_file}' using {columns[0]}:xtic(1) title 'vLLM', \\\n"
                f"     '' using {columns[1]}:xtic(1) title 'Transformers'"
            ),
            "",
        ]
    )

    gp_path.write_text("\n".join(lines), encoding="utf-8")
    return gp_path


def run_gnuplot(gp_paths: list[Path], cwd: Path) -> None:
    if shutil.which("gnuplot") is None:
        raise RuntimeError("gnuplot was not found in PATH. Install gnuplot or run the .gp files manually later.")

    for gp in gp_paths:
        subprocess.run(["gnuplot", gp.name], cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vllm-json", type=Path, required=True)
    p.add_argument("--transformers-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("plots_ic2"))
    p.add_argument("--title", default="Transformers vs vLLM")
    p.add_argument("--throughput-yrange", default=None, help="Example: [0:140]. Default: auto")
    p.add_argument("--power-yrange", default=None, help="Example: [0:300]. Default: auto")
    p.add_argument("--run-gnuplot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vllm = load_json(args.vllm_json)
    transformers = load_json(args.transformers_json)

    data_path = write_data_file(args.out_dir, vllm, transformers)
    data_name = data_path.name

    legend_lines = build_legend_lines(get_model_names(vllm, transformers), vllm, transformers)

    gp_paths: list[Path] = []
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir,
            metric_name="throughput",
            ylabel="Throughput (tokens/s)",
            data_file=data_name,
            output_png="throughput_transformers_vs_vllm.png",
            title=args.title,
            columns=(2, 3),
            yrange=args.throughput_yrange or "[0:*]",
            legend_lines=legend_lines,
        )
    )
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir,
            metric_name="utilization",
            ylabel="Average Utilization (%)",
            data_file=data_name,
            output_png="utilization_transformers_vs_vllm.png",
            title=args.title,
            columns=(4, 5),
            yrange="[0:100]",
            legend_lines=legend_lines,
        )
    )
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir,
            metric_name="power",
            ylabel="Average Power Consumption (W)",
            data_file=data_name,
            output_png="power_transformers_vs_vllm.png",
            title=args.title,
            columns=(6, 7),
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
        for name in [
            "throughput_transformers_vs_vllm.png",
            "utilization_transformers_vs_vllm.png",
            "power_transformers_vs_vllm.png",
        ]:
            print(f"  {args.out_dir / name}")


if __name__ == "__main__":
    main()