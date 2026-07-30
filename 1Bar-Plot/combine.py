#!/usr/bin/env python3
"""
Create combined comparison bar charts for:
  - Jetson-Thor vLLM
  - Jetson-Thor Transformers
  - IC2 vLLM
  - IC2 Transformers

This keeps the same chart style as the original script, but puts all four
datasets on each graph so you can compare them directly.

Example:
  python3 combined_plot_transformers_vs_vllm.py \
    --jt-vllm-json /Users/seantai/Downloads/vllm_benchmark_results_jt.json \
    --jt-transformers-json /Users/seantai/Downloads/transformers_benchmark_results_jt.json \
    --ic2-vllm-json /Users/seantai/Downloads/vllm_benchmark_results_ic2_EnforceEager.json \
    --ic2-transformers-json /Users/seantai/Downloads/transformers_benchmark_results_ic2.json \
    --out-dir plots_combined \
    --title "Jetson-Thor vs IC2"

If one input file is missing or empty, the script will still run and plot NaN for
that series.
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
METRICS: list[MetricSpec] = [
    ("throughput", "software", "tokens_per_second", "Throughput (tokens/s)", None),
    ("utilization", "hardware", "gpu_util_pct_avg", "Average Utilization (%)", "[0:100]"),
    ("power", "hardware", "gpu_power_w_avg", "Average Power Consumption (W)", None),
]

SERIES = [
    ("jt_vllm", "Jetson-Thor vLLM"),
    ("jt_transformers", "Jetson-Thor Transformers"),
    ("ic2_vllm", "IC2 vLLM"),
    ("ic2_transformers", "IC2 Transformers"),
]


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


def write_data_file(
    out_dir: Path,
    datasets: dict[str, dict[str, Any] | None],
) -> Path:
    models = get_model_names(*datasets.values())
    path = out_dir / "combined_transformers_vs_vllm.dat"

    with path.open("w", encoding="utf-8") as f:
        f.write(
            "# model "
            "jt_vllm_tps jt_transformers_tps ic2_vllm_tps ic2_transformers_tps "
            "jt_vllm_gpu_util jt_transformers_gpu_util ic2_vllm_gpu_util ic2_transformers_gpu_util "
            "jt_vllm_gpu_power jt_transformers_gpu_power ic2_vllm_gpu_power ic2_transformers_gpu_power\n"
        )
        for model in models:
            row = [f'"{model}"']
            for section, key in [
                ("software", "tokens_per_second"),
                ("hardware", "gpu_util_pct_avg"),
                ("hardware", "gpu_power_w_dyn_avg"),
            ]:
                for dataset_name in ["jt_vllm", "jt_transformers", "ic2_vllm", "ic2_transformers"]:
                    value = get_metric(datasets.get(dataset_name), model, section, key)
                    row.append(gp_value(value))
            f.write(" ".join(row) + "\n")

    return path


def write_gnuplot_script(
    out_dir: Path,
    metric_name: str,
    ylabel: str,
    data_file: str,
    output_png: str,
    title: str,
    columns: tuple[int, int, int, int],
    yrange: str | None,
    legend_lines: list[str] | None = None,
) -> Path:
    gp_path = out_dir / f"{metric_name}_bar.gp"
    legend_lines = legend_lines or []
    canvas_height = 650
    canvas_width = 1320 if legend_lines else 1000

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
    lines.extend(
        [
            f"set output '{output_png}'",
            (
                f"plot '{data_file}' using {columns[0]}:xtic(1) title 'JT vLLM', \\\n"
                f"     '' using {columns[1]}:xtic(1) title 'JT Transformers', \\\n"
                f"     '' using {columns[2]}:xtic(1) title 'IC2 vLLM', \\\n"
                f"     '' using {columns[3]}:xtic(1) title 'IC2 Transformers'"
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
    p.add_argument("--jt-vllm-json", type=Path, required=True)
    p.add_argument("--jt-transformers-json", type=Path, required=True)
    p.add_argument("--ic2-vllm-json", type=Path, required=True)
    p.add_argument("--ic2-transformers-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("plots_combined"))
    p.add_argument("--title", default="Jetson-Thor vs IC2")
    p.add_argument("--throughput-yrange", default=None, help="Example: [0:140]. Default: auto")
    p.add_argument("--power-yrange", default=None, help="Example: [0:300]. Default: auto")
    p.add_argument("--run-gnuplot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "jt_vllm": load_json_or_none(args.jt_vllm_json),
        "jt_transformers": load_json_or_none(args.jt_transformers_json),
        "ic2_vllm": load_json_or_none(args.ic2_vllm_json),
        "ic2_transformers": load_json_or_none(args.ic2_transformers_json),
    }

    data_path = write_data_file(args.out_dir, datasets)
    legend_lines = []
    models = get_model_names(*datasets.values())
    for model in models:
        model_id = get_model_label(*datasets.values(), model_name=model)
        if model_id != model:
            legend_lines.append(f"{model} = {model_id}")

    gp_paths: list[Path] = []
    gp_paths.append(
        write_gnuplot_script(
            args.out_dir,
            metric_name="throughput",
            ylabel="Throughput (tokens/s)",
            data_file=data_path.name,
            output_png="throughput_combined.png",
            title=args.title,
            columns=(2, 3, 4, 5),
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
            columns=(6, 7, 8, 9),
            yrange="[0:100]",
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
            columns=(10, 11, 12, 13),
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
