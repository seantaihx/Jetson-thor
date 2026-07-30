"""
Compare vLLM benchmark results: enforce_eager=False (CUDA graphs) vs enforce_eager=True.

Metrics compared per model (llama, gemma, gpt):
  - Tokens per second (throughput)
  - GPU power consumption (avg watts)
  - GPU utilization (avg %)

Usage:
    python3 compare_eager.py <no_eager.json> <eager.json> [output_dir]
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_metrics(data):
    """Pull mean tokens/sec, mean GPU power, mean GPU util per model from a results file."""
    metrics = {}
    for model_name, model_summary in data["summary"].items():
        sw = model_summary["software"]
        hw = model_summary["hardware"]
        metrics[model_name] = {
            "tokens_per_second": sw["mean_tokens_per_second"],
            "tokens_per_second_std": sw["std_tokens_per_second"],
            "gpu_power_w": hw["mean_gpu_power_w_avg"],
            "gpu_power_w_std": hw["std_gpu_power_w_avg"],
            "gpu_util_pct": hw["mean_gpu_util_pct_avg"],
            "gpu_util_pct_std": hw["std_gpu_util_pct_avg"],
            "generation_time_s": sw["mean_generation_time_s"],
            "generated_tokens": sw["mean_generated_token_count"],
        }
    return metrics


def pct_change(no_eager_val, eager_val):
    """% change going from no-eager (CUDA graphs) to eager. Negative = eager is lower."""
    if no_eager_val == 0:
        return float("nan")
    return (eager_val - no_eager_val) / no_eager_val * 100.0


def print_comparison_table(no_eager, eager):
    models = sorted(set(no_eager.keys()) | set(eager.keys()))
    metric_specs = [
        ("tokens_per_second", "Tokens/sec", "{:.2f}"),
        ("gpu_power_w", "GPU Power (W)", "{:.2f}"),
        ("gpu_util_pct", "GPU Util (%)", "{:.2f}"),
    ]

    col_w = 22
    print("=" * (col_w * 4 + 4))
    print("vLLM Benchmark Comparison: CUDA Graphs (no --enforce-eager) vs --enforce-eager")
    print("=" * (col_w * 4 + 4))

    for model in models:
        if model not in no_eager or model not in eager:
            print(f"\n[{model}] missing from one of the files, skipping")
            continue

        print(f"\n--- Model: {model} ---")
        header = f"{'Metric':<18}{'No-Eager (CUDA graph)':<24}{'Eager':<18}{'Change':<14}"
        print(header)
        print("-" * len(header))

        for key, label, fmt in metric_specs:
            v_ne = no_eager[model][key]
            v_e = eager[model][key]
            change = pct_change(v_ne, v_e)
            sign = "+" if change >= 0 else ""
            print(
                f"{label:<18}{fmt.format(v_ne):<24}{fmt.format(v_e):<18}{sign}{change:.1f}%"
            )

    print()


def make_comparison_chart(no_eager, eager, out_path):
    models = sorted(set(no_eager.keys()) & set(eager.keys()))
    if not models:
        print("No common models between the two files; skipping chart.")
        return

    metric_specs = [
        ("tokens_per_second", "tokens_per_second_std", "Tokens / sec (higher = better)"),
        ("gpu_power_w", "gpu_power_w_std", "Avg GPU Power (W)"),
        ("gpu_util_pct", "gpu_util_pct_std", "Avg GPU Utilization (%)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "vLLM on IC2: CUDA Graphs (no --enforce-eager) vs --enforce-eager",
        fontsize=14,
        fontweight="bold",
    )

    x = np.arange(len(models))
    width = 0.35

    colors = {"no_eager": "#2E86AB", "eager": "#E76F51"}

    for ax, (key, std_key, title) in zip(axes, metric_specs):
        ne_vals = [no_eager[m][key] for m in models]
        e_vals = [eager[m][key] for m in models]
        ne_err = [no_eager[m][std_key] for m in models]
        e_err = [eager[m][std_key] for m in models]

        bars1 = ax.bar(
            x - width / 2, ne_vals, width, yerr=ne_err, capsize=4,
            label="No --enforce-eager", color=colors["no_eager"]
        )
        bars2 = ax.bar(
            x + width / 2, e_vals, width, yerr=e_err, capsize=4,
            label="--enforce-eager", color=colors["eager"]
        )

        for bars in (bars1, bars2):
            for b in bars:
                h = b.get_height()
                ax.annotate(
                    f"{h:.1f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.7, -0.12), ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {out_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_eager.py <no_eager.json> <eager.json> [output_dir]")
        sys.exit(1)

    no_eager_path = sys.argv[1]
    eager_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    os.makedirs(out_dir, exist_ok=True)

    no_eager_data = load_results(no_eager_path)
    eager_data = load_results(eager_path)

    no_eager_metrics = extract_metrics(no_eager_data)
    eager_metrics = extract_metrics(eager_data)

    print_comparison_table(no_eager_metrics, eager_metrics)

    chart_path = os.path.join(out_dir, "vllm_eager_vs_noeager_comparison.png")
    make_comparison_chart(no_eager_metrics, eager_metrics, chart_path)


if __name__ == "__main__":
    main()