# Jetson-thor
# LLM Benchmark — NVIDIA Jetson Thor

Benchmarks Llama 3.1 8B, Gemma 4 E4B, and GPT-OSS 20B using vLLM
on NVIDIA Jetson hardware, measuring tokens/sec, GPU utilization, and power.

## Requirements

### Hardware
- NVIDIA GPU (tested on NVIDIA Jetson Thor)
- Linux only (vLLM does not support Windows or Mac)
- Minimum 60 GiB RAM recommended for all 3 models

### Python
- Python 3.10 - 3.12

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourname/yourrepo.git
cd yourrepo

# 2. Create a virtual environment
python3 -m venv benchmark-env
source benchmark-env/bin/activate

# 3. Install dependencies
pip install vllm pynvml psutil
```

## Environment Variables (required before running)

Find your CUDA path first:
```bash
find /usr/local -name "nvcc" 2>/dev/null
# OR if using pip-installed nvcc:
find ~/.local -name "nvcc" 2>/dev/null
```

Then export:
```bash
export CUDA_HOME=/path/to/your/cuda        # e.g. /usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0       # avoids JIT compile issues, and CUDA version mismatch
```

To make permanent, add those lines to ~/.bashrc and run `source ~/.bashrc`

## HuggingFace Access

Some models require HuggingFace authentication:

```bash
pip install huggingface_hub
huggingface-cli login
# paste your token from https://huggingface.co/settings/tokens
```

Models that require accepting license on HuggingFace website first:
- meta-llama/Meta-Llama-3.1-8B-Instruct → https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- google/gemma-4-E4B-it → https://huggingface.co/google/gemma-4-E4B-it
- openai/gpt-oss-20b → https://huggingface.co/openai/gpt-oss-20b

## Running

Basic run:
```bash
python get_metrics3.py --gpu-memory-utilization 0.6
```

If you get `sm_XXX not defined` Triton error (Blackwell/Jetson 
```bash
python get_metrics3.py --gpu-memory-utilization 0.6 --enforce-eager
```

Full options:
```bash
python get_metrics3.py \
  --gpu-memory-utilization 0.6 \
  --runs 3 \
  --warmup-runs 1 \
  --max-tokens 4096 \
  --out-dir my_results
```

## Output

Results saved to `benchmark_out/benchmark_results.json`
Summary table saved to `benchmark_out/summary.tsv`

## Common Errors

| Error | Fix |
|-------|-----|
| `sm_110a not defined` | Add `--enforce-eager` |
| `Free memory less than desired` | Lower `--gpu-memory-utilization` |
| `Could not find nvcc` | Set `CUDA_HOME` env var |
| `KV cache memory not enough` | Add `--max-model-len (whatever better number)` |
| Memory full between models | drop cache

## Tested On

- Hardware: NVIDIA Jetson Thor (122 GiB unified memory)
- OS: Ubuntu 24.04 (aarch64)
- vLLM: 0.22.1
- Python: 3.12