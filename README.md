# llama-cpp-local

Scripts and configuration for running [llama.cpp](https://github.com/ggml-org/llama.cpp) locally as an OpenAI-compatible inference server for use with Claude Code via [clawgate](https://github.com/goclawgate/clawgate).

## Getting Started (what you need to change)

1. **`tune_ncpu.py` and `verify_vram_256k.py`** — update `LLAMA_EXE` and `MODEL_PATH` at the top of each file to match your llama.cpp install and model path
2. **Sibling directory** — these scripts look for the benchmark fixture at `../llm-bench/shared/fixture_real_request.json`. Clone [llm-bench](https://github.com/drohack/llm-bench) as a sibling directory or update the `FIXTURE` path in the scripts
3. **`run_server.ps1`** — update `$MODEL_PATH` if using a different model

## Setup

### 1. Download llama.cpp

Get the latest release from [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases).

Use `llama-bNNNN-bin-win-cuda-12.4-x64.zip` (bin-only, not the `cudart-` package). Extract to `C:\llama_cpp\`.

**CUDA DLL fix — required:** The bin-only package is missing CUDA runtime DLLs. Download `cudart-llama-bNNNN-bin-win-cuda-12.4-x64.zip` and copy these files into `C:\llama_cpp\` alongside the binaries:
- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cudart64_12.dll`

Without these, llama-server silently falls back to CPU-only. Verify in the startup log: you should see `load_backend: loaded CUDA backend from ggml-cuda.dll`.

Tested with: **b9143**

### 2. Download a model

```powershell
pip install huggingface_hub hf-transfer
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF --include "*IQ3_XXS*" --local-dir C:\llama_cpp\models\
```

Tested model: **Qwen3.6-35B-A3B IQ3_XXS** (12.3 GiB, 3.05 bpw)

### 3. Tune n-cpu-moe for your GPU

The model is too large to fit fully in VRAM. `--n-cpu-moe` offloads MoE expert layers to RAM. Find the optimal value using the binary search script:

```powershell
# 128K context
python tune_ncpu.py --ctx 131072 --lo 20 --hi 30

# 256K context
python tune_ncpu.py --ctx 262144 --lo 28 --hi 40
```

Results are saved to `tuning_results.md`. Verified values for RTX 3080 10GB:

| Context | n-cpu-moe | Peak VRAM |
|---------|-----------|-----------|
| 128K    | 24        | 9,190 MB  |
| 256K    | 31        | 9,184 MB  |

### 4. Start the server

```powershell
.\run_server.ps1
```

Edit `run_server.ps1` to change the model, context size, or n-cpu-moe. See comments in the script.

## Files

| File | Purpose |
|------|---------|
| `run_server.ps1` | Start llama-server with tuned settings |
| `tune_ncpu.py` | Binary search for optimal n-cpu-moe under real load |
| `verify_vram_256k.py` | Verify peak VRAM for a specific n-cpu-moe value |
| `tuning_results.md` | Verified n-cpu-moe values per model/context/GPU |

## Verified performance (RTX 3080 10GB / 96GB DDR4-3400)

- **Generation:** ~55 tok/s (128K context)
- **Warm TTFT:** 0.1s (KV cache hit after attribution header fix)
- **Cold prefill:** ~440 tok/s (~12s for 86K context)
- **VRAM:** 9,190 MB peak under 86K inference load
