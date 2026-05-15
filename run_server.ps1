# Launch llama.cpp server with OpenAI-compatible API
# Edit LLAMA_EXE and MODEL_PATH before running.

# Standard llama.cpp b9118 -- has SWA hybrid attention cache fix (no turbo KV types)
$LLAMA_EXE  = "C:\llama_cpp\llama-server.exe"
# Turboquant v0.1.1 -- has turbo4/turbo3 KV types but NOT the SWA fix
# $LLAMA_EXE  = "C:\llama_cpp_turbo\llama-server.exe"

# Qwen3.6-35B-A3B MoE IQ3_XXS (12.8 GB) -- fastest gen (46 tok/s), best EvalPlus (92.7%)  [RECOMMENDED]
$MODEL_PATH = "C:\llama_cpp\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
$NGL = 999

# Qwen3.6-35B-A3B MoE IQ4_XS (17.7 GB) -- slightly lower quality, slower (32 tok/s)
# $MODEL_PATH = "C:\llama_cpp\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
# $NGL = 999

# Qwen3.5-122B-A10B MoE IQ2_XXS (36.6 GB) -- 10B active params; experts in RAM, backbone on GPU
# $MODEL_PATH = "C:\llama_cpp\models\Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf"
# $NGL = 999

# Llama 4 Scout MoE IQ2_XXS (37.4 GB) -- 17B active params; experts in RAM, backbone on GPU
# Requires turboquant binary to support Llama 4 architecture.
# $MODEL_PATH = "C:\llama_cpp\models\Llama-4-Scout-17B-16E-Instruct-UD-IQ2_XXS.gguf"
# $NGL = 999

# Gemma 4 26B-A4B MoE IQ4_XS (13.6 GB) -- needs --jinja for tool calling
# $MODEL_PATH = "C:\llama_cpp\models\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
# $NGL = 999

# Context size.
# 131072 (128K): n-cpu-moe=24, VRAM=9,190 MB peak  [default, verified under 86K load]
# 262144 (256K): n-cpu-moe=31, VRAM=9,184 MB peak  [swap CTX and update n-cpu-moe below]
$CTX = 131072

# Number of parallel request slots (1 is fine for single-user dev).
$PARALLEL = 1

# Add --jinja when running Gemma 4 (required for its tool-calling chat template).
# $EXTRA_ARGS = "--jinja"
$EXTRA_ARGS = ""

# --- Log file config ---
# Output (stderr, which is where llama-server writes request/token/timing info)
# is tee'd to both the terminal and this log file so you can review it later.
# Log rotates when it exceeds MAX_LOG_MB: the old file is kept as .old (one backup).
$LOG_PATH    = Join-Path $PSScriptRoot "logs\llama-server-live.log"
$LOG_OLD     = Join-Path $PSScriptRoot "logs\llama-server-live.log.old"
$MAX_LOG_MB  = 50

if (-not (Test-Path $LLAMA_EXE)) {
    Write-Error "llama-server.exe not found at: $LLAMA_EXE"
    Write-Host "Download from: https://github.com/ggerganov/llama.cpp/releases"
    exit 1
}

if (-not (Test-Path $MODEL_PATH)) {
    Write-Error "Model not found at: $MODEL_PATH"
    exit 1
}

# Rotate log if it has grown too large
New-Item -ItemType Directory -Force -Path (Split-Path $LOG_PATH) | Out-Null
if (Test-Path $LOG_PATH) {
    $sizeMB = (Get-Item $LOG_PATH).Length / 1MB
    if ($sizeMB -gt $MAX_LOG_MB) {
        Write-Host "Log exceeded ${MAX_LOG_MB}MB -- rotating to .old"
        Move-Item -Force $LOG_PATH $LOG_OLD
    }
}

# Write session header to log
$header = @"
================================================================================
  llama-server session started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
  Model:   $MODEL_PATH
  Context: $CTX tokens
================================================================================
"@
Add-Content -Path $LOG_PATH -Value $header -Encoding UTF8

Write-Host "Starting llama-server on http://localhost:8081"
Write-Host "  Model:       $MODEL_PATH"
Write-Host "  GPU layers:  $NGL"
Write-Host "  Context:     $CTX"
Write-Host "  Log:         $LOG_PATH  (rotates at ${MAX_LOG_MB}MB)"
Write-Host ""

$cmd = @(
    "--model", $MODEL_PATH,
    "--n-gpu-layers", $NGL,
    "--ctx-size", $CTX,
    "--parallel", $PARALLEL,
    "--port", "8081",
    "--host", "0.0.0.0",
    "-ctk", "q8_0",    # standard llama.cpp: use q8_0 (turbo4/3 only in turboquant fork)
    "-ctv", "q8_0",
    "--no-mmap",
    "--mlock",
    "--n-cpu-moe", "24",   # IQ3_XXS 128K: verified 2026-05-14, 9,190 MB peak under 86K load
                           # Use 31 for 256K context (verified 2026-05-14, 9,184 MB peak)
    "--reasoning", "auto",  # auto-detect from template; proxy injects /no_think on normal
                            # turns and lets model think on error/planning turns
    "-b", "8192"           # larger batch = faster prefill on long prompts
                           # Test b=16384 with bench/test_batch_compact.py (use n-cpu-moe=26)
)
if ($EXTRA_ARGS) { $cmd += $EXTRA_ARGS.Split(" ") }

# Launch llama-server with stderr redirected to the log file.
# PS 5.1 can't cleanly tee native stderr to both terminal and file, so we run the
# process with stderr going to the log, and tail the log in a background job so
# you still see output in the terminal.
$proc = Start-Process -FilePath $LLAMA_EXE `
    -ArgumentList $cmd `
    -RedirectStandardError $LOG_PATH `
    -NoNewWindow -PassThru

Write-Host "PID $($proc.Id) -- tailing log (Ctrl+C to stop)..."
Write-Host ""

# Tail the log file to the terminal so you can see what the server is doing
Get-Content $LOG_PATH -Wait -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host $_
    if ($proc.HasExited) { break }
}
