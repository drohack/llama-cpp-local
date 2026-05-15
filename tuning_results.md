# n-cpu-moe Tuning Results

Hardware: RTX 3080 10GB VRAM, 96GB DDR4-3400 RAM  
Binary: llama.cpp b9143 (standard, not turboquant)  
KV type: q8_0 (turbo4/turbo3 not available in standard llama.cpp)  
Method: binary search using real 86K Claude Code fixture, kill switch at 9,500 MB

## Qwen3.6-35B-A3B IQ3_XXS (12.29 GiB, 3.05 bpw) — VERIFIED 2026-05-14

| Context | n-cpu-moe | Peak VRAM | Gen tok/s | Notes |
|---------|-----------|-----------|-----------|-------|
| 128K (131072) | **24** | 9,190 MB | 55.5 | KV cache = 1,360 MB |
| 256K (262144) | **31** | 9,184 MB | 47.5 | KV cache = 2,720 MB, -14% gen speed |

**How to switch context in run_server.ps1:**
- 128K: `$CTX = 131072`, `--n-cpu-moe 24`
- 256K: `$CTX = 262144`, `--n-cpu-moe 31`

## Qwen3.6-35B-A3B IQ4_XS (17.7 GiB) — NOT YET RE-TUNED FOR B9143

| Context | n-cpu-moe | VRAM | Notes |
|---------|-----------|------|-------|
| 128K | ~28 (stale) | unknown | Needs re-tuning with b9143 + q8_0 KV |

## Qwen3.5-122B-A10B IQ2_XXS (36.6 GiB) — NOT RE-TUNED FOR B9143

| Context | n-cpu-moe | VRAM | Notes |
|---------|-----------|------|-------|
| 128K | 999 (all CPU) | minimal | Backbone on GPU, all experts in RAM |

## Notes

- n-cpu-moe controls how many MoE expert layers are computed on CPU. Lower = more GPU = faster but more VRAM.
- KV cache size scales linearly with context: 128K = 1,360 MB, 256K = 2,720 MB (q8_0, 10 attention layers)
- After changing context size, always re-tune n-cpu-moe — the KV cache eats into model VRAM budget
- Tuning script: `bench/local/tune_ncpu.py` — binary search with real 86K fixture and kill switch
- Old method (20s wait, no real load) gave inaccurate readings — always use tune_ncpu.py

## Tuning History

| Date | Model | Ctx | n-cpu-moe | Peak VRAM | Method |
|------|-------|-----|-----------|-----------|--------|
| 2026-05-13 | IQ3_XXS | 128K | 26 | 8,956 MB | Auto-tuner, no load test (stale) |
| 2026-05-14 | IQ3_XXS | 128K | **24** | **9,190 MB** | Binary search, 86K fixture ✓ |
| 2026-05-14 | IQ3_XXS | 256K | **31** | **9,184 MB** | Binary search, 86K fixture ✓ |
