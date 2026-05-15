"""
Binary search for optimal n-cpu-moe using real 86K fixture load.

Starts conservatively (high n-cpu-moe) and steps down until VRAM
exceeds the limit under actual inference load. Saves results to
tuning_results.md.

Usage:
    python bench/local/tune_ncpu.py --ctx 262144 --lo 26 --hi 36
    python bench/local/tune_ncpu.py --ctx 131072 --lo 20 --hi 26
"""
import argparse, json, subprocess, time, httpx, sys
from pathlib import Path

LLAMA_EXE  = r"C:\llama_cpp\llama-server.exe"
MODEL_PATH = r"C:\llama_cpp\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
LOG_PATH   = str(Path(__file__).parent / "logs" / "llama-server-live.log")
FIXTURE    = str(Path(__file__).parent.parent / "llm-bench" / "shared" / "fixture_real_request.json")
RESULTS_MD = str(Path(__file__).parent / "tuning_results.md")
VRAM_MAX   = 9400
VRAM_KILL  = 9500

def get_vram():
    r = subprocess.run(
        ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    try: return int(r.stdout.strip())
    except: return 0

def kill_server():
    subprocess.run(["taskkill","/F","/IM","llama-server.exe"], capture_output=True)
    time.sleep(4)

def test_ncpu(ncpu, ctx):
    """Start server, send 86K fixture, measure peak VRAM. Returns (peak, safe)."""
    kill_server()
    cmd = [LLAMA_EXE,
           "--model", MODEL_PATH, "--n-gpu-layers","999",
           "--ctx-size", str(ctx), "--parallel","1",
           "--port","8081","--host","0.0.0.0",
           "-ctk","q8_0","-ctv","q8_0",
           "--no-mmap","--mlock","--n-cpu-moe",str(ncpu),
           "--reasoning","off","-b","8192"]
    with open(LOG_PATH,"w") as log:
        proc = subprocess.Popen(cmd, stderr=log, stdout=subprocess.DEVNULL)

    # Wait for ready
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            if httpx.get("http://127.0.0.1:8081/health", timeout=2).status_code == 200:
                break
        except: pass
        time.sleep(3)
    else:
        print(f"  timeout waiting for server")
        kill_server()
        return 0, False

    # Send real 86K fixture
    print(f"  sending 86K fixture...", end=" ", flush=True)
    fix = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))
    payload = {"model":"local","messages":fix["messages"],
               "tools":fix.get("tools",[]),"max_tokens":10}
    peak = get_vram()
    killed = False

    def send():
        try: httpx.post("http://127.0.0.1:8081/v1/chat/completions",
                        json=payload, timeout=300)
        except: pass

    import threading
    t = threading.Thread(target=send, daemon=True)
    t.start()

    # Monitor during inference
    for _ in range(24):  # 2 min max
        time.sleep(5)
        v = get_vram()
        if v > peak: peak = v
        if v >= VRAM_KILL:
            print(f"KILLED at {v} MB", flush=True)
            kill_server()
            killed = True
            break
        if not t.is_alive():
            break

    if not killed:
        v = get_vram()
        if v > peak: peak = v

    kill_server()
    safe = peak <= VRAM_MAX and not killed
    print(f"peak={peak} MB  {'SAFE' if safe else 'OVER'}")
    return peak, safe

def update_md(ncpu, ctx_k, peak, safe, method):
    path = Path(RESULTS_MD)
    content = path.read_text(encoding="utf-8")
    row = f"| {time.strftime('%Y-%m-%d')} | IQ3_XXS | {ctx_k}K | {ncpu} | {peak:,} MB | {method} |"
    if row not in content:
        content = content.rstrip() + "\n" + row + "\n"
        path.write_text(content, encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx",  type=int, default=262144)
    p.add_argument("--lo",   type=int, default=26, help="lowest n-cpu-moe to try")
    p.add_argument("--hi",   type=int, default=36, help="highest n-cpu-moe (known safe)")
    args = p.parse_args()
    ctx_k = args.ctx // 1024

    print(f"\n=== Binary search: {ctx_k}K context, n-cpu-moe in [{args.lo}, {args.hi}] ===")
    print(f"Testing under real 86K fixture load. VRAM limit: {VRAM_MAX} MB\n")

    lo, hi = args.lo, args.hi
    best_ncpu, best_vram = hi, None  # hi is known safe starting point

    while hi - lo > 2:
        mid = (lo + hi) // 2
        print(f"n-cpu-moe={mid} ...", end=" ", flush=True)
        peak, safe = test_ncpu(mid, args.ctx)
        time.sleep(5)
        if safe:
            best_ncpu, best_vram = mid, peak
            hi = mid
            print(f"  -> fits, trying lower")
        else:
            lo = mid
            print(f"  -> over budget, trying higher")

    print(f"\n{'='*50}")
    print(f"OPTIMAL for {ctx_k}K: n-cpu-moe={best_ncpu}  peak={best_vram} MB")
    update_md(best_ncpu, ctx_k, best_vram,  True, f"binary search, 86K fixture, verified")
    print(f"Saved to tuning_results.md")

if __name__ == "__main__":
    main()
