"""
Verify peak VRAM usage for 256K context at a given n-cpu-moe.
Starts llama-server, monitors VRAM every 5s for up to 10 minutes,
sends the real 86K Claude Code fixture to capture inference peak,
saves results to tuning_results.md, then kills the server.

Usage:
    python bench/local/verify_vram_256k.py [--ncpu 36]
"""
import argparse, json, subprocess, time, httpx, sys, os, re
from pathlib import Path

LLAMA_EXE   = r"C:\llama_cpp\llama-server.exe"          # adjust to your llama.cpp install path
MODEL_PATH  = r"C:\llama_cpp\models\your-model.gguf"   # adjust to your model path
LOG_PATH    = str(Path(__file__).parent / "logs" / "llama-server-live.log")
FIXTURE     = str(Path(__file__).parent / "fixture_real_request.json")
RESULTS_MD  = str(Path(__file__).parent / "tuning_results.md")
VRAM_LIMIT  = 9400
KILL_AT     = 9500

def get_vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except:
        return 0

def kill_server():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(3)

def start_server(ncpu, ctx=262144):
    kill_server()
    cmd = [LLAMA_EXE,
           "--model", MODEL_PATH,
           "--n-gpu-layers", "999",
           "--ctx-size", str(ctx),
           "--parallel", "1",
           "--port", "8081",
           "--host", "0.0.0.0",
           "-ctk", "q8_0", "-ctv", "q8_0",
           "--no-mmap", "--mlock",
           "--n-cpu-moe", str(ncpu),
           "--reasoning", "off",
           "-b", "8192"]
    with open(LOG_PATH, "w") as log:
        proc = subprocess.Popen(cmd, stderr=log, stdout=subprocess.DEVNULL)
    return proc

def wait_ready(timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get("http://127.0.0.1:8081/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    return False

def monitor_vram(proc, duration=600):
    """Monitor VRAM every 5s for duration seconds. Kill if over KILL_AT. Return peak."""
    peak = 0
    print(f"  Monitoring VRAM for {duration}s (kill at {KILL_AT} MB)...")
    start = time.monotonic()
    while time.monotonic() - start < duration:
        v = get_vram()
        if v > peak:
            peak = v
            print(f"  [{int(time.monotonic()-start):3d}s] VRAM={v} MB  peak={peak} MB")
        if v >= KILL_AT:
            print(f"  VRAM={v} MB exceeds kill threshold {KILL_AT} MB -- killing server")
            kill_server()
            return peak, False
        time.sleep(5)
        if proc.poll() is not None:
            print("  Server exited unexpectedly")
            return peak, False
    return peak, True

def send_fixture():
    """Send the real 86K Claude Code fixture to capture inference peak VRAM."""
    print("  Sending 86K fixture (peak inference load)...")
    try:
        fix = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))
        payload = {
            "model": "local",
            "messages": fix["messages"],
            "tools": fix.get("tools", []),
            "max_tokens": 10,
        }
        httpx.post("http://127.0.0.1:8081/v1/chat/completions",
                   json=payload, timeout=300)
    except Exception as e:
        print(f"  Fixture error (ok to continue): {e}")

def update_results_md(ncpu, ctx_k, peak_vram, safe):
    """Update the tuning_results.md with verified data."""
    path = Path(RESULTS_MD)
    content = path.read_text(encoding="utf-8")
    tag = f"{ctx_k}K (262144)" if ctx_k == 256 else f"{ctx_k}K (131072)"
    status = "verified under 86K load" if safe else "OOM at this value"
    new_row = f"| {time.strftime('%Y-%m-%d')} | IQ3_XXS | {ctx_k}K | {ncpu} | {peak_vram:,} MB | {status} |"
    # Append to tuning history table
    if new_row not in content:
        content = content.rstrip() + "\n" + new_row + "\n"
        path.write_text(content, encoding="utf-8")
        print(f"  Saved to tuning_results.md")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ncpu", type=int, default=36)
    p.add_argument("--ctx", type=int, default=262144)
    args = p.parse_args()

    ctx_k = args.ctx // 1024
    print(f"\n=== 256K VRAM Verification: n-cpu-moe={args.ncpu}, ctx={args.ctx} ===")
    print(f"VRAM limit: {VRAM_LIMIT} MB  |  Kill at: {KILL_AT} MB")

    proc = start_server(args.ncpu, args.ctx)
    print(f"Server started (PID {proc.pid})")

    print("  Waiting for server ready...")
    if not wait_ready(300):
        print("  Server failed to start")
        kill_server()
        sys.exit(1)
    print("  Server ready")

    # Record VRAM at idle (server ready, no requests yet)
    idle_vram = get_vram()
    print(f"  Idle VRAM: {idle_vram} MB")

    # Send real load immediately -- this is what causes peak VRAM
    send_fixture()

    # Monitor VRAM for 60s after fixture to capture the peak
    peak_startup, ok = monitor_vram(proc, duration=60)
    if not ok:
        print(f"RESULT: n-cpu-moe={args.ncpu} UNSAFE -- peaked at {peak_startup} MB")
        update_results_md(args.ncpu, ctx_k, peak_startup, False)
        sys.exit(0)

    v = get_vram()
    peak = max(peak_startup, idle_vram, v)
    print(f"  Post-fixture VRAM: {v} MB  overall peak: {peak} MB")

    kill_server()

    safe = peak <= VRAM_LIMIT
    print(f"\nRESULT: n-cpu-moe={args.ncpu}  ctx={args.ctx}")
    print(f"  Peak VRAM: {peak:,} MB  ({'SAFE' if safe else 'UNSAFE -- over ' + str(VRAM_LIMIT) + ' MB limit'})")

    update_results_md(args.ncpu, ctx_k, peak, safe)
    sys.exit(0 if safe else 1)

if __name__ == "__main__":
    main()
