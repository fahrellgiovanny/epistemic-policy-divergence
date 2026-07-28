"""Sequential orchestrator — runs 10 workers per model, waits, merges."""

import subprocess, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MODELS = [
    ("gpt", "GPT-5.4 Mini"),
    ("gemini", "Gemini-3.1 Flash-Lite"),
    ("glm", "GLM-4.5-Air"),
]

for model_key, model_name in MODELS:
    workers = 10
    print(f"\n=== STARTING {model_name} ({workers} workers) ===", flush=True)
    
    procs = []
    for i in range(1, workers + 1):
        log = open(f"/tmp/validate_{model_key}_{i}.log", "w")
        p = subprocess.Popen(
            [sys.executable, "-u", "run_worker.py", "--model", model_key,
             "--shard", str(i), "--total-shards", str(workers)],
            stdout=log, stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
        )
        procs.append((i, p, log))
        print(f"  Worker {i}/{workers}: PID {p.pid}", flush=True)
    
    print(f"  Waiting for {model_name} workers...", flush=True)
    
    for i, p, log in procs:
        p.wait()
        log.close()
        # Print last progress line
        with open(f"/tmp/validate_{model_key}_{i}.log") as f:
            lines = f.readlines()
            last_progress = next((l.strip() for l in reversed(lines) if "Progress:" in l), "completed")
        print(f"  {model_name} worker {i}/{workers}: {last_progress}", flush=True)
    
    print(f"\n  {model_name} done. Merging...", flush=True)
    subprocess.run([sys.executable, "merge_shards.py", "--model", model_key], cwd=str(SCRIPT_DIR))
    print(f"  {model_name} merged.", flush=True)

print("\n=== ALL MODELS COMPLETE ===", flush=True)
