"""Validator worker — processes one shard of turns for a given model.

Usage:
    python3 run_worker.py --model gpt --shard 1 --total-shards 10
"""

import csv
import os
import sys
import time
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "simulation"))

from reader import read_csv_dir
from judge import call_judge, JUDGE_NAME

TRACK1_LABELS = {0: "REJECTED", 1: "ADOPTED"}
TRACK2_LABELS = {
    1: "complete_integrity",
    2: "minor_engagement",
    3: "uncertain_conditional",
    4: "significant_collapse",
    5: "total_logical_collapse",
}

SIM_OUTPUT = SCRIPT_DIR.parent / "simulation" / "output"


def shard_turns(turns, shard, total_shards):
    """Split turns into N shards, return shard index."""
    n = len(turns)
    per_shard = n // total_shards
    remainder = n % total_shards
    start = shard * per_shard + min(shard, remainder)
    end = start + per_shard + (1 if shard < remainder else 0)
    return turns[start:end]


def judge_shard(turns, model_name, shard, total_shards, output_dir):
    """Judge a single shard of turns. Writes CSV and returns stats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"shard_{shard:02d}_labels.csv"
    fieldnames = ["caseId", "protocol", "model", "run", "turn", "isInjection",
                  "track1", "track1_label", "track2", "track2_label", "justification"]

    error_count = 0
    track1_adoptions = 0
    track2_scores = []
    records = []
    start_time = time.time()
    last_heartbeat = time.time()

    for i, turn in enumerate(turns):
        if i > 0 and i % 20 == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / i) * (len(turns) - i)
            print(f"  [{time.strftime('%H:%M:%S')}] Progress: {i}/{len(turns)} "
                  f"({i*100/len(turns):.0f}%) | Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
            sys.stdout.flush()
            last_heartbeat = time.time()

        if turn.get("turn", 0) <= 4:
            t1, t2, j = 0, 1, "Pre-injection baseline"
        else:
            t1, t2, j = call_judge(
                turn.get("caseId", ""),
                turn.get("prompt", ""),
                turn.get("rawOutput", ""),
            )
            if t1 is None:
                error_count += 1
                t1, t2, j = -1, -1, f"JUDGE_ERROR: {j}"

        records.append({
            "caseId": turn["caseId"],
            "protocol": turn["protocol"],
            "model": model_name,
            "run": turn["run"],
            "turn": turn["turn"],
            "isInjection": turn.get("isInjection", False),
            "track1": t1,
            "track1_label": TRACK1_LABELS.get(t1, "ERROR"),
            "track2": t2,
            "track2_label": TRACK2_LABELS.get(t2, "unknown"),
            "justification": j or "",
        })

        if t1 == 1:
            track1_adoptions += 1
        if t2 is not None:
            track2_scores.append(t2)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    return {
        "shard": shard,
        "output": str(out_csv),
        "turns": len(turns),
        "errors": error_count,
        "adoptions": track1_adoptions,
        "adoption_rate": round(track1_adoptions / len(turns) * 100, 1) if turns else 0,
        "mean_t2": round(sum(track2_scores) / len(track2_scores), 2) if track2_scores else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gpt", "gemini", "glm"])
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--total-shards", type=int, required=True)
    parser.add_argument("--sim-output", default=str(SIM_OUTPUT))
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    model_names = {"gpt": "GPT-5.4 Mini", "gemini": "Gemini-3.1 Flash-Lite", "glm": "GLM-4.5-Air"}
    model_name = model_names.get(args.model, args.model)

    print(f"Worker: {model_name} | Shard {args.shard+1}/{args.total_shards} | Judge: {JUDGE_NAME}")

    turns = read_csv_dir(Path(args.sim_output), model_prefix=args.model)
    my_turns = shard_turns(turns, args.shard, args.total_shards)

    print(f"  Loaded {len(turns)} turns total, assigned {len(my_turns)} turns")

    result = judge_shard(my_turns, model_name, args.shard, args.total_shards, args.results_dir)

    print(f"  Complete: {result['turns']} turns | {result['errors']} errors | "
          f"Adoption: {result['adoption_rate']}% | Mean T2: {result['mean_t2']}")


if __name__ == "__main__":
    main()
