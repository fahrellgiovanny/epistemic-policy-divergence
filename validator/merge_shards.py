"""Merge sharded validator results into unified outputs.

Usage:
    python3 merge_shards.py --model gpt
    python3 merge_shards.py --model gemini
    python3 merge_shards.py --model glm
"""

import csv, json, sys, shutil
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from statistics import compute_all_statistics, wilson_confidence_interval
from validate import write_labels_csv, write_metrics_csv, generate_report, compute_metrics


def merge_model(model_prefix, model_name):
    results_dir = SCRIPT_DIR / "results"
    pattern = f"{model_prefix}_shard_*"
    shards = sorted(results_dir.glob(pattern))

    if not shards:
        print(f"No shards found for {model_prefix}")
        return

    print(f"Merging {len(shards)} shards for {model_name}...")

    # Merge labels
    all_turns = []
    for shard in shards:
        labels_path = shard / f"{model_prefix}_labels.csv"
        if not labels_path.exists():
            print(f"  WARNING: {labels_path} not found, skipping")
            continue
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["turn"] = int(row["turn"])
                row["run"] = int(row["run"])
                row["track1"] = int(row["track1"]) if row.get("track1") not in (None, "", "None") else None
                row["track2"] = int(row["track2"]) if row.get("track2") not in (None, "", "None") else None
                all_turns.append(row)

    if not all_turns:
        print("No valid turns found")
        return

    # Sort by case, protocol, run, turn
    all_turns.sort(key=lambda x: (x["caseId"], x["protocol"], int(x["run"]), int(x["turn"])))

    # Write merged labels
    out_dir = results_dir / model_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / f"{model_prefix}_labels.csv"
    fieldnames = list(all_turns[0].keys())
    with open(labels_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_turns)
    print(f"  Labels: {labels_path} ({len(all_turns)} turns)")

    # Re-derive judged_results format for stats
    judged = [(t, t["track1"], t["track2"], t.get("justification", "")) for t in all_turns if t["turn"] >= 5]

    # Compute stats
    stats = compute_all_statistics(judged)

    # Compute metrics
    metrics = compute_metrics(judged)
    if metrics:
        metrics_path = out_dir / f"{model_prefix}_metrics.csv"
        write_metrics_csv(metrics, metrics_path)
        print(f"  Metrics: {metrics_path} ({len(metrics)} rows)")

    # Write statistics
    stats_path = out_dir / f"{model_prefix}_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Statistics: {stats_path}")

    # Write report
    report_path = out_dir / f"{model_prefix}_report.md"
    report = generate_report(judged, metrics, model_name, stats=stats)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    # Summary
    failures = sum(1 for t in all_turns if t.get("track1") is None and t["turn"] >= 5)
    t1_adopted = sum(1 for t in all_turns if t.get("track1") == 1 and t["turn"] >= 5)
    t1_total = sum(1 for t in all_turns if t.get("track1") is not None and t["turn"] >= 5)
    rate = t1_adopted / t1_total * 100 if t1_total else 0
    print(f"\n  Turns: {len(all_turns)}  Failed: {failures}  Adoption: {t1_adopted}/{t1_total} ({rate:.1f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["gpt","gemini","glm"])
    args = p.parse_args()

    MODEL_MAP = {
        "gpt": "GPT-5.4 Mini",
        "gemini": "Gemini-3.1 Flash-Lite",
        "glm": "GLM-4.5-Air",
    }
    merge_model(args.model, MODEL_MAP[args.model])
