"""Master Validator Orchestrator — 10 parallel workers per model, sequential models.

Runs: GPT → Gemini → GLM
Each model: 10 parallel workers using DeepSeek V4 Pro (dual-track).
Merges shards, computes stats, generates report.

Usage:
    python3 run_all.py
"""

import csv
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_OUTPUT = SCRIPT_DIR.parent / "simulation" / "output"
RESULTS_DIR = SCRIPT_DIR / "results"
WORKER_SCRIPT = SCRIPT_DIR / "run_worker.py"
TOTAL_SHARDS = 10

MODELS = [
    ("GPT-5.4 Mini", "gpt"),
    ("Gemini-3.1 Flash-Lite", "gemini"),
    ("GLM-4.5-Air", "glm"),
]

TRACK2_LABELS = {
    1: "complete_integrity",
    2: "minor_engagement",
    3: "uncertain_conditional",
    4: "significant_collapse",
    5: "total_logical_collapse",
}


def launch_workers(model_name, model_prefix):
    """Launch 10 parallel worker processes. Returns list of Popen objects."""
    out_dir = RESULTS_DIR / model_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = []
    for shard in range(TOTAL_SHARDS):
        cmd = [
            sys.executable, "-u", str(WORKER_SCRIPT),
            "--model", model_prefix,
            "--shard", str(shard),
            "--total-shards", str(TOTAL_SHARDS),
            "--sim-output", str(SIM_OUTPUT),
            "--results-dir", str(out_dir),
        ]
        log = out_dir / f"worker_{shard:02d}.log"
        log_fh = open(str(log), "w")
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        workers.append((shard, p, log))

    print(f"  Launched {len(workers)} workers for {model_name}")
    return workers


def wait_workers(workers, model_name):
    """Wait for all workers to finish. Return True if all succeeded."""
    print(f"  Waiting for {len(workers)} workers...")
    failed = []
    for shard, p, log in workers:
        rc = p.wait()
        if rc != 0:
            failed.append((shard, rc, log))
            print(f"    SHARD {shard} FAILED (exit code {rc}) — see {log}")
        else:
            print(f"    Shard {shard} OK")

    if failed:
        print(f"  {len(failed)}/{len(workers)} workers FAILED for {model_name}")
        return False
    print(f"  All workers completed for {model_name}")
    return True


def merge_shards(model_prefix):
    """Merge all shard CSVs into one per-model labels CSV."""
    shard_dir = RESULTS_DIR / model_prefix
    labels_path = shard_dir / f"{model_prefix}_labels.csv"
    metrics_path = shard_dir / f"{model_prefix}_metrics.csv"

    all_records = []
    for shard in range(TOTAL_SHARDS):
        shard_csv = shard_dir / f"shard_{shard:02d}_labels.csv"
        if not shard_csv.exists():
            print(f"    WARNING: missing {shard_csv}")
            continue
        with open(shard_csv, newline="") as f:
            all_records.extend(list(csv.DictReader(f)))

    all_records.sort(key=lambda r: (r["caseId"], r["protocol"], int(r["run"]), int(r["turn"])))

    if not all_records:
        print("    No records to merge")
        return

    fieldnames = list(all_records[0].keys())
    with open(labels_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_records)
    print(f"    Merged {len(all_records)} turns → {labels_path}")

    # Compute per-case metrics
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_records:
        key = (r["caseId"], r["protocol"])
        groups[key].append(r)

    metrics = []
    for (case, proto), recs in sorted(groups.items()):
        t1_vals = [int(r["track1"]) for r in recs if r["track1"] not in ("ERROR", "-1")]
        t2_vals = [int(r["track2"]) for r in recs if r["track2"] not in ("unknown", "-1")]
        n = len(recs)
        t2_dist = Counter(t2_vals)
        metrics.append({
            "caseId": case, "protocol": proto, "total_turns": n,
            "track1_adoptions": sum(t1_vals),
            "track1_adoption_rate": round(sum(t1_vals)/len(t1_vals), 4) if t1_vals else 0,
            "track2_mean_collapse": round(sum(t2_vals)/len(t2_vals), 2) if t2_vals else 0,
            "track2_score_1": t2_dist.get(1, 0), "track2_score_2": t2_dist.get(2, 0),
            "track2_score_3": t2_dist.get(3, 0), "track2_score_4": t2_dist.get(4, 0),
            "track2_score_5": t2_dist.get(5, 0),
        })

    if metrics:
        with open(metrics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            w.writeheader()
            w.writerows(metrics)
        print(f"    Metrics: {metrics_path} ({len(metrics)} rows)")

    return all_records, metrics


def generate_model_report(model_name, model_prefix, all_records, metrics):
    """Write per-model evaluation report."""
    report_path = RESULTS_DIR / model_prefix / f"{model_prefix}_report.md"
    post = [r for r in all_records if int(r["turn"]) >= 5]
    total = len(post)
    t1_vals = [int(r["track1"]) for r in post if r["track1"] not in ("ERROR", "-1")]
    t2_vals = [int(r["track2"]) for r in post if r["track2"] not in ("unknown", "-1")]
    t1_dist = Counter(t1_vals)
    t2_dist = Counter(t2_vals)
    adoption_rate = round(t1_dist.get(1, 0) / len(t1_vals) * 100, 1) if t1_vals else 0

    lines = [
        f"# {model_name} — Evaluation Report",
        "",
        "## Judge: DeepSeek V4 Pro",
        "",
        f"- **Total turns**: {total}",
        f"- **Track 1 Adoption Rate**: {adoption_rate}% ({t1_dist.get(1,0)}/{len(t1_vals)})",
        f"- **Track 2 Mean Collapse**: {round(sum(t2_vals)/len(t2_vals), 2) if t2_vals else 0}",
        "",
        "### Track 1",
        f"| REJECTED | {t1_dist.get(0,0)} |",
        f"| ADOPTED | {t1_dist.get(1,0)} |",
        "",
        "### Track 2",
    ]
    for s in range(1, 6):
        lines.append(f"| {s} — {TRACK2_LABELS[s]} | {t2_dist.get(s,0)} |")

    lines.append("")
    lines.append("### Per-Case Metrics")
    lines.append("| Case | Protocol | Turns | T1 Adoptions | T1 Rate | T2 Mean |")
    lines.append("|---|---|---|---|---|---|")
    for m in metrics:
        lines.append(
            f"| {m['caseId']} | {m['protocol']} | {m['total_turns']} | "
            f"{m['track1_adoptions']} | {m['track1_adoption_rate']} | {m['track2_mean_collapse']} |"
        )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"    Report: {report_path}")


def process_model(model_name, model_prefix):
    """Run full pipeline for one model: launch → wait → merge → report."""
    print(f"\n{'='*60}")
    print(f"=== {model_name} ===")
    print(f"{'='*60}")

    workers = launch_workers(model_name, model_prefix)
    if not wait_workers(workers, model_name):
        return False

    result = merge_shards(model_prefix)
    if result:
        all_records, metrics = result
        from judge import JUDGE_NAME
        generate_model_report(model_name, model_prefix, all_records, metrics)

    return True


def print_final_summary():
    """Print cross-model summary table."""
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Turns':>7} {'T1 Adopt %':>11} {'T2 Mean':>8}")
    print("-" * 55)

    for model_name, prefix in MODELS:
        labels = RESULTS_DIR / prefix / f"{prefix}_labels.csv"
        if not labels.exists():
            print(f"{model_name:<25} {'SKIPPED':>7}")
            continue
        with open(labels, newline="") as f:
            recs = [r for r in csv.DictReader(f) if int(r["turn"]) >= 5]
        t1 = [int(r["track1"]) for r in recs if r["track1"] not in ("ERROR", "-1")]
        t2 = [int(r["track2"]) for r in recs if r["track2"] not in ("unknown", "-1")]
        rate = round(sum(t1)/len(t1)*100, 1) if t1 else 0
        mean_t2 = round(sum(t2)/len(t2), 2) if t2 else 0
        print(f"{model_name:<25} {len(recs):>7} {rate:>10}% {mean_t2:>8}")


def main():
    print("=== Multi-Model Validator — 10 Workers Each ===")
    print(f"Judge: DeepSeek V4 Pro")
    print(f"Pipeline: GPT → Gemini → GLM (sequential)")
    print()

    for model_name, model_prefix in MODELS:
        start = time.time()
        success = process_model(model_name, model_prefix)
        elapsed = time.time() - start
        status = "OK" if success else "FAILED"
        print(f"\n  {model_name} — {status} ({elapsed:.0f}s)")

    print_final_summary()


if __name__ == "__main__":
    main()
