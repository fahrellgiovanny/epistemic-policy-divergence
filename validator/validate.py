"""Session-Level Hallucination Judge — Shared Library.

Dual-Track Protocol:
  Track 1: Binary Epistemic Adoption (0=rejected, 1=affirmed)
  Track 2: Progressive Collapse Score (1=integrity, 5=total collapse)

Exports:
  run_validation(model, input_dir, output_dir, limit=None, kappa=False)
  — reads CSV batches, judges all turns, writes CSV outputs + report.
"""

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reader import read_csv_dir
from judge import judge_turns, JUDGE_NAME
from statistics import compute_all_statistics, wilson_confidence_interval

GOLD_STANDARD_PATH = SCRIPT_DIR / "gold_standard.jsonl"

TRACK1_LABELS = {0: "REJECTED", 1: "ADOPTED"}
TRACK2_LABELS = {
    1: "complete_integrity",
    2: "minor_engagement",
    3: "uncertain_conditional",
    4: "significant_collapse",
    5: "total_logical_collapse",
}


def compute_cohens_kappa(judge_labels, gold_labels):
    n = len(judge_labels)
    if n != len(gold_labels) or n == 0:
        return 0.0
    po = sum(1 for j, g in zip(judge_labels, gold_labels) if j == g) / n
    jc = Counter(judge_labels)
    gc = Counter(gold_labels)
    pe = sum(jc[k] * gc[k] for k in jc) / (n * n)
    if po == pe:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def load_gold_standard():
    if not GOLD_STANDARD_PATH.exists():
        return []
    records = []
    with open(GOLD_STANDARD_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_kappa_check(turns, gold_records):
    gold_by_key = {}
    for g in gold_records:
        key = (g.get("caseId"), g.get("protocol"), g.get("run"), g.get("turn"))
        gold_by_key[key] = g

    matched = []
    for t in turns:
        key = (t.get("caseId"), t.get("protocol"), t.get("run"), t.get("turn"))
        if key in gold_by_key:
            matched.append((t, gold_by_key[key]))

    if not matched:
        print("No gold standard matches found in input data")
        return {}

    print(f"\n=== COHEN'S KAPPA VALIDATION ===")
    print(f"Gold standard turns: {len(gold_records)}")
    print(f"Matched turns: {len(matched)}")

    judged = judge_turns([m[0] for m in matched])
    judge_t1 = [t1 for _, t1, _, _ in judged]
    judge_t2 = [t2 for _, _, t2, _ in judged]
    gold_t1 = [m[1].get("track1") for m in matched]
    gold_t2 = [m[1].get("track2") for m in matched]

    valid_t1 = [(j, g) for j, g in zip(judge_t1, gold_t1) if j is not None and g is not None]
    valid_t2 = [(j, g) for j, g in zip(judge_t2, gold_t2) if j is not None and g is not None]

    result = {}

    if valid_t1:
        kappa_t1 = compute_cohens_kappa([v[0] for v in valid_t1], [v[1] for v in valid_t1])
        print(f"\nTrack 1 (Binary) Cohen's Kappa: {kappa_t1:.3f}")
        print(f"  Valid pairs: {len(valid_t1)}/{len(matched)}")
        result["kappa_t1"] = kappa_t1

    if valid_t2:
        kappa_t2 = compute_cohens_kappa([v[0] for v in valid_t2], [v[1] for v in valid_t2])
        print(f"Track 2 (Likert) Cohen's Kappa: {kappa_t2:.3f}")
        print(f"  Valid pairs: {len(valid_t2)}/{len(matched)}")
        result["kappa_t2"] = kappa_t2

    if valid_t1 and valid_t2:
        avg = (result["kappa_t1"] + result["kappa_t2"]) / 2
        result["kappa_avg"] = avg
        print(f"\nAverage Cohen's Kappa: {avg:.3f}")
        if avg >= 0.81:
            print("RESULT: PASS — 'almost perfect' agreement per Landis & Koch (1977; κ ≥ 0.81)")
        else:
            print(f"RESULT: Below almost-perfect threshold ({avg:.3f} < 0.81) — review needed")

    return result


def compute_metrics(judged_results):
    rows = []
    groups = defaultdict(list)
    for turn, t1, t2, _ in judged_results:
        key = (turn["caseId"], turn["protocol"])
        groups[key].append((t1, t2))

    for (case, proto), scores in sorted(groups.items()):
        track1_vals = [s[0] for s in scores if s[0] is not None]
        track2_vals = [s[1] for s in scores if s[1] is not None]
        n = len(scores)

        adoption = sum(track1_vals) if track1_vals else 0
        adoption_rate = round(adoption / len(track1_vals), 4) if track1_vals else 0
        mean_collapse = round(sum(track2_vals) / len(track2_vals), 2) if track2_vals else 0
        t2_dist = Counter(track2_vals)

        rows.append({
            "caseId": case,
            "protocol": proto,
            "total_turns": n,
            "track1_adoptions": adoption,
            "track1_adoption_rate": adoption_rate,
            "track2_mean_collapse": mean_collapse,
            "track2_score_1": t2_dist.get(1, 0),
            "track2_score_2": t2_dist.get(2, 0),
            "track2_score_3": t2_dist.get(3, 0),
            "track2_score_4": t2_dist.get(4, 0),
            "track2_score_5": t2_dist.get(5, 0),
        })
    return rows


def write_labels_csv(judged_results, path):
    """Write per-turn labels as CSV."""
    fieldnames = ["caseId", "protocol", "model", "run", "turn", "isInjection",
                  "track1", "track1_label", "track2", "track2_label", "justification"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for turn, t1, t2, just in judged_results:
            w.writerow({
                "caseId": turn["caseId"],
                "protocol": turn["protocol"],
                "model": turn.get("model", ""),
                "run": turn["run"],
                "turn": turn["turn"],
                "isInjection": turn.get("isInjection", False),
                "track1": t1,
                "track1_label": TRACK1_LABELS.get(t1, "ERROR"),
                "track2": t2,
                "track2_label": TRACK2_LABELS.get(t2, "unknown"),
                "justification": just or "",
            })


def write_metrics_csv(metrics, path):
    """Write per-case metrics as CSV."""
    if not metrics:
        return
    fieldnames = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(metrics)


def generate_report(judged_results, metrics, model_name, stats=None):
    """Generate markdown report string with statistical tests and confidence intervals."""
    turns_with_scores = [(t, t1, t2, j) for t, t1, t2, j in judged_results if t["turn"] >= 5]

    total = len(turns_with_scores)
    t1_dist = Counter(s[1] for s in turns_with_scores)
    t2_dist = Counter(s[2] for s in turns_with_scores)
    adoption_rate = round(t1_dist.get(1, 0) / total * 100, 1) if total else 0

    # Wilson CI for overall rate
    overall_lo, overall_hi, _ = wilson_confidence_interval(t1_dist.get(1, 0), total)

    lines = [
        f"# {model_name} — Session-Level Hallucination Evaluation Report",
        "",
        "## Judge",
        f"- **Model**: {JUDGE_NAME}",
        "- **Method**: Chain-of-Thought prompting with dual-track scoring",
        "- **Validation**: Cohen's Kappa against gold standard human labels",
        "",
        "## Overall Results",
        "",
        f"- **Total turns evaluated**: {total}",
        f"- **Track 1 — Epistemic Adoption Rate**: {adoption_rate}% ({t1_dist.get(1, 0)}/{total}) 95% CI: [{overall_lo:.1%}, {overall_hi:.1%}]",
        f"- **Track 2 — Mean Collapse Score**: {round(sum(s[2] for s in turns_with_scores) / total, 2) if total else 0}",
        "",
        "### Track 1 Distribution",
        "| Label | Count | % |",
        "|---|---|---|",
        f"| REJECTED (0) | {t1_dist.get(0, 0)} | {round(t1_dist.get(0,0)/total*100,1) if total else 0}% |",
        f"| ADOPTED (1) | {t1_dist.get(1, 0)} | {round(t1_dist.get(1,0)/total*100,1) if total else 0}% |",
        "",
        "### Track 2 Distribution",
        "| Score | Label | Count | % |",
        "|---|---|---|---|",
    ]
    for score in range(1, 6):
        count = t2_dist.get(score, 0)
        lines.append(f"| {score} | {TRACK2_LABELS[score]} | {count} | {round(count/total*100,1) if total else 0}% |")

    lines.extend(["", "## Protocol-Level Breakdown (with 95% CI)", ""])
    lines.append("| Protocol | Adoptions | Total | Rate | 95% CI | T2 Mean |")
    lines.append("|---|---|---|---|---|---|")

    if stats and stats.get("protocol_ci"):
        for proto, pci in sorted(stats["protocol_ci"].items()):
            t2_mean = "-"
            if stats.get("dunn_posthoc"):
                # Extract T2 means from judged data
                proto_t2 = [t2 for t, t1, t2, _ in judged_results
                           if t["turn"] >= 5 and t2 is not None
                           and (t.get("protocol","") or "").endswith(proto)]
                t2_mean = f"{sum(proto_t2)/len(proto_t2):.2f}" if proto_t2 else "-"
            lines.append(
                f"| {proto} | {pci['adoptions']} | {pci['total']} | "
                f"{pci['rate']:.1%} | [{pci['ci_lower']:.1%}, {pci['ci_upper']:.1%}] | {t2_mean} |"
            )
    else:
        lines.append("| _(no protocol-level data)_ |||||")

    lines.extend(["", "## Domain-Level Adoption Rates (with 95% CI)", ""])
    lines.append("| Domain | Adoptions | Total | Rate | 95% CI |")
    lines.append("|---|---|---|---|---|")
    if stats and stats.get("domain_ci"):
        for domain, dci in sorted(stats["domain_ci"].items()):
            lines.append(
                f"| {domain} | {dci['adoptions']} | {dci['total']} | "
                f"{dci['rate']:.1%} | [{dci['ci_lower']:.1%}, {dci['ci_upper']:.1%}] |"
            )
    else:
        lines.append("| _(no domain-level data)_ ||||")

    lines.extend(["", "## Statistical Tests", ""])

    if stats:
        chi = stats.get("chi_square", {})
        if chi.get("p_value") is not None:
            lines.append(f"- **Chi-square (protocol independence)**:  χ²({chi.get('dof','?')}) = {chi.get('statistic')}, p = {chi.get('p_value')}")
        else:
            lines.append("- **Chi-square**: insufficient data")

        kw = stats.get("kruskal_wallis", {})
        if kw.get("p_value") is not None:
            lines.append(f"- **Kruskal-Wallis (protocol collapse scores)**: H = {kw.get('h_statistic')}, p = {kw.get('p_value')}")
        else:
            lines.append("- **Kruskal-Wallis**: insufficient data")

        dunn = stats.get("dunn_posthoc", [])
        if dunn:
            lines.extend(["", "### Dunn's Post-Hoc (Bonferroni-corrected)", ""])
            sig_pairs = [d for d in dunn if d["significant"]]
            if sig_pairs:
                lines.append("**Significant pairwise differences:**")
                for d in sig_pairs:
                    lines.append(f"- {d['group_a']} vs {d['group_b']}: z = {d['z_score']}, p = {d['p_corrected']}")
            else:
                lines.append("No significant pairwise differences after correction.")
            lines.append("")
            lines.append("<details><summary>All pairwise comparisons</summary>")
            lines.append("")
            lines.append("| Group A | Group B | z-score | p (corrected) | Significant? |")
            lines.append("|---|---|---|---|---|")
            for d in dunn:
                sig = "✓" if d["significant"] else ""
                lines.append(f"| {d['group_a']} | {d['group_b']} | {d['z_score']} | {d['p_corrected']} | {sig} |")
            lines.append("</details>")
        else:
            lines.append("- **Dunn's post-hoc**: insufficient data")

    lines.extend(["", "## Per-Case Adoption Rates", ""])
    lines.append("| Case | Protocol | Turns | T1 Adoptions | T1 Rate | T2 Mean |")
    lines.append("|---|---|---|---|---|---|")
    for m in metrics:
        lines.append(
            f"| {m['caseId']} | {m['protocol']} | {m['total_turns']} | "
            f"{m['track1_adoptions']} | {m['track1_adoption_rate']} | {m['track2_mean_collapse']} |"
        )

    return "\n".join(lines)


def run_validation(model, input_dir, output_dir, limit=None, kappa=False, model_prefix=None, batch_start=None, batch_end=None):
    """Full validation pipeline.

    Args:
        model: display name (e.g. "GPT-5.4 Mini")
        input_dir: path to simulation output CSVs
        output_dir: where to write results
        limit: only judge first N turns (for smoketest)
        kappa: compute Cohen's Kappa against gold standard
        model_prefix: CSV file prefix (e.g. "gpt", "gemini", "glm")
        batch_start: 1-based start of batch range (for parallel workers)
        batch_end: 1-based end of batch range (for parallel workers)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = model_prefix or model.lower().split()[0].replace(".", "")

    shard_label = f" [batches {batch_start}-{batch_end}]" if batch_start else ""
    print(f"=== {model} Validation{shard_label} ===", flush=True)

    turns = read_csv_dir(input_dir, model_prefix=prefix, batch_start=batch_start, batch_end=batch_end)
    print(f"Loaded {len(turns)} turns from {input_dir}", flush=True)

    if limit:
        turns = turns[:limit]
        print(f"Limited to {len(turns)} turns")

    print(f"Judging {len(turns)} turns with {JUDGE_NAME}...")

    if kappa:
        gold = load_gold_standard()
        if gold:
            run_kappa_check(turns, gold)
        else:
            print(f"No gold standard found at {GOLD_STANDARD_PATH}")

    start = time.time()
    judged = judge_turns(turns)
    elapsed = time.time() - start

    print(f"\nJudging complete: {len(judged)} turns in {elapsed:.0f}s ({elapsed/len(judged):.1f}s/turn)")

    errors = sum(1 for _, t1, _, _ in judged if t1 is None)
    if errors:
        print(f"WARNING: {errors} turns failed to judge (track1=None)")

    # Write labels/metrics FIRST (before stats — stats may crash on edge cases)
    labels_path = output_dir / f"{prefix}_labels.csv"
    write_labels_csv(judged, labels_path)
    print(f"  Labels: {labels_path} ({len(judged)} turns)")

    metrics = compute_metrics(judged)
    if metrics:
        metrics_path = output_dir / f"{prefix}_metrics.csv"
        write_metrics_csv(metrics, metrics_path)
        print(f"  Metrics: {metrics_path} ({len(metrics)} rows)")

    # Compute statistics (done after data is safely saved)
    print("Computing statistical tests...")
    stats = compute_all_statistics(judged)
    if stats:
        stats_path = output_dir / f"{prefix}_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  Statistics: {stats_path}")

    report_path = output_dir / f"{prefix}_report.md"
    report = generate_report(judged, metrics, model, stats=stats)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    return judged, metrics
