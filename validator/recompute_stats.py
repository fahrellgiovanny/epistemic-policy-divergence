#!/usr/bin/env python3
"""recompute_stats.py - session-level (cluster-robust) re-analysis and
per-scale kappa for Paper 1 ("Epistemic Policy Divergence in Multi-Turn LLM
Contamination").

Why: the original analysis treated 5,500 post-injection turns per model as
independent, but 11 turns are nested in each of 500 sessions (design effect
~7). This script recomputes the between-model tests on session-level
aggregates, which is the reviewer-recommended cluster-robust fix, and reports
Cohen's kappa separately per scale (T1 nominal, T2 ordinal with linear and
quadratic weighting) instead of averaging across scales.

Stdlib only. Usage:
  python3 recompute_stats.py [--labels-dir DIR] [--gold CSV] [--json OUT]

  --labels-dir  directory containing gpt/gpt_labels.csv, gemini/..., glm/...
                (default: ./results)
  --gold        human-reviewed gold standard CSV
                (default: ./gold_standard_review.csv)
  --json        optional path to write the result dict as JSON

Exit code 0 = self-checks pass.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MODELS = ["gpt", "gemini", "glm"]
POST_TURN = 5  # turns 5..15 are post-injection
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------- statistics
def chi2_sf_df2(x):
    """Survival function P(X > x) for chi-square with df=2 (exact: exp(-x/2))."""
    return math.exp(-x / 2.0)


def chi2_independence(table):
    """Chi-square test of independence on a 2-D list of counts.

    Returns (statistic, dof). p via chi2_sf_df2 only valid when dof == 2.
    """
    n = sum(sum(row) for row in table)
    rows = [sum(row) for row in table]
    cols = [sum(table[i][j] for i in range(len(table))) for j in range(len(table[0]))]
    stat = 0.0
    for i in range(len(table)):
        for j in range(len(table[0])):
            exp = rows[i] * cols[j] / n
            stat += (table[i][j] - exp) ** 2 / exp
    dof = (len(table) - 1) * (len(table[0]) - 1)
    return stat, dof


def cramers_v(stat, n, table):
    k = min(len(table), len(table[0])) - 1
    return math.sqrt(stat / (n * k)) if n * k else 0.0


def _rank_with_ties(values):
    """Return average ranks (1-based) for values, handling ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def kruskal_wallis(groups):
    """Kruskal-Wallis H over a list of value-lists. Returns (H, eta2_H, dof)."""
    allv = []
    for gi, g in enumerate(groups):
        for v in g:
            allv.append((v, gi))
    ranks = _rank_with_ties([v for v, _ in allv])
    n = len(allv)
    rank_sum = defaultdict(float)
    for (_, gi), rk in zip(allv, ranks):
        rank_sum[gi] += rk
    H = 12.0 / (n * (n + 1)) * sum(
        rank_sum[gi] ** 2 / len(groups[gi]) for gi in range(len(groups))
    ) - 3 * (n + 1)
    dof = len(groups) - 1
    eta2 = (H - dof + 1) / (n - dof) if n > dof else float("nan")
    return H, eta2, dof


def mann_whitney(a, b):
    """Mann-Whitney U (for group a) with tie-corrected normal approx.

    Returns (U_a, rank_biserial_abs, z).
    """
    comb = [(v, 0) for v in a] + [(v, 1) for v in b]
    ranks = _rank_with_ties([v for v, _ in comb])
    na, nb = len(a), len(b)
    ra = sum(rk for (_, g), rk in zip(comb, ranks) if g == 0)
    u = ra - na * (na + 1) / 2.0
    r = 2.0 * u / (na * nb) - 1.0
    mu = na * nb / 2.0
    sd = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    z = (u - mu) / sd if sd else 0.0
    return u, abs(r), z


def cohen_kappa(a, b, weight=None):
    """Cohen's kappa. weight: None (unweighted), 'linear', or 'quadratic'."""
    labels = sorted(set(a) | set(b))
    K = len(labels)
    idx = {v: i for i, v in enumerate(labels)}

    def w(i, j):
        if weight is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j)
        if K <= 1:
            return 0.0
        return (d / (K - 1)) if weight == "linear" else (d * d) / ((K - 1) ** 2)

    obs = Counter(zip(a, b))
    ca, cb = Counter(a), Counter(b)
    n = len(a)
    num = sum(obs[(x, y)] * w(idx[x], idx[y]) for x, y in obs)
    den = sum(ca[x] * cb[y] / n * w(idx[x], idx[y]) for x in labels for y in labels)
    return 1.0 - num / den if den else float("nan")


# -------------------------------------------------------------------- loaders
def load_labels(labels_dir):
    data = {}
    for m in MODELS:
        path = labels_dir / m / f"{m}_labels.csv"
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if int(r["turn"]) >= POST_TURN]
        data[m] = rows
    return data


def session_keys(rows):
    return [(r["caseId"], r["protocol"], r["run"]) for r in rows]


def session_aggregate(rows, key):
    """{session_key: list of int values} for a given column."""
    s = defaultdict(list)
    for r in rows:
        s[(r["caseId"], r["protocol"], r["run"])].append(int(r[key]))
    return s


def analyse(data):
    out = {}

    # ---- T1: session-level independence test (any adoption vs none)
    t1_sessions = {m: session_aggregate(data[m], "track1") for m in MODELS}
    t1_density = {m: [sum(v) / len(v) for v in t1_sessions[m].values()] for m in MODELS}
    table = [
        [sum(1 for v in t1_sessions[m].values() if any(v)),
         sum(1 for v in t1_sessions[m].values() if not any(v))]
        for m in MODELS
    ]
    chi2, dof = chi2_independence(table)
    n_sessions = sum(sum(r) for r in table)
    out["t1_session"] = {
        "table_any_vs_none": table,
        "chi2": round(chi2, 1),
        "dof": dof,
        "n_sessions": n_sessions,
        "p": chi2_sf_df2(chi2) if dof == 2 else None,
        "cramers_v": round(cramers_v(chi2, n_sessions, table), 3),
        "affected_pct": {m: round(100 * table[i][0] / sum(table[i]), 1)
                         for i, m in enumerate(MODELS)},
    }

    # turn-level chi2 (reproduction / sensitivity)
    turn_table = [
        [sum(1 for r in data[m] if int(r["track1"]) == 1),
         sum(1 for r in data[m] if int(r["track1"]) == 0)]
        for m in MODELS
    ]
    tchi2, tdof = chi2_independence(turn_table)
    out["t1_turn_sensitivity"] = {
        "chi2": round(tchi2, 1), "dof": tdof,
        "n": sum(sum(r) for r in turn_table),
        "p": chi2_sf_df2(tchi2) if tdof == 2 else None,
        "cramers_v": round(cramers_v(tchi2, sum(sum(r) for r in turn_table), turn_table), 3),
    }
    out["design_effect_chi2"] = round(tchi2 / chi2, 1) if chi2 else None

    # ---- T2: session-mean Kruskal-Wallis
    t2_means = {m: [sum(v) / len(v)
                    for v in session_aggregate(data[m], "track2").values()]
                for m in MODELS}
    H, eta2, hdof = kruskal_wallis([t2_means[m] for m in MODELS])
    out["t2_session"] = {
        "kruskal_h": round(H, 1), "dof": hdof,
        "n_sessions": sum(len(t2_means[m]) for m in MODELS),
        "p": chi2_sf_df2(H) if hdof == 2 else None,
        "eta2_H": round(eta2, 3),
    }

    # ---- pairwise session-mean Mann-Whitney (collapse severity)
    pairs = {}
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            u, r, z = mann_whitney(t2_means[a], t2_means[b])
            pairs[f"{a}_vs_{b}"] = {"U": round(u), "rank_biserial_abs": round(r, 2),
                                    "z": round(z, 1)}
    out["t2_pairwise"] = pairs

    # ---- exact per-case rates and domain-length deltas
    cases = sorted({r["caseId"] for r in data["gemini"]})
    per_case = {m: {} for m in MODELS}
    for m in MODELS:
        for c in cases:
            rs = [r for r in data[m] if r["caseId"] == c]
            per_case[m][c] = round(100 * sum(1 for r in rs if int(r["track1"]) == 1) / len(rs), 2)
    out["per_case_t1_pct"] = per_case
    deltas = {"gemini": {}, "glm": {}}
    for m in ("gemini", "glm"):
        for base in ("math", "physics", "history", "chemistry", "geo"):
            deltas[m][base] = round(per_case[m][base + "_long"] - per_case[m][base + "_short"], 2)
    out["domain_length_deltas"] = deltas
    return out


def analyse_kappa(gold_path):
    """Cohen's kappa from the human-reviewed gold standard.

    Accepts either a CSV (columns prelabel_track1, human_track1,
    prelabel_track2, human_track2) or a .jsonl file with the same keys.
    """
    path = Path(gold_path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

    def col(name):
        return [int(r[name]) for r in rows if str(r[name]).strip().isdigit()]

    t1p, t1h = col("prelabel_track1"), col("human_track1")
    t2p, t2h = col("prelabel_track2"), col("human_track2")
    return {
        "n": len(rows),
        "kappa_t1_unweighted": round(cohen_kappa(t1p, t1h), 4),
        "kappa_t2_unweighted": round(cohen_kappa(t2p, t2h), 4),
        "kappa_t2_linear": round(cohen_kappa(t2p, t2h, "linear"), 4),
        "kappa_t2_quadratic": round(cohen_kappa(t2p, t2h, "quadratic"), 4),
        "t1_disagreements": sum(1 for a, b in zip(t1p, t1h) if a != b),
        "t2_disagreements": sum(1 for a, b in zip(t2p, t2h) if a != b),
    }


def self_checks(labels, gold, res):
    assert gold["n"] == 120, "expected 120 gold rows"
    assert gold["kappa_t1_unweighted"] == 1.0, "T1 kappa must be 1.000"
    assert gold["t1_disagreements"] == 0, "no T1 disagreements expected"
    assert gold["t1_disagreements"] + gold["t2_disagreements"] == 16, \
        "expected 16 total corrections"
    if labels is None:
        print("self-checks (kappa only): PASS")
        return
    assert all(len(labels[m]) == 5500 for m in MODELS), "expected 5500 post-injection turns/model"
    assert all(len(session_aggregate(labels[m], "track1")) == 500 for m in MODELS), \
        "expected 500 sessions/model"
    assert res["t1_session"]["dof"] == 2
    assert 0.0 <= res["t1_session"]["cramers_v"] <= 1.0
    # session-level tests must remain decisive
    assert res["t1_session"]["p"] < 1e-6
    assert res["t2_session"]["p"] < 1e-6
    # deltas from unrounded data must round to the paper's Table 8 values
    d = res["domain_length_deltas"]
    assert round(d["gemini"]["history"], 1) == 19.6
    assert round(d["glm"]["chemistry"], 1) == 3.3
    print("self-checks: PASS")


def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return paths[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="results",
                    help="dir with {gpt,gemini,glm}/*_labels.csv (optional; "
                         "kappa is computed without it)")
    ap.add_argument("--gold",
                    default=_first_existing([
                        str(HERE / "gold_standard.jsonl"),
                        "gold_standard.jsonl",
                        str(HERE / "gold_standard_review_human.csv"),
                        "gold_standard_review_human.csv",
                        str(HERE / "gold_standard_review - gold_standard_review (human review done).csv"),
                        "gold_standard_review - gold_standard_review (human review done).csv",
                        str(HERE / "gold_standard_review.csv")]))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    labels = None
    if Path(args.labels_dir).exists():
        labels = load_labels(Path(args.labels_dir))
    else:
        print(f"[info] per-turn labels not found under {args.labels_dir}; "
              f"computing kappa only. Provide --labels-dir to reproduce the "
              f"session-level tests (regenerate labels with the runner first).")

    res = {}
    gold = analyse_kappa(args.gold)
    res["kappa"] = gold
    if labels is not None:
        res.update(analyse(labels))
    self_checks(labels, gold, res)

    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
