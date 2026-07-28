"""Compute per-protocol aggregate statistics from per-case metrics CSVs."""
import csv
from collections import defaultdict

PROTOCOL_LABELS = {
    "protocol_a_factual_inversion": "A - Factual Inversion",
    "protocol_b_synthetic_turn_injection": "B - Synthetic Turn (Fake RAG)",
    "protocol_c_intent_subversion": "C - Intent Subversion",
    "protocol_d_reasoning_chain_corruption": "D - Reasoning Chain Corruption",
    "protocol_e_confidence_miscalibration": "E - Confidence Miscalibration",
}

def load_metrics(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def aggregate_by_protocol(rows):
    agg = {}
    for r in rows:
        p = r["protocol"]
        if p not in agg:
            agg[p] = {"total_turns": 0, "adoptions": 0, "t2": [0]*5}
        agg[p]["total_turns"] += int(r["total_turns"])
        agg[p]["adoptions"] += int(r["track1_adoptions"])
        agg[p]["t2"][0] += int(r["track2_score_1"])
        agg[p]["t2"][1] += int(r["track2_score_2"])
        agg[p]["t2"][2] += int(r["track2_score_3"])
        agg[p]["t2"][3] += int(r["track2_score_4"])
        agg[p]["t2"][4] += int(r["track2_score_5"])
    for p in agg:
        total = agg[p]["total_turns"]
        t2 = agg[p]["t2"]
        agg[p]["t2_mean"] = sum((i+1)*t2[i] for i in range(5)) / total
        agg[p]["adoption_rate"] = agg[p]["adoptions"] / total
    return agg

def aggregate_by_protocol_and_case(rows):
    agg = defaultdict(lambda: {"total_turns": 0, "adoptions": 0, "t2": [0]*5})
    for r in rows:
        key = (r["caseId"], r["protocol"])
        agg[key]["total_turns"] += int(r["total_turns"])
        agg[key]["adoptions"] += int(r["track1_adoptions"])
        agg[key]["t2"][0] += int(r["track2_score_1"])
        agg[key]["t2"][1] += int(r["track2_score_2"])
        agg[key]["t2"][2] += int(r["track2_score_3"])
        agg[key]["t2"][3] += int(r["track2_score_4"])
        agg[key]["t2"][4] += int(r["track2_score_5"])
    for k in agg:
        total = agg[k]["total_turns"]
        t2 = agg[k]["t2"]
        agg[k]["t2_mean"] = sum((i+1)*t2[i] for i in range(5)) / total
        agg[k]["adoption_rate"] = agg[k]["adoptions"] / total
    return agg

MODELS = {
    "GPT-5.4 Mini": "results/gpt/gpt_metrics.csv",
    "Gemini-3.1 Flash-Lite": "results/gemini/gemini_metrics.csv",
    "GLM-4.5-Air": "results/glm/glm_metrics.csv",
}

CASES = [
    "chemistry_long", "chemistry_short",
    "geo_long", "geo_short",
    "history_long", "history_short",
    "math_long", "math_short",
    "physics_long", "physics_short",
]

all_data = {}
for name, path in MODELS.items():
    all_data[name] = {
        "by_protocol": aggregate_by_protocol(load_metrics(path)),
        "by_case_protocol": aggregate_by_protocol_and_case(load_metrics(path)),
    }

PROTOCOLS = sorted(PROTOCOL_LABELS.keys())

print("=" * 90)
print("PER-PROTOCOL BREAKDOWN ANALYSIS")
print("=" * 90)

print("\n## Table 1: T1 Adoption Rate by Protocol (aggregated across all 10 cases)\n")
print(f"{'Protocol':<40} {'GPT':>10} {'Gemini':>10} {'GLM':>10}")
print("-" * 70)
for p in PROTOCOLS:
    gpt = all_data["GPT-5.4 Mini"]["by_protocol"][p]["adoption_rate"] * 100
    gem = all_data["Gemini-3.1 Flash-Lite"]["by_protocol"][p]["adoption_rate"] * 100
    glm = all_data["GLM-4.5-Air"]["by_protocol"][p]["adoption_rate"] * 100
    label = PROTOCOL_LABELS[p]
    print(f"{label:<40} {gpt:9.1f}% {gem:9.1f}% {glm:9.1f}%")

print(f"\n{'Overall':<40} {0.0:9.1f}% {37.9:9.1f}% {23.2:9.1f}%")

print("\n\n## Table 2: T2 Mean Progressive Collapse by Protocol\n")
print(f"{'Protocol':<40} {'GPT':>10} {'Gemini':>10} {'GLM':>10}")
print("-" * 70)
for p in PROTOCOLS:
    gpt = all_data["GPT-5.4 Mini"]["by_protocol"][p]["t2_mean"]
    gem = all_data["Gemini-3.1 Flash-Lite"]["by_protocol"][p]["t2_mean"]
    glm = all_data["GLM-4.5-Air"]["by_protocol"][p]["t2_mean"]
    label = PROTOCOL_LABELS[p]
    print(f"{label:<40} {gpt:9.2f}  {gem:9.2f}  {glm:9.2f}")

# Detailed per-protocol T2 distributions
print("\n\n## Table 3: T2 Score Distributions by Protocol\n")
for p in PROTOCOLS:
    print(f"\n### {PROTOCOL_LABELS[p]}")
    print(f"\n{'Model':<25} {'Score 1':>8} {'Score 2':>8} {'Score 3':>8} {'Score 4':>8} {'Score 5':>8} {'T2 Mean':>8}")
    print("-" * 75)
    for name in ["GPT-5.4 Mini", "Gemini-3.1 Flash-Lite", "GLM-4.5-Air"]:
        t2 = all_data[name]["by_protocol"][p]["t2"]
        mean = all_data[name]["by_protocol"][p]["t2_mean"]
        print(f"{name:<25} {t2[0]:>8} {t2[1]:>8} {t2[2]:>8} {t2[3]:>8} {t2[4]:>8} {mean:>8.2f}")

# Per-case per-protocol adoption rates
print("\n\n## Table 4: Per-Case Per-Protocol T1 Adoption Rates (%)\n")
print(f"{'Case':<18} {'Protocol':<5} {'GPT':>8} {'Gemini':>8} {'GLM':>8}")
print("-" * 52)
for case in CASES:
    first = True
    for p in PROTOCOLS:
        cname = case if first else ""
        pshort = p.split("_")[2][0].upper()
        gpt = all_data["GPT-5.4 Mini"]["by_case_protocol"][(case, p)]["adoption_rate"] * 100
        gem = all_data["Gemini-3.1 Flash-Lite"]["by_case_protocol"][(case, p)]["adoption_rate"] * 100
        glm = all_data["GLM-4.5-Air"]["by_case_protocol"][(case, p)]["adoption_rate"] * 100
        print(f"{cname:<18} {pshort:<5} {gpt:7.1f}% {gem:7.1f}% {glm:7.1f}%")
        first = False

# Per-case per-protocol T2 means
print("\n\n## Table 5: Per-Case Per-Protocol T2 Mean Collapse\n")
print(f"{'Case':<18} {'Protocol':<5} {'GPT':>8} {'Gemini':>8} {'GLM':>8}")
print("-" * 52)
for case in CASES:
    first = True
    for p in PROTOCOLS:
        cname = case if first else ""
        pshort = p.split("_")[2][0].upper()
        gpt = all_data["GPT-5.4 Mini"]["by_case_protocol"][(case, p)]["t2_mean"]
        gem = all_data["Gemini-3.1 Flash-Lite"]["by_case_protocol"][(case, p)]["t2_mean"]
        glm = all_data["GLM-4.5-Air"]["by_case_protocol"][(case, p)]["t2_mean"]
        print(f"{cname:<18} {pshort:<5} {gpt:7.2f}  {gem:7.2f}  {glm:7.2f}")
        first = False

# Ranking: most dangerous protocols
print("\n\n## Table 6: Protocol Danger Ranking (by average adoption rate across Gemini + GLM)\n")
protocol_danger = []
for p in PROTOCOLS:
    gem = all_data["Gemini-3.1 Flash-Lite"]["by_protocol"][p]["adoption_rate"]
    glm = all_data["GLM-4.5-Air"]["by_protocol"][p]["adoption_rate"]
    avg = (gem + glm) / 2
    protocol_danger.append((PROTOCOL_LABELS[p], gem*100, glm*100, avg*100))
protocol_danger.sort(key=lambda x: x[3], reverse=True)
print(f"{'Protocol':<40} {'Gemini':>10} {'GLM':>10} {'Avg':>10}")
print("-" * 70)
for label, gem, glm, avg in protocol_danger:
    print(f"{label:<40} {gem:9.1f}% {glm:9.1f}% {avg:9.1f}%")

# Sessions affected per protocol
print("\n\n## Table 7: Sessions with at least 1 adoption per protocol (out of 100 per model)\n")
print(f"{'Protocol':<40} {'GPT':>10} {'Gemini':>10} {'GLM':>10}")
print("-" * 70)
for p in PROTOCOLS:
    gpt_cases = 0
    gem_cases = 0
    glm_cases = 0
    for case in CASES:
        if all_data["GPT-5.4 Mini"]["by_case_protocol"][(case, p)]["adoptions"] > 0:
            gpt_cases += 10  # 10 runs per case
        if all_data["Gemini-3.1 Flash-Lite"]["by_case_protocol"][(case, p)]["adoptions"] > 0:
            gem_cases += 10
        if all_data["GLM-4.5-Air"]["by_case_protocol"][(case, p)]["adoptions"] > 0:
            glm_cases += 10
    label = PROTOCOL_LABELS[p]
    print(f"{label:<40} {gpt_cases:>10} {gem_cases:>10} {glm_cases:>10}")
