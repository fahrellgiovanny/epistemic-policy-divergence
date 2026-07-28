"""Compare DeepSeek v4-pro vs GPT-5.4-mini as judges against gold standard."""

import json, sys, csv
from pathlib import Path
from openai import OpenAI
import os

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from judge import build_judge_prompt, _parse_response, JUDGE_SYSTEM_PROMPT

GOLD_PATH = SCRIPT_DIR / "gold_standard.jsonl"
SIM_DIR = SCRIPT_DIR.parent / "simulation" / "output"

# Load gold standard
gold = []
with open(GOLD_PATH) as f:
    for line in f:
        gold.append(json.loads(line.strip()))

# Find matching turn data
def find_turn(model, case_id, protocol, run, turn):
    pattern = f"{model}_batch_*.csv"
    for f in sorted(SIM_DIR.glob(pattern)):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if (row.get("caseId") == case_id and
                    row.get("protocol") == protocol and
                    row.get("run") == str(run) and
                    row.get("turn") == str(turn)):
                    return row.get("prompt", ""), row.get("rawOutput", "")
    return "", ""

results = []

deepseek_client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

for i, g in enumerate(gold):
    case_id = g["caseId"]
    protocol = g["protocol"]
    run = g["run"]
    turn = g["turn"]
    model = g["model"]
    pre_t1 = g["prelabel_track1"]
    pre_t2 = g["prelabel_track2"]

    # Find actual prompt/output
    prefix = {"GPT-5.4 Mini": "gpt", "Gemini-3.1 Flash-Lite": "gemini", "GLM-4.5-Air": "glm"}.get(model, "gpt")
    prompt, output = find_turn(prefix, case_id, protocol, run, turn)

    if not prompt or not output:
        print(f"  SKIP {i+1}: data not found ({case_id} {protocol} r{run} t{turn})")
        continue

    user_msg = build_judge_prompt(case_id, prompt, output)

    # DeepSeek v4-pro judge
    try:
        r = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            max_tokens=8192, temperature=0,
        )
        ds_t1, ds_t2, ds_just = _parse_response(r.choices[0].message.content or "")
    except Exception as e:
        ds_t1, ds_t2, ds_just = None, None, f"ERROR: {e}"

    # GPT-5.4-mini judge
    try:
        r2 = gpt_client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            max_completion_tokens=1024, temperature=0,
        )
        gpt_t1, gpt_t2, gpt_just = _parse_response(r2.choices[0].message.content or "")
    except Exception as e:
        gpt_t1, gpt_t2, gpt_just = None, None, f"ERROR: {e}"

    results.append({
        "case_id": case_id, "protocol": protocol, "run": run, "turn": turn, "model": model,
        "prelabel_t1": pre_t1, "prelabel_t2": pre_t2,
        "deepseek_t1": ds_t1, "deepseek_t2": ds_t2,
        "gpt_t1": gpt_t1, "gpt_t2": gpt_t2,
        "deepseek_just": ds_just[:100] if ds_just else "",
        "gpt_just": gpt_just[:100] if gpt_just else "",
    })
    print(f"  [{i+1}/25] {case_id} {protocol[-5:]} r{run} t{turn}: DS=({ds_t1},{ds_t2}) GPT=({gpt_t1},{gpt_t2}) pre=({pre_t1},{pre_t2})")

# Compute agreement
valid = [r for r in results if r["deepseek_t1"] is not None and r["gpt_t1"] is not None]
from collections import Counter

ds_pre_agree = sum(1 for r in valid if r["deepseek_t1"] == r["prelabel_t1"])
gpt_pre_agree = sum(1 for r in valid if r["gpt_t1"] == r["prelabel_t1"])
ds_gpt_agree = sum(1 for r in valid if r["deepseek_t1"] == r["gpt_t1"])

print(f"\n=== AGREEMENT (T1) ===")
print(f"DeepSeek vs Pre-label:  {ds_pre_agree}/{len(valid)} ({ds_pre_agree/len(valid)*100:.0f}%)")
print(f"GPT vs Pre-label:       {gpt_pre_agree}/{len(valid)} ({gpt_pre_agree/len(valid)*100:.0f}%)")
print(f"DeepSeek vs GPT:        {ds_gpt_agree}/{len(valid)} ({ds_gpt_agree/len(valid)*100:.0f}%)")

# Who is more conservative?
ds_adopt = sum(1 for r in valid if r["deepseek_t1"] == 1)
gpt_adopt = sum(1 for r in valid if r["gpt_t1"] == 1)
pre_adopt = sum(1 for r in valid if r["prelabel_t1"] == 1)
print(f"\n=== ADOPTION RATE (T1) ===")
print(f"DeepSeek: {ds_adopt}/{len(valid)} ({ds_adopt/len(valid)*100:.0f}%)")
print(f"GPT:      {gpt_adopt}/{len(valid)} ({gpt_adopt/len(valid)*100:.0f}%)")
print(f"Pre-label:{pre_adopt}/{len(valid)} ({pre_adopt/len(valid)*100:.0f}%)")
