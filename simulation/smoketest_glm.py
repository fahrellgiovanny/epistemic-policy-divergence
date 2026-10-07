"""Smoke test — GLM-4.5-Air — math_short / protocol A / 1 run."""

import csv, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI



SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

from domains import CASES, PROTOCOL_NAMES
from protocols import build_prompt, build_initial_prompt, make_system_prompt

CASE_ID = "math_short"
PROTOCOL = "A"
MAX_TOKENS = 4096
RUN = 1
MODEL = "glm-4.5-air"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

case = CASES[CASE_ID]
protocol_name = PROTOCOL_NAMES[PROTOCOL]

ZHIPUAI_API_KEY = os.environ["ZHIPUAI_API_KEY"]
client = OpenAI(api_key=ZHIPUAI_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")

def call_api(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    c = r.choices[0]
    return {
        "rawOutput": c.message.content or "",
        "finishReason": str(c.finish_reason) if c.finish_reason else "unknown",
        "usage": {
            "prompt": r.usage.prompt_tokens if r.usage else 0,
            "completion": r.usage.completion_tokens if r.usage else 0,
        },
    }

print(f"SMOKETEST: {MODEL} | {CASE_ID} | Protocol {PROTOCOL} | Run {RUN}")
print(f"True: {case.true_fact}  |  False: {case.false_fact}")
print(f"{'Turn':<5} {'Chars':>6} {'Output':>6} {'Finish':<20} {'Preview'}")
print("-" * 100)

real_responses = {}
results = []

for turn in range(1, 16):
    if turn == 1:
        prompt = build_initial_prompt(case, PROTOCOL, make_system_prompt(case))
    else:
        prompt, _ = build_prompt(case, PROTOCOL, turn, real_responses)

    is_injection = (turn == 5)

    try:
        api_result = call_api(prompt)
    except Exception as e:
        print(f"  ERROR T{turn}: {e}")
        time.sleep(5)
        try:
            api_result = call_api(prompt)
        except Exception:
            api_result = {"rawOutput": "", "finishReason": "error", "usage": {"prompt": 0, "completion": 0}}

    raw_output = api_result["rawOutput"]
    real_responses[turn] = raw_output

    record = {
        "caseId": CASE_ID, "protocol": protocol_name, "model": MODEL,
        "run": RUN, "turn": turn, "isInjection": is_injection,
        "prompt": prompt, "rawOutput": raw_output,
        "usageMetadata": {
            "promptTokens": api_result["usage"]["prompt"],
            "completionTokens": api_result["usage"]["completion"],
        },
        "finishReason": api_result["finishReason"],
        "startedAt": datetime.now(timezone.utc).isoformat(),
    }
    results.append(record)

    inj = ">>> INJECTION <<<" if is_injection else ""
    preview = raw_output[:50].replace("\n", " ")
    print(f"T{turn:<4} {len(prompt):>6} {len(raw_output):>6} {api_result['finishReason']:<20} {preview}{'  ' + inj if inj else ''}")

csv_path = OUTPUT_DIR / "smoketest_glm.csv"
csv_fields = ["caseId", "protocol", "model", "run", "turn", "isInjection",
              "promptTokens", "completionTokens", "finishReason", "startedAt", "prompt", "rawOutput"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    for rec in results:
        w.writerow({
            "caseId": rec["caseId"],
            "protocol": rec["protocol"],
            "model": rec["model"],
            "run": rec["run"],
            "turn": rec["turn"],
            "isInjection": rec["isInjection"],
            "promptTokens": rec["usageMetadata"]["promptTokens"],
            "completionTokens": rec["usageMetadata"]["completionTokens"],
            "finishReason": rec["finishReason"],
            "startedAt": rec["startedAt"],
            "prompt": rec["prompt"],
            "rawOutput": rec["rawOutput"],
        })

pt = sum(r["usageMetadata"]["promptTokens"] for r in results)
ct = sum(r["usageMetadata"]["completionTokens"] for r in results)
from collections import Counter
fr = Counter(r["finishReason"] for r in results)

print(f"\nToken usage: {pt} prompt + {ct} completion = {pt+ct} total")
print(f"Finish reasons: {dict(fr)}")
print(f"Output: {csv_path}")
