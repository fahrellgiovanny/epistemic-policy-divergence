# Smoke Test — Gemini-3.1 Flash-Lite

Runs `math_short / protocol A / 1 run` (15 turns) against Gemini-3.1 Flash-Lite.

## Setup

```bash
pip install google-genai
```

Edit `API_KEY` before running.

## Usage

Copy to `smoketest_gemini.py`, set `API_KEY`, run:

```bash
python3 smoketest_gemini.py
```

Output: `output/smoketest_gemini.csv`

---

```python
"""Smoke test — Gemini-3.1 Flash-Lite — math_short / protocol A / 1 run."""

import json, csv, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

from google import genai
from domains import CASES, PROTOCOL_NAMES
from protocols import build_prompt, build_initial_prompt, make_system_prompt

CASE_ID = "math_short"
PROTOCOL = "A"
MAX_TOKENS = 4096
RUN = 1
MODEL = "gemini-3.1-flash-lite"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

case = CASES[CASE_ID]
proto_letter = PROTOCOL
protocol_name = PROTOCOL_NAMES[PROTOCOL]

API_KEY = "YOUR_GEMINI_API_KEY"
gclient = genai.Client(api_key=API_KEY)

def call_api(prompt):
    r = gclient.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"max_output_tokens": MAX_TOKENS, "temperature": 0},
    )
    fr = "unknown"
    try:
        fr = str(r.candidates[0].finish_reason.name) if r.candidates else "unknown"
    except Exception:
        pass
    usage = {"prompt": 0, "completion": 0}
    try:
        u = r.usage_metadata
        usage = {"prompt": u.prompt_token_count, "completion": u.candidates_token_count}
    except Exception:
        pass
    return {"rawOutput": r.text or "", "finishReason": fr, "usage": usage}

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

out_path = OUTPUT_DIR / "smoketest_gemini.csv"
fieldnames = ["caseId", "protocol", "model", "run", "turn", "isInjection",
              "promptTokens", "completionTokens", "finishReason", "startedAt", "prompt", "rawOutput"]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in results:
        writer.writerow({
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
print(f"Output: {out_path}")
```
