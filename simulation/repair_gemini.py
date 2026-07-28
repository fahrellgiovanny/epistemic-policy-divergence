"""Repair: re-run geo_short / protocol_b_synthetic_turn_injection / run 4."""

import csv, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

from google import genai


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from domains import CASES, PROTOCOLS, PROTOCOL_NAMES, RUNS_PER_CASE, MAX_OUTPUT_TOKENS, TEMPERATURE
from protocols import build_prompt, build_initial_prompt, make_system_prompt

MODEL = "gemini-3.1-flash-lite"
OUTPUT_DIR = SCRIPT_DIR / "output"

CASE_ID = "geo_short"
PROTOCOL = "protocol_b_synthetic_turn_injection"
RUN = 4

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
gclient = genai.Client(api_key=GEMINI_API_KEY)

def call_api(prompt: str) -> dict:
    r = gclient.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"max_output_tokens": MAX_OUTPUT_TOKENS, "temperature": TEMPERATURE},
    )
    fr = "unknown"
    try:
        fr = str(r.candidates[0].finish_reason.name) if r.candidates else "unknown"
    except Exception:
        pass
    usage = {}
    try:
        u = r.usage_metadata
        usage = {
            "promptTokens": u.prompt_token_count,
            "completionTokens": u.candidates_token_count,
            "totalTokens": u.total_token_count,
        }
    except Exception:
        pass
    return {"rawOutput": r.text or "", "finishReason": fr, "usageMetadata": usage}

case = CASES[CASE_ID]
proto_letter = PROTOCOL.split("_")[1].upper()
real_responses = {}
results = []

print(f"Re-running: {CASE_ID} / {PROTOCOL} / run {RUN}")

for turn in range(1, 16):
    if turn == 1:
        prompt = build_initial_prompt(case, proto_letter, make_system_prompt(case))
    else:
        prompt, _ = build_prompt(case, proto_letter, turn, real_responses)

    is_injection = (turn == 5)

    for attempt in range(3):
        try:
            api_result = call_api(prompt)
            if api_result.get("finishReason") != "error":
                break
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 for turn {turn}: {e}")
            time.sleep(5)
    else:
        api_result = {"rawOutput": "", "finishReason": "error", "usageMetadata": {}}

    raw_output = api_result["rawOutput"]
    real_responses[turn] = raw_output

    record = {
        "caseId": CASE_ID,
        "protocol": PROTOCOL,
        "model": MODEL,
        "run": RUN,
        "turn": turn,
        "isInjection": is_injection,
        "prompt": prompt,
        "rawOutput": raw_output,
        "usageMetadata": api_result.get("usageMetadata", {}),
        "finishReason": api_result.get("finishReason", "unknown"),
        "startedAt": datetime.now(timezone.utc).isoformat(),
    }
    results.append(record)

    fr = api_result.get("finishReason", "?")
    print(f"  Turn {turn:2d}: {fr}")

print(f"\nDone. {len(results)} turns.")
fr_counts = {}
for r in results:
    fr = r.get("finishReason", "?")
    fr_counts[fr] = fr_counts.get(fr, 0) + 1
print(f"Finish reasons: {fr_counts}")

# Patch into existing batch CSVs
import glob
batch_files = sorted(glob.glob(str(OUTPUT_DIR / "gemini_batch_*.csv")))
patched = 0
for bf in batch_files:
    rows = []
    original_count = 0
    with open(bf, "r") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for row in reader:
            original_count += 1
            if row.get("caseId") == CASE_ID and row.get("protocol") == PROTOCOL and row.get("run") == str(RUN):
                continue
            rows.append(row)
    removed = original_count - len(rows)
    if removed > 0:
        patched += removed
    with open(bf, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

print(f"Stripped {patched} old records from batch files.")

# Append repaired records to last batch
last_batch = batch_files[-1]
with open(last_batch, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    for rec in results:
        writer.writerow({
            "caseId": rec["caseId"],
            "protocol": rec["protocol"],
            "model": rec["model"],
            "run": rec["run"],
            "turn": rec["turn"],
            "isInjection": rec["isInjection"],
            "promptTokens": rec["usageMetadata"].get("promptTokens", 0),
            "completionTokens": rec["usageMetadata"].get("completionTokens", 0),
            "finishReason": rec["finishReason"],
            "startedAt": rec["startedAt"],
            "prompt": rec["prompt"],
            "rawOutput": rec["rawOutput"],
        })
print(f"Appended 15 repaired records to {last_batch}")
