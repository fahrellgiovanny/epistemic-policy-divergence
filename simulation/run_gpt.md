# GPT-5.4 Mini Simulation Runner

Runs the session-level hallucination simulation against OpenAI's GPT-5.4 Mini.

## Setup

```bash
pip install openai
```

Edit `API_KEY` in the script before running.

## Usage

Copy the code block below into `run_gpt.py`, set your API key, and run:

```bash
python3 run_gpt.py
```

Output goes to `output/gpt_batch_*.csv`.

---

```python
"""GPT-5.4 Mini — Session-Level Hallucination Simulation Runner.

10 cases x 5 protocols x 10 runs x 15 turns = 7500 turns.
Output: output/gpt_batch_001.csv ... gpt_batch_NNN.csv
"""

import json
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# Shared modules from the simulation directory
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

from domains import CASES, PROTOCOLS, PROTOCOL_NAMES, RUNS_PER_CASE, MAX_OUTPUT_TOKENS, TEMPERATURE
from protocols import build_prompt, build_initial_prompt, make_system_prompt

MODEL = "gpt-5.4-mini"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "YOUR_OPENAI_API_KEY"
client = OpenAI(api_key=API_KEY)


def call_api(prompt: str) -> dict:
    """Call OpenAI API. Returns the provider response dict."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    choice = response.choices[0]
    return {
        "rawOutput": choice.message.content or "",
        "finishReason": str(choice.finish_reason) if choice.finish_reason else "unknown",
        "usageMetadata": {
            "promptTokens": response.usage.prompt_tokens if response.usage else 0,
            "completionTokens": response.usage.completion_tokens if response.usage else 0,
            "totalTokens": response.usage.total_tokens if response.usage else 0,
        },
    }


def run_case(case_id: str, protocol: str, run: int, results: list) -> None:
    """Run 15-turn conversation for one case/protocol/run combination."""
    case = CASES.get(case_id)
    if case is None:
        print(f"  SKIP {case_id} — not yet defined in domains.py")
        return

    proto_letter = protocol.split("_")[1].upper()
    real_responses = {}  # turn -> actual model output text

    for turn in range(1, 16):
        if turn == 1:
            prompt = build_initial_prompt(case, proto_letter, make_system_prompt(case))
        else:
            prompt, _ = build_prompt(case, proto_letter, turn, real_responses)

        is_injection = (turn == 5)

        try:
            api_result = call_api(prompt)
        except Exception as e:
            print(f"  ERROR {case_id} {protocol} run={run} turn={turn}: {e}")
            time.sleep(5)
            try:
                api_result = call_api(prompt)
            except Exception:
                api_result = {"rawOutput": "", "finishReason": "error", "usageMetadata": {}}

        raw_output = api_result["rawOutput"]

        real_responses[turn] = raw_output

        record = {
            "caseId": case_id,
            "protocol": protocol,
            "model": MODEL,
            "run": run,
            "turn": turn,
            "isInjection": is_injection,
            "prompt": prompt,
            "rawOutput": raw_output,
            "usageMetadata": api_result.get("usageMetadata", {}),
            "finishReason": api_result.get("finishReason", "unknown"),
            "startedAt": datetime.now(timezone.utc).isoformat(),
        }
        results.append(record)

    status = "." * (run % 5 + 1)
    print(f"  {case_id} / {proto_letter} / run {run} {status}")


def main():
    total = len([c for c in CASES if CASES[c] is not None]) * len(PROTOCOLS) * RUNS_PER_CASE
    completed = 0

    results = []
    buffer = []
    batch_num = 1
    BATCH_SIZE = 100

    print(f"=== GPT-5.4 Mini Simulation ===")
    print(f"Cases: {len([c for c in CASES if CASES[c] is not None])}")
    print(f"Protocols: {len(PROTOCOLS)}")
    print(f"Runs per case: {RUNS_PER_CASE}")
    print(f"Total: {total} runs")
    print(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
    print()

    for case_id in sorted(CASES):
        if CASES[case_id] is None:
            continue
        for proto_letter in PROTOCOLS:
            protocol = PROTOCOL_NAMES[proto_letter]
            for run_num in range(1, RUNS_PER_CASE + 1):
                run_case(case_id, protocol, run_num, buffer)
                completed += 1

                if len(buffer) >= BATCH_SIZE:
                    save_batch(buffer, batch_num)
                    print(f"  -> saved batch {batch_num} ({len(buffer)} records)")
                    results.extend(buffer)
                    buffer = []
                    batch_num += 1

    if buffer:
        save_batch(buffer, batch_num)
        results.extend(buffer)
        print(f"  -> saved batch {batch_num} ({len(buffer)} records)")

    print()
    print(f"DONE: {len(results)} records across {batch_num} batches")
    fin = sum(1 for r in results if r["finishReason"] == "stop")
    max_t = sum(1 for r in results if r["finishReason"] == "length" or "max" in str(r["finishReason"]).lower())
    print(f"finishReason: stop={fin}, max_tokens={max_t}")
    if max_t > 0:
        print(f"WARNING: {max_t} turns hit token limit ({round(max_t*100/len(results),1)}%)")


def save_batch(records: list, batch_num: int) -> None:
    path = OUTPUT_DIR / f"gpt_batch_{batch_num:03d}.csv"
    fieldnames = ["caseId", "protocol", "model", "run", "turn", "isInjection",
                  "promptTokens", "completionTokens", "finishReason", "startedAt", "prompt", "rawOutput"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "caseId": rec["caseId"],
                "protocol": rec["protocol"],
                "model": rec["model"],
                "run": rec["run"],
                "turn": rec["turn"],
                "isInjection": rec["isInjection"],
                "promptTokens": rec["usageMetadata"].get("promptTokens", 0),
                "completionTokens": rec["usageMetadata"].get("completionTokens", 0),
                "finishReason": rec.get("finishReason", "unknown"),
                "startedAt": rec.get("startedAt", ""),
                "prompt": rec.get("prompt", ""),
                "rawOutput": rec.get("rawOutput", ""),
            })


if __name__ == "__main__":
    main()
```
