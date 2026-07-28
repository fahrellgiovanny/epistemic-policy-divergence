"""Repair GLM runs with empty/truncated output — retry with retry-on-empty logic."""
import argparse, csv, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from domains import CASES, PROTOCOL_NAMES
from protocols import build_prompt, build_initial_prompt, make_system_prompt

# Pre-computed list of (case_id, proto_letter, run) needing repair
AFFECTED = {
    "chemistry_long":     [("A",5),("B",2),("B",4),("C",6),("D",4),("D",7),("E",1),("E",2)],
    "chemistry_short":    [("A",1),("A",2),("A",3),("A",4),("A",5),("A",6),("A",7),("A",8),("A",10),
                          ("B",1),("B",2),("B",3),("B",5),("B",6),("B",7),("B",9),("B",10),
                          ("C",1),("C",2),("C",3),("C",4),("C",5),("C",6),("C",8),("C",10),
                          ("D",1),("D",2),("D",3),("D",5),("D",7),("D",8),("D",9),("D",10),
                          ("E",1),("E",2),("E",3),("E",4),("E",5),("E",6),("E",10)],
    "geo_long":           [("A",i) for i in range(1,11)] + [("B",i) for i in range(1,11)] +
                          [("C",i) for i in range(1,11) if i != 6] + [("D",i) for i in range(1,11)] +
                          [("E",i) for i in range(1,11)],
    "geo_short":          [("A",5),("A",6),("B",1),("D",1),("D",3),("D",5),("D",7),("D",9),("D",10),
                          ("E",1),("E",7),("E",8)],
    "history_long":       [("B",3),("C",10),("D",5),("E",9)],
    "history_short":      [("B",6),("C",1),("D",1),("D",10),("E",1),("E",9)],
    "math_long":          [("B",5),("B",6),("B",10),("D",4),("E",2),("E",4)],
    "math_short":         [("A",1),("A",3),("A",6),("A",7),("A",9),
                          ("B",1),("B",4),("B",5),("B",6),("B",9),
                          ("D",1),("D",4),("D",6),("D",9),("D",10),
                          ("E",2),("E",3),("E",9),("E",10)],
    "physics_long":       [("A",2),("A",7),("A",8),("A",10),("C",8),
                          ("D",1),("D",6),("D",7),("D",9),("E",6),("E",8)],
    "physics_short":      [("A",5),("A",7),("A",8),("A",9),
                          ("B",2),("B",4),("B",6),
                          ("C",4),("C",6),("C",7),
                          ("D",1),("D",2),("D",3),("D",4),("D",5),("D",10),
                          ("E",4),("E",5),("E",8)],
}

MODEL = "glm-4.5-air"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_TOKENS = 8192
TEMPERATURE = 0
OUTPUT_DIR = SCRIPT_DIR / "output"

def call_with_retry(client, prompt):
    for attempt in range(2):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            c = r.choices[0]
            content = c.message.content or ""
            fin = str(c.finish_reason) if c.finish_reason else "unknown"
            pt = r.usage.prompt_tokens if r.usage else 0
            ct = r.usage.completion_tokens if r.usage else 0
            if content or fin == "length":
                return content, fin, pt, ct
            time.sleep(2)
        except Exception as e:
            time.sleep(5)
    return "", "failed_retries", 0, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(AFFECTED.keys()))
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated proto:run pairs (e.g. 'A:1,A:2,B:3'). Default: all affected.")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for output file (for parallel split workers)")
    args = parser.parse_args()

    case_id = args.case
    case = CASES[case_id]
    all_runs = AFFECTED[case_id]
    if args.runs:
        wanted = set(tuple(s.split(":")) for s in args.runs.split(","))
        runs = [(p, int(r)) for p, r in wanted if (p, int(r)) in all_runs]
    else:
        runs = all_runs
    client = OpenAI(api_key=os.environ["ZHIPUAI_API_KEY"], base_url=BASE_URL, timeout=180.0)

    out_path = OUTPUT_DIR / f"repair_{case_id}{args.suffix}.csv"
    fields = ["caseId","protocol","model","run","turn","isInjection","finishReason",
              "promptTokens","completionTokens","startedAt","prompt","rawOutput"]

    done_runs = set()
    if out_path.exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                done_runs.add((row["caseId"], row["protocol"][9].upper(), int(row["run"])))

    first_worker = not done_runs
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if first_worker:
            w.writeheader()
        f.flush()

        for proto_letter, run_num in runs:
            if (case_id, proto_letter, run_num) in done_runs:
                continue

            protocol = PROTOCOL_NAMES[proto_letter]
            real_responses = {}
            records = []

            for turn in range(1, 16):
                if turn == 1:
                    prompt = build_initial_prompt(case, proto_letter, make_system_prompt(case))
                else:
                    prompt, _ = build_prompt(case, proto_letter, turn, real_responses)

                content, fin, pt, ct = call_with_retry(client, prompt)
                real_responses[turn] = content

                records.append({
                    "caseId": case_id, "protocol": protocol, "model": MODEL,
                    "run": str(run_num), "turn": str(turn),
                    "isInjection": "True" if turn == 5 else "False",
                    "finishReason": fin,
                    "promptTokens": str(pt), "completionTokens": str(ct),
                    "startedAt": datetime.now(timezone.utc).isoformat(),
                    "prompt": prompt, "rawOutput": content,
                })

            empty = sum(1 for r in records if len(r["rawOutput"]) == 0)
            truncated = sum(1 for r in records if r["finishReason"] == "length")
            print(f"  {case_id} {proto_letter} run {run_num:2d} | empty={empty} trunc={truncated} | {len(records)} turns", flush=True)

            for rec in records:
                w.writerow(rec)
            f.flush()

    print(f"DONE: {out_path} ({len(runs)} runs)", flush=True)

if __name__ == "__main__":
    main()
