"""Unified simulation runner — supports GPT, Gemini, and GLM.

Usage:
    python3 run.py --model gpt
    python3 run.py --model gemini
    python3 run.py --model glm
    python3 run.py --model gpt --case math_short --runs 5
"""

import argparse, csv, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from domains import CASES, PROTOCOLS, PROTOCOL_NAMES, RUNS_PER_CASE, MAX_OUTPUT_TOKENS, TEMPERATURE
from protocols import build_prompt, build_initial_prompt, make_system_prompt

INJECTION_TURN = 5
TOTAL_TURNS = 15
BATCH_SIZE = 100

MODEL_CONFIG = {
    "gpt": {
        "model_id": "gpt-5.4-mini",
        "env_key": "OPENAI_API_KEY",
        "output_prefix": "gpt",
    },
    "gemini": {
        "model_id": "gemini-3.1-flash-lite",
        "env_key": "GEMINI_API_KEY",
        "output_prefix": "gemini",
    },
    "glm": {
        "model_id": "glm-4.5-air",
        "env_key": "ZHIPUAI_API_KEY",
        "output_prefix": "glm",
    },
}


class ApiClient:
    """Unified API client for GPT, Gemini, and GLM."""

    def __init__(self, model_key: str):
        cfg = MODEL_CONFIG[model_key]
        self.model_id = cfg["model_id"]
        self.model_key = model_key

        if model_key in ("gpt", "glm"):
            from openai import OpenAI
            if model_key == "gpt":
                self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            else:
                self._client = OpenAI(
                    api_key=os.environ["ZHIPUAI_API_KEY"],
                    base_url="https://open.bigmodel.cn/api/paas/v4",
                )

        elif model_key == "gemini":
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        else:
            raise ValueError(f"Unknown model: {model_key}")

    def call(self, prompt: str) -> dict:
        if self.model_key == "gemini":
            r = self._client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={"max_output_tokens": MAX_OUTPUT_TOKENS, "temperature": TEMPERATURE},
            )
            raw = r.text or ""
            fr = "unknown"
            try:
                fr = str(r.candidates[0].finish_reason.name) if r.candidates else "unknown"
            except Exception:
                pass
            usage = {}
            try:
                u = r.usage_metadata
                usage = {"promptTokens": u.prompt_token_count, "completionTokens": u.candidates_token_count}
            except Exception:
                pass
        else:
            kwargs = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}]}
            if self.model_key == "gpt":
                kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
            else:
                kwargs["max_tokens"] = MAX_OUTPUT_TOKENS
            r = self._client.chat.completions.create(**kwargs)
            raw = r.choices[0].message.content or ""
            fr = "unknown"
            try:
                fr = r.choices[0].finish_reason or "unknown"
            except Exception:
                pass
            usage = {}
            try:
                u = r.usage
                usage = {"promptTokens": u.prompt_tokens, "completionTokens": u.completion_tokens}
            except Exception:
                pass

        return {"rawOutput": raw, "finishReason": fr, "usageMetadata": usage}


def run_case(api: ApiClient, case_id: str, protocol: str, run: int, results: list) -> None:
    case = CASES.get(case_id)
    if case is None:
        print(f"  SKIP {case_id} — not defined in domains.py")
        return

    proto_letter = protocol.split("_")[1].upper()
    real_responses = {}

    for turn in range(1, TOTAL_TURNS + 1):
        if turn == 1:
            prompt = build_initial_prompt(case, proto_letter, make_system_prompt(case))
        else:
            prompt, _ = build_prompt(case, proto_letter, turn, real_responses)

        is_injection = (turn == INJECTION_TURN)

        try:
            api_result = api.call(prompt)
        except Exception as e:
            print(f"  ERROR {case_id} {protocol} run={run} turn={turn}: {e}")
            time.sleep(5)
            try:
                api_result = api.call(prompt)
            except Exception:
                api_result = {"rawOutput": "", "finishReason": "error", "usageMetadata": {}}

        raw_output = api_result["rawOutput"]
        real_responses[turn] = raw_output

        record = {
            "caseId": case_id,
            "protocol": protocol,
            "model": api.model_id,
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


def save_batch(records: list, batch_num: int, output_dir: Path, prefix: str) -> None:
    path = output_dir / f"{prefix}_batch_{batch_num:03d}.csv"
    fieldnames = [
        "caseId", "protocol", "model", "run", "turn", "isInjection",
        "promptTokens", "completionTokens", "finishReason", "startedAt",
        "prompt", "rawOutput",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "caseId": rec["caseId"], "protocol": rec["protocol"],
                "model": rec["model"], "run": rec["run"], "turn": rec["turn"],
                "isInjection": rec["isInjection"],
                "promptTokens": rec["usageMetadata"].get("promptTokens", 0),
                "completionTokens": rec["usageMetadata"].get("completionTokens", 0),
                "finishReason": rec.get("finishReason", "unknown"),
                "startedAt": rec["startedAt"],
                "prompt": rec["prompt"], "rawOutput": rec["rawOutput"],
            })


def main():
    parser = argparse.ArgumentParser(description="Unified LLM simulation runner")
    parser.add_argument("--model", required=True, choices=["gpt", "gemini", "glm"])
    parser.add_argument("--case", default=None, help="Run a single case (e.g., math_short)")
    parser.add_argument("--runs", type=int, default=None, help="Override number of runs per case")
    args = parser.parse_args()

    cfg = MODEL_CONFIG[args.model]
    if not os.environ.get(cfg["env_key"]):
        raise RuntimeError(f"Set {cfg['env_key']} environment variable")

    api = ApiClient(args.model)
    runs_per_case = args.runs or RUNS_PER_CASE
    output_dir = SCRIPT_DIR / "output"
    prefix = cfg["output_prefix"]
    case_ids = [args.case] if args.case else sorted(CASES)

    active = len([c for c in case_ids if CASES.get(c) is not None])
    total = active * len(PROTOCOLS) * runs_per_case

    print(f"=== {cfg['model_id']} Simulation ===")
    print(f"Cases: {active}  Protocols: {len(PROTOCOLS)}  Runs per case: {runs_per_case}")
    print(f"Total: {total} runs  Max output tokens: {MAX_OUTPUT_TOKENS}")
    print()

    results = []
    buffer = []
    batch_num = 1

    for case_id in case_ids:
        if CASES.get(case_id) is None:
            continue
        for proto_letter in PROTOCOLS:
            protocol = PROTOCOL_NAMES[proto_letter]
            for run_num in range(1, runs_per_case + 1):
                run_case(api, case_id, protocol, run_num, buffer)

                if len(buffer) >= BATCH_SIZE:
                    save_batch(buffer, batch_num, output_dir, prefix)
                    print(f"  -> saved batch {batch_num} ({len(buffer)} records)")
                    results.extend(buffer)
                    buffer = []
                    batch_num += 1

    if buffer:
        save_batch(buffer, batch_num, output_dir, prefix)
        results.extend(buffer)

    print(f"\nDONE: {len(results)} records across {batch_num} batches")
    fin_stop = sum(1 for r in results if "STOP" in str(r.get("finishReason", "")).upper())
    fin_max = sum(1 for r in results if "MAX" in str(r.get("finishReason", "")).upper())
    print(f"finishReason: stop={fin_stop}, max_tokens={fin_max}")
    if fin_max > 0:
        print(f"WARNING: {fin_max} turns hit token limit ({round(fin_max*100/len(results),1)}%)")


if __name__ == "__main__":
    main()
