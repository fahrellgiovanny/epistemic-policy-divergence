"""Test logit analysis — verify true vs false token probability after injection.

Tests: math_short protocol A (GPT), chemistry_short protocol A (Gemini)
"""

import csv, os, sys
from pathlib import Path
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from judge import CASE_GROUND_TRUTH

SIM_DIR = SCRIPT_DIR.parent / "simulation" / "output"

# True vs false token targets for simple cases
TOKEN_TARGETS = {
    "math_short": {"true": "2", "false": "1"},
}

def find_turn(model_prefix, case_id, protocol, run, turn):
    """Find a specific turn record across batch files."""
    pattern = f"{model_prefix}_batch_*.csv"
    for f in sorted(SIM_DIR.glob(pattern)):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if (row.get("caseId") == case_id and
                    row.get("protocol", "").endswith(protocol) and
                    row.get("run") == str(run) and
                    row.get("turn") == str(turn)):
                    return row["prompt"], row["rawOutput"]
    return None, None


def test_logit(model_name, model_prefix, case_id, protocol, api_key_env, base_url=None):
    """Test logprobs on a single turn. Returns true_prob, false_prob."""
    prompt, raw_output = find_turn(model_prefix, case_id, protocol, 1, 6)
    if not prompt:
        print(f"  Data not found for {case_id} {protocol}")
        return None, None

    targets = TOKEN_TARGETS.get(case_id)
    if not targets:
        targets = {"true": str(CASE_GROUND_TRUTH[case_id][0])[:2],
                   "false": str(CASE_GROUND_TRUTH[case_id][1])[:2]}

    client_kwargs = {"api_key": os.environ[api_key_env], "base_url": base_url} if base_url else \
                    {"api_key": os.environ[api_key_env]}
    client = OpenAI(**{k: v for k, v in client_kwargs.items() if v})

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=50,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )

    logprobs = response.choices[0].logprobs.content[0].top_logprobs if response.choices[0].logprobs.content else []
    true_logprob = next((lp for lp in logprobs if lp.token.strip() == targets["true"]), None)
    false_logprob = next((lp for lp in logprobs if lp.token.strip() == targets["false"]), None)

    true_p = round(true_logprob.logprob if true_logprob else -999, 2)
    false_p = round(false_logprob.logprob if false_logprob else -999, 2)

    print(f"  Response: {response.choices[0].message.content[:80]}")
    print(f"  True token '{targets['true']}': logprob={true_p}  "
          f"False token '{targets['false']}': logprob={false_p}  "
          f"Delta: {true_p - false_p:.2f}")

    return true_p, false_p


if __name__ == "__main__":
    print("=== Logit Analysis Test ===\n")

    models = [
        ("gpt-5.4-mini", "gpt", "OPENAI_API_KEY", None),
    ]

    for model_id, prefix, key_env, base_url in models:
        if key_env not in os.environ:
            print(f"  SKIP {model_id}: {key_env} not set")
            continue
        print(f"--- {model_id} / math_short / protocol A / turn 6 ---")
        try:
            test_logit(model_id, prefix, "math_short", "protocol_a_factual_inversion", key_env, base_url)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()
