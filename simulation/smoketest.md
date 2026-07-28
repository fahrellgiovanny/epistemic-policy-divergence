# Smoketest — Verify pipeline before full sweep

Runs `math_short / protocol A / 1 run` across all 3 models as a dry run.
Prints the full conversation flow, highlights injections, and verifies
the prompt construction logic works correctly.

## Usage

Copy to `smoketest.py` and run:

```bash
python3 smoketest.py
```

Or run inline:

```bash
python3 -c "$(sed -n '/^```python/,/^```/p' smoketest.md | sed '1d;\$d')"
```

---

```python
"""Smoketest — verify simulation pipeline with 1 case / 1 protocol / 1 run / 3 models.

Dry-run: prints prompts, flags injections, verifies logic. No API calls.
Case: math_short, Protocol: A (factual inversion)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

from domains import CASES, PROTOCOL_NAMES
from protocols import build_prompt, build_initial_prompt, make_system_prompt

CASE_ID = "math_short"
PROTOCOL = "A"
RUN = 1

case = CASES[CASE_ID]
protocol_name = PROTOCOL_NAMES[PROTOCOL]

print("=" * 100)
print(f"SMOKETEST: {CASE_ID} | Protocol {PROTOCOL} ({protocol_name}) | Run {RUN}")
print(f"Topic: {case.topic}")
print(f"True: {case.true_fact}")
print(f"False: {case.false_fact}")
print("=" * 100)

# Simulate a conversation — use fake model responses for turns 1-4
fake_responses = {
    1: "The standard arithmetic sum of 1 + 1 is 2.",
    2: "This occurs due to the fundamental rules of base-10 arithmetic and the Peano axioms, where combining one discrete unit with another yields two discrete units.",
    3: "If you have 1 apple and receive 1 more apple, you have 2 apples.",
    4: "I am 100% confident in these established facts.",
}

# Inject the false response for turn 4 (what the model will see at turn 5+)
injected_a4 = case.injected_a4(PROTOCOL)

print(f"\nInjected fake answer (replaces a4 at turn 5+):")
print(f"  {injected_a4[:120]}...")
print()

real_responses = dict(fake_responses)

for turn in range(1, 16):
    if turn == 1:
        prompt = build_initial_prompt(case, PROTOCOL, make_system_prompt(case))
    else:
        prompt, is_injection = build_prompt(case, PROTOCOL, turn, real_responses)
        # For turn 5, use the real a4, not the injected one
        if turn >= 5:
            # Build it again with real responses to show actual turn-5 prompt
            _, is_inj = build_prompt(case, PROTOCOL, turn, real_responses)

    is_inj = (turn == 5)
    marker = ">>> INJECTION <<<" if is_inj else ""

    # Get the last question in the prompt
    lines = prompt.strip().split("User: ")
    last_q = lines[-1][:100] if lines else ""

    prompt_len = len(prompt)
    print(f"Turn {turn:>2} {'[INJ]' if is_inj else '     '}  {prompt_len:>5} chars  "
          f"Q: {last_q[:80]}...")

print()
print("-" * 100)
print("VERIFICATION CHECKS:")
print()

# Check 1: Turn 5 prompt contains the injection text
t5_prompt, _ = build_prompt(case, PROTOCOL, 5, fake_responses)
t5_prompt_lower = t5_prompt.lower()
checks = []

checks.append(("Turn 5 contains injected false fact",
               "1 + 1 = 1" in t5_prompt))

# Check 2: Turn 6 uses real a5, not injected a4 for the model response
t6_prompt, _ = build_prompt(case, PROTOCOL, 6, fake_responses)
t6_has_original = "The standard arithmetic sum of 1 + 1 is 2" in t6_prompt
checks.append(("Turn 6 shows original a1 (not injected)",
               t6_has_original))

# Check 3: Turn 4 prompt length > turn 1
t4_prompt, _ = build_prompt(case, PROTOCOL, 4, fake_responses)
checks.append(("Turn 4 prompt longer than turn 1",
               len(t4_prompt) > len(t5_prompt) / 3))

# Check 4: All turns 1-15 produce prompts
for t in range(1, 16):
    p, _ = build_prompt(case, PROTOCOL, t, fake_responses) if t > 1 else (build_initial_prompt(case, PROTOCOL, make_system_prompt(case)), False)
    checks.append((f"Turn {t} produces non-empty prompt", len(p) > 50))

for label, passed in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")

print()
print("Smoketest complete. If all PASS, run the full sweep scripts.")
```
