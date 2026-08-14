# RUNBOOK - Full Reproduction

Step-by-step reproduction of the evaluation described in the paper. All
commands run from this directory.

## 1. Environment

- Python 3.9+
- Dependencies: `google-genai`, `openai`, `numpy`, `scipy`

```
pip install google-genai openai numpy scipy
```

## 2. API keys

Set as environment variables before running:

```
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Roles: GPT, Gemini, and GLM are the evaluated models; DeepSeek is the judge.

## 3. Run the simulation

Sessions are written to `simulation/output/` (git-ignored; every run
regenerates them).

```
cd simulation
python3 run.py --model gpt      # GPT-5.4 Mini
python3 run.py --model gemini   # Gemini-3.1 Flash-Lite
python3 run.py --model glm      # GLM-4.5-Air
```

Or a single case: `python3 run.py --model gpt --case math_short --runs 3`.

A no-API smoke test is available: `python3 smoketest.py`.

## 4. Judge and analyze

```
cd ../validator
python3 run_all.py      # scores adoption (track 1) and severity (track 2)
python3 statistics.py   # chi-square, Kruskal-Wallis, Mann-Whitney U
python3 kappa_check.py  # judge agreement against the gold standard
```

The gold standard (120 human-annotated turns) is in
`validator/gold_standard.jsonl`; the scoring rubric is in
`validator/rules/rubric.json`.

## Structure

```
simulation/            # runner and domain definitions
  run.py               # unified runner (--model gpt|gemini|glm)
  domains.py           # 10 domain cases
  protocols.py         # injection logic for the 5 protocols
  plans/               # complete attack texts (plan_*.txt)
  mitigation_gemini.py # verified-history defense experiment
  ablation_temperature.py  # temperature ablation study
validator/             # evaluation pipeline
  judge.py             # DeepSeek V4 Pro automated judge
  run_all.py           # master orchestrator (parallel workers)
  run_worker.py        # single worker process
  statistics.py        # statistical tests
  logit_analysis.py    # GPT logit-level analysis
  kappa_check.py       # judge agreement vs gold standard
  build_gold_standard.py   # gold standard construction
  gold_standard.jsonl  # human-annotated gold standard (120 turns)
  rules/rubric.json    # scoring rubric
```

## Notes

- No experiment data is committed. Raw model outputs vary across API
  versions, so every run must regenerate its own data; run the pipeline
  above to reproduce the evaluation.
- Model and SDK versions are pinned in the run logs.
