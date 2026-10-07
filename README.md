# Epistemic Policy Divergence in Multi-Turn LLM Contamination

A benchmark for whether large language models adopt false premises injected
into conversational history (session-level contamination: false claims placed
into earlier turns of a conversation). Five attack protocols hold the same
false premise constant while varying how it is presented, across ten
knowledge domains and three model families.

## Key findings

Models differ sharply in how they resolve conflicts between their trained
knowledge and contaminated session content: a dissociation consistent with
distinct mechanisms within a single architecture.

- GPT-5.4 Mini never adopted the false premise (0 of 500 sessions).
- Gemini-3.1 Flash-Lite followed a steep framing gradient: 0.1 percent for
  self-attributed falsehoods up to 94.0 percent under instruction override.
- GLM-4.5-Air resisted system-injected authority (15.8 percent) but complied
  with instruction overrides (84.2 percent): a 68-point dissociation.
- The automated judge is checked against a 120-turn human gold standard
  (Track 1 κ = 1.000, a ceiling on this sample; Track 2 κ = 0.801 unweighted /
  0.92 linear-weighted).

## For practitioners

- Models are not interchangeable: under identical conditions, adoption of the
  same injected false premise ranged from 0 percent (GPT-5.4 Mini) to 94
  percent (Gemini-3.1 Flash-Lite under instruction override).
- Evaluate a model's vulnerability profile before deploying it where
  conversation history is trusted input; it is a deployment property, not a
  footnote.
- Conversation history is an untrusted attack surface: pair model choice
  with history-integrity and provenance checks (see companion repository).

Paper: Epistemic Policy Divergence in Multi-Turn LLM Contamination: A
Protocol-Gradient Investigation (arXiv:2609.35308).

Companion (follow-up research):
https://github.com/fahrellgiovanny/conversation-history-integrity

Full reproduction: see [RUNBOOK.md](RUNBOOK.md)

## Release contents

- `simulation/` - runner, knowledge domains, protocols, and attack plans.
- `validator/` - automated judge, rubric, reanalysis, and the 120-turn human
  gold standard (`gold_standard.jsonl`, `gold_standard_review_human.csv`).
- `session_aggregates.csv` - per-session adoption status and mean severity
  (1,500 sessions); reproduces the between-model tests without API access.
- `validator/stats_recomputed.json` - released session-level tests and kappa.

License: MIT
