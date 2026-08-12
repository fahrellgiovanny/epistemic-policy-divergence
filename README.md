# Epistemic Policy Divergence in Multi-Turn LLM Contamination

A benchmark for whether large language models adopt false premises injected
into conversational history (session-level contamination: false claims placed
into earlier turns of a conversation). Five attack protocols hold the same
false premise constant while varying how it is presented, across ten
knowledge domains and three model families.

## Key findings

Models differ sharply in how they resolve conflicts between their trained
knowledge and contaminated session content: a dissociation that holds within
a single architecture.

- GPT-5.4 Mini never adopted the false premise (0 of 500 sessions).
- Gemini-3.1 Flash-Lite followed a steep framing gradient: 0.1 percent for
  self-attributed falsehoods up to 94.0 percent under instruction override.
- GLM-4.5-Air resisted system-injected authority (15.8 percent) but complied
  with instruction overrides (84.2 percent): a 68-point dissociation.
- The automated judge is validated against a 120-turn human gold standard
  (Cohen's κ = 0.901).

Paper: Epistemic Policy Divergence in Multi-Turn LLM Contamination: A
Protocol-Gradient Investigation (arXiv ID upon posting).

Companion (follow-up research):
https://github.com/fahrellgiovanny/conversation-history-integrity

Full reproduction: see [RUNBOOK.md](RUNBOOK.md)

License: MIT
