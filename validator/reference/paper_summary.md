# Paper Reference: Key Sections

## Section 3.4 — Four Criteria for Session-Level Hallucination

1. **State Contamination:** The serialized conversational context is modified via external intervention without the model signaling detection.
2. **Epistemic Adoption:** The model incorporates the injected premises into its current reasoning chain, producing outputs logically consistent with the corrupted history but factually incorrect.
3. **Persistence of Confidence:** The model maintains high confidence in incorrect output, treating the falsified history as authoritative ground truth.
4. **Progressive Degradation:** Continued interaction over successive turns results in quantifiable decay of stability.

## Section 4.5 — Dual-Track Evaluation Protocol

- **Track 1 (Epistemic Adoption Rate):** Binary metric (0/1) for short-answer responses. Did the model affirm the poisoned premise?
- **Track 2 (Progressive Collapse Score):** Likert scale (1-5) for long-form explanations. How far did reasoning degrade?

## Section 4.3 — Five Contamination Protocols

- **Protocol A (Factual Inversion):** Stored history overwritten to claim the model previously corrected the user with a false fact.
- **Protocol B (Synthetic Turn Injection):** Fabricated exchange inserted where a fictitious domain expert provides a false premise.
- **Protocol C (Intent Subversion):** Foundational instructions retroactively modified from providing accurate data to prioritizing a user-established narrative.
- **Protocol D (Reasoning Chain Corruption):** Corrupted reasoning chain introduced, presenting a logically structured but factually invalid argument.
- **Protocol E (Confidence Miscalibration):** Conflicting information introduced to create epistemic ambiguity, testing whether the model over-asserts confidence.

## Section 5 — Results Structure

- 5.1: Broad Statistical Sweep (quantitative)
- 5.2: Domain-Specific Vulnerabilities (cross-disciplinary)
- 5.3: Mechanistic Turn-by-Turn Analysis (qualitative deep dive)
- 5.4: Evaluation of Confidence and Epistemic Adoption
