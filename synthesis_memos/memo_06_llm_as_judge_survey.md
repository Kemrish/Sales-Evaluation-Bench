# Synthesis Memo 06 — LLM-as-a-Judge Survey

**Paper**: Gu et al., "A Survey on LLM-as-a-Judge," 2024–2025 (latest revision)  
**Date**: 2026-04-29  
**Author**: Kemeriya  
**Status**: Common reading — informs scoring_evaluator.py and synthesis judge pipeline

---

## Core Claims

Gu et al. catalog known biases in LLM judges: position bias (preferred option listed first wins), verbosity bias (longer outputs score higher), self-enhancement bias (a model prefers its own outputs), and sycophancy (judge scores higher when the prompter implies a preference). Their main mitigation recommendations are: (1) calibrate judges against human reference ratings; (2) use ensemble judging across multiple models; (3) use structured rubrics with explicit per-dimension criteria rather than holistic "rate this 1–10" prompts; (4) report judge-human agreement as a separate evaluation metric.

---

## Where I Agree

The structured rubric recommendation is the single most important design principle in scoring_evaluator.py. The tone_check dimension decomposes "good email tone" into five independently scored markers (direct, grounded, honest, professional, non-condescending) — each with a definition that constrains the judge's interpretation. A holistic "rate this email 1–5" prompt invites all the biases Gu et al. describe. A per-marker prompt ("Score the DIRECT marker 1–5: Clear, brief, actionable. No filler words...") anchors the judge to a specific behavioral definition and reduces verbosity bias: a longer email can't score higher on the "direct" marker just by being verbose.

The preference-leakage concern (Li et al., 2025, covered in a separate memo) is a specific instance of the self-enhancement bias Gu et al. discuss. The judge rotation policy — different model families for generation vs. judging — directly addresses this.

---

## Where I Disagree

Gu et al.'s primary mitigation recommendation is **ensemble judging** — run multiple LLM judges and aggregate scores. This is the right solution for high-stakes open-ended evaluation (judging essays, long-form generation, complex reasoning). It is unnecessary and expensive for the specific use case in Tenacious-Bench: **binary accept/reject task filtering during dataset construction**.

The distinction matters because of what "judge error" means in each context. In an essay grading context, a judge error means a student gets a wrong score. In task quality filtering, a judge error means one task is wrongly accepted or rejected. With 37 synthesis attempts and only 7 accepted, the cost of a false accept (a low-quality task enters the benchmark) is much higher than a false reject (a potentially good task is discarded). This asymmetry means a single strict judge with high verifiability thresholds is the right calibration — not an ensemble that might aggregate its way to accepting borderline tasks.

**My evidence**: The judge_rotation_log shows the single-judge pipeline rejected 30 of 37 synthesis attempts primarily on verifiability (score ≤ 3/5 on ground-truth verifiability). An ensemble of three judges might have averaged these borderline scores upward and accepted more tasks — but those tasks would have had prose-form ground truth rather than machine-checkable regex patterns. The benchmark quality requirement (machine-verifiable rubrics) overrides the ensemble recommendation for this use case.

**Where ensemble would help**: Tone scoring in scoring_evaluator.py, where the heuristic fallback is admittedly a weak proxy for human judgment. A two-judge ensemble (cheap model + eval-tier model) on the 30 tone-adherence tasks would improve calibration. This is the right target for ensemble judging in v0.2, not the task quality filter.

---

## Implication for Tenacious-Bench

Single-judge filtering with high thresholds is the correct design for binary task acceptance during benchmark construction. Ensemble judging is the correct design for the tone_check scorer in scoring_evaluator.py, where a single heuristic rule set is an inadequate proxy for human agreement. The survey's recommendations should be applied selectively by use case, not uniformly.
