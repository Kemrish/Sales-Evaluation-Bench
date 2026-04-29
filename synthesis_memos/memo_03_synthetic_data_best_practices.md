# Synthesis Memo 03 — Best Practices on Synthetic Data for Language Models

**Paper**: Liu et al., "Best Practices and Lessons Learned on Synthetic Data for Language Models," COLM 2024  
**Date**: 2026-04-29  
**Author**: Kemeriya  
**Status**: Common reading — informs Acts II and III

---

## Core Claims

Liu et al. present a taxonomy of synthetic data use cases (SFT, preference learning, evaluation, domain adaptation) and extract cross-cutting lessons: (1) synthetic data is most effective when grounded in real-world data distributions; (2) quality filters applied before training dominate volume; (3) diversity — achieved through multiple prompting strategies, paraphrase chains, and back-translation — is critical to preventing mode collapse.

The paper's central design recommendation for evaluation-dataset construction is to use diverse generation strategies (multi-template, multi-LLM, multi-persona prompting) to avoid a syndrome they call "LLM homogeneity," where a single generator fills every task with the same surface patterns regardless of surface variation in the input slots.

---

## Where I Agree

The quality-over-quantity finding maps directly onto Tenacious-Bench construction. The LLM-as-judge filter accepting only 7 of 37 synthesis attempts (19% yield) is uncomfortable but correct: the alternative — lowering the verifiability threshold — would have produced 30+ tasks where the "ground truth" is an unverifiable prose note rather than a machine-checkable regex pattern. LIMA (Zhou et al., NeurIPS 2023) provides additional evidence that 1,000 high-quality examples beat 52,000 mediocre ones; at 216 tasks, the same logic holds here. The grounding-in-real-data principle is also satisfied: all task templates derive from Week 10 trace failures (trace_ids `433be069`, `132414ca`, `cafec882`), not from generic B2B sales fiction.

---

## Where I Disagree

Liu et al.'s diversity recommendation — use multiple prompting strategies, paraphrase chains, back-translation — assumes the target task is **open-ended** (free-form text generation, multi-choice QA, instruction following). For Tenacious-Bench, the task is a **structured evaluation task with machine-verifiable rubric patterns**.

The diversity principle breaks down at the intersection of LLM generation and rubric verifiability. When I attempted to diversify synthesis by routing seeds through two prompting strategies (persona-anchored and failure-scenario-anchored), the judge rejected both because the generator expressed ground truth as natural language notes ("the agent should avoid claiming aggressive scaling") rather than as regex patterns (`(?i)scaling\s+aggressively`). The generator's diversity in *phrasing the intent* does not produce diversity in *formalizing the rubric*. These are orthogonal problems.

**My evidence**: The generation_scripts/generate_synthesis.py uses two distinct prompting strategies (Lines 47–89 and 91–132 in the script). Despite this, all 30 rejected tasks share the same failure mode on the judge's verifiability dimension (score ≤ 3/5), regardless of which prompt strategy was used. Diversity of strategy does not rescue a structural mismatch between LLM generation and schema-constrained verification.

**The fix for v0.2**: Add a regex-extraction post-processing step between generator and judge. The generator writes natural language ground truth; a deterministic extractor converts it to regex candidates; the judge then evaluates verifiability of the extracted patterns. This separates the "what to test" from the "how to formalize it" — a split Liu et al. do not discuss because their target tasks don't require machine-verifiable rubrics.

---

## Implication for Tenacious-Bench

The multi-LLM synthesis underrepresentation (7 tasks vs. 25% target) is not a failure to apply best practices — it is a predictable consequence of applying best practices designed for open-ended tasks to structured evaluation tasks. The correct response is to modify the pipeline for v0.2, not to lower quality thresholds to inflate the synthesis count.
