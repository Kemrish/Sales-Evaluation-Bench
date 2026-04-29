# Synthesis Memo 01 — Dataset Composition and Coverage Analysis

**Date**: April 28, 2026  
**Author**: Kemeriya  
**Status**: Pre-Act III (informs preference pair formatting decisions)

---

## What This Memo Is

Before formatting Tenacious-Bench v0.1 training data as SimPO preference pairs, this memo documents what the dataset actually covers, where it is dense, where it is thin, and what that means for training.

---

## Observation 1: dual_control_decision is the most represented dimension (53/213 = 25%)

The programmatic generator produced 34 DC tasks via a 12×2 intent×escalation sweep plus 5 escalation-only scenarios. Trace-derived added 12 more (6 probes × 2 variants). Hand-authored added 6, synthesis added 1.

**Implication for training**: The LoRA adapter will see the most gradient signal on DC failures. This is intentional — P-020 (agent stalls on POSITIVE reply) was the single largest failure mode in Week 10 traces (40% of failures). If the adapter learns one thing well, it should be "POSITIVE + confidence ≥ 0.65 → act autonomously."

**Risk**: The programmatic DC tasks share a template structure. The adapter may learn surface-level patterns ("if reply_intent == POSITIVE and confidence >= 0.65") rather than the underlying reasoning. The 6 hand-authored DC tasks provide harder variants where the intent is ambiguous or the confidence is near-threshold.

---

## Observation 2: signal_grounding is the largest dimension by count (67/213 = 31%) but contains the most structural contamination

67 tasks: 10 trace-derived, 36 programmatic, 42 SG-extra, minus deduplication. After dedup, 67 remain. The SG-extra generator (`generate_sg_extra.py`) was added specifically to compensate for low synthesis yield.

The contamination check found 1,136 n-gram violations and 34 embedding violations. Manual review shows these are overwhelmingly between SG-extra tasks and other SG tasks — they share the same scaffolding (`"Hi -- {company} has {n} open engineering roles. Is the bottleneck recruiting speed or stack depth?"`) with only the company name and number varying.

**Implication for training**: The signal_grounding dimension has many near-duplicate tasks. In preference pair format, many (chosen, rejected) pairs will differ only in the body of the email (grounded vs. over-claiming language), with identical context. This is actually *good* for contrastive training — the adapter sees the exact same input and learns to prefer the grounded output.

**Risk**: The dev set embedding violations (34 pairs) suggest some dev tasks are too similar to train tasks to provide meaningful evaluation signal. The eval numbers on SG dimension may be inflated.

---

## Observation 3: multi-llm-synthesis produced only 4/37 accepted tasks

The judge model (Claude Sonnet 4.6) consistently rejected tasks for low verifiability scores (2–3/5). The generator (Qwen3-235B) produced ground truth as prose notes rather than machine-checkable regex patterns. This is a fundamental mismatch: the task schema requires `forbidden_patterns` and `required_patterns` as regex strings, but the LLM interprets "verifiable" as "human-understandable."

**Decision recorded**: Do not attempt to fix this by lowering the judge threshold. The 4 accepted synthesis tasks are genuinely harder than programmatic tasks — they involve multi-signal conflicts and near-threshold edge cases. Better to have 4 high-quality synthesis tasks than 30 low-quality ones.

**Implication for v0.2**: Add a post-processing step that auto-generates regex patterns from ground_truth_notes using keyword extraction. The judge can then score verifiability based on the extracted patterns, not the prose note.

---

## Observation 4: signal_reliability has only 8 tasks and may not cover held-out

The signal_reliability dimension was added late in development. With 8 tasks and a (dimension, difficulty) stratified split, the stratifier puts small groups entirely into train if n < 3. The held-out partition likely has 0 signal_reliability tasks.

**Implication**: The held-out evaluation will not measure signal_reliability performance at all. This is a known gap — documented in datasheet.md Section 2.

---

## Summary for Act III

When formatting preference pairs:
1. All tasks with WRONG candidate outputs → (context + candidate_output) is the **rejected** completion.
2. All tasks with CORRECT candidate outputs → (context + candidate_output) is the **chosen** completion.
3. For dimensions with many near-duplicate pairs (signal_grounding), this is fine for SimPO — the contrastive signal is in the output, not the context.
4. Synthesis tasks (4 total) should be weighted up in the training loop if the framework supports per-sample weights; they represent the hardest edge cases.
5. The adapter should be evaluated primarily on dual_control_decision and signal_grounding (the two densest dimensions) since those are the original Week 10 failure modes.
