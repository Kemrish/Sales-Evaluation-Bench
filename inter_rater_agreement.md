# Inter-Rater Agreement Protocol — Tenacious-Bench v0.1

## Purpose

This document records the inter-rater agreement methodology used to validate annotation quality for the hand-authored and trace-derived tasks in Tenacious-Bench v0.1. The goal is to verify that the binary correctness label assigned to each `candidate_output` (CORRECT / WRONG) is reproducible and not an artifact of a single labeler's reading.

---

## Protocol

### Sample

30 tasks were selected for double-labeling:
- 15 from `raw_hand_authored.jsonl` (every other task, TB-HA-001 through TB-HA-029)
- 10 from `raw_trace_derived.jsonl` (TB-DC-001, TB-SG-001, TB-BC-001, TB-IC-001, TB-TA-001 and their -002 variants)
- 5 from `raw_programmatic.jsonl` (randomly sampled: one per dimension)

### Labeling Procedure

Each task was evaluated independently by the same labeler on two separate occasions at least 24 hours apart (self-consistency protocol, as no second human annotator is available for this internal artifact).

For each task, the labeler recorded:
1. **Action label** (CORRECT / WRONG): Does the `candidate_output.action` match `ground_truth.expected_action`?
2. **Grounding label** (CORRECT / WRONG): Does the `candidate_output.email_body` avoid all `ground_truth.forbidden_patterns` and include at least one `ground_truth.required_patterns` match?
3. **Overall label** (CORRECT / WRONG): Would a Tenacious policy expert consider this output acceptable?

### Agreement Metric

Cohen's κ (kappa) is computed for each labeling dimension. Target: κ ≥ 0.80 (near-perfect agreement).

---

## Results

### Session 1 Labels (April 27, 2026)

| Task ID | Source | Dimension | Action | Grounding | Overall |
|---------|--------|-----------|--------|-----------|---------|
| TB-HA-001 | hand_authored | dual_control_decision | WRONG | N/A | WRONG |
| TB-HA-003 | hand_authored | signal_grounding | WRONG | WRONG | WRONG |
| TB-HA-005 | hand_authored | bench_capacity_honesty | WRONG | N/A | WRONG |
| TB-HA-007 | hand_authored | icp_segment_classification | WRONG | N/A | WRONG |
| TB-HA-009 | hand_authored | tone_adherence | WRONG | WRONG | WRONG |
| TB-HA-011 | hand_authored | dual_control_decision | WRONG | N/A | WRONG |
| TB-HA-013 | hand_authored | signal_grounding | WRONG | WRONG | WRONG |
| TB-HA-015 | hand_authored | bench_capacity_honesty | WRONG | N/A | WRONG |
| TB-HA-017 | hand_authored | icp_segment_classification | WRONG | N/A | WRONG |
| TB-HA-019 | hand_authored | tone_adherence | CORRECT | CORRECT | CORRECT |
| TB-HA-021 | hand_authored | dual_control_decision | CORRECT | N/A | CORRECT |
| TB-HA-023 | hand_authored | signal_grounding | CORRECT | CORRECT | CORRECT |
| TB-HA-025 | hand_authored | bench_capacity_honesty | CORRECT | N/A | CORRECT |
| TB-HA-027 | hand_authored | icp_segment_classification | CORRECT | N/A | CORRECT |
| TB-HA-029 | hand_authored | tone_adherence | WRONG | WRONG | WRONG |
| TB-DC-001 | trace_derived | dual_control_decision | WRONG | N/A | WRONG |
| TB-DC-002 | trace_derived | dual_control_decision | CORRECT | N/A | CORRECT |
| TB-SG-001 | trace_derived | signal_grounding | WRONG | WRONG | WRONG |
| TB-SG-002 | trace_derived | signal_grounding | CORRECT | CORRECT | CORRECT |
| TB-BC-001 | trace_derived | bench_capacity_honesty | WRONG | N/A | WRONG |
| TB-BC-002 | trace_derived | bench_capacity_honesty | CORRECT | N/A | CORRECT |
| TB-IC-001 | trace_derived | icp_segment_classification | WRONG | N/A | WRONG |
| TB-IC-002 | trace_derived | icp_segment_classification | CORRECT | N/A | CORRECT |
| TB-TA-001 | trace_derived | tone_adherence | WRONG | WRONG | WRONG |
| TB-TA-002 | trace_derived | tone_adherence | CORRECT | CORRECT | CORRECT |
| TB-PG-DC-001 | programmatic | dual_control_decision | WRONG | N/A | WRONG |
| TB-PG-BC-001 | programmatic | bench_capacity_honesty | WRONG | N/A | WRONG |
| TB-PG-IC-001 | programmatic | icp_segment_classification | WRONG | N/A | WRONG |
| TB-PG-SG-001 | programmatic | signal_grounding | WRONG | WRONG | WRONG |
| TB-PG-TA-001 | programmatic | tone_adherence | WRONG | WRONG | WRONG |

### Session 2 Labels (April 28, 2026 — 24h later)

All 30 tasks re-labeled without reference to Session 1 results.

**Result: 30/30 exact matches across all three sub-labels (Action, Grounding, Overall).**

### Cohen's κ Computation

Since both sessions produced identical labels, the agreement matrix for each dimension is:

```
              Session 2: CORRECT  Session 2: WRONG
Session 1: CORRECT       P_c           0
Session 1: WRONG          0           P_w
```

Where P_c + P_w = 1.0. With zero disagreements:

**κ = 1.00 (perfect agreement)**

### Interpretation

The high kappa is expected given:
1. Tasks with WRONG outputs have obvious policy violations (wrong action type, explicit forbidden phrases, or fabricated capacity numbers).
2. Tasks with CORRECT outputs satisfy all rubric criteria in an unambiguous way.
3. The labeler was the same person who authored the tasks, so recall of authoring intent was still fresh.

**Limitation**: Self-consistency labeling (single annotator, two sessions) overstates true inter-annotator agreement. The protocol documents reproducibility of the author's own intent, not independence of labeling judgment. A second annotator (e.g., a Tenacious sales manager) would be required for genuine IRR validation.

---

## Disagreement Analysis

No disagreements were found in this round. Expected sources of future disagreement if a second annotator is added:

| Scenario | Likely disagreement | Root cause |
|----------|--------------------|-|
| Tone-adherence tasks with borderline condescension | Overall: CORRECT vs. WRONG | "non-condescending" marker is subjective; rubric note is guidance, not binary |
| Signal-grounding tasks where company has 4 roles | Grounding: CORRECT vs. WRONG | Threshold for "few roles" is implicit; forbidden_pattern `(?i)aggress` may not fire for "ambitious" |
| Dual-control tasks with confidence = 0.64 | Action: WRONG vs. CORRECT | One unit below policy threshold; labeler may disagree on rounding |

These cases are addressed in v0.2 by adding explicit threshold tables to the task schema's `ground_truth.notes` field.

---

## Rubric Calibration Notes

During the labeling sessions, two rubric edge cases were identified and recorded:

1. **TB-HA-009 (tone_adherence)**: The candidate output uses "we understand your concern" which is borderline condescending. Session 1 labeled WRONG based on author intent; confirmed WRONG in Session 2 because the sentence pattern matches the condescension failure mode documented in P-030. Rubric is clear enough.

2. **TB-SG-001 (signal_grounding)**: The candidate email claims "aggressive scaling" with 4 roles open. The `forbidden_patterns` list includes `"(?i)(scaling aggressively|aggressive hiring|rapid expansion)"`. Pattern fires correctly. No ambiguity.

---

## Files

| File | Description |
|------|-------------|
| `week11/inter_rater_agreement.md` | This document |
| `week11/tenacious_bench_v0.1/raw_hand_authored.jsonl` | 30 hand-authored tasks labeled in Session 1 |
| `week11/contamination_check.json` | Contamination check results (separate from IRR) |

---

## Sign-off

> Protocol completed: April 28, 2026  
> Labeler: Kemeriya (dataset author)  
> κ (overall label): **1.00**  
> Status: **PASS** — proceed to Act III preference pair formatting
