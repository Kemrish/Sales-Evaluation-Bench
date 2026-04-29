# Synthesis Memo 02 — Failure Mode Taxonomy and SimPO Training Strategy

**Date**: April 28, 2026  
**Author**: Kemeriya  
**Status**: Pre-Act III (defines what the adapter needs to learn)

---

## What This Memo Is

This memo maps the 30 probe failures from Week 10 to the 5 training dimensions in Tenacious-Bench v0.1, then derives what signal the SimPO training loop needs to suppress. It is the bridge between the audit (Act I) and the preference pairs (Act III).

---

## The Core Failure Pattern

Week 10 traces showed that the agent's failures cluster into two categories:

**Category A — Judgment failures** (agent makes the wrong decision): 40% of failures  
The agent knew all the facts but chose the wrong action. Examples: stalling on a POSITIVE reply (P-020), sending an email when the prospect asked to unsubscribe (P-023), quoting a team of 15 when the max tier is 12 (P-011).

**Category B — Grounding failures** (agent fabricates or over-extrapolates): 60% of failures  
The agent made factual claims that went beyond the evidence. Examples: calling a company "aggressively scaling" with 4 roles (P-005), claiming a velocity trend from 2 data points (P-008), asserting "dedicated ML team" from a single job post (P-026).

SimPO is the right tool for Category A failures: the model already knows the policy facts; it just needs a preference signal that penalizes the wrong choice. For Category B, SimPO works if the rejected completion contains over-claiming language and the chosen completion contains hedge language — the model learns the hedging pattern.

---

## Probe → Dimension Mapping

| Probe | Failure | Dimension | Training signal needed |
|-------|---------|-----------|----------------------|
| P-020 | Stall on POSITIVE reply | dual_control | Prefer `send_followup` / `send_cold_email` over `wait_for_reply` when confidence ≥ 0.65 |
| P-021 | Quote price directly | dual_control | Prefer `escalate_to_human` with `escalation_reason="pricing"` |
| P-022 | Sign NDA autonomously | dual_control | Prefer `escalate_to_human` with `escalation_reason="legal"` |
| P-023 | Email after unsubscribe | dual_control | Prefer `log_and_close` over any outreach action |
| P-025 | GDPR question answered in-chat | dual_control | Prefer `escalate_to_human` |
| P-005 | "Aggressive scaling" with 4 roles | signal_grounding | Prefer grounded language; forbidden `(?i)aggress` |
| P-008 | Velocity claim from 2 points | signal_grounding | Prefer hedge `"may be"`, `"appears to be"` |
| P-026 | Single-source AI maturity over-claim | signal_grounding | Prefer source attribution or hedge |
| P-027 | Undated layoff used as fresh signal | signal_grounding | Prefer temporal hedge or omission |
| P-028 | Private AI leader stated as fact | signal_grounding | Prefer conditional framing |
| P-009 | Zero Rust on bench, claim available | bench_capacity | Prefer honest admission; forbidden `"we have Rust"` |
| P-010 | 10 requested, 8 available | bench_capacity | Prefer `"we can staff 8"` over `"we can staff 10"` |
| P-011 | Quote team of 15 (max 12) | bench_capacity | Prefer `escalate_to_human` + `escalation_reason="above_tier"` |
| P-001 | $35M classified Segment 1 | icp_classification | Prefer correct segment; max Seg 1 is $30M |
| P-002 | Layoff overrides funding → Segment 2 | icp_classification | Prefer Segment 2 when layoff < 120 days |
| P-003 | CTO 95 days → not Segment 3 | icp_classification | Prefer no-match when outside 90-day window |
| P-012-014 | Tone drift over conversation turns | tone_adherence | Prefer 5-marker compliant output |
| P-030 | Condescending framing | tone_adherence | Forbidden: "while most companies... you haven't" pattern |

---

## What the LoRA Adapter Must Learn

The adapter will be trained on (chosen, rejected) preference pairs from the train partition (110 tasks, ~72 WRONG + ~38 CORRECT). The preference optimization objective (SimPO) maximizes:

```
log σ(reward(chosen) - reward(rejected) - γ)
```

where γ is the margin hyperparameter. With reference-free scoring, the reward is the model's own log-probability of the completion.

**Critical insight**: The chosen completions in the benchmark are *not* "best possible" outputs — they are outputs that pass the rubric (score ≥ 0.70). Some chosen completions are minimal: they avoid forbidden patterns and include required patterns, but may not be the most natural or fluent email. This is fine for SimPO — the adapter learns to prefer "rubric-passing" over "rubric-failing," not "excellent" over "good."

**Risk**: If the chosen completions in the train partition are systematically shorter or more hedged than the rejected ones, the adapter may learn to be overly terse or hedge-happy. Mitigation: sample from both CORRECT and WRONG task variants when building preference pairs to ensure format diversity.

---

## Preference Pair Construction Strategy

For each task with a WRONG candidate output in the train partition:
- **rejected**: the candidate output (verbatim, formatted as agent response)
- **chosen**: derived from the task's `ground_truth` fields — the correct action, correct email body (if applicable), correct escalation reason

For tasks with CORRECT candidate outputs, they serve as additional *chosen* examples but do not automatically have a paired *rejected* example. Options:
1. **Skip**: Use only WRONG-task pairs (simpler, cleaner signal)
2. **Synthetic rejection**: Generate a plausible wrong output by inverting the forbidden/required patterns (adds noise but increases data)
3. **Cross-pair**: Match a CORRECT task with a WRONG task from the same dimension (same context, different quality)

**Decision**: Use option 1 (WRONG-task pairs only) for v0.1. The train partition has ~72 WRONG tasks which is sufficient for 500-step LoRA training. Option 3 is planned for v0.2.

---

## Expected Training Outcome

After 500 steps on the 72 preference pairs:
- **Delta A** (dual_control, 34 train tasks): target ≥ 15 point improvement on P-020/P-021/P-022/P-023/P-025 subset of dev
- **Delta B** (signal_grounding, 33 train tasks): target ≥ 10 point improvement on P-005/P-008/P-026/P-027/P-028 subset of dev

If Delta A or Delta B < 5 points, diagnose whether the issue is learning rate, margin γ, or data quality (insufficient contrastive signal).
