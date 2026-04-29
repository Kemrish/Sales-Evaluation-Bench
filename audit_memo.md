# Audit Memo — What τ²-Bench Retail Misses for Tenacious

**Date**: 2026-04-28  
**Author**: Kemeriya  
**Scope**: Gap analysis between τ²-Bench retail and Tenacious-specific B2B sales behavior  
**Word count**: ~580

---

## The Core Gap

τ²-Bench retail evaluates a generic e-commerce support agent: cancel orders, track shipments, process exchanges, apply promo codes. The Conversion Engine is a B2B sales agent that classifies company signal, generates grounded outreach emails, and decides when to act autonomously vs. escalate. These are structurally different problems. A benchmark designed for one cannot grade the other.

Examining the 300 trace-log entries (`eval/trace_log.jsonl`, run IDs `433be069`, `132414ca`, `8c4c9218`, `cafec882`, `bf3008e6`) confirms this: every task is a retail scenario with binary pass/fail against an expected outcome like `order_cancelled_refund_initiated` or `exchange_initiated`. These outcomes do not appear anywhere in the Tenacious workflow. The trace failures are overwhelmingly API provider errors (HTTP 400 from Google Vertex, Azure OpenAI) — not agent logic failures. τ²-Bench cannot distinguish a bad Tenacious email from a good one because it was never designed to look at email quality at all.

## Eight Specific Gaps, Indexed to Week 10 Evidence

**Gap 1 — Dual-control decision quality (40% of Tenacious failures)**  
The most consequential failure mode in `eval/failure_taxonomy.md`: the agent classifies a POSITIVE reply correctly but then stalls, waiting for human approval instead of acting. This was fixed via policy.py (P-020), but τ²-Bench has no construct for "autonomous action vs. human escalation." Every retail task is binary. Probe P-020 and the mechanism run delta (38.7% → 46.7% pass@1) are invisible to the bench.

**Gap 2 — Signal grounding (claims vs. source data)**  
Probe P-005 tests whether the agent asserts "aggressive hiring" when fewer than 5 open roles exist. This requires knowing what the agent *said* and what the *evidence was*. τ²-Bench checks for a task outcome; it never reads the email body.

**Gap 3 — Bench capacity honesty**  
Probes P-009 and P-010 test whether the agent over-commits capacity the `bench_summary.json` does not show. A generic retail benchmark has no concept of a bench document. Verifying this claim requires loading `bench_summary.json` and checking every capacity assertion in the output against it.

**Gap 4 — Tone drift across multi-turn sequences**  
Probes P-012, P-013, P-014 track whether the agent drifts from Tenacious tone markers after 3–5 adversarial prospect turns. τ²-Bench tasks are 1–2 turns. Tone drift is invisible in single-turn evaluation.

**Gap 5 — ICP segment misclassification**  
Probe P-001 (Segment 1 assigned to a $35M company) and P-004 (Segment 4 assigned to an AI-score-1 company) test classification correctness against structured input parameters. τ²-Bench has no segment taxonomy.

**Gap 6 — Single-source signal reliability**  
Probe P-026 tests whether the agent hedges when AI maturity = 3 but all signals come from one source (job posts only). This requires checking not just the output phrasing but the signal provenance metadata in the brief. τ²-Bench has no signal-provenance concept.

**Gap 7 — Competitor gap over-claiming**  
Probe P-028 (gap framing when prospect has private AI leadership not on public page) and P-030 (condescending vs. peer framing) test whether claims are scoped to public signal and use peer-level language. These are compositional, multi-signal checks impossible to grade with a binary outcome.

**Gap 8 — Scheduling edge cases with compliance constraints**  
Probe P-023 (timezone natural-language ambiguity) and P-025 (GDPR routing) require checking policy decisions against regulatory constraints. τ²-Bench retail has no compliance layer.

## What Tenacious-Bench Must Provide

A Tenacious-specific benchmark must grade:
1. **Autonomous-vs-escalate decision correctness** given intent + confidence + policy state
2. **Signal grounding**: every factual claim traceable to an input field
3. **Bench capacity honesty**: output assertions verified against `bench_summary.json`
4. **Tone marker adherence**: all 5 Tenacious markers scored 1–5 per message
5. **ICP segment classification**: correct segment given funding, employees, AI maturity, layoff history
6. **Rubric application is machine-verifiable** — no rubric dimension depends solely on human judgment

The scoring evaluator must be a Python script that takes a task JSON and an agent output and returns a numeric score with no human in the loop. The LLM-judge component is limited to the tone markers; all other dimensions are deterministic.
