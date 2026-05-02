# Tenacious-Bench v0.1 — Executive Memo
**To**: Tenacious Intelligence Corporation CEO and CFO  
**From**: Kemeriya  
**Date**: 2026-05-02  
**Re**: Trained Sales-Agent Judge — Deployment Decision

---

## Page 1 — The Decision

**What was built.** We identified eight capability gaps between public retail benchmarks (τ²-Bench) and the Tenacious B2B sales workflow, authored a 216-task evaluation benchmark (Tenacious-Bench v0.1) covering dual-control routing, signal grounding, bench-capacity honesty, ICP classification, tone adherence, and signal reliability, and trained a LoRA preference-optimization adapter (ORPO, Qwen3.5-0.8B backbone) on 158 human-curated preference pairs derived exclusively from Week 10 failure traces.

**Headline result.**

| Metric | Value |
|--------|-------|
| Week 10 baseline (held-out avg) | 74.59% |
| Prompt-engineered base model (no training) | 50.07% |
| Trained adapter (held-out avg) | 72.25% |
| **Delta A**: trained vs Week 10 baseline | **−2.34 pts** (95% CI: −11.09 to +6.20; p = 0.71 — not significant) |
| **Delta B**: trained vs prompt-engineering alone | **+22.18 pts** (p = 0.0 — highly significant) |
| Avg latency per task with adapter | 19.7 seconds |
| Avg latency per task without adapter | ~0 seconds |
| Training compute cost | $0 (Colab T4 free tier) |
| Total API spend for dataset authoring | $0.10 |

**Delta A** is slightly negative but statistically indistinguishable from zero: the adapter nearly matches the Week 10 baseline but does not yet clearly surpass it. The confidence interval includes +6 points, meaning a positive true effect remains plausible with more training steps.

**Delta B** is strongly positive and highly significant: the trained adapter outperforms an explicit policy-prompt applied to the same untrained backbone by 22 percentage points. Training instilled Tenacious policy knowledge that no prompt formulation recovered — this is the core justification for Path B.

**Recommendation: Deploy as rejection-sampling layer with caveat.** Do not replace the Week 10 generator. Insert the adapter as a downstream critic: generate with the existing Conversion Engine, pass candidate outputs through the judge, and escalate to human review when the judge scores below threshold. This preserves the Week 10 generation quality while adding policy-grounded filtering. **Do not deploy as sole decision-maker** until Delta A is positive and statistically significant (requires full 750-step training run and dual-control dimension recovery).

---

## Page 2 — The Skeptic's Appendix

**Four failure modes Tenacious-Bench v0.1 still does not capture**

1. **Multi-turn coherence across a full discovery sequence.** Every task in v0.1 is single-turn (one prospect reply → one agent action). A judge trained on single-turn pairs cannot detect cumulative tone drift or policy contradiction across a five-turn thread. Fixing this requires trajectory-level tasks, which are the domain of a process reward model (Path C), not preference pairs.

2. **Live bench-state staleness.** Bench summaries in the dataset are static snapshots. In production the bench changes daily — engineers go on assignment, return, or are unavailable. A task that was correct at authoring time may be wrong 48 hours later. v0.2 must introduce a time-parameterised bench state and test whether the adapter generalises across bench configurations it was not trained on.

3. **Real-prospect distributional shift.** All company names, funding figures, and prospect replies in v0.1 are synthetic. The ICP thresholds and tone markers are calibrated to Tenacious's documented playbook, but a real prospect may use language that pattern-matches to no signal source in the training vocabulary. The adapter's failure rate on live traffic could be higher than the 27.75% held-out error rate suggests.

4. **Pricing and NDA negotiation depth.** The escalation tasks in v0.1 only require the agent to route correctly — they do not evaluate whether the escalation message itself is well-formed or whether the timing is appropriate. A more complete benchmark would score the quality of the escalation hand-off, not just the binary route/don't-route decision.

**Public-signal lossiness.** Ground truth in signal-grounding tasks is anchored to synthetic signal sources (Wellfound, LinkedIn, Crunchbase ODM sample). Real signals are noisier, more ambiguous, and sometimes contradictory. The adapter was rewarded for hedging on single-source signals — but in production, a single strong signal (e.g., a direct LinkedIn message from the hiring manager) may justify stronger language than the rubric currently allows. The rubric's forbidden-pattern list may be over-conservative for high-confidence live signals.

**One honest unresolved failure.** The dual-control dimension regressed post-training (0.632 → 0.587, −4.5 pts) despite being the second-largest training dimension (33/158 pairs). The most likely cause is the early stop at ~200 of 750 configured steps: dual-control failures require the model to learn a threshold boundary (confidence ≥ 0.65 AND intent = POSITIVE → autonomous), and threshold learning is known to require more gradient steps than pattern avoidance. The regression is unresolved; a full 750-step run is the first diagnostic action.

**Kill-switch trigger.** Disable the adapter and fall back to the Week 10 generator without judge filtering if any of the following are observed in production: (1) judge pass rate on live traffic drops below 40% (indicating distributional shift the adapter was not trained to handle); (2) adapter latency per task exceeds 30 seconds on production hardware (unacceptable for real-time follow-up workflows); (3) a false-negative rate above 15% on dual-control decisions is confirmed by human audit of 50 consecutive escalated cases.
