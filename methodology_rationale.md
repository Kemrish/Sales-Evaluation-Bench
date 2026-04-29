# Methodology Rationale

## Selected Path

I selected Path B: a preference-tuned judge/critic trained with SimPO-style preference pairs.

The Week 10 evidence points to inconsistent self-evaluation rather than weak prose generation. The recurring failures were cases where the agent had enough information to choose the right sales action but either stalled or over-escalated. Trace IDs `433be069`, `132414ca`, and `cafec882` are representative: the agent classified a high-intent or policy-relevant situation but selected the wrong control action. The associated probes `P-020`, `P-021`, and `P-026` also show the same structure: a candidate output must be rejected because the decision policy is wrong, not because the email wording is merely awkward.

That failure mode makes a trained critic more production-relevant than a generation adapter. The critic can sit in front of the Week 10 generator and reject outputs that violate autonomous-action, capacity, signal-grounding, or tone rules. It also supports cheap rollback or rejection sampling without replacing the whole Conversion Engine.

## Paper-Grounded Design

LIMA supports the decision to keep the preference set small and high-quality: at this scale, diagnostic coverage and clean labels matter more than raw volume. The dataset therefore emphasizes probe-triggered failures and corrected outputs rather than bulk synthetic paraphrases.

The LLM-as-a-Judge survey motivates separating task-quality filtering from final scoring and using explicit rubrics per dimension. This is why `scoring_evaluator.py` decomposes output quality into action correctness, grounding, tone, and format rather than using a single subjective "good outreach" label.

SimPO is preferred over DPO for the first training run because it is reference-free and lighter for Colab T4. The preference pairs here are intentionally asymmetric: rejected outputs often violate clear policies, while chosen outputs satisfy machine-checkable rubrics. That margin-style setup is a natural fit for SimPO.

## Leakage Controls

The held-out partition is drawn from hand-authored adversarial and multi-LLM synthesis tasks, while train/dev use programmatic and trace-derived variants. This keeps template families out of the sealed slice and gives the contamination checker a meaningful chance to detect leakage.

Preference leakage is handled by separating generation and judging roles in `generation_scripts/judge_rotation_log.jsonl`. The committed local build uses deterministic filtering and documents where future dev-tier LLM calls should be routed.
