# Tenacious Path B Judge Adapter

This model card is a publication scaffold for the Path B judge/critic adapter.

## Model Details

- Base model: Qwen/Qwen3-0.6B or Qwen/Qwen3-1.7B, pinned in the final training notebook.
- Adapter type: LoRA.
- Training method: SimPO preference optimization.
- Training framework: Unsloth + TRL.
- Intended artifact: LoRA adapter only, not a merged backbone.

## Intended Use

The adapter scores candidate Tenacious Conversion Engine outputs as acceptable or rejectable for a specific prospect context. It is intended as a rejection-sampling, rollback, or human-escalation layer, not as a standalone sales agent.

## Training Data

Training data comes from `training_data/preference_pairs.jsonl`, derived only from the train partition of Tenacious-Bench v0.1. The current committed build contains 62 preference pairs.

## Limitations

- The dataset is small and Tenacious-specific.
- Held-out results must be interpreted as domain alignment, not general sales-agent quality.
- Public-signal fields are synthetic or redacted snapshots; they should not be treated as live commercial data.
- The adapter should not autonomously send outreach without the Week 10 safety and routing controls.

## Evaluation

All results are on the sealed held-out partition (37 tasks). Training used 62 SimPO preference pairs from the train partition only.

| Metric | Value |
|--------|-------|
| Week 10 generation baseline (held-out avg) | 74.59% |
| Prompt-engineered base model, no training (held-out avg) | 52.22% |
| SimPO-trained adapter (held-out avg) | 49.66% |
| Delta A: trained vs Week 10 baseline | −24.92 pts |
| Delta A 95% CI | [−33.98, −15.60] |
| Delta A p-value | 1.0 (not significant in positive direction) |
| Delta B: trained vs prompt-engineered | −2.56 pts |
| Delta B p-value | 0.9675 |
| Training loss (final) | 0.453 |
| Training steps | 500 (125 epochs on 62 pairs) |

## Negative Result Analysis

Training degraded held-out performance by 24.92 percentage points relative to the Week 10 baseline. Three likely causes:

**1. Catastrophic forgetting on the base task.** Qwen3-0.6B at 0.6B parameters is at the lower bound for instruction following. SimPO preference optimization at this scale may have overwritten general instruction-following capability faster than it instilled Tenacious-specific policy knowledge.

**2. Preference pair sparsity per dimension.** icp_segment_classification had only 2 pairs and signal_reliability had 3. The training signal is too thin for the model to generalise within these dimensions, and the optimisation may have over-fitted to signal_grounding (23 pairs, 37% of the training set).

**3. Evaluation mismatch.** The Week 10 baseline uses the embedded `candidate_output` fields directly — these were authored as correct outputs and score well by construction. The trained model generates free-form text which the heuristic tone scorer and regex grounding checker penalise more harshly.

## Recommended Next Steps

- Expand preference pairs to ≥200 before retraining, with balanced coverage across all 6 dimensions.
- Switch backbone to Qwen3-1.7B or larger to reduce catastrophic forgetting risk.
- Consider ORPO instead of SimPO to preserve reference-model behaviour as a regulariser.
- All numeric claims are cross-referenced in `evidence_graph.json`.
