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

Final held-out results will be added after the Colab training run and ablation pass. Numeric claims should be mapped in `evidence_graph.json`.
