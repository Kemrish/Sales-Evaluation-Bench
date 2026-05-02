# Tenacious Path B Judge Adapter

## Model Details

- Base model: `unsloth/Qwen3.5-0.8B`
- Adapter type: LoRA (r=16, lora_alpha=32, dropout=0.05)
- Training method: ORPO (Monolithic Preference Optimization without Reference Model, Hong et al., EMNLP 2024)
- Training framework: Unsloth + TRL ORPOTrainer
- Intended artifact: LoRA adapter only, not a merged backbone

## Intended Use

The adapter scores candidate Tenacious Conversion Engine outputs as acceptable or rejectable for a specific prospect context. It is intended as a rejection-sampling, rollback, or human-escalation layer, not as a standalone sales agent.

## Training Data

Training data comes from `training_data/preference_pairs.jsonl`, derived only from the train partition of Tenacious-Bench v0.1. The committed build contains **158 preference pairs** (62 wrong tasks × up to 3 chosen tasks each, train-only, with cross-dimension fallback).

| Dimension | Pairs |
|-----------|-------|
| signal_grounding | 69 |
| dual_control_decision | 33 |
| tone_adherence | 27 |
| bench_capacity_honesty | 14 |
| signal_reliability | 9 |
| icp_segment_classification | 6 |
| **Total** | **158** |

144 pairs matched a real correct task from the same dimension. 14 pairs used a minimal correct output constructed from ground_truth fields.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | unsloth/Qwen3.5-0.8B |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| ORPO beta (λ) | 0.2 |
| Learning rate | 2e-5 |
| Steps | Early stopped (~200) |
| Effective batch size | 16 (batch=2, grad_accum=8) |
| Precision | 16-bit LoRA (fp16 on T4) |
| Seed | 42 |

## Limitations

- The dataset is small and Tenacious-specific.
- Held-out results must be interpreted as domain alignment, not general sales-agent quality.
- Public-signal fields are synthetic or redacted snapshots; they should not be treated as live commercial data.
- The adapter should not autonomously send outreach without the Week 10 safety and routing controls.

## Evaluation

All results are on the sealed held-out partition (37 tasks). Training used 158 ORPO preference pairs from the train partition only.

| Metric | Value |
|--------|-------|
| Week 10 generation baseline (held-out avg) | 74.59% |
| Prompt-engineered base model, no training (held-out avg) | 50.07% |
| ORPO-trained adapter (held-out avg) | 72.25% |
| Delta A: trained vs Week 10 baseline | −2.34 pts |
| Delta A 95% CI | [−11.09, +6.20] pts |
| Delta A p-value | 0.7095 (not significant) |
| Delta B: trained vs prompt-engineered | **+22.18 pts** |
| Delta B p-value | **0.0 (highly significant)** |
| Avg latency per task (with adapter) | 19,712 ms |

### Per-Dimension Results (held-out)

| Dimension | Baseline | Post-Training | Delta |
|-----------|----------|---------------|-------|
| dual_control_decision | 0.632 | 0.587 | −4.5 pts |
| signal_grounding | 0.718 | 0.836 | **+11.8 pts** |
| All dimensions avg | 0.746 | 0.723 | −2.3 pts |

## Result Interpretation

**Delta B is strongly positive (+22.18 pts, p=0.0)**: the trained adapter far outperforms a prompt-engineered base model, confirming that ORPO preference training successfully instilled Tenacious policy knowledge that cannot be recovered by prompting alone.

**Delta A is slightly negative but not significant (−2.34 pts, p=0.71)**: the adapter nearly matches the Week 10 baseline (which used authored correct outputs — a very high bar) but does not clearly surpass it. The CI of [−11.09, +6.20] means the true effect could plausibly be positive.

**Signal grounding improved substantially** (+11.8 pts), the dimension with the most training pairs (69/158). Dual control regressed slightly, consistent with having fewer pairs (33/158) and the early stop at ~200 steps.

## Recommended Next Steps

- Run full 750 steps with LoRA r=32 to confirm the positive Delta B holds and check if Delta A turns positive with more training.
- Add more dual_control_decision pairs (currently 33) to recover the regression on that dimension.
- All numeric claims are cross-referenced in `evidence_graph.json`.
