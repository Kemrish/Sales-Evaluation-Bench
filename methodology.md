# Methodology — Tenacious-Bench v0.1

**Author**: Kemeriya  
**Date**: 2026-04-28  
**Version**: v0.1 (Act I draft)

---

## 1. Path Declaration

**Selected path: Path B — Preference-tuned judge / critic (DPO/SimPO/ORPO)**

### Justification

The Week 10 failure evidence points unambiguously to an inconsistency failure mode, not a generation-quality failure. The three most consequential failures in `eval/failure_taxonomy.md` are:

**Dual-Control Stalling (40% of failures, ~37 of 92 failures in baseline run)**  
The agent classified a POSITIVE reply correctly but then waited for human approval before acting. The policy engine fix (P-020, `agent/policy.py`) addresses this structurally, but the underlying problem — the agent cannot reliably judge when its own output meets the threshold for autonomous action — remains. Trace evidence from the baseline run (run_id `433be069`, `132414ca`, `cafec882`) shows repeated incomplete outcomes where the action should have been autonomous but wasn't.

**Over-Escalation on Ambiguous Requests (25% of failures)**  
The agent routed `UNCLEAR` intents to humans instead of sending one clarifying question. Again: the action was wrong not because the email was badly written, but because the agent misjudged the appropriate response policy.

**Policy Hallucination (15% of failures)**  
The agent invented constraints not in the system prompt. This is a detection problem — the agent cannot identify when it is confabulating policy vs. citing real policy.

All three failures share a common structure: the agent produces an output, but cannot reliably evaluate whether that output is the correct action for the situation. A trained judge that sits in front of the generator, scoring each candidate output against Tenacious policy before it is sent, directly addresses this structure.

Path A (SFT) would improve generation quality — the email phrasing, tone, and grounding. But the emails in the failure cases were not badly written; the wrong action was taken. A better email generator cannot fix a wrong routing decision.

Path C (PRM) would address trajectory failures. The Week 10 evidence shows only 13% of failures are multi-tool latency / turn-limit issues. These are operationally significant but not the dominant failure mode.

Path B trains a component that, given (prospect_context, agent_action, email_body), outputs a preference score: is this output correct for this situation? This is precisely what the failures require.

**Training algorithm choice: ORPO** (Monolithic Preference Optimization without Reference Model, Hong, Lee & Thorne, EMNLP 2024)  
Rationale: ORPO combines the SFT loss and preference loss in a single pass, requiring no reference model. This makes it more data-efficient than SimPO for small preference datasets (~100 pairs) because the SFT component anchors the model's output distribution while the odds-ratio loss steers it away from rejected responses simultaneously. SimPO (Meng, Xia & Chen, NeurIPS 2024) is the alternative; ORPO is preferred here because the training set is small (101 pairs) and the model backbone is at the lower end of instruction-following capacity (0.6B), making catastrophic forgetting a significant risk. ORPO's joint SFT+preference loss acts as a regulariser that SimPO — which has no SFT component — cannot provide. At larger dataset sizes (500+ pairs), SimPO's simpler length-normalised reward would be the preferred choice.

---

## 2. Benchmark Design Principles

### 2.1 Machine-verifiability constraint

Every rubric dimension is machine-verifiable or LLM-judge-scored with a documented threshold. The specific constraint: a rubric that cannot be scored by `scoring_evaluator.py` without human intervention is not in the benchmark. This eliminates subjective criteria like "sounds on-brand" and forces operational definitions like "zero forbidden patterns + at least one required pattern + tone markers ≥ 4/5 from judge".

### 2.2 Scoring evaluator design

`scoring_evaluator.py` implements four independently-scored dimensions:

| Dimension | Method | Description |
|-----------|--------|-------------|
| `action_correctness` | Exact match / flexible | Agent action matches expected action; autonomous flag correct |
| `grounding_check` | Regex | Forbidden patterns absent; required patterns present; bench ceiling respected |
| `tone_check` | LLM judge (heuristic fallback) | 5 Tenacious tone markers, 1–5 each; normalized to 0–1 |
| `format_check` | Regex | ≤120 words, ≤60-char subject, no emojis, signature present |

Pass threshold: **≥ 0.70 total weighted score**. This threshold is calibrated against the 30-task hand-label inter-rater exercise (Day 3).

### 2.3 Dimension weights by task type

Different failure modes weight different dimensions:

| Dimension | dual_control | signal_grounding | bench_capacity | tone_adherence | icp_classification |
|-----------|-------------|-----------------|----------------|----------------|--------------------|
| action_correctness | 0.50 | 0.10 | 0.10 | 0.05 | 0.70 |
| grounding_check | 0.00 | 0.50 | 0.60 | 0.20 | 0.15 |
| tone_check | 0.30 | 0.30 | 0.20 | 0.65 | 0.10 |
| format_check | 0.20 | 0.10 | 0.10 | 0.10 | 0.05 |

---

## 3. Dataset Construction

### 3.1 Authoring modes and targets

| Mode | Target share | Current accepted count | Actual share | Notes |
|------|-------------|---------------------|--------------|-------|
| Trace-derived | ~30% | 60 | 27.8% | On target |
| Programmatic (parameter sweeps) | ~30% | 119 | 55.1% | Over target; absorbed synthesis shortfall |
| Multi-LLM synthesis | ~25% | 7 | 3.2% | Severely under target; generator could not produce machine-verifiable regex rubric patterns — see memo_03 |
| Hand-authored adversarial | ~15% | 30 | 13.9% | On target |

### 3.2 Dimension distribution

| Dimension | Target tasks |
|-----------|-------------|
| dual_control_decision | 40 |
| signal_grounding | 40 |
| bench_capacity_honesty | 35 |
| tone_adherence | 35 |
| icp_segment_classification | 30 |
| reply_classification | 20 |
| signal_reliability | 20 (combined with above) |

### 3.3 Partitioning protocol

- **Training partition** (111 tasks): input to Path B preference-pair construction
- **Dev partition** (68 tasks): public; used during dataset authoring iteration and prompt-engineering baselines
- **Held-out partition** (37 tasks): sealed; used only for final ablation scoring

Partitioning is contamination-aware: hand-authored adversarial and accepted synthesis tasks form held-out, while programmatic and trace-derived template families are split into train/dev. This sacrifices perfectly matched source-mode proportions to keep template variants out of the sealed slice.

### 3.4 Contamination prevention

Three checks before any task enters the held-out partition:

1. **N-gram overlap**: < 8-gram overlap between held-out input fields and training/dev fields (script: `generation_scripts/contamination_check.py`)
2. **Embedding similarity**: cosine similarity < 0.85 between held-out and training/dev tasks using `sentence-transformers/all-MiniLM-L6-v2` when cached/available, with a documented TF-IDF fallback for offline runs
3. **Time-shift verification**: any task referencing public data (Crunchbase, layoffs.fyi) uses signal windows documented in task metadata; no generic placeholders

### 3.5 LLM-as-a-judge pipeline

Every synthetically generated task passes a three-stage judge filter before entering the dataset:

1. **Pointwise scoring** (dev-tier model: Qwen3-8B via OpenRouter) on three dimensions:
   - Input coherence (1–5): Is the prospect context internally consistent?
   - Ground-truth verifiability (1–5): Can the rubric be mechanically applied?
   - Rubric-application clarity (1–5): Is the scoring unambiguous?
   - Inclusion threshold: ≥ 3.5 on all three dimensions

2. **Pairwise comparison** when two synthesis paths produce similar tasks: prefer the more diagnostically distinct one

3. **Generation-judge rotation**: the model that generates a task is never the model that judges it. Rotation documented in `generation_scripts/judge_rotation_log.jsonl`.

### 3.6 Inter-rater agreement protocol

Hand-label 30 tasks against the rubric. Re-label 24 hours later. Record agreement matrix in `inter_rater_agreement.md`. Trigger rubric revision if agreement < 80% on any dimension.

---

## 4. Training Plan (Path B)

### 4.1 Training data format

Preference pairs: `(prompt, chosen, rejected)`

- **prompt**: serialized task input (prospect_context, bench_state, prior_thread, reply)
- **chosen**: the correct agent output (action + email_body if applicable) — sourced from:
  - Hand-fixed outputs from Week 10 probe failures
  - Dev-tier model rewrites that score ≥ 0.85 on `scoring_evaluator.py`
- **rejected**: the incorrect output — sourced from:
  - Week 10 baseline-run failures (dual-control stalls, over-escalations, hallucinations)
  - Probe-triggered failures (P-020, P-001, P-009, P-026)

**Preference-leakage prevention**: the model used to generate `chosen` rewrites is a different family from the model used as judge. If Claude Sonnet 4.6 generates chosen rewrites, Qwen3-80B judges. If Qwen3-80B generates chosen, Claude judges. Rotation is logged.

### 4.2 Backbone and training config

| Parameter | Value |
|-----------|-------|
| Backbone | unsloth/Qwen3.5-0.8B |
| Training framework | Unsloth + TRL ORPOTrainer |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| ORPO beta (λ) | 0.2 (increased from 0.1 default — strengthens preference signal for categorical policy failures) |
| Learning rate | 2e-5 (increased from 8e-6 — needed to overcome base model bias in 500 steps at 0.8B scale) |
| Batch size | 2 (grad_accum=8 → effective 16) |
| Max steps | 500 |
| Precision | 16-bit LoRA; fp16 on T4, bf16 on A100/4090 (per Unsloth Qwen3 guide) |
| Random seed | 42 |

### 4.3 Ablation plan

| Ablation | Measurement | Purpose |
|----------|-------------|---------|
| Delta A | Trained judge vs. Week 10 baseline on held-out | Does training lift? |
| Delta B | Trained judge vs. prompt-engineered judge (same backbone, no training) on held-out | Does training beat prompting? |
| Cost-Pareto | Per-task latency and cost with/without trained judge | Is the lift worth the overhead? |

---

## 5. Cost Budget

| Bucket | Allocation | Notes |
|--------|-----------|-------|
| Dataset authoring (dev-tier LLM) | $3.00 | OpenRouter Qwen3-8B (~$0.06/1M tokens) |
| Training | $0.00 | Unsloth on Colab T4 (free) |
| Held-out evaluation (eval-tier) | $2.00 | Claude Sonnet 4.6 via Anthropic, 3–4 passes |
| Reserve | $1.00 | Late-week probe additions, re-runs |
| **Total** | **$6.00** | Under $10 envelope |

---

*This document will be updated as Acts II–IV complete. All numeric claims will be back-referenced in `evidence_graph.json`.*
