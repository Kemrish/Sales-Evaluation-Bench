# Synthesis Memo 07 — SimPO: Simple Preference Optimization

**Paper**: Meng, Xia & Chen, "SimPO: Simple Preference Optimization with a Reference-Free Reward," NeurIPS 2024  
**Date**: 2026-04-29  
**Author**: Kemeriya  
**Status**: Path B reading — training algorithm for the Tenacious judge/critic

---

## Core Claims

SimPO replaces DPO's reference-model-based reward with a length-normalized log-probability reward: the reward for a completion is the average token log-probability under the policy model. This makes SimPO reference-free (no separate reference model is needed during training) and lighter on compute, which is why it fits a Colab T4 budget while DPO on the same backbone would require a reference model copy in memory.

The key hyperparameter additions over DPO are: (1) the **margin** γ, which enforces a minimum gap between chosen and rejected reward; and (2) **length normalization**, which divides each completion's log-sum by its token length to prevent the model from preferring long completions trivially.

Meng et al. report SimPO outperforming DPO on AlpacaEval 2.0 and MT-Bench across multiple backbone sizes (Llama-3-8B, Mistral-7B) with lower training cost. Their recommended starting hyperparameters: β=2.0, γ=0.5, lr=5e-7.

---

## Where I Agree

The reference-free property is the decisive factor for Tenacious-Bench training. On a T4 with 16GB VRAM, running a reference model copy alongside the training model leaves approximately 6–7GB for the policy model — insufficient for anything above Qwen3-0.6B without quantization. SimPO's removal of the reference model effectively doubles usable VRAM for the policy, which is what makes Qwen3-1.7B feasible as the backbone. ORPO (Hong et al., EMNLP 2024) also removes the reference model but modifies the loss function more aggressively; SimPO's closer structural relationship to DPO makes it easier to debug when training loss diverges.

The margin hyperparameter γ is well-motivated for Tenacious preference pairs: the rejected completions are *categorically wrong* (fabricated capacity, wrong action, forbidden patterns present), not merely *slightly worse*. A margin-based objective that penalizes small reward gaps is appropriate when the training signal is sharp — which it is here.

---

## Where I Disagree

**Length normalization is the wrong default for Tenacious preference pairs.**

SimPO normalizes each completion's log-probability by its token length. The motivation is sound for general instruction-following: without normalization, a model is incentivized to generate long completions because more tokens means more log-probability mass. But this assumption about chosen/rejected length symmetry does not hold for Tenacious-Bench.

**My evidence**: In the 62 preference pairs in training_data/preference_pairs.jsonl, the *chosen* completions are systematically shorter than the *rejected* completions. A chosen dual-control response is: `"ACTION: book_discovery_call\nAUTONOMOUS: True\n..."` (typically 40–60 tokens). The corresponding rejected completion is a longer fabricated email with false capacity claims and over-claiming signal language (typically 80–120 tokens). After length normalization, the per-token reward of the shorter chosen completion is penalized relative to its raw log-probability, while the longer rejected completion gets a per-token normalization boost.

This is the opposite of what length normalization is designed to prevent. For asymmetric-length preference pairs where shorter = better, length normalization works against training signal.

**Proposed fix**: Disable length normalization or set a high normalization weight (close to 0) for Tenacious training. Evaluate on dev set after 250 steps with and without normalization. If the unnormalized run shows better dual_control_decision improvement, document this as a design deviation from Meng et al.'s default with specific justification.

---

## Implication for Tenacious-Bench Training

The training notebook (training/train_simpo_colab.ipynb) should test two configurations: (1) SimPO with default length normalization (Meng et al. default) and (2) SimPO with length normalization disabled or weight=0.01. The chosen/rejected length asymmetry in Tenacious pairs makes this an empirical question worth a 30-minute ablation, not a default to accept.
