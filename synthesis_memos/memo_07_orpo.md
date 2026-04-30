# Synthesis Memo 07 — ORPO: Monolithic Preference Optimization without Reference Model

**Paper**: Hong, Lee & Thorne, "ORPO: Monolithic Preference Optimization without Reference Model," EMNLP 2024  
**Date**: 2026-04-29  
**Author**: Kemeriya  
**Status**: Path B reading — chosen training algorithm for the Tenacious judge/critic

---

## Core Claims

ORPO eliminates the reference model entirely by modifying the standard cross-entropy SFT loss with an odds-ratio penalty term. The combined loss is:

```
L_ORPO = L_SFT + λ · L_OR
```

where `L_SFT` is the standard next-token prediction loss on the chosen completion, and `L_OR` is a log-sigmoid of the log-odds-ratio between chosen and rejected completions under the current policy. The key parameter λ (called `beta` in TRL's implementation) controls the weight of the preference signal; the default is 0.1.

Because there is no frozen reference model, ORPO needs only one model copy in memory during training. This is the same VRAM advantage as SimPO, but ORPO additionally incorporates an SFT objective, which acts as a regulariser preventing the policy from drifting too far from its pre-trained distribution.

Hong et al. report ORPO matching or outperforming DPO and RLHF baselines on AlpacaEval 2.0, MT-Bench, and IFEval at multiple backbone sizes (Mistral-7B, Llama-3-8B, Phi-2). The paper demonstrates ORPO is particularly effective when training data is small, because the SFT component anchors the model's output distribution while the odds-ratio term provides the preference signal.

---

## Where I Agree

**The joint SFT + preference loss is the decisive advantage for Tenacious training.**

The training set contains 101 preference pairs. At this scale, SimPO's reference-free reward (length-normalised log-probability) has no regularising component — the model is free to drift toward degenerate outputs that maximise the margin without maintaining coherent instruction-following. ORPO's SFT term prevents this by simultaneously training the model to produce the chosen completion as a supervised target, not just to score higher than the rejected one.

This is directly relevant to the catastrophic forgetting risk at 0.6B scale. A 0.6B model has limited parameter capacity. SimPO training on 101 pairs can overwrite instruction-following capability faster than it installs Tenacious-specific policy knowledge. ORPO's SFT component provides the continuity that prevents this.

---

## Where I Disagree

**The λ=0.1 default may be too weak for Tenacious preference pairs.**

Hong et al. set λ=0.1 as a general default across diverse benchmarks (AlpacaEval, MT-Bench, IFEval). These benchmarks contain preference pairs where chosen and rejected outputs differ in quality but not in kind — both are coherent responses, one is slightly better.

Tenacious preference pairs are categorically different: the rejected outputs contain *forbidden patterns* (fabricated capacity claims, over-claiming signal phrases), or take the *wrong action entirely* (escalating when autonomous action was required). The margin between chosen and rejected is not a matter of degree; it is binary correctness.

For this reason, λ=0.1 may not apply enough preference pressure. A value of 0.2–0.3 would weight the odds-ratio term more heavily relative to the SFT loss, and may produce sharper separation on the Tenacious evaluation dimensions. This was not tested in the current training run due to compute constraints; it is the first hyperparameter to tune in a follow-up run.

**Proposed verification**: run a 250-step checkpoint comparison between λ=0.1 and λ=0.2 on the dev set. If dual_control_decision accuracy improves with λ=0.2 without degrading tone_adherence, adopt the higher value in the final run.

---

## Implication for Tenacious-Bench Training

ORPO is the correct algorithm for this training setup: 101 pairs, 0.6B backbone, Colab T4 VRAM budget. The joint SFT+preference loss addresses the catastrophic forgetting risk that a pure preference objective (SimPO) would not. The main tuning question remaining is whether λ=0.1 applies sufficient preference pressure for categorically-wrong rejected outputs — this is the primary ablation for a follow-up run.
