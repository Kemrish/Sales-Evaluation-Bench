# Synthesis Memo 10 — DPO: Direct Preference Optimization

**Paper**: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," NeurIPS 2023  
**Date**: 2026-04-29  
**Author**: Kemeriya  
**Status**: Path B reading — foundational preference optimization algorithm

---

## Core Claims

DPO reframes the RLHF objective as a supervised learning problem on preference pairs. Starting from the KL-constrained reward maximisation objective used in RLHF, Rafailov et al. show that the optimal policy has a closed-form relationship to the reward function:

```
r(x, y) = β · log [π*(y|x) / π_ref(y|x)] + β · log Z(x)
```

Substituting this into the Bradley-Terry preference model and simplifying, the optimal policy can be trained directly with:

```
L_DPO = -E [ log σ( β · log[π_θ(y_w|x)/π_ref(y_w|x)] - β · log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]
```

This eliminates the need for an explicit reward model and RL training loop. A frozen reference model `π_ref` is still required to compute the log-ratio, but no separate reward model, value function, or PPO rollout is needed.

Rafailov et al. validate on sentiment control, summarisation (TL;DR), and dialogue tasks, showing DPO matches or exceeds PPO-RLHF baselines at lower computational cost. The key hyperparameter is β, which controls the KL penalty — how far the trained policy is permitted to deviate from the reference model.

---

## Where I Agree

DPO's core insight — that the optimal RLHF policy is derivable in closed form from the preference data — is sound, and the simplification from PPO to supervised learning is practically valuable. The elimination of the reward model removes a significant source of reward hacking and training instability.

The KL penalty via the frozen reference model also provides a natural regulariser: the policy cannot deviate arbitrarily far from its pre-trained distribution. For small datasets and small backbones, this is an important stability property that pure preference objectives like SimPO lack.

---

## Where I Disagree

**The frozen reference model is the wrong choice for Tenacious's compute constraints.**

DPO requires holding two copies of the model in memory simultaneously: the trainable policy and the frozen reference. On a Colab T4 with 16 GB VRAM, this halves the effective parameter budget. For Qwen3-0.6B (approximately 1.2 GB in 16-bit), this is technically feasible, but it eliminates the option to scale to Qwen3-1.7B (~3.4 GB × 2 = 6.8 GB for the pair) without tight memory management.

More importantly, DPO's reference model is only useful when the training distribution is close to the reference model's distribution — i.e., when the preference pairs are subtle (chosen and rejected are both plausible outputs, one is slightly better). For Tenacious-Bench, the rejected outputs are *categorically wrong*: they contain fabricated capacity claims, trigger forbidden patterns, or take the wrong routing action entirely. In this regime, the KL constraint acts as a brake on learning rather than a regulariser, because the reference model has no prior knowledge of Tenacious policy and therefore no useful signal to anchor to.

**Conclusion**: DPO's reference model is well-motivated for general alignment tasks (where the base model is already close to the target distribution) but counterproductive for narrow domain adaptation on categorically-wrong rejected outputs. ORPO's joint SFT+preference loss is the better fit for Tenacious training, and is the algorithm selected for this project.

---

## Implication for Tenacious-Bench Training

DPO is the foundational algorithm that all subsequent reference-free variants (SimPO, ORPO, IPO) improve upon. The key design choices it introduced — treating preference optimization as supervised learning, using log-ratios over a reference model, and β as the KL coefficient — are directly inherited by ORPO. Understanding DPO's reference model motivation explains why ORPO's elimination of it (via the SFT+odds-ratio formulation) is the correct architectural choice for small-data, small-backbone, narrow-domain training.
