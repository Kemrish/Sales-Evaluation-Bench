# Synthesis Memo 05 — Contamination Survey

**Paper**: Chen et al., "Recent Advances in Large Language Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation," EMNLP 2025  
**Date**: 2026-04-29  
**Author**: Kemeriya  
**Status**: Common reading — informs contamination_check.py and held-out partition design

---

## Core Claims

Chen et al. survey contamination detection and prevention techniques across static, semi-dynamic, and fully dynamic evaluation paradigms. Their main finding is that static benchmarks — fixed test sets released publicly — are structurally vulnerable: any model trained after the benchmark's publication date may have seen the test set, and n-gram overlap checks are a weak signal because paraphrased contamination evades them. They recommend dynamic evaluation (randomized question ordering, programmatic test-set generation at inference time, periodic re-sealing of held-out data) as the primary defense. Embedding-similarity checks are presented as a stronger but still imperfect intermediate solution.

---

## Where I Agree

The n-gram overlap check being "weak" is directly observable in Tenacious-Bench. The signal_grounding tasks have a template family where many tasks share the scaffold `"Hi -- {company} has {n} open engineering roles..."`. The n-gram check at n=8 catches near-identical instances but misses tasks that share this scaffold with different company names and role counts — which is exactly the "paraphrased contamination" Chen et al. describe. The embedding similarity check (cosine ≥ 0.85) is a better signal for this case, and the committed contamination_check.json now uses `sentence-transformers/all-MiniLM-L6-v2` rather than the TF-IDF fallback (corrected 2026-04-29).

Chen et al.'s observation that held-out composition matters more than metric thresholds is the key design insight applied in Tenacious-Bench: the held-out is composed entirely of hand-authored adversarial and multi-LLM synthesis tasks, while the train/dev partitions use programmatic and trace-derived tasks. This structural separation means the held-out tasks come from a different generative process than the training data — the gold standard defense Chen et al. point toward even if they don't name it exactly.

---

## Where I Disagree

Chen et al. recommend **dynamic evaluation** — generating new test instances at evaluation time — as the primary long-term defense. For a production evaluation dataset at 216 tasks authored by one person over one week, this recommendation is practically infeasible, and it targets the wrong threat model.

The threat model in Chen et al. is: a model was *trained on* the test set because the test set was published before training cutoff. This is the contamination problem for MMLU, HumanEval, and similar benchmarks used to evaluate frontier models trained on web-scale data.

**The Tenacious-Bench threat model is different**: the risk is not that a model was trained on the held-out partition, but that the *evaluator* (me) inadvertently wrote held-out tasks that are templated variants of training tasks — what I call "intra-author contamination." Dynamic evaluation doesn't help with this; structural partition design does. The adversarial held-out composition addresses this directly: hand-authored tasks are written by the same author but specifically to defeat the train-partition patterns, not to replicate them.

**My evidence**: The held-out partition has 0 programmatic tasks and 0 trace-derived tasks. All 37 held-out tasks are hand-authored (30) or synthesis-accepted (7). The contamination checks pass not primarily because the metrics say so, but because the generative process for held-out tasks is categorically different from the generative process for train/dev tasks. Chen et al.'s n-gram and embedding metrics are confirmatory, not foundational.

---

## Implication for Tenacious-Bench

Dynamic evaluation is the right long-term goal for v1.0 (a programmatic generator that can produce held-out instances on demand without human authoring). For v0.1, structural partition design — different source modes for held-out vs. train/dev — is more defensible at our scale than attempting a full dynamic evaluation pipeline.
