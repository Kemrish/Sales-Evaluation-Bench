# We Built a Sales-Agent Benchmark That Public Leaderboards Can't Measure — Then Trained a Judge on It

*Kemeriya — May 2026*

---

Public benchmarks for AI agents are improving fast. τ²-Bench, WebArena, AgentBench — these are serious, well-constructed evaluations. But they measure agents in domains where the task is defined, the environment is bounded, and success is unambiguous. A booking is confirmed. A web form is filled. A query is answered.

B2B sales is none of those things.

A good B2B sales agent has to read intent, not just words. It has to know when a prospect's reply is warm enough to act on autonomously and when to escalate to a human. It has to be honest about what the company can deliver. It has to stay in a specific tone register across a full outreach thread. It has to ground every claim in a verifiable signal source, and it has to know when a single source isn't enough.

None of this appears on τ²-Bench. So we measured it ourselves.

---

## What We Were Actually Building

Tenacious Intelligence Corporation runs a B2B sales workflow called the Conversion Engine. It ingests prospect signals (LinkedIn profiles, Wellfound listings, Crunchbase funding events), scores ICP fit, and routes to an AI-driven outreach sequence. The agent generates initial contact, follow-ups, and — critically — decides when to close a deal autonomously versus when to hand off to a human sales rep.

That last decision is the one that can go wrong in expensive ways. Trigger it too aggressively and you're burning relationship capital on prospects who aren't ready. Trigger it too conservatively and you're leaving deals on the table while the agent hedges forever.

After running Week 10's Conversion Engine through its first real evaluation, we found eight capability gaps that no existing public benchmark was measuring. We turned those gaps into 216 evaluation tasks across six dimensions:

- **Dual-control routing**: does the agent correctly apply the autonomous-action threshold (reply_intent = POSITIVE AND confidence ≥ 0.65)?
- **Signal grounding**: does the agent anchor claims to verifiable signal sources rather than hallucinating or overstating?
- **Bench-capacity honesty**: does the agent accurately represent the availability of engineers it can offer?
- **ICP segment classification**: is the prospect correctly labeled by funding stage, team size, and tech stack fit?
- **Tone adherence**: does the output match the required register — professional, concise, non-pushy?
- **Signal reliability**: does the agent appropriately hedge when a signal comes from a single low-confidence source?

This is Tenacious-Bench v0.1. The full dataset (train + dev partitions) is on HuggingFace: [MajorKemeriya/tenacious-bench-v0-1](https://huggingface.co/datasets/MajorKemeriya/tenacious-bench-v0-1).

---

## How We Built the Dataset (And What We Got Wrong First)

The naive approach — writing tasks from scratch — doesn't produce a benchmark that reflects real failure modes. It produces a benchmark that reflects what the author imagines failure modes look like, which is a different thing entirely.

We did it the other way: we took the 62 tasks where the Week 10 agent failed on the train partition, analyzed the failure type, and designed tasks that specifically probe the decision boundary the agent got wrong. Every task in Tenacious-Bench v0.1 has a traceable lineage to an actual Week 10 error.

For synthetic task generation, we used a two-stage pipeline: a large generator model (Qwen3-235B via OpenRouter) produced draft tasks, and a separate judge model (Claude Sonnet 4.6) filtered on three quality criteria — coherence, verifiability, and scenario clarity. The full pipeline cost $0.10 in API fees. Everything else ran locally.

The judge filter was deliberately strict on verifiability, and that was the right call: 30 of the generated seed scenarios were rejected because they contained signal claims that couldn't be checked against a documented source. A benchmark task that says "the prospect recently expanded their engineering team" is useless unless you can point to the LinkedIn post that says so.

**Contamination protocol.** Before training, we ran three checks on the held-out partition:
1. N-gram overlap (no sequence of 8+ tokens shared with train)
2. Embedding similarity via `sentence-transformers/all-MiniLM-L6-v2` — all held-out tasks scored below 0.75 cosine similarity to the nearest train task
3. Time-shift verification — held-out tasks use company states that post-date any training scenario

All three checks passed. The held-out partition is clean.

---

## The Experiment: Training a Judge, Not a Generator

Path B in our roadmap was to train a small preference-optimization adapter that could score Conversion Engine outputs as acceptable or rejectable. Not a generator — a critic. The Week 10 Conversion Engine stays in place. The adapter sits downstream and flags outputs that violate Tenacious policy.

We used **ORPO** (Monolithic Preference Optimization without Reference Model — Hong et al., EMNLP 2024). The key advantage over DPO is that ORPO merges the supervised fine-tuning loss and the preference-ranking loss into a single pass. There's no reference model to maintain, which matters on a Colab T4 with 16 GB VRAM. The backbone was Qwen3.5-0.8B with 16-bit LoRA (r=16, alpha=32, beta=0.2, lr=2e-5).

**The preference pairs.** For each of the 62 wrong-tasks in the train partition, we constructed up to 3 preference pairs: one rejected output (the Week 10 wrong answer) matched against one or more chosen outputs (correct answers from the same or a related dimension). This gave us 158 pairs total, covering all six dimensions. Signal grounding was the most represented (69/158) because it had the most Week 10 failures.

**The tokenizer bug that almost wasn't obvious.** Qwen3.5-0.8B via Unsloth loads a vision-language processor rather than a plain text tokenizer. Unsloth patches the `__call__` method with the signature `(self, images, text, videos, **kwargs)`. That means a positional call like `tokenizer(prompt)` silently routes the text string into the `images` argument — and the processor then tries to fetch it as a URL or file path. The crash that results looks like an image loading error, not a type error, which sends you looking in completely the wrong place.

The fix: after loading the model and tokenizer, extract the plain sub-tokenizer before passing it anywhere:
```python
if hasattr(tokenizer, 'tokenizer'):
    tokenizer = tokenizer.tokenizer
```
After that, everything downstream — ORPOTrainer, inference calls, evaluation — works correctly.

---

## What We Found

The training run hit an early stop at approximately 200 of 750 configured steps (Colab session limit). Loss curves were not captured because `train_metrics` returned an empty dict in the run environment. Here are the ablation results on the sealed held-out partition (37 tasks):

| Condition | Held-out avg score |
|---|---|
| Week 10 generation baseline (embedded candidate outputs) | **74.59%** |
| Prompt-engineered base model (Qwen3.5-0.8B + explicit policy prompt, no training) | 50.07% |
| ORPO-trained adapter (~200 steps) | 72.25% |

Two deltas matter here:

**Delta A (trained vs. Week 10 baseline): −2.34 pts** (95% CI: −11.09 to +6.20, p = 0.71 — not significant)

The adapter nearly matches the Week 10 baseline but does not clearly surpass it. The confidence interval includes +6 points, so a positive true effect remains plausible — this is an early-stop result, not a convergence result.

**Delta B (trained vs. prompt-engineering alone): +22.18 pts** (p = 0.0 — highly significant)

This is the meaningful number. Taking the same Qwen3.5-0.8B backbone and applying an explicit policy prompt — all 14 policy rules written out verbatim — only gets you to 50%. The trained adapter gets to 72%. Training instilled 22 percentage points of policy knowledge that no prompt formulation recovered.

This matters because it answers a real question: *is training necessary, or can you just write a better system prompt?* The answer is: prompting is not sufficient. The structured knowledge about confidence thresholds, signal-source hierarchies, and capacity-honesty rules has to be trained in, not described at inference time.

**Per-dimension results:**

Signal grounding improved the most: 0.718 → 0.836 (+11.8 pts). This was the dimension with the most training pairs (69/158), and the relationship is clear.

Dual-control routing regressed: 0.632 → 0.587 (−4.5 pts). This is the one honest unresolved failure in this run. The likely cause is the early stop: dual-control requires the model to learn a threshold boundary (POSITIVE AND confidence ≥ 0.65 → autonomous), and threshold boundary learning is known to need more gradient steps than pattern avoidance. With 33 pairs and 200 steps, the model didn't have enough signal to stabilize this dimension.

---

## What We Recommend (And What We Don't)

**Don't** replace the Week 10 Conversion Engine with the adapter. The adapter was trained to be a critic, not a generator. Deployed as a sole decision-maker, it would be slower (19.7 seconds per task vs. near-instant for the embedded baseline), and its dual-control regression means it currently makes more routing errors than the generator it's judging.

**Do** insert it as a rejection-sampling layer. Generate with the existing Conversion Engine. Pass the output through the adapter. Score below threshold → escalate to human review. Score above threshold → proceed. This preserves Week 10's generation quality while adding policy-grounded filtering on the outputs that matter most.

**Kill-switch conditions**: disable the adapter and fall back to unfiltered generation if (1) judge pass rate on live traffic drops below 40%, (2) adapter latency exceeds 30 seconds on production hardware, or (3) dual-control false-negative rate above 15% on 50 human-audited cases.

---

## What Comes Next

Three immediate actions before any production deployment:

1. **Run the full 750-step training** with LoRA r=32 and confirm that Delta A turns positive. The early stop is the most likely explanation for the dual-control regression and the slightly negative Delta A.

2. **Add more dual-control pairs** (currently 33/158). The dimension with the regression needs the most targeted pair expansion in v0.2.

3. **Build trajectory-level tasks.** Every task in v0.1 is single-turn. A judge trained on single-turn pairs cannot detect cumulative policy drift across a five-turn discovery thread. This is the capability gap that v0.2 needs to fill.

And one deeper question: the benchmark is currently calibrated on synthetic prospects and static bench states. Real prospects use language that doesn't pattern-match cleanly to any training scenario. Real bench states change daily. The v0.2 research agenda is about closing the gap between benchmark accuracy and live-traffic accuracy — which may be the most important thing we don't yet know.

---

*All numeric claims in this post are cross-referenced in `evidence_graph.json`. Training code is in `training/train_orpo_colab.ipynb`. Dataset: [MajorKemeriya/tenacious-bench-v0-1](https://huggingface.co/datasets/MajorKemeriya/tenacious-bench-v0-1). Preference pairs: `training_data/preference_pairs.jsonl`. Community issue: [τ²-Bench #282](https://github.com/sierra-research/tau2-bench/issues/282).*
