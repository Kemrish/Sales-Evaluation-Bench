# Methodology Rationale

## Selected Path

I selected Path B: a preference-tuned judge/critic trained with ORPO preference pairs.

The Week 10 evidence points to inconsistent self-evaluation rather than weak prose generation. The recurring failures were cases where the agent had enough information to choose the right sales action but either stalled or over-escalated. Trace IDs `433be069`, `132414ca`, and `cafec882` are representative: the agent classified a high-intent or policy-relevant situation but selected the wrong control action. The associated probes `P-020`, `P-021`, and `P-026` also show the same structure: a candidate output must be rejected because the decision policy is wrong, not because the email wording is merely awkward.

That failure mode makes a trained critic more production-relevant than a generation adapter. The critic can sit in front of the Week 10 generator and reject outputs that violate autonomous-action, capacity, signal-grounding, or tone rules. It also supports cheap rollback or rejection sampling without replacing the whole Conversion Engine.

## Paper-Grounded Design

LIMA supports the decision to keep the preference set small and high-quality: at this scale, diagnostic coverage and clean labels matter more than raw volume. The dataset therefore emphasizes probe-triggered failures and corrected outputs rather than bulk synthetic paraphrases.

The LLM-as-a-Judge survey motivates separating task-quality filtering from final scoring and using explicit rubrics per dimension. This is why `scoring_evaluator.py` decomposes output quality into action correctness, grounding, tone, and format rather than using a single subjective "good outreach" label.

ORPO is preferred over DPO and SimPO for this training run. It is reference-free (no frozen model copy needed, saving VRAM on Colab T4) and combines an SFT loss with the odds-ratio preference loss in one pass. The SFT component acts as a regulariser that prevents catastrophic forgetting — a critical property at 0.8B scale with only 158 preference pairs. SimPO was the initial choice but lacks this regulariser; at small dataset sizes where forgetting is the dominant failure mode, ORPO's joint objective is the better fit.

## Leakage Controls

The held-out partition is drawn from hand-authored adversarial and multi-LLM synthesis tasks, while train/dev use programmatic and trace-derived variants. This keeps template families out of the sealed slice and gives the contamination checker a meaningful chance to detect leakage.

Preference leakage is handled by separating generation and judging roles in `generation_scripts/judge_rotation_log.jsonl`. The committed local build uses deterministic filtering and documents where future dev-tier LLM calls should be routed.

---

## Contamination Check Results

Full machine-readable report: `contamination_check.json`. Checks were run after partitioning with `generation_scripts/contamination_check.py`. The held-out slice (37 tasks) was checked against the combined train+dev reference corpus (179 tasks).

### Method 1 — N-gram Overlap (n=8)

**Rule**: reject any held-out task whose input text shares 8 or more consecutive tokens with any train or dev task input.

| Metric | Value |
|--------|-------|
| Held-out tasks checked | 37 |
| Reference tasks (train+dev) | 179 |
| Pairs flagged | 0 |
| Action taken | None required |
| Result | **PASS** |

**Why zero violations**: held-out tasks are drawn exclusively from hand-authored adversarial and multi-LLM synthesis modes. These modes do not reuse sentence-level templates from the programmatic or trace-derived families that populate train/dev. The structural partition design is what eliminates overlap — the checker confirms it.

### Method 2 — Embedding Similarity (cosine ≥ 0.85)

**Rule**: reject any held-out task whose input embedding has cosine similarity ≥ 0.85 with any train or dev task embedding.

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional sentence embeddings; the script falls back to TF-IDF if the model is unavailable, but this run used the full transformer model).

| Metric | Value |
|--------|-------|
| Pairs flagged (cosine ≥ 0.85) | 0 |
| Highest observed cosine similarity | < 0.85 (no pair logged) |
| Action taken | None required |
| Result | **PASS** |

**Why zero violations**: hand-authored tasks use distinct company names, specific multi-signal conflict scenarios (GDPR + pricing, undated layoff + high AI score), and adversarial constructions not present in parametric templates. LLM-synthesis tasks were generated with seed prompts explicitly drawn from failure evidence outside the programmatic parameter space.

### Method 3 — Time-shift / Placeholder Verification

**Rule**: reject any task with an unfilled template placeholder (`[DATE]`, `[COMPANY]`, `<CAPS_PLACEHOLDER>`, or similar).

| Metric | Value |
|--------|-------|
| Tasks with unfilled placeholders | 0 |
| Action taken | None required |
| Result | **PASS** |

### Summary

| Check | Flagged | Resolution | Final status |
|-------|---------|------------|--------------|
| N-gram overlap (n=8) | 0 | N/A | PASS |
| Embedding similarity (cos ≥ 0.85) | 0 | N/A | PASS |
| Time-shift placeholders | 0 | N/A | PASS |

**Overall assessment**: the held-out partition is clean across all three methods. The structural decision to source held-out tasks exclusively from non-template modes (hand-authored + synthesis) was the primary control; the contamination checker serves as a verification layer rather than a corrective one. Had any pair been flagged, the protocol was to drop the held-out task and replace it from the same source mode — no rewrites, to preserve adversarial intent.
