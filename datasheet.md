# Datasheet: Tenacious-Bench v0.1

> Gebru et al. (2021) 7-section format with Pushkarna et al. (2022) layered detail.

---

## 1. Motivation

**Why was this dataset created?**
Tenacious-Bench v0.1 was created to fill a critical evaluation gap for the Tenacious Intelligence Corporation sales agent. The τ²-Bench retail dataset (cancel-order, exchange-items, track-shipment tasks) shares no domain vocabulary, decision logic, or grounding constraints with Tenacious's B2B engineering staff augmentation workflow. Eight specific capability gaps were identified in audit_memo.md, anchored to 30 probe IDs from the Week 10 adversarial test suite.

**What tasks does it support?**
The dataset supports evaluation of a B2B sales agent on six dimensions: dual-control decision-making (when to act autonomously vs. escalate), signal grounding (matching outreach claims to evidence quality), bench capacity honesty (never committing beyond available engineers), ICP segment classification (four segments with numerical thresholds), tone adherence (five markers: direct, grounded, honest, professional, non-condescending), and signal reliability (hedging when signal provenance is weak or single-source). A seventh dimension, reply_classification, was originally specified in the schema but removed from v0.1 as no tasks were authored for it; deferred to v0.2.

**Who funded / authorized creation?**
Internal research artifact for Tenacious Consulting Week 11 challenge. No external funding. Dataset created by Kemeriya using a combination of trace-derived authoring, programmatic generation, multi-LLM synthesis, and hand-authored adversarial cases.

---

## 2. Composition

**What does each instance represent?**
Each instance is a single agent evaluation task. It contains:
- `input`: prospect context (company name, funding, employee count, engineering roles open, layoff events, AI maturity score, signal sources), bench state, prior conversation thread, prospect reply, reply intent, and confidence score.
- `candidate_output`: an agent action (action type, autonomous flag, email body, subject line, capacity claim, escalation reason).
- `ground_truth`: expected action, expected autonomous flag, forbidden regex patterns, required regex patterns, and a ground-truth note.
- `rubric`: per-dimension scoring weights summing to 1.0 with method (exact_match, regex, llm_judge).

**How many instances?**
216 total tasks after deduplication, partitioned as:
- Train: 111 (51.4%)
- Dev: 68 (31.5%)
- Held-out: 37 (17.1%)

**Breakdown by dimension:**

| Dimension | Count |
|-----------|-------|
| dual_control_decision | 53 |
| signal_grounding | 67 |
| bench_capacity_honesty | 32 |
| icp_segment_classification | 29 |
| tone_adherence | 26 |
| signal_reliability | 8 |

**Breakdown by source mode:**

| Source mode | Count |
|-------------|-------|
| programmatic | 119 |
| trace-derived | 60 |
| hand-authored | 30 |
| multi-llm-synthesis | 7 |

**Breakdown by difficulty:**

| Difficulty | Count |
|------------|----------------|
| hard | 82 |
| medium | 75 |
| adversarial | 38 |
| easy | 21 |

**Does the dataset contain all possible instances, or a sample?**
The programmatic tasks cover a combinatorial sweep of parameter values (funding amounts, role counts, intent categories, stack combinations) defined in `generation_scripts/generate_programmatic.py` and `generate_sg_extra.py`. The trace-derived tasks are anchored 1:1 to the 30 probe IDs in `probes/probe_catalog.json` (2 variants each). The hand-authored and synthesis tasks are a non-exhaustive adversarial sample.

**Is there missing information?**
- `multi-llm-synthesis` mode produced only 7 accepted tasks after filtering. This mode is underrepresented relative to the original 25% target.
- `signal_reliability` dimension has only 8 tasks due to late addition; held-out partition does not yet have coverage for this dimension.

**Does the dataset contain data that might be considered confidential or sensitive?**
No. All company names, funding amounts, employee counts, and prospect replies are synthetic. No real prospect data, PII, or proprietary deal records are included.

---

## 3. Collection Process

**How was the data collected?**

Four authoring modes were used:

1. **Trace-derived** (`generate_trace_derived.py`): Each of the 30 probe IDs in `probes/probe_catalog.json` was instantiated as 2 tasks — one with a wrong candidate output (the failure mode the probe targets) and one with a correct output. Company pool: 8 synthetic companies. Produces 60 deterministic tasks.

2. **Programmatic** (`generate_programmatic.py`, `generate_sg_extra.py`): Parameter sweeps across ICP segment thresholds, bench stack availability, intent × confidence combinations, and signal source quality. All logic is deterministic given `random.seed(42)` / `random.seed(99)`. Produces 119 tasks.

3. **Multi-LLM synthesis** (`generate_synthesis.py`): seed prompts derived from probe failure evidence are sent through a generator route and judged by a different model family to prevent preference leakage. Acceptance criterion: coherence >= 3.5, verifiability >= 3.5, clarity >= 3.5 on a 1-5 scale. Judge rotation log: `generation_scripts/judge_rotation_log.jsonl`. Produces 7 accepted tasks in the committed build.

4. **Hand-authored adversarial** (`raw_hand_authored.jsonl`): 30 tasks covering adversarial combinations not reached by the programmatic sweep (multi-signal ICP conflicts, cumulative tone drift, undated layoff + high AI score, GDPR + pricing in one message). 19 wrong / 11 correct candidate outputs.

**Who collected the data?**
Authoring was performed by Kemeriya (human author) for trace-derived and hand-authored modes. Programmatic generation uses deterministic code; no external annotators. LLM synthesis uses two distinct model families with automated judge scoring.

**Over what time period was the data collected?**
April 27–28, 2026.

**Were any ethical review processes conducted?**
Not applicable — dataset contains no human subject data, no personal information, and no real prospect or deal records.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing / cleaning / labeling done?**

Yes:
- **Deduplication**: `partition.py:deduplicate()` removes any task with a duplicate `task_id` before splitting.
- **Contamination-aware partitioning**: Hand-authored adversarial and accepted synthesis tasks form the sealed held-out slice. Programmatic and trace-derived template families are reserved for train/dev, then split by `(dimension, difficulty)` to preserve coverage.
- **Contamination checks** (`contamination_check.py`):
  - *N-gram overlap* (n=8): 0 violations after moving templated families out of held-out.
  - *Embedding similarity* (cosine >= 0.85): 0 violations in the local offline TF-IDF fallback. The script attempts `sentence-transformers/all-MiniLM-L6-v2` first when the model is available.
  - *Time-shift placeholder check*: 0 violations. No unfilled `[DATE]`, `[COMPANY]`, or `<CAPS>` placeholders remain.

**Was the "raw" data saved in addition to the preprocessed/cleaned data?**
Yes. Five raw JSONL files are preserved in `tenacious_bench_v0.1/`:
- `raw_trace_derived.jsonl`
- `raw_programmatic.jsonl`
- `raw_synthesis.jsonl`
- `raw_hand_authored.jsonl`
- `raw_programmatic_sg.jsonl`

**Is the software used to preprocess/clean/label the data available?**
Yes — all generation and processing scripts are in `week11/generation_scripts/`.

---

## 5. Uses

**What tasks has the dataset been used for already?**
Dataset was authored specifically for this project; it has not been used in any prior publication.

**What are the intended uses?**
- **Primary**: Offline evaluation of the Tenacious sales agent on the five core dimensions (dual-control, signal grounding, bench capacity honesty, ICP classification, tone adherence).
- **Secondary**: ORPO preference pair training (Act III). The train partition is formatted into (chosen, rejected) pairs for LoRA fine-tuning of Qwen3.5-0.8B backbone.
- **Tertiary**: Regression testing — held-out partition is sealed and used only for final benchmark reporting after training converges.

**What are the out-of-scope uses?**
- Evaluating general-purpose sales agents outside the Tenacious workflow. The ICP segments, bench state, and pricing logic are specific to Tenacious.
- Training a generation model end-to-end. The dataset is designed for preference optimization of an existing capable backbone, not for supervised fine-tuning from scratch.
- Use as ground truth for any domain other than B2B engineering staff augmentation.

**Are there tasks for which the dataset should not be used?**
Do not use this dataset to evaluate models on email generation quality in isolation — the rubric weights tone (0.2–0.3) and action correctness (0.1–0.4) differently per task, and a model that scores well on email body alone can still fail the benchmark.

---

## 6. Distribution

**Will the dataset be distributed?**
Yes. The train and dev partitions are publicly released on HuggingFace Hub under CC BY 4.0. The held-out partition is released after the leaderboard is published (v1.0). Distribution follows a staggered timeline: train/dev are available now; held-out is released post-training to prevent benchmark gaming. All company names, funding figures, and prospect replies are synthetic; no personal information or proprietary deal data is included.

**Under what license?**
CC BY 4.0. Rationale: (1) all task data is synthetic — no real prospect records, PII, or proprietary deal information; (2) the probe catalog is derived from publicly observable sales-agent failure modes, not confidential Tenacious trade secrets; (3) CC BY 4.0 permits academic reuse, derivatives, and leaderboard submissions with attribution. The scoring evaluator and generation scripts are released under the same license. See `methodology.md` for the full license rationale.

**Is the dataset self-contained?**
Yes. All tasks are self-contained JSON objects. The scoring evaluator (`scoring_evaluator.py`) is included and requires only `httpx` (for LLM judge calls) and optionally `sentence-transformers` (for embedding contamination check).

---

## 7. Maintenance

**Who is maintaining the dataset?**
Kemeriya / Tenacious Consulting internal team.

**Will the dataset be updated?**
Planned updates:
- v0.2: Expand multi-LLM synthesis to 50+ tasks by relaxing verifiability criterion and adding regex post-processing step to auto-generate required/forbidden patterns from ground_truth_notes.
- v0.3: Add signal_reliability dimension tasks to match other dimensions (~25 tasks).
- v1.0: Full held-out re-seal after LoRA training completes.

**Are there mechanisms for others to contribute?**
Not currently. Future versions may accept pull requests via HuggingFace dataset repository.

**Will older versions be retained?**
v0.1 raw JSONL files and partition outputs are preserved in `tenacious_bench_v0.1/`. Git history provides version provenance.

---

## Contamination Report Summary

Full report: `week11/contamination_check.json`

| Check | Result | Details |
|-------|--------|---------|
| N-gram overlap (n=8) | PASS | 0 violations |
| Embedding similarity (cos >= 0.85) | PASS | 0 violations using offline TF-IDF fallback in this environment |
| Time-shift placeholders | PASS | 0 unfilled placeholders |

**Assessment**: The held-out partition is clean under the committed local checks. The sealed slice is intentionally composed from hand-authored adversarial and multi-LLM synthesis tasks to avoid leakage from programmatic template families.

---

## Pushkarna (2022) Layered Detail

### Layer 1: Quick Reference

| Property | Value |
|----------|-------|
| Total tasks | 216 |
| Splits | 111 train / 68 dev / 37 held-out |
| Dimensions | 5 primary + 1 secondary (6 total in v0.1; reply_classification deferred to v0.2) |
| Source modes | 4 |
| Scoring methods | exact_match, regex, llm_judge |
| Pass threshold | ≥ 0.70 weighted score |
| Created | April 2026 |

### Layer 2: Annotator Profile

No human annotators were used for labeling. Ground truth is either:
- **Deterministic**: Derived from Tenacious policy documents (ICP thresholds, bench_summary.json, pricing sheet, style guide). Any Tenacious employee familiar with the playbook can verify.
- **LLM-adjudicated**: Synthesis tasks judged by Claude Sonnet 4.6 with a structured 3-criterion rubric. Judge logs retained.

### Layer 3: Known Limitations

1. **Low synthesis yield**: 7 synthesis tasks accepted. The generator model struggled to produce machine-verifiable rubric patterns, defaulting to free-text ground-truth notes. This is a fundamental limitation of prompting-based task generation for structured evaluation.
2. **Signal reliability underrepresentation**: 8 tasks out of 216 (3.7%). The dimension was added late and lacks full combinatorial coverage.
3. **Held-out composition tradeoff**: The sealed slice is cleaner because it excludes programmatic template families, but this means held-out source-mode distribution differs from train/dev.
4. **Single-domain scope**: All tasks are anchored to Tenacious's specific ICP segments and bench state. Generalization to other staff augmentation companies would require re-authoring ICP thresholds and bench constraints.
