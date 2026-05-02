# Tenacious-Bench v0.1

Tenacious-Bench is a small, machine-scored benchmark for Tenacious-style B2B sales agents. It evaluates the behaviors that a generic retail/task benchmark misses: signal-grounded outreach, bench-capacity honesty, dual-control routing, ICP segment classification, signal reliability, and Tenacious tone adherence.

## Public Artifacts

| Artifact | Link |
|---|---|
| Dataset (train + dev, CC-BY-4.0) | https://huggingface.co/datasets/MajorKemeriya/tenacious-bench-v0-1 |
| Blog post | https://substack.com/profile/176991119-kemeriya-major/note/c-252497550 |
| Community issue (τ²-Bench) | https://github.com/sierra-research/tau2-bench/issues/282 |
| LoRA adapter (HuggingFace model) | pending — upload after full 750-step run; adapter trained at ~200 steps on Colab T4 |

## Status

Current state: Acts I–IV complete. Adapter upload pending full training run.

- 216 authored tasks across train/dev/held-out partitions.
- Held-out contamination check passes (n-gram, sentence-transformers embedding, time-shift).
- Path B selected: a preference-tuned judge/critic using ORPO.
- 158 preference pairs generated from the train partition (multi-paired, train-only).
- 10 synthesis memos committed (4 common-reading + 4 Path B + 2 project analysis).
- ORPO adapter trained (~200 steps on Colab T4): Delta B +22.18 pts (p=0.0), Delta A −2.34 pts (not significant).
- Dataset published on HuggingFace. Blog post and τ²-Bench community issue live.

## Repository Layout

- `audit_memo.md`: benchmark-gap audit (8 gaps, 30 probe IDs).
- `datasheet.md`: Gebru + Pushkarna datasheet (7 sections + layered detail).
- `methodology.md`: Path B justification, benchmark design, training plan, and cost budget.
- `methodology_rationale.md`: extended rationale for design decisions.
- `schema.json`: Tenacious-Bench task schema plus three scored examples.
- `scoring_evaluator.py`: machine-verifiable scoring evaluator.
- `contamination_check.json`: contamination report (n-gram, embedding, time-shift results).
- `cost_log.md`: API cost log for all LLM calls during dataset authoring.
- `evidence_graph.json`: machine-readable claim registry with artifact pointers.
- `inter_rater_agreement.md`: single-annotator agreement protocol and results.
- `tenacious_bench_v0.1/`: raw tasks plus `train/`, `dev/`, and `held_out/` partitions.
- `generation_scripts/`: task generation, partitioning, judge routing, and contamination checks.
  - `generate_programmatic.py` — deterministic parameter sweeps; no LLM calls.
  - `generate_trace_derived.py` — deterministic probe-to-task conversion; no LLM calls.
  - `generate_sg_extra.py` — deterministic SG extras (`random.seed(99)`); no LLM calls.
  - `generate_synthesis.py` — multi-LLM synthesis with judge rotation (requires `OPENROUTER_API_KEY`).
  - `judge_filter.py` — **standalone LLM-as-judge filter pipeline**: pointwise scoring (coherence/verifiability/clarity, threshold 3.5), pairwise near-duplicate resolution, JSONL logging. Run independently or imported by `generate_synthesis.py`.
  - `judge_rotation_log.jsonl` — **execution evidence from the actual synthesis run**: 37 judge decisions (7 accepted, 30 rejected) logged by `generate_synthesis.py` during dataset authoring. Each entry records task ID, generator model, judge model, per-dimension scores, and accept/reject decision.
  - `judge_filter_summary.json` — human-readable summary of the judge filter run: acceptance rate, per-dimension average scores, dominant rejection reason (verifiability failures = 30/30 rejections).
  - `partition.py` — contamination-aware train/dev/held-out split.
  - `contamination_check.py` — n-gram, embedding-similarity, and time-shift checks.
- `training_data/`: Path B preference-pair formatter and generated pairs.
- `training/`: Colab training notebook for ORPO LoRA (`train_orpo_colab.ipynb`).
- `ablations/`: evaluation scripts and result templates.
- `synthesis_memos/`: paper-reading and design memos.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Optional environment variables:

```powershell
$env:OPENROUTER_API_KEY="..."
```

The evaluator works without an LLM key by using the deterministic tone heuristic.

## Reproduce The Interim Build

```powershell
python generation_scripts\partition.py
python generation_scripts\contamination_check.py
python scoring_evaluator.py --self-test
python training_data\format_preference_pairs.py
```

Expected current counts:

- Train: 111
- Dev: 68
- Held-out: 37
- Preference pairs: 158

## Score A Task

```powershell
python scoring_evaluator.py --self-test
```

For batch scoring, provide one JSONL of tasks and one JSONL of candidate outputs in matching order:

```powershell
python scoring_evaluator.py --task tenacious_bench_v0.1\dev\dev.jsonl --output outputs.jsonl --batch --out dev_scores.json
```

## Path B Training

The selected intervention is a judge/critic trained on preference pairs:

```powershell
python training_data\format_preference_pairs.py
```

Upload or open `training/train_orpo_colab.ipynb` in Colab with a T4 runtime. The adapter should be published as a LoRA artifact only.

## License

Dataset and all artifacts in this repository are released under **CC BY 4.0**.
See [LICENSE](LICENSE) and `datasheet.md` for use limitations.

## Attribution and Credits

**Author**: Kemeriya (kemeriya@10academy.org)

**External benchmarks**
- τ²-Bench — Sierra Research. Used as the public baseline whose retail-domain gaps motivated this work. [GitHub](https://github.com/sierra-research/tau2-bench)

**Libraries and frameworks**
- [Unsloth](https://github.com/unslothai/unsloth) — efficient LoRA fine-tuning
- [TRL](https://github.com/huggingface/trl) (HuggingFace) — ORPOTrainer
- [sentence-transformers](https://www.sbert.net/) — contamination embedding check (`all-MiniLM-L6-v2`)
- [httpx](https://www.python-httpx.org/) — OpenRouter API calls
- [OpenRouter](https://openrouter.ai/) — multi-LLM routing (Qwen3-235B generator, Claude Sonnet judge)

**Models**
- Generator: `qwen/qwen3-235b-a22b` via OpenRouter
- Judge / filter: `anthropic/claude-sonnet-4-6` via OpenRouter
- Training backbone: `unsloth/Qwen3.5-0.8B`
- Contamination embeddings: `sentence-transformers/all-MiniLM-L6-v2`

**Coursework context**: Tenacious-Bench v0.1 was developed as Week 11 coursework for the 10 Academy AI Mastery programme.
