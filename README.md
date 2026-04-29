# Tenacious-Bench v0.1

Tenacious-Bench is a small, machine-scored benchmark for Tenacious-style B2B sales agents. It evaluates the behaviors that a generic retail/task benchmark misses: signal-grounded outreach, bench-capacity honesty, dual-control routing, ICP segment classification, signal reliability, and Tenacious tone adherence.

## Status

Current state: Acts I-III complete. Act IV (training + ablations) in progress.

- 216 authored tasks across train/dev/held-out partitions.
- Held-out contamination check passes (n-gram, sentence-transformers embedding, time-shift).
- Path B selected: a preference-tuned judge/critic using SimPO.
- 62 preference pairs generated from the train partition.
- 9 synthesis memos committed (4 common-reading + 3 Path B + 2 project analysis).
- Training notebook ready in `training/train_simpo_colab.ipynb` — run on Colab T4 to produce adapter.

## Repository Layout

- `audit_memo.md`: benchmark-gap audit.
- `schema.json`: Tenacious-Bench task schema plus three scored examples.
- `scoring_evaluator.py`: machine-verifiable scoring evaluator.
- `tenacious_bench_v0.1/`: raw tasks plus `train/`, `dev/`, and `held_out/` partitions.
- `generation_scripts/`: task generation, partitioning, judge routing, and contamination checks.
- `training_data/`: Path B preference-pair formatter and generated pairs.
- `training/`: Colab training notebook for SimPO LoRA.
- `ablations/`: evaluation scripts and result templates.
- `synthesis_memos/`: reading and design memos.

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
- Preference pairs: 62

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

Upload or open `training/train_simpo_colab.ipynb` in Colab with a T4 runtime. The adapter should be published as a LoRA artifact only.

## License

Dataset license target: CC-BY-4.0. See `datasheet.md` and `methodology.md` for rationale and use limitations.
