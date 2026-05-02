# Cost Log

All API and compute charges for Week 11 are recorded here. Local deterministic scripts cost $0.

| Timestamp | Bucket | Service | Purpose | Cost USD | Notes |
|---|---|---|---|---:|---|
| 2026-04-28 | Dataset authoring | Local scripts | Programmatic, trace-derived, and hand-authored generation | 0.00 | No paid calls; all generation is deterministic |
| 2026-04-28 | Dataset authoring | OpenRouter — qwen/qwen3-235b-a22b | Multi-LLM synthesis: 37 generation calls for 8 seed tasks (multiple retry rounds) | 0.01 | ~37K input tokens + ~18.5K output tokens @ $0.13/$0.40 per 1M |
| 2026-04-28 | Dataset authoring | OpenRouter — anthropic/claude-sonnet-4-6 | Multi-LLM synthesis: 37 judge filter calls (coherence, verifiability, clarity scores) | 0.09 | ~26K input tokens + ~1K output tokens @ $3/$15 per 1M; all 30 rejected seeds rejected on verifiability |
| 2026-04-28 | Dataset QA | Local scripts | Partitioning and contamination checks | 0.00 | Embedding check re-run 2026-04-29 with sentence-transformers/all-MiniLM-L6-v2 (local); TF-IDF fallback was used in first committed run on 2026-04-28 |
| 2026-04-28 | Training prep | Local scripts | Path B preference-pair construction | 0.00 | 62 pairs generated from train partition |
| 2026-05-01 | Training run | Google Colab T4 (free tier) | ORPO adapter training — Qwen3.5-0.8B, LoRA r=16, 158 pairs, ~200 steps (early stopped) | 0.00 | Free-tier T4; session limit caused early stop at ~200/750 configured steps |

Budget cap: $10.00.

Running total: $0.10.
Remaining: $9.90.
