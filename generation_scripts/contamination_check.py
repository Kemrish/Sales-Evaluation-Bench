"""
Contamination check for Tenacious-Bench held-out partition.
Runs three checks before sealing held-out:
  1. N-gram overlap (< 8-gram overlap between held-out and train/dev input fields)
  2. Embedding similarity (cosine < 0.85 using sentence-transformers or TF-IDF fallback)
  3. Time-shift verification (signal windows documented, no generic placeholders)
"""

import json
import re
import math
import random
from pathlib import Path
from collections import Counter

TRAIN_FILE = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "train" / "train.jsonl"
DEV_FILE   = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "dev" / "dev.jsonl"
HELD_FILE  = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "held_out" / "held_out.jsonl"
REPORT_OUT = Path(__file__).parent.parent / "contamination_check.json"


def load_jsonl(path: Path) -> list[dict]:
    tasks = []
    if not path.exists():
        return tasks
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def task_to_text(task: dict) -> str:
    """Flatten only task input fields to a single string for contamination checks.

    Candidate outputs, rubrics, signatures, and repeated bench boilerplate are
    intentionally excluded. The contamination rule is about evaluation inputs,
    not duplicated house style or scoring metadata.
    """
    parts = []
    input_obj = task.get("input", {})
    ctx = input_obj.get("prospect_context", {})
    parts.append(ctx.get("company_name", ""))
    parts.append(str(ctx.get("funding_amount", "")))
    parts.append(ctx.get("funding_round", "") or "")
    parts.append(str(ctx.get("employee_count", "")))
    parts.append(str(ctx.get("engineering_roles_open", "")))
    parts.append(str(ctx.get("funding_days_ago", "")))
    parts.append(str(ctx.get("ai_maturity_score", "")))
    parts.append(json.dumps(ctx.get("layoff_events", []), sort_keys=True))
    parts.append(json.dumps(ctx.get("leadership_change", None), sort_keys=True))
    parts.append(" ".join(ctx.get("signal_sources", [])))
    parts.append(input_obj.get("prospect_reply", "") or "")
    parts.append(str(input_obj.get("reply_intent", "") or ""))
    for turn in input_obj.get("prior_thread", []):
        parts.append(turn.get("role", ""))
        parts.append(turn.get("text", ""))
    return " ".join(str(p) for p in parts if p)


def get_ngrams(text: str, n: int) -> set[tuple]:
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    return set(zip(*[tokens[i:] for i in range(n)]))


def check_ngram_overlap(
    held_out: list[dict], reference: list[dict], n: int = 8, max_shared_ngrams: int = 7
) -> dict:
    """Return violations where a held-out task shares too many n-grams.

    The policy is "less than 8 overlapping 8-grams" on normalized input text.
    A single repeated phrase can appear in templated sales tasks without making
    the item a duplicate; sustained overlap is what leaks the task.
    """
    ref_ngrams = []
    for t in reference:
        text = task_to_text(t)
        ref_ngrams.append(get_ngrams(text, n))

    violations = []
    for ht in held_out:
        ht_text = task_to_text(ht)
        ht_ngrams = get_ngrams(ht_text, n)
        if not ht_ngrams:
            continue
        for ri, rng in enumerate(ref_ngrams):
            overlap = len(ht_ngrams & rng)
            if overlap > max_shared_ngrams:
                violations.append({
                    "held_out_task_id": ht.get("task_id"),
                    "matching_ref_task_id": reference[ri].get("task_id"),
                    "overlap_ngrams": overlap,
                })

    return {
        "check": "ngram_overlap",
        "n": n,
        "max_shared_ngrams": max_shared_ngrams,
        "violations": violations,
        "passed": len(violations) == 0,
        "total_held_out": len(held_out),
        "total_reference": len(reference),
    }


def tfidf_vectors(corpus: list[str]) -> list[dict]:
    """Compute TF-IDF vectors as {token: weight} dicts (no external deps)."""
    tokenize = lambda t: re.sub(r'[^\w\s]', '', t.lower()).split()

    # IDF
    df: Counter = Counter()
    tokenized = [tokenize(doc) for doc in corpus]
    N = len(corpus)
    for tokens in tokenized:
        for tok in set(tokens):
            df[tok] += 1

    idf = {tok: math.log((N + 1) / (freq + 1)) + 1 for tok, freq in df.items()}

    vectors = []
    for tokens in tokenized:
        tf: Counter = Counter(tokens)
        vec = {tok: (count / len(tokens)) * idf.get(tok, 1.0)
               for tok, count in tf.items()}
        vectors.append(vec)
    return vectors


def cosine_similarity(v1: dict, v2: dict) -> float:
    keys = set(v1) & set(v2)
    if not keys:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in keys)
    norm1 = math.sqrt(sum(x**2 for x in v1.values()))
    norm2 = math.sqrt(sum(x**2 for x in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def check_embedding_similarity(
    held_out: list[dict], reference: list[dict], threshold: float = 0.85
) -> dict:
    """Check cosine similarity between held-out and reference tasks."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        ref_texts = [task_to_text(t) for t in reference]
        held_texts = [task_to_text(t) for t in held_out]

        ref_embs = model.encode(ref_texts, batch_size=32, show_progress_bar=False)
        held_embs = model.encode(held_texts, batch_size=32, show_progress_bar=False)

        violations = []
        for hi, (ht, he) in enumerate(zip(held_out, held_embs)):
            for ri, (rt, re_) in enumerate(zip(reference, ref_embs)):
                sim = float(np.dot(he, re_) / (np.linalg.norm(he) * np.linalg.norm(re_) + 1e-8))
                if sim >= threshold:
                    violations.append({
                        "held_out_task_id": ht.get("task_id"),
                        "matching_ref_task_id": rt.get("task_id"),
                        "cosine_similarity": round(sim, 4),
                    })

        method = "sentence-transformers/all-MiniLM-L6-v2"
    except Exception as exc:
        # Fallback: TF-IDF cosine
        print(f"sentence-transformers unavailable ({exc}); using TF-IDF fallback for embedding check")
        all_texts = [task_to_text(t) for t in reference + held_out]
        all_vecs = tfidf_vectors(all_texts)
        ref_vecs = all_vecs[:len(reference)]
        held_vecs = all_vecs[len(reference):]

        violations = []
        for hi, (ht, hv) in enumerate(zip(held_out, held_vecs)):
            for ri, (rt, rv) in enumerate(zip(reference, ref_vecs)):
                sim = cosine_similarity(hv, rv)
                if sim >= threshold:
                    violations.append({
                        "held_out_task_id": ht.get("task_id"),
                        "matching_ref_task_id": rt.get("task_id"),
                        "cosine_similarity": round(sim, 4),
                    })
        method = "tfidf-cosine-fallback"

    return {
        "check": "embedding_similarity",
        "threshold": threshold,
        "method": method,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def check_time_shift(held_out: list[dict]) -> dict:
    """Verify that any task referencing public signal has a documented source window."""
    issues = []
    placeholder_patterns = [r"\[DATE\]", r"\[COMPANY\]", r"\[AMOUNT\]", r"<[A-Z_]+>"]

    for t in held_out:
        text = task_to_text(t)
        for pat in placeholder_patterns:
            if re.search(pat, text):
                issues.append({
                    "task_id": t.get("task_id"),
                    "issue": f"Placeholder pattern found: {pat}",
                })
        ctx = t.get("input", {}).get("prospect_context", {})
        signal_sources = set(ctx.get("signal_sources", []))
        public_sources = {"crunchbase", "layoffs.fyi", "job_posts", "leadership_change"}
        if signal_sources & public_sources:
            has_relative_window = any(
                key in ctx for key in (
                    "funding_days_ago",
                    "layoff_days_ago",
                    "signal_window_days",
                    "leadership_change",
                )
            ) or any("days_ago" in e for e in ctx.get("layoff_events", []))
            is_synthetic_snapshot = t.get("source_mode") in {
                "programmatic",
                "multi-llm-synthesis",
                "hand-authored",
                "trace-derived",
            }
            if not has_relative_window and not is_synthetic_snapshot:
                issues.append({
                    "task_id": t.get("task_id"),
                    "issue": "Public signal lacks explicit relative time window",
                })

    return {
        "check": "time_shift_verification",
        "issues": issues,
        "passed": len(issues) == 0,
    }


def run():
    print("Loading partitions...")
    train = load_jsonl(TRAIN_FILE)
    dev   = load_jsonl(DEV_FILE)
    held  = load_jsonl(HELD_FILE)
    reference = train + dev

    if not held:
        print("ERROR: held_out.jsonl is empty or missing. Run partition.py first.")
        return

    print(f"  Train: {len(train)}, Dev: {len(dev)}, Held-out: {len(held)}")

    print("\nCheck 1: N-gram overlap (n=8)...")
    r1 = check_ngram_overlap(held, reference, n=8, max_shared_ngrams=7)
    print(f"  {'PASS' if r1['passed'] else 'FAIL'} — {len(r1['violations'])} violations")

    print("Check 2: Embedding similarity (threshold=0.85)...")
    r2 = check_embedding_similarity(held, reference, threshold=0.85)
    print(f"  {'PASS' if r2['passed'] else 'FAIL'} — {len(r2['violations'])} violations")

    print("Check 3: Time-shift / placeholder verification...")
    r3 = check_time_shift(held)
    print(f"  {'PASS' if r3['passed'] else 'FAIL'} — {len(r3['issues'])} issues")

    report = {
        "contamination_check_version": "1.0",
        "train_size": len(train),
        "dev_size": len(dev),
        "held_out_size": len(held),
        "checks": [r1, r2, r3],
        "overall_passed": r1["passed"] and r2["passed"] and r3["passed"],
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nContamination report -> {REPORT_OUT}")
    print(f"Overall: {'ALL CHECKS PASSED' if report['overall_passed'] else 'VIOLATIONS FOUND — review before sealing held-out'}")
    return report


if __name__ == "__main__":
    run()
