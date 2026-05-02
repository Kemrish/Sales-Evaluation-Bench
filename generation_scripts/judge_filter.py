"""
Judge Filter Pipeline — Tenacious-Bench v0.1

Standalone LLM-as-a-judge quality filter for candidate tasks.

Two modes
---------
1. Pointwise:  score each task on coherence, verifiability, and rubric clarity.
               Reject if any dimension < POINTWISE_THRESHOLD.
2. Pairwise:   when two tasks are near-duplicates, compare them head-to-head and
               keep only the higher-quality one.

Threshold constants
-------------------
POINTWISE_THRESHOLD  = 3.5   (score 1-5; all three dimensions must meet this)
PAIRWISE_MIN_DELTA   = 0.0   (keep the higher-scoring task; ties go to task_a)
NEAR_DUPLICATE_JACCARD = 0.70 (4-gram Jaccard; tasks above this are near-duplicates)

Run standalone
--------------
  python generation_scripts/judge_filter.py \\
      --input tenacious_bench_v0.1/raw_synthesis.jsonl \\
      --output tenacious_bench_v0.1/filtered_synthesis.jsonl \\
      --log    generation_scripts/judge_filter_log.jsonl

Requires OPENROUTER_API_KEY in the environment or .env file.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_MODEL = "anthropic/claude-sonnet-4-6"   # must differ from generator model
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

POINTWISE_THRESHOLD: float = 3.5    # min score (1–5) on each of 3 dimensions
PAIRWISE_MIN_DELTA: float = 0.0     # keep task with strictly higher sum; ties → task_a
NEAR_DUPLICATE_JACCARD: float = 0.70  # 4-gram Jaccard threshold for near-duplicate detection

DEFAULT_LOG = Path(__file__).parent / "judge_filter_log.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int = 4) -> set:
    words = text.lower().split()
    return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _task_text(task: dict) -> str:
    """Extract searchable text from a task dict."""
    parts = [
        json.dumps(task.get("input", {}).get("prospect_context", {})),
        task.get("description", ""),
        json.dumps(task.get("ground_truth", {})),
    ]
    return " ".join(parts)


def is_near_duplicate(task_a: dict, task_b: dict) -> bool:
    """Return True if 4-gram Jaccard similarity >= NEAR_DUPLICATE_JACCARD."""
    grams_a = _ngrams(_task_text(task_a))
    grams_b = _ngrams(_task_text(task_b))
    return _jaccard(grams_a, grams_b) >= NEAR_DUPLICATE_JACCARD


def _call_judge(messages: list[dict], max_tokens: int = 200) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tenacious.consulting",
        "X-Title": "Tenacious-Bench Judge Filter",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = httpx.post(BASE_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json_response(raw: str, fallback: dict) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return fallback


# ---------------------------------------------------------------------------
# Pointwise judge
# ---------------------------------------------------------------------------

def pointwise_judge(task: dict) -> dict:
    """
    Score a task on three dimensions (1–5 each).

    Dimensions
    ----------
    coherence      : Is the prospect_context internally consistent and realistic?
    verifiability  : Can the rubric be applied mechanically (regex / exact-match)?
    clarity        : Is it unambiguous what the correct agent behaviour should be?

    Threshold: accept = True iff all three scores >= POINTWISE_THRESHOLD (3.5).

    Returns
    -------
    dict with keys: coherence, verifiability, clarity, sum, accept, judge_model
    """
    prompt = f"""You are a quality judge for sales-agent evaluation tasks.
Score this task on three dimensions from 1 to 5:

1. coherence     — Is the prospect_context internally consistent and realistic?
2. verifiability — Can the rubric be applied mechanically (regex/exact-match)?
3. clarity       — Is it unambiguous what the correct agent behaviour should be?

Task:
{json.dumps(task, indent=2)}

Rules:
- Score each dimension 1–5 (integer or .5 steps).
- accept = true  if ALL THREE scores >= {POINTWISE_THRESHOLD}
- accept = false if ANY score < {POINTWISE_THRESHOLD}
- Respond ONLY with a JSON object, no markdown:
  {{"coherence": X, "verifiability": X, "clarity": X, "accept": true/false}}"""

    raw = _call_judge([{"role": "user", "content": prompt}], max_tokens=120)
    result = _parse_json_response(raw, {"coherence": 3, "verifiability": 3, "clarity": 3, "accept": False})

    result["sum"] = (
        result.get("coherence", 0)
        + result.get("verifiability", 0)
        + result.get("clarity", 0)
    )
    result["judge_model"] = JUDGE_MODEL

    # Enforce threshold even if model said accept=True
    result["accept"] = (
        result.get("coherence", 0) >= POINTWISE_THRESHOLD
        and result.get("verifiability", 0) >= POINTWISE_THRESHOLD
        and result.get("clarity", 0) >= POINTWISE_THRESHOLD
    )
    return result


# ---------------------------------------------------------------------------
# Pairwise judge
# ---------------------------------------------------------------------------

def pairwise_judge(task_a: dict, task_b: dict) -> dict:
    """
    Compare two near-duplicate tasks and decide which to keep.

    Used when is_near_duplicate(task_a, task_b) is True.
    Asks the judge which task is of higher quality overall.

    Returns
    -------
    dict with keys:
        preferred     : "A" | "B" | "tie"
        keep_task_id  : task_id of the task to retain
        reasoning     : one-sentence rationale from the judge
        judge_model   : model used
    """
    prompt = f"""Two evaluation tasks are near-duplicates. Choose which one is higher quality.

TASK A:
{json.dumps(task_a, indent=2)}

TASK B:
{json.dumps(task_b, indent=2)}

Criteria (in order of importance):
1. Verifiability: can the rubric be applied mechanically?
2. Clarity: is the correct behaviour unambiguous?
3. Coherence: is the scenario realistic?

Respond ONLY with JSON, no markdown:
{{"preferred": "A" or "B" or "tie", "reasoning": "<one sentence>"}}"""

    raw = _call_judge([{"role": "user", "content": prompt}], max_tokens=150)
    result = _parse_json_response(raw, {"preferred": "tie", "reasoning": "parse error"})

    preferred = result.get("preferred", "tie")
    if preferred == "A":
        keep_id = task_a.get("task_id", "task_a")
    elif preferred == "B":
        keep_id = task_b.get("task_id", "task_b")
    else:
        keep_id = task_a.get("task_id", "task_a")  # ties go to task_a

    result["keep_task_id"] = keep_id
    result["judge_model"] = JUDGE_MODEL
    return result


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------

def filter_tasks(tasks: list[dict], log_path: Path = DEFAULT_LOG) -> list[dict]:
    """
    Run the full judge filter pipeline over a list of tasks.

    Steps
    -----
    1. Pointwise filter: reject tasks below POINTWISE_THRESHOLD on any dimension.
    2. Near-duplicate detection: identify pairs with Jaccard >= NEAR_DUPLICATE_JACCARD.
    3. Pairwise resolution: for each duplicate pair, keep the preferred task.

    All decisions are logged to log_path (JSONL).

    Returns
    -------
    Filtered list of accepted tasks.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_entries = []

    # Step 1 — pointwise
    print(f"[judge_filter] Pointwise scoring {len(tasks)} tasks (threshold={POINTWISE_THRESHOLD})...")
    pointwise_accepted = []
    for task in tasks:
        scores = pointwise_judge(task)
        entry = {
            "stage": "pointwise",
            "task_id": task.get("task_id"),
            "scores": scores,
            "decision": "accept" if scores["accept"] else "reject",
        }
        log_entries.append(entry)
        if scores["accept"]:
            task["_judge_scores"] = scores
            pointwise_accepted.append(task)
        else:
            print(f"  REJECT {task.get('task_id')} — scores: {scores}")
        time.sleep(0.5)

    print(f"[judge_filter] After pointwise: {len(pointwise_accepted)}/{len(tasks)} accepted.")

    # Step 2 & 3 — pairwise near-duplicate resolution
    print(f"[judge_filter] Checking near-duplicates (Jaccard threshold={NEAR_DUPLICATE_JACCARD})...")
    kept = []
    dropped_ids: set = set()

    for i, task_a in enumerate(pointwise_accepted):
        if task_a.get("task_id") in dropped_ids:
            continue
        for task_b in pointwise_accepted[i + 1:]:
            if task_b.get("task_id") in dropped_ids:
                continue
            if is_near_duplicate(task_a, task_b):
                result = pairwise_judge(task_a, task_b)
                loser_id = (
                    task_b.get("task_id")
                    if result["preferred"] in ("A", "tie")
                    else task_a.get("task_id")
                )
                dropped_ids.add(loser_id)
                entry = {
                    "stage": "pairwise",
                    "task_a": task_a.get("task_id"),
                    "task_b": task_b.get("task_id"),
                    "preferred": result["preferred"],
                    "keep": result["keep_task_id"],
                    "drop": loser_id,
                    "reasoning": result.get("reasoning", ""),
                }
                log_entries.append(entry)
                print(f"  DUPLICATE pair ({task_a.get('task_id')} vs {task_b.get('task_id')}) → keep {result['keep_task_id']}")
                time.sleep(0.5)
        if task_a.get("task_id") not in dropped_ids:
            kept.append(task_a)

    print(f"[judge_filter] After pairwise: {len(kept)} tasks retained.")

    with open(log_path, "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"[judge_filter] Log written to {log_path}")

    return kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge quality filter for Tenacious-Bench tasks.")
    parser.add_argument("--input",  required=True, help="Input JSONL of candidate tasks")
    parser.add_argument("--output", required=True, help="Output JSONL of accepted tasks")
    parser.add_argument("--log",    default=str(DEFAULT_LOG), help="Log file path (JSONL)")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.")
        return

    tasks = [json.loads(l) for l in open(args.input) if l.strip()]
    accepted = filter_tasks(tasks, log_path=Path(args.log))

    with open(args.output, "w") as f:
        for task in accepted:
            task.pop("_judge_scores", None)
            task.pop("_raw", None)
            f.write(json.dumps(task) + "\n")

    print(f"\nDone. {len(accepted)}/{len(tasks)} tasks written to {args.output}")


if __name__ == "__main__":
    main()
