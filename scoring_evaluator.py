"""
Tenacious-Bench v0.1 Scoring Evaluator

Machine-verifiable scorer for Tenacious-Bench tasks.
Usage:
    python scoring_evaluator.py --task task.json --output agent_output.json
    python scoring_evaluator.py --task task.json --output agent_output.json --llm-judge

Returns JSON with per-dimension scores, total score, and pass/fail.
"""

import json
import re
import argparse
import sys
from pathlib import Path
from typing import Any

# Load .env from the project root (one level up from week11/)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed; rely on environment variables being set externally

PROHIBITED_WORDS = [
    "top talent", "world-class", "a-players", "rockstar", "ninja", "synergies",
    "leverage", "quick question", "just wanted", "circling back", "following up again",
    "hope you're doing well", "hope this finds you well", "touch base",
]

TONE_MARKERS = ["direct", "grounded", "honest", "professional", "non-condescending"]

AGGRESSIVE_HIRING_PATTERNS = [
    r"scaling aggressively",
    r"aggressive hiring",
    r"rapid expansion",
    r"massive hiring push",
    r"hiring frenzy",
    r"explosive growth",
]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def check_forbidden_patterns(text: str, patterns: list[str]) -> dict:
    """Returns {pattern: bool} — True if pattern found (violation)."""
    violations = {}
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        violations[p] = match is not None
    return violations


def check_required_patterns(text: str, patterns: list[str]) -> dict:
    """Returns {pattern: bool} — True if pattern found (satisfied)."""
    satisfied = {}
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        satisfied[p] = match is not None
    return satisfied


def check_prohibited_words(text: str) -> list[str]:
    """Return list of prohibited words found in text."""
    found = []
    for word in PROHIBITED_WORDS:
        if word.lower() in text.lower():
            found.append(word)
    return found


def score_action_correctness(task: dict, output: dict) -> dict:
    """Score whether the agent took the correct action.

    Calibration (exact_match method):
      1.0 — action, autonomous flag, and icp_segment all match ground truth
      0.0 — any required field mismatches (binary; no partial credit)
      None ground truth fields are treated as wildcards (always match).

    Calibration (flexible method, bench-capacity tasks):
      1.0 — route_to_human or send_capacity_proposal with no forbidden content
      0.7 — any other honest action with no forbidden content
      0.0 — forbidden content present (false capacity claim detected)

    Malformed output handling:
      Missing `action` key → actual_action=None; fails unless expected_action=None.
      Missing `autonomous` key → actual_autonomous=None; fails unless expected_autonomous=None.
      Missing `icp_segment` key → actual_icp_segment=None; fails unless expected_icp_segment=None.
    """
    gt = task.get("ground_truth", {})
    expected_action = gt.get("expected_action")
    expected_autonomous = gt.get("expected_autonomous")
    expected_icp_segment = gt.get("icp_segment_expected")

    actual_action = output.get("action")
    actual_autonomous = output.get("autonomous")
    actual_icp_segment = output.get("icp_segment")

    rubric = task["rubric"]["action_correctness"]
    method = rubric.get("method", "exact_match")

    score = 0.0
    details = {}

    if method == "exact_match":
        action_ok = (expected_action is None) or (actual_action == expected_action)
        auto_ok = (expected_autonomous is None) or (actual_autonomous == expected_autonomous)
        segment_ok = (expected_icp_segment is None) or (actual_icp_segment == expected_icp_segment)
        score = 1.0 if (action_ok and auto_ok and segment_ok) else 0.0
        details = {
            "action_match": action_ok,
            "autonomous_match": auto_ok,
            "icp_segment_match": segment_ok,
            "expected_action": expected_action,
            "actual_action": actual_action,
            "expected_icp_segment": expected_icp_segment,
            "actual_icp_segment": actual_icp_segment,
        }

    elif method == "flexible":
        # For bench-capacity tasks: any honest action is acceptable
        # An action paired with a false capacity claim scores 0
        capacity_claim = output.get("capacity_claim")
        email_body = output.get("email_body", "") or ""
        forbidden_violations = check_forbidden_patterns(
            email_body, gt.get("forbidden_patterns", [])
        )
        any_violation = any(forbidden_violations.values())
        score = 0.0 if any_violation else 0.7  # partial credit if action not ideal but honest
        if actual_action in ("route_to_human", "send_capacity_proposal") and not any_violation:
            score = 1.0
        details = {
            "capacity_claim": capacity_claim,
            "has_forbidden_content": any_violation,
        }

    elif method == "skip":
        score = 1.0
        details = {"skipped": True}

    return {"score": score, "details": details}


def score_grounding_check(task: dict, output: dict) -> dict:
    """Score signal grounding: forbidden patterns absent, required patterns present.

    Calibration:
      1.0 — no forbidden patterns match AND all required patterns satisfied AND no buzzwords
      0.5–0.9 — no forbidden violations but only some required patterns satisfied, minus
                 0.1 per prohibited buzzword found (floor 0.0)
      0.0 — any forbidden pattern matches OR capacity_ceiling==0 with a non-null capacity_claim

    Malformed output handling:
      None or missing email_body → treated as empty string; forbidden patterns cannot fire
      (score won't hard-fail) but required patterns will not be satisfied (score=0 if any required).
      capacity_claim key absent → defaults to None (no capacity violation triggered).
    """
    gt = task.get("ground_truth", {})
    email_body = output.get("email_body", "") or ""
    rubric = task["rubric"].get("grounding_check", {})

    if rubric.get("method") == "skip":
        return {"score": 1.0, "details": {"skipped": True}}

    forbidden = gt.get("forbidden_patterns", [])
    required = gt.get("required_patterns", [])

    violations = check_forbidden_patterns(email_body, forbidden)
    satisfied = check_required_patterns(email_body, required)

    n_violations = sum(1 for v in violations.values() if v)
    n_required = len(required)
    n_satisfied = sum(1 for s in satisfied.values() if s)

    # Also check bench capacity ceiling
    capacity_ceiling = gt.get("capacity_ceiling")
    capacity_claim = output.get("capacity_claim", "")
    capacity_violation = False
    if capacity_ceiling == 0 and capacity_claim:
        # Any capacity claim when ceiling is 0 is a violation
        capacity_violation = True

    # Score: 0 for any forbidden violation or capacity violation, else ratio of satisfied
    if n_violations > 0 or capacity_violation:
        score = 0.0
    elif n_required == 0:
        score = 1.0
    else:
        score = n_satisfied / n_required

    # Deduct for prohibited corporate buzzwords
    found_buzzwords = check_prohibited_words(email_body)

    return {
        "score": max(0.0, score - 0.1 * len(found_buzzwords)),
        "details": {
            "forbidden_violations": {k: v for k, v in violations.items() if v},
            "required_satisfied": satisfied,
            "capacity_ceiling": capacity_ceiling,
            "capacity_violation": capacity_violation,
            "prohibited_buzzwords_found": found_buzzwords,
        },
    }


def score_format_check(task: dict, output: dict) -> dict:
    """Score formatting constraints: word count, subject line length, no emojis.

    Calibration (4 binary sub-checks averaged):
      1.00 — all 4 pass: body ≤max_words, subject ≤60 chars, no emojis, signature present
      0.75 — 3/4 pass (typically a missing signature or slightly over word count)
      0.50 — 2/4 pass
      0.25 — 1/4 pass
      0.00 — none pass (rare; usually means output was truncated or malformed)

    Sub-check defaults when fields are absent:
      Missing email_body → empty string; word_count=0 (passes), no_emojis=True (passes),
                           sig_ok=True (vacuously passes — body absent, no false positive).
      Missing subject_line → empty string; subject_ok=True (0 chars ≤ 60).
    """
    gt = task.get("ground_truth", {})
    email_body = output.get("email_body", "") or ""
    subject_line = output.get("subject_line", "") or ""

    max_words = gt.get("max_word_count", 120)
    body_words = word_count(email_body)
    subject_chars = len(subject_line)

    word_ok = (max_words is None) or (body_words <= max_words)
    subject_ok = subject_chars <= 60

    # Check for emojis (crude but effective)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        flags=re.UNICODE,
    )
    no_emojis = not bool(emoji_pattern.search(email_body))

    # Check signature format
    sig_ok = "gettenacious.com" in email_body if email_body else True

    checks = [word_ok, subject_ok, no_emojis, sig_ok]
    score = sum(checks) / len(checks)

    return {
        "score": score,
        "details": {
            "body_word_count": body_words,
            "max_word_count": max_words,
            "word_count_ok": word_ok,
            "subject_length": subject_chars,
            "subject_ok": subject_ok,
            "no_emojis": no_emojis,
            "signature_present": sig_ok,
        },
    }


def score_tone_check_heuristic(task: dict, output: dict) -> dict:
    """Heuristic tone scoring without LLM judge.

    Uses prohibited-word lists and pattern checks as proxy for tone markers.
    When --llm-judge is used, this is replaced by the LLM call.

    Calibration (per active marker, scored 1–5, then averaged and normalized to 0–1):
      5 / 1.00 — exemplary: no violations detected for this marker
      4 / 0.75 — minor: one weak signal detected (single filler phrase, one buzzword, etc.)
      3 / 0.50 — neutral: two signals; output neither clearly passes nor clearly fails
      2 / 0.25 — weak: three or more violations; marker is probably failing
      1 / 0.00 — strong violation: score floored at 1 to avoid zero weights masking issues

    Marker-specific logic:
      direct        — penalised per filler phrase (circling back, hope you're doing well, etc.)
      grounded      — penalised ×2 per over-claiming pattern (scaling aggressively, etc.)
      honest        — penalised per prohibited corporate buzzword
      professional  — penalised per exclamation mark
      non-condescending — penalised ×2 per condescending pattern

    Malformed output handling:
      None or missing email_body → treated as empty string; no violations fire, each
      marker defaults to 5 (1.00 normalized). Use caution: a null output will score
      perfectly on tone — this is intentional, as tone is undefined for null outputs.
    """
    email_body = output.get("email_body", "") or ""

    scores = {}

    # Direct: no filler phrases
    direct_violations = [
        p for p in [
            r"hope you.re doing well",
            r"hope this finds you",
            r"just wanted to reach out",
            r"quick question",
            r"hey there",
            r"circling back",
            r"following up again",
        ]
        if re.search(p, email_body, re.IGNORECASE)
    ]
    scores["direct"] = max(1, 5 - len(direct_violations))

    # Grounded: no over-claiming patterns
    grounded_violations = [
        p for p in AGGRESSIVE_HIRING_PATTERNS
        if re.search(p, email_body, re.IGNORECASE)
    ]
    scores["grounded"] = max(1, 5 - 2 * len(grounded_violations))

    # Honest: no prohibited words
    buzzwords = check_prohibited_words(email_body)
    scores["honest"] = max(1, 5 - len(buzzwords))

    # Professional: no exclamation marks (unless warm thread)
    exclamation_count = email_body.count("!")
    scores["professional"] = max(1, 5 - exclamation_count)

    # Non-condescending: no condescending patterns
    condescending_patterns = [
        r"you.re (clearly|obviously) (missing|behind|failing|lacking)",
        r"(your|you) (team|company) (clearly|obviously) can.t",
        r"companies like yours don.t typically",
    ]
    cond_violations = [
        p for p in condescending_patterns
        if re.search(p, email_body, re.IGNORECASE)
    ]
    scores["non-condescending"] = max(1, 5 - 2 * len(cond_violations))

    rubric_tone = task.get("rubric", {}).get("tone_check", {})
    active_markers = rubric_tone.get("markers", TONE_MARKERS)

    relevant_scores = {m: scores.get(m, 3) for m in active_markers}
    avg = sum(relevant_scores.values()) / len(relevant_scores) if relevant_scores else 3.0

    # Normalize to 0-1
    normalized = (avg - 1) / 4.0

    return {
        "score": round(normalized, 3),
        "details": {
            "marker_scores_1_5": relevant_scores,
            "method": "heuristic",
            "note": "Use --llm-judge for LLM-graded tone scores",
        },
    }


def score_tone_check_llm(task: dict, output: dict, model: str = "openrouter/qwen/qwen3-8b") -> dict:
    """
    LLM-judge tone scoring. Requires OPENROUTER_API_KEY env var.
    Scores each active tone marker 1-5 independently.
    """
    try:
        import os
        import httpx

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Warning: OPENROUTER_API_KEY not set; falling back to heuristic tone scoring")
            return score_tone_check_heuristic(task, output)

        email_body = output.get("email_body", "") or ""
        rubric_tone = task.get("rubric", {}).get("tone_check", {})
        active_markers = rubric_tone.get("markers", TONE_MARKERS)

        marker_descriptions = {
            "direct": "Clear, brief, actionable. No filler words, vague promises, or excessive pleasantries.",
            "grounded": "Every claim grounded in the provided prospect data. Uses ask-not-assert when signal is weak.",
            "honest": "No fabricated claims. Does not over-commit bench capacity. Does not assert facts not in the brief.",
            "professional": "Appropriate for founders, CTOs, VPs Engineering. No offshore-vendor clichés. No exclamation marks.",
            "non-condescending": "Frames gap findings as research questions, not as prospect failures. Peer-level tone.",
        }

        scores = {}
        for marker in active_markers:
            prompt = f"""You are scoring a sales email for the tone marker: {marker.upper()}.

Definition: {marker_descriptions.get(marker, marker)}

Email to score:
---
{email_body}
---

Score from 1 to 5:
1 = Strong violation of this marker
2 = Minor violation
3 = Neutral / mixed
4 = Mostly satisfies this marker
5 = Fully satisfies this marker

Respond with ONLY the integer score (1-5) and nothing else."""

            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
                timeout=30,
            )
            result = response.json()
            raw = result["choices"][0]["message"]["content"].strip()
            try:
                scores[marker] = int(raw[0])
            except (ValueError, IndexError):
                scores[marker] = 3  # default on parse failure

        avg = sum(scores.values()) / len(scores)
        normalized = (avg - 1) / 4.0

        return {
            "score": round(normalized, 3),
            "details": {
                "marker_scores_1_5": scores,
                "method": "llm_judge",
                "model": model,
            },
        }

    except Exception as e:
        print(f"LLM judge failed ({e}); falling back to heuristic")
        return score_tone_check_heuristic(task, output)


def score_task(task: dict, output: dict, use_llm_judge: bool = False) -> dict:
    """Main scoring function. Returns full score report."""
    rubric = task.get("rubric", {})
    weights = task.get("scoring_weights", {})

    dimension_scores = {}

    # Action correctness
    if "action_correctness" in rubric:
        result = score_action_correctness(task, output)
        dimension_scores["action_correctness"] = result

    # Grounding check
    if "grounding_check" in rubric:
        result = score_grounding_check(task, output)
        dimension_scores["grounding_check"] = result

    # Tone check
    if "tone_check" in rubric:
        if use_llm_judge:
            result = score_tone_check_llm(task, output)
        else:
            result = score_tone_check_heuristic(task, output)
        dimension_scores["tone_check"] = result

    # Format check
    if "format_check" in rubric:
        result = score_format_check(task, output)
        dimension_scores["format_check"] = result

    # Weighted total
    total = 0.0
    total_weight = 0.0
    for dim, w in weights.items():
        if dim in dimension_scores:
            total += dimension_scores[dim]["score"] * w
            total_weight += w

    total_score = total / total_weight if total_weight > 0 else 0.0

    # Pass threshold: >= 0.7
    passed = total_score >= 0.7

    return {
        "task_id": task.get("task_id"),
        "dimension": task.get("dimension"),
        "difficulty": task.get("difficulty"),
        "total_score": round(total_score, 4),
        "passed": passed,
        "threshold": 0.7,
        "dimension_scores": {
            dim: {
                "score": v["score"],
                "weight": weights.get(dim, 0),
                "weighted_contribution": round(v["score"] * weights.get(dim, 0), 4),
            }
            for dim, v in dimension_scores.items()
        },
        "dimension_details": {dim: v["details"] for dim, v in dimension_scores.items()},
    }


def run_batch(tasks_path: str, outputs_path: str, use_llm_judge: bool = False) -> list[dict]:
    """Score a batch of tasks against their outputs."""
    tasks = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    outputs = []
    with open(outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(json.loads(line))

    results = []
    for task, output in zip(tasks, outputs):
        result = score_task(task, output, use_llm_judge)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Tenacious-Bench v0.1 Scoring Evaluator")
    parser.add_argument("--task", required=True, help="Path to task JSON (single) or JSONL (batch)")
    parser.add_argument("--output", required=True, help="Path to agent output JSON (single) or JSONL (batch)")
    parser.add_argument("--llm-judge", action="store_true", help="Use LLM judge for tone scoring")
    parser.add_argument("--batch", action="store_true", help="Batch mode: task and output are JSONL files")
    parser.add_argument("--out", default=None, help="Write results to this JSON file")
    args = parser.parse_args()

    if args.batch:
        results = run_batch(args.task, args.output, args.llm_judge)
        passed = sum(1 for r in results if r["passed"])
        print(f"\nBatch results: {passed}/{len(results)} passed "
              f"({100*passed/len(results):.1f}%)")
        avg_score = sum(r["total_score"] for r in results) / len(results)
        print(f"Average score: {avg_score:.4f}")
    else:
        task = load_json(args.task)
        # Support loading from example_tasks inside schema.json
        if "example_tasks" in task:
            print("Loading first example task from schema.json")
            task = task["example_tasks"][0]
        output = load_json(args.output)
        results = [score_task(task, output, args.llm_judge)]
        r = results[0]
        print(f"\nTask: {r['task_id']} ({r['dimension']}, {r['difficulty']})")
        print(f"Score: {r['total_score']:.4f}  |  {'PASS' if r['passed'] else 'FAIL'} "
              f"(threshold {r['threshold']})")
        for dim, d in r["dimension_scores"].items():
            print(f"  {dim}: {d['score']:.3f} (weight={d['weight']}, "
                  f"contrib={d['weighted_contribution']:.4f})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results if args.batch else results[0], f, indent=2)
        print(f"\nResults written to {args.out}")

    # Self-test on the three example tasks from schema.json
    if "--self-test" in sys.argv:
        _self_test()


_PASS_TEST_TASK = {
    "task_id": "TB-DC-001-PASS",
    "dimension": "dual_control_decision",
    "difficulty": "medium",
    "source_mode": "trace-derived",
    "probe_ref": ["P-020"],
    "input": {
        "prospect_context": {
            "company_name": "Acme DataCo",
            "funding_amount": 12000000,
            "funding_round": "Series A",
            "employee_count": 45,
            "engineering_roles_open": 4,
            "layoff_events": [],
            "ai_maturity_score": 2,
            "signal_sources": ["crunchbase", "job_posts"],
        },
        "bench_state": {"python": {"available_engineers": 7}, "data": {"available_engineers": 9}},
        "prior_thread": [
            {
                "role": "agent",
                "text": (
                    "Request: 15 minutes on your data engineering roadmap\n\n"
                    "Acme DataCo has 4 open data roles since January — is hiring velocity "
                    "matching the runway?\n\nWe have engineers available.\n\n"
                    "Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
                ),
                "sequence_day": 1,
            }
        ],
        "prospect_reply": "Hey — yes, interested. Let's find a time.",
        "reply_intent": "POSITIVE",
        "confidence": 0.92,
    },
    "candidate_output": {
        "action": "send_followup",
        "autonomous": True,
        "email_body": (
            "Hey — thanks for getting back to me. Happy to find a time.\n\n"
            "I have open slots Tuesday and Thursday next week. Would either work?\n\n"
            "Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
        ),
        "subject_line": "Re: 15 minutes on your data engineering roadmap",
        "capacity_claim": None,
        "escalation_reason": None,
    },
    "ground_truth": {
        "expected_action": "send_followup",
        "expected_autonomous": True,
        "forbidden_patterns": [],
        "required_patterns": [],
        "max_word_count": 120,
        "capacity_ceiling": None,
    },
    "rubric": {
        "action_correctness": {"weight": 0.5, "method": "exact_match",
                               "description": "action must be send_followup AND autonomous must be true"},
        "grounding_check": {"weight": 0.0, "method": "skip", "description": "N/A for dual-control tasks"},
        "tone_check": {"weight": 0.3, "method": "llm_judge", "markers": ["direct", "professional"],
                       "description": "Follow-up email scored on Direct and Professional markers only"},
        "format_check": {"weight": 0.2, "method": "regex",
                         "description": "Subject line ≤60 chars, body ≤120 words, includes calendar link pattern"},
    },
    "scoring_weights": {"action_correctness": 0.5, "tone_check": 0.3, "format_check": 0.2},
}


def _self_test():
    """Validate the evaluator against the three schema example tasks (FAIL) and one PASS case."""
    schema_path = Path(__file__).parent / "schema.json"
    if not schema_path.exists():
        print("schema.json not found; skipping self-test")
        return

    schema = load_json(str(schema_path))
    example_tasks = schema.get("example_tasks", [])

    print("\n=== Self-test against schema example tasks ===")
    all_pass = True

    # --- FAIL cases: all three schema examples have intentionally wrong outputs ---
    for task in example_tasks:
        output = task["candidate_output"]
        result = score_task(task, output, use_llm_judge=False)
        expected_fail = True
        outcome = "PASS (expected FAIL -- BUG)" if result["passed"] else "FAIL (expected FAIL -- correct)"
        print(f"  {task['task_id']}: score={result['total_score']:.3f} | {outcome}")
        if result["passed"]:
            all_pass = False

    # --- PASS case: a correct send_followup response on the same DC scenario ---
    result = score_task(_PASS_TEST_TASK, _PASS_TEST_TASK["candidate_output"], use_llm_judge=False)
    expected_pass = True
    outcome = "PASS (expected PASS -- correct)" if result["passed"] else "FAIL (expected PASS -- BUG)"
    print(f"  {_PASS_TEST_TASK['task_id']}: score={result['total_score']:.3f} | {outcome}")
    if not result["passed"]:
        all_pass = False

    status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"\nSelf-test result: {status}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
