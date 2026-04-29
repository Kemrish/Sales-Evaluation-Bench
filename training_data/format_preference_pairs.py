"""
Format Tenacious-Bench v0.1 train partition into SimPO preference pairs.

Output format (HuggingFace TRL CPOTrainer / SimPO):
  {"prompt": "...", "chosen": "...", "rejected": "..."}

Strategy (from synthesis_memos/memo_02):
  - rejected: candidate_output from WRONG tasks (has forbidden_patterns)
  - chosen: candidate_output from a matched CORRECT task in the same dimension
  - Fallback: construct minimal correct output from ground_truth fields
"""

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

TRAIN_FILE = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "train" / "train.jsonl"
OUT_FILE   = Path(__file__).parent / "preference_pairs.jsonl"
STATS_FILE = Path(__file__).parent / "pair_stats.json"


def load_jsonl(path: Path) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def build_prompt(task: dict) -> str:
    """Serialize input context to a natural-language prompt string."""
    inp = task.get("input", {})
    ctx = inp.get("prospect_context", {})
    bench = inp.get("bench_state", {})
    thread = inp.get("prior_thread", [])
    reply = inp.get("prospect_reply")
    intent = inp.get("reply_intent")
    confidence = inp.get("confidence")

    lines = ["<|system|>"]
    lines.append("You are the Tenacious Intelligence Corporation sales agent.")
    lines.append(
        "Tenacious provides engineering staff augmentation. "
        "You classify prospects into ICP segments, generate grounded outreach emails, "
        "and decide when to act autonomously vs. escalate to a human."
    )
    lines.append("")
    lines.append("Bench state (available engineers):")
    for stack, info in bench.items():
        lines.append(f"  {stack}: {info.get('available_engineers', 0)} available")
    lines.append("")
    lines.append("Policies:")
    lines.append("  - Act autonomously only if reply_intent is POSITIVE AND confidence >= 0.65")
    lines.append("  - Escalate for: pricing questions, legal/NDA requests, GDPR questions, unsubscribe")
    lines.append("  - Never commit bench capacity beyond what is available")
    lines.append("  - Outreach emails: max 120 words, direct, grounded, honest, professional, non-condescending")
    lines.append("<|end_system|>")
    lines.append("")
    lines.append("<|context|>")

    company = ctx.get("company_name", "Unknown")
    funding = ctx.get("funding_amount")
    round_ = ctx.get("funding_round", "")
    employees = ctx.get("employee_count")
    roles_open = ctx.get("engineering_roles_open")
    layoffs = ctx.get("layoff_events", [])
    ai_score = ctx.get("ai_maturity_score")
    sources = ctx.get("signal_sources", [])

    lines.append(f"Company: {company}")
    if funding:
        lines.append(f"Funding: ${funding:,} {round_}".strip())
    if employees:
        lines.append(f"Employee count: {employees}")
    if roles_open is not None:
        lines.append(f"Engineering roles open: {roles_open}")
    if layoffs:
        for le in layoffs:
            date_str = le.get("date") or "undated"
            pct = le.get("pct_cut", "")
            lines.append(f"Layoff event: {pct}% cut, date={date_str}")
    if ai_score is not None:
        lines.append(f"AI maturity score: {ai_score}/4")
    if sources:
        lines.append(f"Signal sources: {', '.join(sources)}")

    if thread:
        lines.append("")
        lines.append("Prior conversation thread:")
        for turn in thread:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"  [{role}]: {content}")

    if reply:
        lines.append("")
        lines.append(f"Prospect reply: {reply}")
        if intent:
            lines.append(f"Reply intent: {intent}")
        if confidence is not None:
            lines.append(f"Confidence: {confidence}")

    lines.append("<|end_context|>")
    lines.append("")
    lines.append("<|task|>Decide the correct action and generate the appropriate output.<|end_task|>")

    return "\n".join(lines)


def format_output(candidate: dict) -> str:
    """Serialize a candidate_output dict to a text completion."""
    action = candidate.get("action", "")
    autonomous = candidate.get("autonomous", False)
    email_body = candidate.get("email_body")
    subject = candidate.get("subject_line")
    capacity = candidate.get("capacity_claim")
    escalation = candidate.get("escalation_reason")

    parts = [f"ACTION: {action}", f"AUTONOMOUS: {autonomous}"]
    if subject:
        parts.append(f"SUBJECT: {subject}")
    if email_body:
        parts.append(f"EMAIL:\n{email_body}")
    if capacity:
        parts.append(f"CAPACITY_CLAIM: {capacity}")
    if escalation:
        parts.append(f"ESCALATION_REASON: {escalation}")
    return "\n".join(parts)


def construct_minimal_chosen(task: dict) -> dict:
    """
    Build a minimal correct output from ground_truth fields when no matched
    correct task is available.
    """
    gt = task.get("ground_truth", {})
    candidate = task.get("candidate_output", {})

    expected_action = gt.get("expected_action") or candidate.get("action")
    expected_auto = gt.get("expected_autonomous")
    if expected_auto is None:
        expected_auto = candidate.get("autonomous", True)

    escalation_for_action = {
        "escalate_to_human": "requires_human_review",
        "log_and_close": None,
    }

    chosen_output = {
        "action": expected_action,
        "autonomous": expected_auto,
        "email_body": None,
        "subject_line": None,
        "capacity_claim": None,
        "escalation_reason": None,
    }

    if expected_action in escalation_for_action:
        chosen_output["escalation_reason"] = (
            candidate.get("escalation_reason") or escalation_for_action[expected_action]
        )

    # For send_cold_email / send_followup: reuse required_patterns as content hint
    if expected_action in ("send_cold_email", "send_followup", "send_objection_response"):
        # Reuse the candidate email but strip forbidden content
        body = candidate.get("email_body") or ""
        forbidden = gt.get("forbidden_patterns", [])
        import re
        for pat in forbidden:
            body = re.sub(pat, "[grounded claim]", body, flags=re.IGNORECASE)
        chosen_output["email_body"] = body if body else None
        chosen_output["subject_line"] = candidate.get("subject_line")
        chosen_output["autonomous"] = True

    return chosen_output


def run():
    print("Loading train partition...")
    tasks = load_jsonl(TRAIN_FILE)
    print(f"  {len(tasks)} tasks loaded")

    # Separate WRONG (has forbidden_patterns) and CORRECT tasks
    wrong_tasks = []
    correct_tasks = []
    for t in tasks:
        fb = t.get("ground_truth", {}).get("forbidden_patterns", [])
        if fb:
            wrong_tasks.append(t)
        else:
            # Also treat tasks where candidate action != expected action as WRONG
            expected = t.get("ground_truth", {}).get("expected_action")
            candidate = t.get("candidate_output", {}).get("action")
            if expected and candidate and expected != candidate:
                wrong_tasks.append(t)
            else:
                correct_tasks.append(t)

    print(f"  WRONG tasks: {len(wrong_tasks)}, CORRECT tasks: {len(correct_tasks)}")

    # Build index of CORRECT tasks by dimension (for cross-pairing)
    correct_by_dim = defaultdict(list)
    for t in correct_tasks:
        correct_by_dim[t.get("dimension", "")].append(t)

    pairs = []
    stats = {
        "total_wrong": len(wrong_tasks),
        "pairs_formed": 0,
        "matched_correct": 0,
        "constructed_chosen": 0,
        "by_dimension": defaultdict(int),
    }

    for wt in wrong_tasks:
        dim = wt.get("dimension", "")
        rejected_text = format_output(wt["candidate_output"])

        # Try to find a correct task in the same dimension
        correct_pool = correct_by_dim.get(dim, [])
        if correct_pool:
            ct = random.choice(correct_pool)
            chosen_output = ct["candidate_output"]
            chosen_text = format_output(chosen_output)
            stats["matched_correct"] += 1
        else:
            # Construct minimal correct output from ground_truth
            chosen_output = construct_minimal_chosen(wt)
            chosen_text = format_output(chosen_output)
            stats["constructed_chosen"] += 1

        prompt_text = build_prompt(wt)

        pair = {
            "task_id": wt["task_id"],
            "dimension": dim,
            "difficulty": wt.get("difficulty", ""),
            "source_mode": wt.get("source_mode", ""),
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "meta": {
                "chosen_source": ct.get("task_id") if correct_pool else "constructed",
                "rejected_task_id": wt["task_id"],
                "forbidden_patterns": wt.get("ground_truth", {}).get("forbidden_patterns", []),
            },
        }
        pairs.append(pair)
        stats["by_dimension"][dim] += 1

    stats["pairs_formed"] = len(pairs)
    stats["by_dimension"] = dict(stats["by_dimension"])

    # Write outputs
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nPreference pairs: {len(pairs)}")
    print(f"  Matched with real correct task: {stats['matched_correct']}")
    print(f"  Constructed from ground_truth: {stats['constructed_chosen']}")
    print(f"\nBy dimension:")
    for d, n in sorted(stats["by_dimension"].items()):
        print(f"  {d:40s} {n}")
    print(f"\nOutput -> {OUT_FILE}")
    print(f"Stats  -> {STATS_FILE}")


if __name__ == "__main__":
    run()
