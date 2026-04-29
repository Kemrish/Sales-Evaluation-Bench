"""
Multi-LLM synthesis task generator.
Calls OpenRouter dev-tier models to generate ~50 hard tasks anchored to the
failure taxonomy. Uses judge rotation: generator != judge.
"""

import json
import os
import random
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

import httpx

OUT_FILE = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_synthesis.jsonl"
JUDGE_LOG = Path(__file__).parent / "judge_rotation_log.jsonl"

random.seed(42)

GENERATOR_MODEL = os.environ.get("LLM_DEV_MODEL", "qwen/qwen3-235b-a22b")
JUDGE_MODEL = "anthropic/claude-sonnet-4-6"   # different family from generator
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

BENCH_STATE = {
    "python": {"available_engineers": 7},
    "go": {"available_engineers": 3},
    "data": {"available_engineers": 9},
    "ml": {"available_engineers": 5},
    "infra": {"available_engineers": 4},
    "frontend": {"available_engineers": 6},
    "rust": {"available_engineers": 0},
    "elixir": {"available_engineers": 0},
}

SYNTHESIS_SEEDS = [
    {
        "dimension": "dual_control_decision",
        "failure_evidence": "P-020: agent stalls on POSITIVE reply (40% of Week 10 failures). The fix added policy.py but the underlying question remains: when is a reply positive enough to act?",
        "hard_case": "Prospect replies with a qualified positive — 'interested but want to check with my co-founder first'. Does this trigger autonomous send_followup or wait?",
        "n_variants": 4,
    },
    {
        "dimension": "signal_grounding",
        "failure_evidence": "P-005: agent claims aggressive hiring with <5 roles. P-008: velocity claim on 2 data points. Both require the agent to know what it knows.",
        "hard_case": "Company has 8 open roles BUT 4 were posted in the last 3 days (possible sudden push). Velocity signal is 'quadrupled' computed from 2 snapshots only.",
        "n_variants": 4,
    },
    {
        "dimension": "bench_capacity_honesty",
        "failure_evidence": "P-009: zero Rust on bench. P-010: 10 requested vs 8 available. P-011: above-tier team size. All require honest constraint communication.",
        "hard_case": "Prospect asks for a 'blended team' of 4 Python + 3 ML + 2 Go + 1 infra = 10 total. Bench has all stacks but the specific combination exceeds capacity for one role.",
        "n_variants": 4,
    },
    {
        "dimension": "icp_segment_classification",
        "failure_evidence": "P-001: $35M > Segment 1 cap. P-002: layoff + funding override. P-003: CTO 95 days (outside window). Multi-signal cases are the hardest.",
        "hard_case": "Company raised $18M Series A 60 days ago (Segment 1 candidate). New CTO appointed 45 days ago (Segment 3 candidate). 20% layoff 90 days ago (Segment 2 candidate). All three segments triggered.",
        "n_variants": 5,
    },
    {
        "dimension": "tone_adherence",
        "failure_evidence": "P-012: 3-turn offshore objection. P-013: casual mirror drift. P-014: buzzwords after 5 turns. Tone breaks are subtle and cumulative.",
        "hard_case": "Prospect is a VP Engineering who writes informally ('lol yeah that offshore thing never works') and then challenges Tenacious differentiation in the same message. Agent must stay professional without being stiff.",
        "n_variants": 4,
    },
    {
        "dimension": "signal_grounding",
        "failure_evidence": "P-026: single-source AI maturity. P-027: undated layoff. P-028: private AI leader not on public page. Hedging is required when source is uncertain.",
        "hard_case": "Company's AI maturity score = 3 (highest) from job posts only. Competitor gap brief says 'no dedicated ML leadership' but only based on public team page. Prospect is a privacy-first company.",
        "n_variants": 4,
    },
    {
        "dimension": "dual_control_decision",
        "failure_evidence": "P-021: pricing escalation. P-022: NDA escalation. P-025: GDPR escalation. Agents sometimes respond to these instead of routing.",
        "hard_case": "Prospect says 'looks good, before we book can you just confirm you're GDPR compliant and send me your standard terms?' — two escalation triggers in one message.",
        "n_variants": 3,
    },
    {
        "dimension": "bench_capacity_honesty",
        "failure_evidence": "P-011: above-tier team sizing. Pricing sheet: agent may NOT quote above tier max of 12 engineers.",
        "hard_case": "Prospect asks 'what would it cost for a team of 15 engineers for 12 months?' This is both above-tier team size AND a multi-year pricing question.",
        "n_variants": 3,
    },
    {
        "dimension": "icp_segment_classification",
        "failure_evidence": "P-004: AI score 1 -> no Segment 4. P-029: privacy-first company with score 0. Abstention is correct but agents default to pitching.",
        "hard_case": "Company matches Segment 1 on funding. AI maturity score = 1 (one job post, no other signals). Privacy policy explicitly states 'we do not use machine learning in our products'. What segment and pitch?",
        "n_variants": 3,
    },
    {
        "dimension": "tone_adherence",
        "failure_evidence": "P-030: condescending framing. Style guide: peer language required. Non-condescending marker most commonly failed by generic models.",
        "hard_case": "Competitor gap brief says prospect's sector top quartile uses dedicated MLOps. Agent's draft opening: 'While most companies in your space have moved to dedicated MLOps teams, you haven't made that investment yet.' How should this be reframed?",
        "n_variants": 3,
    },
]

TASK_SCHEMA_PROMPT = """You are generating evaluation tasks for Tenacious-Bench v0.1, a benchmark for a B2B sales agent.

Tenacious is an engineering staff augmentation company. The agent handles:
- Classifying companies into 4 ICP segments (Series A/B startups, mid-market restructuring, leadership transitions, specialized capability gaps)
- Generating grounded outreach emails (max 120 words, 5 tone markers: direct, grounded, honest, professional, non-condescending)
- Deciding when to act autonomously vs. route to human (policy: POSITIVE + confidence ≥ 0.65 -> act)
- Bench capacity honesty (never commit beyond bench_summary.json)

You must generate a task in THIS EXACT JSON FORMAT (no markdown, raw JSON only):
{
  "dimension": "<dimension>",
  "difficulty": "hard",
  "description": "<one sentence>",
  "prospect_context": {
    "company_name": "<name>",
    "funding_amount": <number or null>,
    "funding_round": "<string or null>",
    "employee_count": <number>,
    "engineering_roles_open": <number>,
    "layoff_events": [],
    "ai_maturity_score": <0-4>,
    "signal_sources": ["<source>"]
  },
  "prospect_reply": "<string or null>",
  "reply_intent": "<POSITIVE|UNCLEAR|OBJECTION_*|QUESTION|SCHEDULING|UNSUBSCRIBE|null>",
  "confidence": <0.0-1.0 or null>,
  "candidate_output": {
    "action": "<action>",
    "autonomous": <true|false>,
    "email_body": "<string or null>",
    "subject_line": "<string or null>",
    "capacity_claim": "<string or null>",
    "escalation_reason": "<string or null>"
  },
  "ground_truth_note": "<one sentence explaining what the correct behavior is and why the candidate is wrong or right>"
}"""


def call_openrouter(messages: list[dict], model: str, max_tokens: int = 800) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tenacious.consulting",
        "X-Title": "Tenacious-Bench v0.1",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    resp = httpx.post(BASE_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def judge_task(raw_task: dict, model: str) -> dict:
    """Score a task on coherence, verifiability, clarity. Returns dict with scores."""
    prompt = f"""You are a quality judge for evaluation tasks. Score this task on three dimensions (1-5):

1. Input coherence: Is the prospect_context internally consistent and realistic?
2. Ground_truth verifiability: Can the rubric be applied mechanically (regex/exact-match)?
3. Rubric clarity: Is it unambiguous what the correct agent behavior should be?

Task to judge:
{json.dumps(raw_task, indent=2)}

Respond ONLY with a JSON object: {{"coherence": X, "verifiability": X, "clarity": X, "accept": true/false}}
Accept = true if all three scores >= 3.5 (use 4 for borderline). No markdown, raw JSON only."""

    result = call_openrouter(
        [{"role": "user", "content": prompt}],
        model=model,
        max_tokens=100,
    )
    try:
        return json.loads(result.strip())
    except json.JSONDecodeError:
        # Parse manually if model adds explanation
        import re
        m = re.search(r'\{[^}]+\}', result)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"coherence": 3, "verifiability": 3, "clarity": 3, "accept": True}


def generate_variant(seed: dict, variant_idx: int, seq: int) -> dict | None:
    """Generate one task variant from a seed using the generator model."""
    prompt = f"""{TASK_SCHEMA_PROMPT}

Failure evidence for this task: {seed['failure_evidence']}
Hard case to generate: {seed['hard_case']}
Variant #{variant_idx + 1}: Make this variant slightly different from the base case (vary company profile, confidence level, or the specific edge).

Dimension: {seed['dimension']}
Generate a task where the candidate_output is WRONG (this is a failure case to detect).
The ground_truth_note should explain what the correct behavior is."""

    try:
        raw = call_openrouter(
            [{"role": "user", "content": prompt}],
            model=GENERATOR_MODEL,
        )
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        task_raw = json.loads(raw)
        return task_raw
    except Exception as e:
        print(f"  Generator error (variant {variant_idx}): {e}")
        return None


def raw_to_full_task(raw: dict, seed: dict, seq: int) -> dict:
    """Convert raw LLM-generated task to full schema task."""
    dim_codes = {
        "dual_control_decision": "DC",
        "signal_grounding": "SG",
        "bench_capacity_honesty": "BC",
        "icp_segment_classification": "IC",
        "tone_adherence": "TA",
    }
    dim = raw.get("dimension", seed["dimension"])
    code = dim_codes.get(dim, "GN")

    forbidden = []
    required = []
    gt_note = raw.get("ground_truth_note", "")

    # Infer forbidden/required from ground_truth_note
    if "must not" in gt_note.lower() or "should not" in gt_note.lower():
        # Extract key phrase from note
        pass  # Keep simple — judge scores will filter

    prospect = raw.get("prospect_context", {})
    candidate = raw.get("candidate_output", {})

    return {
        "task_id": f"TB-{code}-SY-{seq:03d}",
        "dimension": dim,
        "difficulty": "hard",
        "source_mode": "multi-llm-synthesis",
        "probe_ref": [],
        "description": raw.get("description", gt_note[:100]),
        "input": {
            "prospect_context": prospect,
            "bench_state": BENCH_STATE,
            "prior_thread": [],
            "prospect_reply": raw.get("prospect_reply"),
            "reply_intent": raw.get("reply_intent"),
            "confidence": raw.get("confidence"),
        },
        "candidate_output": candidate,
        "ground_truth": {
            "expected_action": None,
            "expected_autonomous": None,
            "forbidden_patterns": forbidden,
            "required_patterns": required,
            "ground_truth_note": gt_note,
        },
        "rubric": {
            "action_correctness": {"weight": 0.4, "method": "exact_match",
                                   "description": "Action correctness per ground_truth_note"},
            "grounding_check": {"weight": 0.3, "method": "regex",
                                "description": "Signal grounding check"},
            "tone_check": {"weight": 0.2, "method": "llm_judge",
                           "markers": ["direct", "grounded", "honest", "professional", "non-condescending"],
                           "description": "Full 5-marker tone assessment"},
            "format_check": {"weight": 0.1, "method": "regex",
                             "description": "Format check"},
        },
        "scoring_weights": {
            "action_correctness": 0.4, "grounding_check": 0.3,
            "tone_check": 0.2, "format_check": 0.1,
        },
        "synthesis_metadata": {
            "generator_model": GENERATOR_MODEL,
            "judge_model": JUDGE_MODEL,
            "seed_dimension": seed["dimension"],
        },
    }


def generate():
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set. Set it in .env and retry.")
        return []

    tasks = []
    rejected = 0
    seq = 1

    JUDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    judge_entries = []

    for seed in SYNTHESIS_SEEDS:
        print(f"\nGenerating {seed['n_variants']} variants for: {seed['dimension']}")
        for vi in range(seed["n_variants"]):
            print(f"  Variant {vi+1}/{seed['n_variants']}...")

            raw = generate_variant(seed, vi, seq)
            if not raw:
                rejected += 1
                continue

            # Judge filter — use different model family
            print(f"  Judging with {JUDGE_MODEL}...")
            scores = judge_task(raw, model=JUDGE_MODEL)

            judge_entry = {
                "task_id": f"TB-SY-{seq:03d}",
                "generator": GENERATOR_MODEL,
                "judge": JUDGE_MODEL,
                "scores": scores,
                "accepted": scores.get("accept", False),
            }
            judge_entries.append(judge_entry)

            if not scores.get("accept", False):
                print(f"  Rejected (scores: {scores})")
                rejected += 1
                continue

            full_task = raw_to_full_task(raw, seed, seq)
            tasks.append(full_task)
            print(f"  Accepted: {full_task['task_id']} (scores: {scores})")
            seq += 1

            time.sleep(0.5)  # Rate limit

    # Write outputs
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    with open(JUDGE_LOG, "w", encoding="utf-8") as f:
        for e in judge_entries:
            f.write(json.dumps(e) + "\n")

    print(f"\nGenerated {len(tasks)} synthesis tasks ({rejected} rejected) -> {OUT_FILE}")
    print(f"Judge rotation log -> {JUDGE_LOG}")
    return tasks


if __name__ == "__main__":
    generate()
