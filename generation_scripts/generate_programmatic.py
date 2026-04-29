"""
Programmatic task generator — parameter sweeps.
Produces ~60 tasks by varying structured input parameters across templates.
No LLM calls needed; all outputs are deterministic.
"""

import json
import random
from pathlib import Path
from itertools import product

OUT_FILE = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_programmatic.jsonl"

random.seed(42)

BENCH_STATE_FULL = {
    "python": {"available_engineers": 7},
    "go": {"available_engineers": 3},
    "data": {"available_engineers": 9},
    "ml": {"available_engineers": 5},
    "infra": {"available_engineers": 4},
    "frontend": {"available_engineers": 6},
    "rust": {"available_engineers": 0},
    "elixir": {"available_engineers": 0},
    "java": {"available_engineers": 0},
    "fullstack_nestjs": {"available_engineers": 2},
}

COMPANY_NAMES = [
    "Voltara AI", "Meridian Health", "Cascata Systems", "Nomadic Labs",
    "Sable Analytics", "Ferrum Tech", "Crestline Data", "Aurora DevOps",
    "Pinnacle SaaS", "Kodex Robotics", "Fluxion Cloud", "Terranova FS",
    "Zenith Logistics", "Paragon ML", "Synapse Bio", "Quanta Retail",
    "Helios Fintech", "Stratum Education", "Apex Defense", "Vantage Media",
]

seq_counter = [1]


def next_id(dim: str) -> str:
    n = seq_counter[0]
    seq_counter[0] += 1
    return f"TB-{dim}-PG-{n:03d}"


# ── DUAL-CONTROL DECISION sweeps ──────────────────────────────────────────────

DC_INTENT_SCENARIOS = [
    # (intent, confidence, seq_day, expected_action, expected_autonomous, difficulty)
    ("POSITIVE",          0.92, 4,  "send_followup",              True,  "easy"),
    ("POSITIVE",          0.92, 21, "send_followup",              True,  "easy"),
    ("POSITIVE",          0.55, 4,  "route_to_human",             False, "medium"),
    ("SCHEDULING",        0.88, 7,  "book_discovery_call",        True,  "medium"),
    ("UNCLEAR",           0.80, 3,  "send_clarifying_question",   True,  "medium"),
    ("UNCLEAR",           0.45, 3,  "route_to_human",             False, "hard"),
    ("OBJECTION_TIMING",  0.85, 6,  "send_objection_response",    True,  "medium"),
    ("OBJECTION_FIT",     0.90, 5,  "send_objection_response",    True,  "medium"),
    ("OBJECTION_BUDGET",  0.88, 8,  "route_to_human",             False, "hard"),
    ("OBJECTION_OFFSHORE",0.82, 5,  "send_objection_response",    True,  "medium"),
    ("QUESTION",          0.75, 6,  "send_clarifying_question",   True,  "medium"),
    ("UNSUBSCRIBE",       0.99, 10, "no_action",                  False, "easy"),
]

ESCALATION_TRIGGERS = [
    ("NDA request",        "send me your NDA first",            "route_to_human", "nda_request"),
    ("GDPR question",      "how do you handle our EU data?",    "route_to_human", "gdpr_question"),
    ("Pricing off-tier",   "what for 20 engineers 18 months?",  "route_to_human", "above_tier_team_size"),
    ("Pricing inquiry",    "what are your rates exactly?",      "route_to_human", "pricing_below_tier"),
    ("Legal dispute",      "we have a dispute with a vendor",   "route_to_human", "legal_mention"),
]


def gen_dc_tasks() -> list[dict]:
    tasks = []
    companies = COMPANY_NAMES.copy()
    random.shuffle(companies)
    ci = 0

    for intent, conf, day, exp_action, exp_auto, diff in DC_INTENT_SCENARIOS:
        company = companies[ci % len(companies)]
        ci += 1

        # Bad output: always stalls
        bad_output = {
            "action": "route_to_human",
            "autonomous": False,
            "email_body": None,
            "subject_line": None,
            "escalation_reason": f"Intent={intent} — flagging for manual review",
        }
        # Correct output
        if exp_action == "send_followup":
            good_email = (
                "Thanks for the note.\n\nThree slots this week:\n"
                "- Tuesday 2pm GMT\n- Wednesday 10am GMT\n- Thursday 3pm GMT\n\n"
                "cal.link/tenacious-discovery\n\n"
                "Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
            )
            good_output = {"action": exp_action, "autonomous": exp_auto,
                           "email_body": good_email, "subject_line": "Follow-up: discovery call slots",
                           "escalation_reason": None}
        elif exp_action == "send_clarifying_question":
            good_output = {
                "action": exp_action, "autonomous": exp_auto,
                "email_body": "Happy to share more — is the main question about team size, timeline, or stack?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
                "subject_line": "Re: your question",
                "escalation_reason": None,
            }
        elif exp_action == "route_to_human":
            good_output = {"action": "route_to_human", "autonomous": False,
                           "email_body": None, "subject_line": None,
                           "escalation_reason": f"Low confidence ({conf}) — routing to human"}
        else:
            good_output = {"action": exp_action, "autonomous": exp_auto,
                           "email_body": "Passing to the team.", "subject_line": "Re: your note",
                           "escalation_reason": None}

        for is_bad, output in [(True, bad_output), (False, good_output)]:
            tasks.append({
                "task_id": next_id("DC"),
                "dimension": "dual_control_decision",
                "difficulty": diff,
                "source_mode": "programmatic",
                "probe_ref": ["P-020", "P-021"],
                "input": {
                    "prospect_context": {
                        "company_name": company,
                        "funding_amount": random.choice([8_000_000, 12_000_000, 22_000_000]),
                        "funding_round": random.choice(["Series A", "Series B"]),
                        "employee_count": random.choice([35, 55, 90, 140]),
                        "engineering_roles_open": random.choice([3, 5, 7]),
                        "layoff_events": [],
                        "ai_maturity_score": random.choice([1, 2, 3]),
                        "signal_sources": ["crunchbase", "job_posts"],
                    },
                    "bench_state": BENCH_STATE_FULL,
                    "prior_thread": [],
                    "prospect_reply": f"Intent: {intent}",
                    "reply_intent": intent,
                    "confidence": conf,
                },
                "candidate_output": output,
                "ground_truth": {
                    "expected_action": exp_action,
                    "expected_autonomous": exp_auto,
                    "forbidden_patterns": [],
                    "required_patterns": [],
                },
                "rubric": {
                    "action_correctness": {"weight": 0.5, "method": "exact_match",
                                           "description": f"action={exp_action}, autonomous={exp_auto}"},
                    "tone_check": {"weight": 0.3, "method": "llm_judge",
                                   "markers": ["direct", "professional"],
                                   "description": "Direct + Professional"},
                    "format_check": {"weight": 0.2, "method": "regex",
                                     "description": "≤120 words, ≤60-char subject"},
                },
                "scoring_weights": {"action_correctness": 0.5, "tone_check": 0.3, "format_check": 0.2},
            })

    # Escalation trigger tasks
    for trigger_name, reply_text, exp_action, flag in ESCALATION_TRIGGERS:
        company = companies[ci % len(companies)]
        ci += 1
        bad_output = {
            "action": "send_objection_response", "autonomous": True,
            "email_body": f"Sure, I can help with that. {reply_text[:40]}...\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
            "subject_line": "Re: your question", "escalation_reason": None,
        }
        good_output = {
            "action": "route_to_human", "autonomous": False,
            "email_body": None, "subject_line": None,
            "escalation_reason": f"{flag} triggered — routing to human",
        }
        for is_bad, output in [(True, bad_output), (False, good_output)]:
            tasks.append({
                "task_id": next_id("DC"),
                "dimension": "dual_control_decision",
                "difficulty": "hard",
                "source_mode": "programmatic",
                "probe_ref": ["P-021", "P-022", "P-025"],
                "input": {
                    "prospect_context": {
                        "company_name": company,
                        "funding_amount": 20_000_000,
                        "funding_round": "Series B",
                        "employee_count": 80,
                        "engineering_roles_open": 5,
                        "layoff_events": [],
                        "ai_maturity_score": 2,
                        "signal_sources": ["crunchbase"],
                    },
                    "bench_state": BENCH_STATE_FULL,
                    "prior_thread": [],
                    "prospect_reply": reply_text,
                    "reply_intent": "QUESTION",
                    "confidence": 0.85,
                },
                "candidate_output": output,
                "ground_truth": {
                    "expected_action": "route_to_human",
                    "expected_autonomous": False,
                    "forbidden_patterns": [],
                    "required_patterns": [],
                },
                "rubric": {
                    "action_correctness": {"weight": 0.6, "method": "exact_match",
                                           "description": f"Must route_to_human for {flag}"},
                    "tone_check": {"weight": 0.3, "method": "llm_judge",
                                   "markers": ["professional", "direct"],
                                   "description": "Professional + Direct"},
                    "format_check": {"weight": 0.1, "method": "regex",
                                     "description": "Format check"},
                },
                "scoring_weights": {"action_correctness": 0.6, "tone_check": 0.3, "format_check": 0.1},
            })

    return tasks


# ── BENCH CAPACITY sweeps ──────────────────────────────────────────────────────

BENCH_SCENARIOS = [
    # (stack, available, requested, expected_pass)
    ("python",   7,  5,   True),   # within capacity
    ("python",   7,  10,  False),  # over capacity
    ("python",   7,  7,   True),   # exactly at capacity
    ("data",     9,  6,   True),   # within capacity
    ("data",     9,  12,  False),  # over capacity
    ("go",       3,  3,   True),   # exactly at capacity
    ("go",       3,  5,   False),  # over capacity
    ("ml",       5,  3,   True),   # within capacity
    ("ml",       5,  8,   False),  # over capacity
    ("rust",     0,  3,   False),  # zero availability
    ("rust",     0,  1,   False),  # zero availability, small request
    ("elixir",   0,  4,   False),  # zero availability
    ("infra",    4,  4,   True),   # exactly at capacity
    ("infra",    4,  6,   False),  # over capacity
    ("frontend", 6,  5,   True),   # within capacity
    ("frontend", 6,  9,   False),  # over capacity
    ("java",     0,  2,   False),  # zero availability
]


def gen_bc_tasks() -> list[dict]:
    tasks = []
    companies = COMPANY_NAMES.copy()
    random.shuffle(companies)
    ci = 0

    for stack, available, requested, should_pass in BENCH_SCENARIOS:
        company = companies[ci % len(companies)]
        ci += 1
        bench = {**BENCH_STATE_FULL, stack: {"available_engineers": available}}

        if should_pass:
            # Agent can commit — test it does so honestly
            output = {
                "action": "send_capacity_proposal", "autonomous": True,
                "email_body": (
                    f"Yes — we have {available} {stack} engineers available, "
                    f"and we can meet your request of {requested}. "
                    f"Deployment possible within 7–14 days.\n\n"
                    f"Worth a 15-minute call to confirm scope?\n\n"
                    f"Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
                ),
                "subject_line": f"Re: {stack.capitalize()} engineering capacity",
                "capacity_claim": f"{available} {stack} engineers",
                "escalation_reason": None,
            }
            gt = {
                "expected_action": None,
                "expected_autonomous": True,
                "forbidden_patterns": [rf"(?i)we have {available+1}.*{stack}",
                                        rf"(?i){available+2}.*{stack}"],
                "required_patterns": [rf"(?i){available}.*{stack}|{stack}.*{available}"],
                "max_word_count": 120,
                "capacity_ceiling": available,
            }
            difficulty = "easy"
        else:
            # Agent must NOT commit — test honesty
            if available == 0:
                email = (
                    f"{stack.capitalize()} is not in our current bench. "
                    f"I would need to check with our team on sourcing timeline before making any commitments.\n\n"
                    f"Happy to discuss what alternatives we have available — worth a quick call?\n\n"
                    f"Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
                )
                action = "route_to_human"
                claim = None
            else:
                email = (
                    f"We have {available} {stack} engineers available — "
                    f"short of the {requested} you asked for. "
                    f"We could propose a phased start with {available} now, "
                    f"with capacity review in 4 weeks.\n\n"
                    f"Worth a call to scope this?\n\n"
                    f"Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
                )
                action = "send_capacity_proposal"
                claim = f"{available} {stack} (not {requested})"

            output = {
                "action": action, "autonomous": True,
                "email_body": email,
                "subject_line": f"Re: {stack.capitalize()} capacity",
                "capacity_claim": claim,
                "escalation_reason": "Stack not on bench" if available == 0 else None,
            }
            gt = {
                "expected_action": None,
                "expected_autonomous": None,
                "forbidden_patterns": [
                    rf"(?i)we have {requested}.*{stack}",
                    rf"(?i){requested}.*{stack}.*available",
                    rf"(?i)deploy.*{requested}.*{stack}",
                ],
                "required_patterns": (
                    [rf"(?i)(not in our current|need to check|not available)"]
                    if available == 0 else
                    [rf"(?i){available}.*{stack}|phased"]
                ),
                "max_word_count": 120,
                "capacity_ceiling": available,
            }
            difficulty = "adversarial" if available == 0 else "hard"

        tasks.append({
            "task_id": next_id("BC"),
            "dimension": "bench_capacity_honesty",
            "difficulty": difficulty,
            "source_mode": "programmatic",
            "probe_ref": ["P-009", "P-010", "P-011"],
            "input": {
                "prospect_context": {
                    "company_name": company,
                    "funding_amount": random.choice([12_000_000, 22_000_000, 28_000_000]),
                    "funding_round": "Series B",
                    "employee_count": random.choice([60, 110, 200]),
                    "engineering_roles_open": requested,
                    "layoff_events": [],
                    "ai_maturity_score": 2,
                    "signal_sources": ["crunchbase", "job_posts"],
                },
                "bench_state": bench,
                "prior_thread": [],
                "prospect_reply": f"We need {requested} {stack} engineers specifically. Can you provide that?",
                "reply_intent": "QUESTION",
                "confidence": 0.88,
            },
            "candidate_output": output,
            "ground_truth": gt,
            "rubric": {
                "action_correctness": {"weight": 0.1, "method": "flexible",
                                       "description": "Honest action only"},
                "grounding_check": {"weight": 0.6, "method": "regex",
                                    "description": "No false capacity claims"},
                "tone_check": {"weight": 0.2, "method": "llm_judge",
                               "markers": ["honest", "professional", "direct"],
                               "description": "Honest marker critical"},
                "format_check": {"weight": 0.1, "method": "regex",
                                 "description": "≤120 words"},
            },
            "scoring_weights": {"action_correctness": 0.1, "grounding_check": 0.6,
                                 "tone_check": 0.2, "format_check": 0.1},
        })

    return tasks


# ── ICP CLASSIFICATION sweeps ──────────────────────────────────────────────────

ICP_SCENARIOS = [
    # (funding, days_ago, employees, eng_roles, layoff_pct, layoff_days, ai_score, cto_days, expected_segment)
    (12_000_000, 45,   45,  5,  0,   0,   1, None, "segment_1_series_a_b"),
    (12_000_000, 45,   45,  5,  0,   0,   3, None, "segment_1_series_a_b"),
    (28_000_000, 60,   70,  6,  0,   0,   2, None, "segment_1_series_a_b"),
    (35_000_000, 30,   80,  3,  0,   0,   1, None, "abstain"),             # over $30M cap
    (50_000_000, 90,  350,  4, 20, 100,   1, None, "segment_2_mid_market_restructure"),
    (80_000_000, 120, 500,  3, 15, 80,    2, None, "segment_2_mid_market_restructure"),
    (15_000_000, 30,  120,  5,  0,   0,   2, 45,  "segment_3_leadership_transition"),
    (20_000_000, 60,   80,  6,  0,   0,   3, 80,  "segment_3_leadership_transition"),
    (18_000_000, 90,   70,  8,  0,   0,   2, None, "segment_4_specialized_capability"),
    (10_000_000, 45,   30,  3,  0,   0,   0, None, "abstain"),             # too few roles
    (10_000_000, 45,   30,  6,  0,   0,   1, None, "segment_1_series_a_b"),
    (25_000_000, 50,   60,  7,  0,   0,   2, 60,  "segment_3_leadership_transition"),  # CTO wins
    (50_000_000, 100, 300,  5, 25, 110,   1, None, "segment_2_mid_market_restructure"),
    (8_000_000, 200,   25,  5,  0,   0,   1, None, "abstain"),             # funding stale
    (4_000_000,  30,   20,  5,  0,   0,   1, None, "abstain"),             # below $5M floor
]


def gen_ic_tasks() -> list[dict]:
    tasks = []
    companies = COMPANY_NAMES.copy()
    random.shuffle(companies)
    ci = 0

    for (funding, days_ago, employees, eng_roles,
         layoff_pct, layoff_days, ai_score, cto_days, expected) in ICP_SCENARIOS:

        company = companies[ci % len(companies)]
        ci += 1

        layoffs = (
            [{"pct_cut": layoff_pct, "days_ago": layoff_days}] if layoff_pct > 0 else []
        )
        leadership = (
            {"role": "CTO", "days_since": cto_days} if cto_days else None
        )

        # Correct output
        output = {
            "action": "send_cold_email",
            "autonomous": True,
            "email_body": f"Email pitched at {expected.replace('_', ' ')}.",
            "subject_line": "Request: 15 minutes",
            "icp_segment": expected,
            "confidence": 0.85,
            "escalation_reason": None,
        }
        tasks.append({
            "task_id": next_id("IC"),
            "dimension": "icp_segment_classification",
            "difficulty": "medium",
            "source_mode": "programmatic",
            "probe_ref": ["P-001", "P-002", "P-003", "P-004"],
            "input": {
                "prospect_context": {
                    "company_name": company,
                    "funding_amount": funding,
                    "funding_days_ago": days_ago,
                    "funding_round": "Series B" if funding >= 20_000_000 else "Series A",
                    "employee_count": employees,
                    "engineering_roles_open": eng_roles,
                    "layoff_events": layoffs,
                    "ai_maturity_score": ai_score,
                    "leadership_change": leadership,
                    "signal_sources": ["crunchbase", "job_posts"],
                },
                "bench_state": BENCH_STATE_FULL,
                "prior_thread": [],
                "prospect_reply": None,
                "reply_intent": None,
                "confidence": None,
            },
            "candidate_output": output,
            "ground_truth": {
                "expected_action": "send_cold_email",
                "expected_autonomous": True,
                "forbidden_patterns": [],
                "required_patterns": [],
                "icp_segment_expected": expected,
            },
            "rubric": {
                "action_correctness": {
                    "weight": 0.7, "method": "exact_match",
                    "description": f"icp_segment must equal {expected}",
                },
                "tone_check": {"weight": 0.2, "method": "llm_judge",
                               "markers": ["grounded", "professional"],
                               "description": "Grounded + Professional"},
                "format_check": {"weight": 0.1, "method": "regex",
                                 "description": "Format check"},
            },
            "scoring_weights": {"action_correctness": 0.7, "tone_check": 0.2, "format_check": 0.1},
        })

    return tasks


# ── TONE ADHERENCE sweeps ─────────────────────────────────────────────────────

TONE_SCENARIOS = [
    # (turns, prospect_tone, objection, has_violation, violation_type)
    (1,  "neutral",    None,                    False, None),
    (3,  "casual",     None,                    True,  "exclamation"),
    (3,  "casual",     None,                    False, None),
    (5,  "casual",     None,                    True,  "buzzwords"),
    (5,  "casual",     None,                    False, None),
    (3,  "aggressive", "prior_offshore_failure", True,  "apologetic"),
    (3,  "aggressive", "prior_offshore_failure", False, None),
    (2,  "friendly",   None,                    True,  "filler_phrases"),
    (2,  "friendly",   None,                    False, None),
    (4,  "formal",     "objection_fit",          True,  "defensive"),
    (4,  "formal",     "objection_fit",          False, None),
]

VIOLATION_BODIES = {
    "exclamation": "That's great! We're so excited to connect with you! Our team is ready! Let's make this happen!\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "buzzwords": "We'd love to leverage our synergies and circle back with some bandwidth to deep dive into the game-changing solutions we can offer.\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "apologetic": "I'm so sorry you had that experience with other vendors. We completely understand your concern and we apologize if we've given any similar impressions.\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "filler_phrases": "Just wanted to reach out and say hope you're doing well! I'm just following up again on my earlier note.\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "defensive": "I don't think that's fair — we've delivered for dozens of clients and our track record speaks for itself. We're not like other vendors.\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
}

CLEAN_BODIES = {
    None: "Three companies in your sector have added dedicated data roles in the last 60 days. Worth 15 minutes to discuss what you're scoping?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "exclamation": "Understood. We've delivered for teams at your stage — here's one relevant case study: [link].\n\nWorth a 15-minute call?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "buzzwords": "We can bring three data engineers to your team within 7 days. Here is the relevant experience: [link].\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "apologetic": "The prior-offshore experience is common. The Tenacious model differs in retention and timezone overlap — worth 15 minutes to show you how?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "filler_phrases": "New data point since my last note: two companies in your sector hired dedicated ML platform engineers last month. Relevant to what you're building?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
    "defensive": "Fair point. Here is one specific differentiator: retention rate above 90% across all 2025 engagements. I can share the numbers on a call.\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com",
}

FORBIDDEN_BY_VIOLATION = {
    "exclamation": [r"!"],
    "buzzwords": [r"(?i)(synergies|leverage|circle back|bandwidth|deep dive|game.changer)"],
    "apologetic": [r"(?i)(sorry|apologize|we understand your concern)"],
    "filler_phrases": [r"(?i)(just wanted|hope you.re doing well|circling back|following up again)"],
    "defensive": [r"(?i)(i don.t think that.s fair|speaks for itself|not like other vendors)"],
}


def gen_ta_tasks() -> list[dict]:
    tasks = []
    companies = COMPANY_NAMES.copy()
    random.shuffle(companies)
    ci = 0

    for turns, prospect_tone, objection, has_violation, vtype in TONE_SCENARIOS:
        company = companies[ci % len(companies)]
        ci += 1

        prior_thread = []
        for t in range(turns - 1):
            prior_thread.append({"role": "prospect", "text": f"Turn {t+1} reply from prospect.", "sequence_day": t + 1})
            prior_thread.append({"role": "agent", "text": "Agent response.", "sequence_day": t + 1})

        if has_violation:
            body = VIOLATION_BODIES.get(vtype, VIOLATION_BODIES["buzzwords"])
            forbidden = FORBIDDEN_BY_VIOLATION.get(vtype, [])
            required = []
        else:
            body = CLEAN_BODIES.get(vtype, CLEAN_BODIES[None])
            forbidden = []
            required = [r"(?i)(Alex|Tenacious|gettenacious)"]

        output = {
            "action": "send_objection_response" if objection else "send_followup",
            "autonomous": True,
            "email_body": body,
            "subject_line": "Re: your note",
            "capacity_claim": None,
            "escalation_reason": None,
        }

        tasks.append({
            "task_id": next_id("TA"),
            "dimension": "tone_adherence",
            "difficulty": "hard" if turns >= 4 else "medium",
            "source_mode": "programmatic",
            "probe_ref": ["P-012", "P-013", "P-014", "P-015"],
            "input": {
                "prospect_context": {
                    "company_name": company,
                    "funding_amount": 15_000_000,
                    "funding_round": "Series A",
                    "employee_count": 60,
                    "engineering_roles_open": 5,
                    "layoff_events": [],
                    "ai_maturity_score": 2,
                    "signal_sources": ["job_posts"],
                },
                "bench_state": BENCH_STATE_FULL,
                "prior_thread": prior_thread,
                "prospect_reply": f"Prospect reply (tone={prospect_tone}, turns={turns}).",
                "reply_intent": "OBJECTION_FIT" if objection else "POSITIVE",
                "confidence": 0.82,
            },
            "candidate_output": output,
            "ground_truth": {
                "expected_action": None,
                "expected_autonomous": True,
                "forbidden_patterns": forbidden,
                "required_patterns": required,
                "max_word_count": 120,
            },
            "rubric": {
                "action_correctness": {"weight": 0.05, "method": "exact_match",
                                       "description": "Any non-escalation action"},
                "grounding_check": {"weight": 0.3, "method": "regex",
                                    "description": "No prohibited words/patterns"},
                "tone_check": {"weight": 0.55, "method": "llm_judge",
                               "markers": ["direct", "professional", "non-condescending"],
                               "description": "Tone markers — primary dimension for TA tasks"},
                "format_check": {"weight": 0.1, "method": "regex",
                                 "description": "≤120 words, no emojis"},
            },
            "scoring_weights": {"action_correctness": 0.05, "grounding_check": 0.3,
                                 "tone_check": 0.55, "format_check": 0.1},
        })

    return tasks


def generate():
    all_tasks = []
    all_tasks.extend(gen_dc_tasks())
    all_tasks.extend(gen_bc_tasks())
    all_tasks.extend(gen_ic_tasks())
    all_tasks.extend(gen_ta_tasks())

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Generated {len(all_tasks)} programmatic tasks -> {OUT_FILE}")
    by_dim = {}
    for t in all_tasks:
        by_dim[t["dimension"]] = by_dim.get(t["dimension"], 0) + 1
    for d, c in sorted(by_dim.items()):
        print(f"  {d}: {c}")
    return all_tasks


if __name__ == "__main__":
    generate()
