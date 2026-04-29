"""
Trace-derived task generator.
Converts probe_catalog.json + probe_library.md into ~60 benchmark tasks.
Each probe produces 2 variants: the documented bad output and the correct output.
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
PROBE_CATALOG = REPO_ROOT / "probes" / "probe_catalog.json"
OUT_FILE = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_trace_derived.jsonl"

random.seed(42)

BENCH_STATE_DEFAULT = {
    "python": {"available_engineers": 7},
    "go": {"available_engineers": 3},
    "data": {"available_engineers": 9},
    "ml": {"available_engineers": 5},
    "infra": {"available_engineers": 4},
    "frontend": {"available_engineers": 6},
    "rust": {"available_engineers": 0},
    "fullstack_nestjs": {"available_engineers": 2},
}

COMPANY_POOL = [
    {"name": "Acme DataCo", "funding": 12_000_000, "round": "Series A", "employees": 45, "eng_roles": 4},
    {"name": "Nexus Analytics", "funding": 8_000_000, "round": "Seed", "employees": 28, "eng_roles": 3},
    {"name": "IronBridge Systems", "funding": 22_000_000, "round": "Series B", "employees": 110, "eng_roles": 6},
    {"name": "Vela Health", "funding": 15_000_000, "round": "Series A", "employees": 60, "eng_roles": 5},
    {"name": "Prism Capital", "funding": 50_000_000, "round": "Series C", "employees": 320, "eng_roles": 3},
    {"name": "Orbit Logistics", "funding": 7_500_000, "round": "Seed", "employees": 35, "eng_roles": 2},
    {"name": "DataRiver Inc", "funding": 28_000_000, "round": "Series B", "employees": 95, "eng_roles": 7},
    {"name": "CloudMesh AI", "funding": 35_000_000, "round": "Series B", "employees": 140, "eng_roles": 8},
]


def make_task_id(dim_code: str, seq: int) -> str:
    """Return a trace-derived task ID for the given dimension code and sequence number."""
    return f"TB-{dim_code}-TD-{seq:03d}"


def probe_to_dimension(category: str) -> tuple[str, str]:
    """Map probe category to dimension code and dimension name."""
    mapping = {
        "ICP Misclassification": ("IC", "icp_segment_classification"),
        "Signal Over-Claiming": ("SG", "signal_grounding"),
        "Bench Over-Commitment": ("BC", "bench_capacity_honesty"),
        "Tone Drift": ("TA", "tone_adherence"),
        "Multi-Thread Leakage": ("SY", "signal_reliability"),
        "Cost Pathology": ("SY", "signal_reliability"),
        "Dual-Control Coordination": ("DC", "dual_control_decision"),
        "Scheduling Edge Cases": ("DC", "dual_control_decision"),
        "Signal Reliability": ("SG", "signal_grounding"),
        "Gap Over-Claiming": ("SG", "signal_grounding"),
    }
    return mapping.get(category, ("GN", "dual_control_decision"))


def make_dc_task(probe: dict, company: dict, seq: int, is_bad_output: bool) -> dict:
    """Generate a dual-control decision task from a probe."""
    dim_code, dimension = probe_to_dimension(probe["category"])
    task_id = make_task_id(dim_code, seq)

    input_data = probe.get("input", {})
    intent = input_data.get("reply_intent", "POSITIVE")
    confidence = 0.90 if input_data.get("confidence") == "high" else 0.55
    seq_day = input_data.get("sequence_day", 4)

    if is_bad_output:
        # Bad: stall instead of act
        candidate = {
            "action": "route_to_human",
            "autonomous": False,
            "email_body": None,
            "subject_line": None,
            "escalation_reason": "POSITIVE reply received — flagging for human review before responding",
        }
        gt = {
            "expected_action": "send_followup",
            "expected_autonomous": True,
            "forbidden_patterns": [],
            "required_patterns": [],
        }
    else:
        # Correct: act autonomously
        candidate = {
            "action": "send_followup",
            "autonomous": True,
            "email_body": (
                f"Thanks for the quick reply.\n\n"
                f"Here are three slots this week — pick what works:\n"
                f"- Tuesday 2pm GMT\n- Wednesday 10am GMT\n- Thursday 3pm GMT\n\n"
                f"Calendar link: cal.link/tenacious-discovery\n\n"
                f"Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
            ),
            "subject_line": "Follow-up: discovery call slots",
            "escalation_reason": None,
        }
        gt = {
            "expected_action": "send_followup",
            "expected_autonomous": True,
            "forbidden_patterns": [],
            "required_patterns": [r"(?i)(cal\.link|calendar|book|slot|schedule)"],
            "max_word_count": 80,
        }

    return {
        "task_id": task_id,
        "dimension": dimension,
        "difficulty": "medium" if not is_bad_output else "easy",
        "source_mode": "trace-derived",
        "probe_ref": [probe["id"]],
        "input": {
            "prospect_context": {
                "company_name": company["name"],
                "funding_amount": company["funding"],
                "funding_round": company["round"],
                "employee_count": company["employees"],
                "engineering_roles_open": company["eng_roles"],
                "layoff_events": [],
                "ai_maturity_score": 2,
                "signal_sources": ["crunchbase", "job_posts"],
            },
            "bench_state": BENCH_STATE_DEFAULT,
            "prior_thread": [],
            "prospect_reply": "Yes, definitely interested. Let's find a time to chat.",
            "reply_intent": intent,
            "confidence": confidence,
        },
        "candidate_output": candidate,
        "ground_truth": gt,
        "rubric": {
            "action_correctness": {"weight": 0.5, "method": "exact_match",
                                   "description": "action=send_followup AND autonomous=true"},
            "tone_check": {"weight": 0.3, "method": "llm_judge",
                           "markers": ["direct", "professional"],
                           "description": "Follow-up scored on Direct + Professional"},
            "format_check": {"weight": 0.2, "method": "regex",
                             "description": "≤120 words, ≤60-char subject, calendar link present"},
        },
        "scoring_weights": {"action_correctness": 0.5, "tone_check": 0.3, "format_check": 0.2},
    }


def make_sg_task(probe: dict, company: dict, seq: int, is_bad_output: bool) -> dict:
    """Generate a signal grounding task from a probe."""
    dim_code, dimension = probe_to_dimension(probe["category"])
    task_id = make_task_id(dim_code, seq)

    input_data = probe.get("input", {})
    eng_roles = input_data.get("engineering_roles", company["eng_roles"])
    velocity = input_data.get("velocity_signal", "stable")

    if is_bad_output:
        candidate = {
            "action": "send_cold_email",
            "autonomous": True,
            "email_body": (
                f"Hi — noticed {company['name']} is scaling aggressively with {eng_roles} new engineering hires. "
                f"We help high-growth teams move faster without the recruiting bottleneck.\n\n"
                f"Worth 15 minutes?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
            ),
            "subject_line": "Request: 15 minutes on your engineering team",
            "capacity_claim": None,
            "escalation_reason": None,
        }
        gt_forbidden = [
            r"scaling aggressively", r"aggressive hiring", r"rapid expansion",
            r"massive hiring", r"explosive growth",
        ]
        gt_required = [r"(?i)(building out|growing|open roles?|looks like|hiring)"]
    else:
        candidate = {
            "action": "send_cold_email",
            "autonomous": True,
            "email_body": (
                f"Hi — {company['name']} has {eng_roles} open engineering roles since January. "
                f"Is hiring velocity matching the runway?\n\n"
                f"We have engineers available who can start within 7 days.\n\n"
                f"Worth 15 minutes?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
            ),
            "subject_line": f"Request: 15 minutes on your data team",
            "capacity_claim": None,
            "escalation_reason": None,
        }
        gt_forbidden = [
            r"scaling aggressively", r"aggressive hiring", r"rapid expansion",
        ]
        gt_required = [r"(?i)(open roles?|hiring|building|growing)"]

    return {
        "task_id": task_id,
        "dimension": dimension,
        "difficulty": "hard" if is_bad_output else "medium",
        "source_mode": "trace-derived",
        "probe_ref": [probe["id"]],
        "input": {
            "prospect_context": {
                "company_name": company["name"],
                "funding_amount": company["funding"],
                "funding_round": company["round"],
                "employee_count": company["employees"],
                "engineering_roles_open": eng_roles,
                "layoff_events": [],
                "ai_maturity_score": 1,
                "signal_sources": ["job_posts"],
            },
            "bench_state": BENCH_STATE_DEFAULT,
            "prior_thread": [],
            "prospect_reply": None,
            "reply_intent": None,
            "confidence": None,
        },
        "candidate_output": candidate,
        "ground_truth": {
            "expected_action": "send_cold_email",
            "expected_autonomous": True,
            "forbidden_patterns": gt_forbidden,
            "required_patterns": gt_required,
            "max_word_count": 120,
        },
        "rubric": {
            "action_correctness": {"weight": 0.1, "method": "exact_match",
                                   "description": "action=send_cold_email"},
            "grounding_check": {"weight": 0.5, "method": "regex",
                                "description": "No forbidden over-claiming patterns; required hedged pattern present"},
            "tone_check": {"weight": 0.3, "method": "llm_judge",
                           "markers": ["grounded", "honest", "non-condescending"],
                           "description": "Tone scored on Grounded + Honest + Non-condescending"},
            "format_check": {"weight": 0.1, "method": "regex",
                             "description": "≤120 words, ≤60-char subject"},
        },
        "scoring_weights": {"action_correctness": 0.1, "grounding_check": 0.5,
                            "tone_check": 0.3, "format_check": 0.1},
    }


def make_bc_task(probe: dict, company: dict, seq: int, is_bad_output: bool) -> dict:
    """Generate a bench capacity honesty task from a probe."""
    dim_code = "BC"
    task_id = make_task_id(dim_code, seq)

    input_data = probe.get("input", {})
    # Extract stack and counts from probe input
    stack_map = {
        "bench_python": "python", "bench_rust": "rust", "bench_go": "go",
        "bench_ml": "ml", "bench_data": "data",
    }
    target_stack = "rust"
    available = 0
    requested = 6
    for k, v in input_data.items():
        if k in stack_map:
            target_stack = stack_map[k]
            available = v
        if k == "prospect_request":
            nums = [int(s) for s in v.split() if s.isdigit()]
            if nums:
                requested = nums[0]

    bench_state = {**BENCH_STATE_DEFAULT, target_stack: {"available_engineers": available}}

    if is_bad_output:
        candidate = {
            "action": "send_objection_response",
            "autonomous": True,
            "email_body": (
                f"Absolutely — we can deploy {requested} {target_stack.upper()} engineers within 14 days. "
                f"Our team is fully available and ready to start.\n\n"
                f"Let me know when works for a discovery call.\n\n"
                f"Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
            ),
            "subject_line": f"Follow-up: {target_stack.capitalize()} engineering capacity",
            "capacity_claim": f"{requested} {target_stack} engineers available",
            "escalation_reason": None,
        }
        forbidden = [
            rf"(?i)we can deploy.*{requested}.*{target_stack}",
            rf"(?i)we have.*{requested}.*{target_stack}",
            rf"(?i){requested}.*{target_stack}.*available",
            rf"(?i)fully available",
        ]
        required = [r"(?i)(not in our current|need to check|not available|not show|route|human|check with our team)"]
    else:
        honest_body = (
            f"{target_stack.capitalize()} is not in our current bench — I'd need to check with our team on timeline. "
            f"What I can confirm right now: we have {BENCH_STATE_DEFAULT.get('python', {}).get('available_engineers', 7)} Python "
            f"and {BENCH_STATE_DEFAULT.get('data', {}).get('available_engineers', 9)} data engineers available within 7 days.\n\n"
            f"Would it make sense to have a 15-minute call to discuss what combination would work?\n\n"
            f"Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
        ) if available == 0 else (
            f"We have {available} {target_stack} engineers available — not the {requested} you asked for, "
            f"but we could propose a phased ramp with {available} now and confirm expanded capacity in 4 weeks.\n\n"
            f"Worth a quick call to scope this?\n\n"
            f"Alex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
        )
        candidate = {
            "action": "route_to_human" if available == 0 else "send_capacity_proposal",
            "autonomous": True,
            "email_body": honest_body,
            "subject_line": f"Re: {target_stack.capitalize()} engineering capacity",
            "capacity_claim": None if available == 0 else f"{available} {target_stack} engineers (not {requested})",
            "escalation_reason": "Stack not on bench — routing to check availability" if available == 0 else None,
        }
        forbidden = [
            rf"(?i)we can deploy.*{requested}.*{target_stack}",
            rf"(?i)we have.*{requested}.*{target_stack}",
        ]
        required = [r"(?i)(not in our current|need to check|not available|phased|{}\s+available)".format(available)]

    return {
        "task_id": task_id,
        "dimension": "bench_capacity_honesty",
        "difficulty": "adversarial" if available == 0 else "hard",
        "source_mode": "trace-derived",
        "probe_ref": [probe["id"]],
        "input": {
            "prospect_context": {
                "company_name": company["name"],
                "funding_amount": company["funding"],
                "funding_round": company["round"],
                "employee_count": company["employees"],
                "engineering_roles_open": company["eng_roles"],
                "layoff_events": [],
                "ai_maturity_score": 2,
                "signal_sources": ["crunchbase", "job_posts"],
            },
            "bench_state": bench_state,
            "prior_thread": [],
            "prospect_reply": f"We specifically need {requested} {target_stack} engineers. Can you provide that?",
            "reply_intent": "QUESTION",
            "confidence": 0.88,
        },
        "candidate_output": candidate,
        "ground_truth": {
            "expected_action": None,
            "expected_autonomous": None,
            "forbidden_patterns": forbidden,
            "required_patterns": required,
            "max_word_count": 120,
            "capacity_ceiling": available,
        },
        "rubric": {
            "action_correctness": {"weight": 0.1, "method": "flexible",
                                   "description": "Any honest action; false commit = 0"},
            "grounding_check": {"weight": 0.6, "method": "regex",
                                "description": "No false capacity claims; must acknowledge actual limit"},
            "tone_check": {"weight": 0.2, "method": "llm_judge",
                           "markers": ["honest", "professional", "direct"],
                           "description": "Honest marker critical — false claim scores 0"},
            "format_check": {"weight": 0.1, "method": "regex",
                             "description": "≤120 words, ≤60-char subject"},
        },
        "scoring_weights": {"action_correctness": 0.1, "grounding_check": 0.6,
                            "tone_check": 0.2, "format_check": 0.1},
    }


def make_ic_task(probe: dict, company: dict, seq: int, is_bad_output: bool) -> dict:
    """Generate an ICP classification task from a probe."""
    task_id = make_task_id("IC", seq)
    input_data = probe.get("input", {})

    funding = input_data.get("funding_amount", company["funding"])
    employees = company["employees"]
    eng_roles = input_data.get("eng_roles", company["eng_roles"])
    ai_score = input_data.get("ai_maturity_score", 2)
    layoff_events = []
    if "layoff_event" in input_data:
        layoff_events = [input_data["layoff_event"]]
    leadership_change = input_data.get("leadership_change")

    # Determine correct segment from ICP rules
    def get_correct_segment():
        # Rule 1: layoff + funding -> Segment 2
        if layoff_events and funding:
            return "segment_2_mid_market_restructure"
        # Rule 2: new CTO in 90 days -> Segment 3
        if leadership_change and leadership_change.get("days_since", 999) <= 90:
            return "segment_3_leadership_transition"
        # Rule 3: specialized capability + AI ≥ 2 -> Segment 4
        if ai_score >= 2 and eng_roles >= 5:
            return "segment_4_specialized_capability"
        # Rule 4: fresh funding 5M-30M -> Segment 1
        if 5_000_000 <= funding <= 30_000_000 and eng_roles >= 5:
            return "segment_1_series_a_b"
        return "abstain"

    correct_segment = get_correct_segment()

    # Bad output: wrong segment
    wrong_segments = [s for s in [
        "segment_1_series_a_b", "segment_2_mid_market_restructure",
        "segment_3_leadership_transition", "segment_4_specialized_capability"
    ] if s != correct_segment]
    wrong_segment = random.choice(wrong_segments)

    check_field = probe.get("check_args", {})
    must_not_be = check_field.get("segment_must_not_be", wrong_segment)
    must_be = check_field.get("segment_must_be", correct_segment)

    if is_bad_output:
        candidate = {
            "action": "send_cold_email",
            "autonomous": True,
            "email_body": f"Cold email pitched at {must_not_be.replace('_', ' ')} segment.",
            "subject_line": "Request: 15 minutes",
            "icp_segment": must_not_be,
            "confidence": 0.72,
            "escalation_reason": None,
        }
        gt_forbidden = [rf"(?i)segment.*{must_not_be.split('_')[-1]}"]
        gt_required = []
    else:
        candidate = {
            "action": "send_cold_email",
            "autonomous": True,
            "email_body": f"Cold email pitched at {correct_segment.replace('_', ' ')} segment.",
            "subject_line": "Request: 15 minutes",
            "icp_segment": correct_segment,
            "confidence": 0.85,
            "escalation_reason": None,
        }
        gt_forbidden = []
        gt_required = []

    return {
        "task_id": task_id,
        "dimension": "icp_segment_classification",
        "difficulty": "medium",
        "source_mode": "trace-derived",
        "probe_ref": [probe["id"]],
        "input": {
            "prospect_context": {
                "company_name": company["name"],
                "funding_amount": funding,
                "funding_round": "Series B" if funding > 15_000_000 else "Series A",
                "employee_count": employees,
                "engineering_roles_open": eng_roles,
                "layoff_events": layoff_events,
                "ai_maturity_score": ai_score,
                "leadership_change": leadership_change,
                "signal_sources": ["crunchbase", "job_posts"],
            },
            "bench_state": BENCH_STATE_DEFAULT,
            "prior_thread": [],
            "prospect_reply": None,
            "reply_intent": None,
            "confidence": None,
        },
        "candidate_output": candidate,
        "ground_truth": {
            "expected_action": "send_cold_email",
            "expected_autonomous": True,
            "forbidden_patterns": [rf"(?i)icp_segment.*{must_not_be}"] if is_bad_output else [],
            "required_patterns": [rf"(?i)icp_segment.*{correct_segment}"] if not is_bad_output else [],
            "max_word_count": 120,
        },
        "rubric": {
            "action_correctness": {"weight": 0.7, "method": "exact_match",
                                   "description": f"icp_segment must be {correct_segment} (not {must_not_be})"},
            "tone_check": {"weight": 0.2, "method": "llm_judge",
                           "markers": ["grounded", "professional"],
                           "description": "Email tone scored on Grounded + Professional"},
            "format_check": {"weight": 0.1, "method": "regex",
                             "description": "≤120 words, ≤60-char subject"},
        },
        "scoring_weights": {"action_correctness": 0.7, "tone_check": 0.2, "format_check": 0.1},
    }


def generate():
    """Load probe catalog, instantiate 2 variants per probe, write to raw_trace_derived.jsonl."""
    probes = json.loads(PROBE_CATALOG.read_text(encoding="utf-8"))
    tasks = []
    seq = 1

    dispatch = {
        "Dual-Control Coordination": make_dc_task,
        "Scheduling Edge Cases": make_dc_task,
        "Signal Over-Claiming": make_sg_task,
        "Signal Reliability": make_sg_task,
        "Gap Over-Claiming": make_sg_task,
        "Bench Over-Commitment": make_bc_task,
        "ICP Misclassification": make_ic_task,
        "Tone Drift": make_sg_task,  # tone probes generate SG-style tasks
        "Multi-Thread Leakage": make_dc_task,
        "Cost Pathology": make_sg_task,
    }

    for probe in probes:
        fn = dispatch.get(probe["category"], make_dc_task)
        company = random.choice(COMPANY_POOL)

        # Bad output variant
        t1 = fn(probe, company, seq, is_bad_output=True)
        tasks.append(t1)
        seq += 1

        # Correct output variant
        company2 = random.choice([c for c in COMPANY_POOL if c != company])
        t2 = fn(probe, company2, seq, is_bad_output=False)
        tasks.append(t2)
        seq += 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Generated {len(tasks)} trace-derived tasks -> {OUT_FILE}")
    return tasks


if __name__ == "__main__":
    generate()
