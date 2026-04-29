"""Extra signal-grounding programmatic tasks to reach 200+ total.

Fully deterministic — no LLM calls. Uses random.seed(99). Sweeps 25 scenarios
across (eng_roles, velocity, signal_sources, ai_score, undated_layoff) parameters.
"""
import json, random
random.seed(99)
from pathlib import Path

OUT = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_programmatic_sg.jsonl"

BENCH = {"python": {"available_engineers": 7}, "go": {"available_engineers": 3},
         "data": {"available_engineers": 9}, "ml": {"available_engineers": 5},
         "infra": {"available_engineers": 4}, "frontend": {"available_engineers": 6},
         "rust": {"available_engineers": 0}}

COMPANIES = ["Astra Compute","Borealis Labs","Canary Data","Delphi AI","Epiq Systems",
             "Foundry ML","Glacier Tech","Harbinger Bio","Indigo Analytics","Juniper FS"]

SCENARIOS = [
    (3, "stable",  ["job_posts"],                 1, False, True),
    (4, "stable",  ["job_posts"],                 1, False, True),
    (6, "growing", ["job_posts","crunchbase"],     2, False, False),
    (2, "stable",  ["job_posts"],                 0, False, True),
    (8, "doubled", ["job_posts"],                 2, False, True),
    (5, "growing", ["job_posts","crunchbase"],     1, False, False),
    (1, "stable",  ["job_posts"],                 0, False, True),
    (10,"tripled", ["job_posts"],                 3, False, True),
    (3, "stable",  ["job_posts"],                 2, True,  True),
    (7, "growing", ["job_posts","crunchbase"],     3, False, False),
    (4, "stable",  ["job_posts"],                 1, True,  True),
    (6, "doubled", ["job_posts"],                 1, False, True),
    (9, "growing", ["crunchbase","leadership"],    3, False, False),
    (2, "stable",  ["job_posts"],                 0, False, True),
    (5, "stable",  ["job_posts","crunchbase"],     2, False, False),
    (3, "doubled", ["job_posts"],                 1, False, True),
    (7, "growing", ["job_posts","crunchbase"],     2, False, False),
    (4, "stable",  ["job_posts"],                 2, False, True),
    (11,"tripled", ["job_posts","crunchbase"],     3, False, False),
    (2, "stable",  ["job_posts"],                 1, True,  True),
    (6, "growing", ["job_posts","crunchbase","leadership"], 3, False, False),
    (3, "stable",  ["job_posts"],                 0, False, True),
    (8, "doubled", ["job_posts"],                 2, True,  True),
    (5, "growing", ["crunchbase"],                1, False, True),
    (4, "stable",  ["job_posts"],                 1, False, True),
]

tasks = []
for i, (eng, vel, sources, ai, undated_layoff, should_hedge) in enumerate(SCENARIOS, 1):
    company = COMPANIES[i % len(COMPANIES)]
    layoffs = [{"company": company, "date": None, "pct_cut": 15}] if undated_layoff else []
    single_src = len(set(sources)) == 1

    bad_body = (
        f"Hi -- {company} is scaling aggressively with {eng} new engineering hires. "
        "Our engineers are ready to accelerate your growth.\n\n"
        "Worth 15 minutes?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
    )
    good_body = (
        f"Hi -- {company} has {eng} open engineering roles. "
        "Is the bottleneck recruiting speed or stack depth?\n\n"
        "Worth 15 minutes?\n\nAlex\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
    )

    for is_bad in ([True, False] if should_hedge else [False]):
        body = bad_body if is_bad else good_body
        forbidden = ["scaling aggressively", "aggressive hiring", "rapid expansion"] if is_bad else []
        required = [r"(?i)(open roles?|hiring|building|growing|bottleneck)"] if not is_bad else []
        seq = i * 2 - (1 if is_bad else 0)
        tasks.append({
            "task_id": f"TB-SG-PG2-{seq:03d}",
            "dimension": "signal_grounding",
            "difficulty": "hard" if (single_src or undated_layoff) else "medium",
            "source_mode": "programmatic",
            "probe_ref": ["P-005", "P-008", "P-026", "P-027"],
            "input": {
                "prospect_context": {
                    "company_name": company,
                    "funding_amount": random.choice([8_000_000, 12_000_000, 20_000_000]),
                    "funding_round": "Series A",
                    "employee_count": random.choice([30, 55, 90]),
                    "engineering_roles_open": eng,
                    "layoff_events": layoffs,
                    "ai_maturity_score": ai,
                    "signal_sources": sources,
                },
                "bench_state": BENCH,
                "prior_thread": [],
                "prospect_reply": None,
                "reply_intent": None,
                "confidence": None,
            },
            "candidate_output": {
                "action": "send_cold_email", "autonomous": True,
                "email_body": body,
                "subject_line": "Request: 15 minutes on your team",
                "capacity_claim": None, "escalation_reason": None,
            },
            "ground_truth": {
                "expected_action": "send_cold_email",
                "expected_autonomous": True,
                "forbidden_patterns": forbidden,
                "required_patterns": required,
                "max_word_count": 120,
            },
            "rubric": {
                "action_correctness": {"weight": 0.1, "method": "exact_match", "description": "send_cold_email"},
                "grounding_check": {"weight": 0.5, "method": "regex", "description": "No over-claim"},
                "tone_check": {"weight": 0.3, "method": "llm_judge",
                               "markers": ["grounded","honest","non-condescending"], "description": "Tone"},
                "format_check": {"weight": 0.1, "method": "regex", "description": "Format"},
            },
            "scoring_weights": {"action_correctness": 0.1, "grounding_check": 0.5,
                                 "tone_check": 0.3, "format_check": 0.1},
        })

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t) + "\n")
print(f"Generated {len(tasks)} SG extra tasks -> {OUT}")
