"""Evaluate candidate outputs already embedded in a Tenacious-Bench task file.

This is useful for baseline sanity checks before a live agent or trained judge
produces separate output JSONL files.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scoring_evaluator import score_task  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bootstrap_ci(scores: list[float], iterations: int = 2000, seed: int = 42) -> dict:
    import random

    rng = random.Random(seed)
    if not scores:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    means = []
    n = len(scores)
    for _ in range(iterations):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "mean": round(sum(scores) / n, 4),
        "ci_low": round(means[int(0.025 * iterations)], 4),
        "ci_high": round(means[int(0.975 * iterations)], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="JSONL task file")
    parser.add_argument("--out", required=True, help="Path for result JSON")
    parser.add_argument("--llm-judge", action="store_true")
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    results = [score_task(t, t.get("candidate_output", {}), args.llm_judge) for t in tasks]
    scores = [r["total_score"] for r in results]
    passed = sum(1 for r in results if r["passed"])

    report = {
        "tasks": str(args.tasks),
        "n": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "score_ci": bootstrap_ci(scores),
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("n", "passed", "pass_rate", "score_ci")}, indent=2))


if __name__ == "__main__":
    main()
