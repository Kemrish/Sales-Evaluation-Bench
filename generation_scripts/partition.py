"""
Partition Tenacious-Bench tasks into train / dev / held-out.
Stratified by dimension and difficulty to ensure each split covers all categories.
Splits: 50% train, 30% dev, 20% held-out.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

RAW_FILES = [
    Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_trace_derived.jsonl",
    Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_programmatic.jsonl",
    Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_synthesis.jsonl",
    Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_hand_authored.jsonl",
    Path(__file__).parent.parent / "tenacious_bench_v0.1" / "raw_programmatic_sg.jsonl",
]

TRAIN_DIR   = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "train"
DEV_DIR     = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "dev"
HELD_DIR    = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "held_out"


def load_all() -> list[dict]:
    tasks = []
    for f in RAW_FILES:
        if not f.exists():
            print(f"  WARNING: {f.name} not found, skipping")
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    return tasks


def deduplicate(tasks: list[dict]) -> list[dict]:
    """Remove duplicate task IDs."""
    seen = set()
    out = []
    for t in tasks:
        tid = t.get("task_id", "")
        if tid not in seen:
            seen.add(tid)
            out.append(t)
    return out


def stratified_split(tasks: list[dict], train_pct=0.5, dev_pct=0.3) -> tuple:
    """
    Split tasks into train/dev/held-out stratified by (dimension, difficulty).
    Returns (train, dev, held_out).
    """
    groups = defaultdict(list)
    for t in tasks:
        key = (t.get("dimension", "unknown"), t.get("difficulty", "medium"))
        groups[key].append(t)

    train, dev, held = [], [], []

    for key, group in groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_pct))
        n_dev   = max(1, round(n * dev_pct))
        n_held  = max(0, n - n_train - n_dev)

        if n_held == 0 and n >= 3:
            # Force at least 1 in held-out for groups of 3+
            n_train = n - 2
            n_dev   = 1
            n_held  = 1
        elif n < 3:
            # Small groups go entirely to train
            n_train = n
            n_dev   = 0
            n_held  = 0

        train.extend(group[:n_train])
        dev.extend(group[n_train:n_train + n_dev])
        held.extend(group[n_train + n_dev:n_train + n_dev + n_held])

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(held)

    return train, dev, held


def contamination_aware_split(tasks: list[dict]) -> tuple:
    """Split tasks while keeping the sealed slice away from template families.

    Programmatic and trace-derived tasks intentionally create many nearby
    variants. Those are useful for training and public dev, but they make a
    poor sealed slice because lexical overlap is expected. For v0.1, the
    sealed held-out is drawn from hand-authored adversarial and multi-LLM
    synthesis tasks, which are the least templated and most diagnostic.
    The remaining tasks are stratified into train/dev at roughly 62.5/37.5,
    yielding an overall 50/30/20 split.
    """
    held_modes = {"hand-authored", "multi-llm-synthesis"}
    held = [t for t in tasks if t.get("source_mode") in held_modes]
    remainder = [t for t in tasks if t.get("source_mode") not in held_modes]

    groups = defaultdict(list)
    for t in remainder:
        key = (t.get("dimension", "unknown"), t.get("difficulty", "medium"))
        groups[key].append(t)

    train, dev = [], []
    for _, group in groups.items():
        random.shuffle(group)
        n_train = max(1, round(len(group) * 0.625))
        train.extend(group[:n_train])
        dev.extend(group[n_train:])

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(held)

    return train, dev, held


def assign_split_field(tasks: list[dict], split_name: str) -> list[dict]:
    return [{**t, "split": split_name} for t in tasks]


def write_jsonl(tasks: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")


def print_stats(train, dev, held):
    total = len(train) + len(dev) + len(held)
    print(f"\nPartition summary ({total} total tasks)")
    print(f"  Train:    {len(train):3d} ({100*len(train)/total:.1f}%)")
    print(f"  Dev:      {len(dev):3d} ({100*len(dev)/total:.1f}%)")
    print(f"  Held-out: {len(held):3d} ({100*len(held)/total:.1f}%)")

    # By dimension
    all_tasks = train + dev + held
    dims = sorted(set(t.get("dimension", "?") for t in all_tasks))
    print("\n  By dimension:")
    for d in dims:
        n_tr = sum(1 for t in train if t.get("dimension") == d)
        n_dv = sum(1 for t in dev   if t.get("dimension") == d)
        n_ho = sum(1 for t in held  if t.get("dimension") == d)
        print(f"    {d:40s} train={n_tr:3d}  dev={n_dv:3d}  held={n_ho:3d}")

    # By source mode
    modes = sorted(set(t.get("source_mode", "?") for t in all_tasks))
    print("\n  By source_mode:")
    for m in modes:
        n = sum(1 for t in all_tasks if t.get("source_mode") == m)
        print(f"    {m:30s} {n:3d}")


def run():
    print("Loading all raw tasks...")
    tasks = load_all()
    print(f"  Loaded {len(tasks)} tasks before dedup")

    tasks = deduplicate(tasks)
    print(f"  {len(tasks)} tasks after dedup")

    train, dev, held = contamination_aware_split(tasks)

    train = assign_split_field(train, "train")
    dev   = assign_split_field(dev,   "dev")
    held  = assign_split_field(held,  "held_out")

    write_jsonl(train, TRAIN_DIR / "train.jsonl")
    write_jsonl(dev,   DEV_DIR   / "dev.jsonl")
    write_jsonl(held,  HELD_DIR  / "held_out.jsonl")

    print_stats(train, dev, held)
    print(f"\nPartitions written:")
    print(f"  {TRAIN_DIR / 'train.jsonl'}")
    print(f"  {DEV_DIR   / 'dev.jsonl'}")
    print(f"  {HELD_DIR  / 'held_out.jsonl'}")


if __name__ == "__main__":
    run()
