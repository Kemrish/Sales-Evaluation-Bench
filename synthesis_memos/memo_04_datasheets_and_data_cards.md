# Synthesis Memo 04 — Datasheets for Datasets and Data Cards

**Papers**:  
- Gebru et al., "Datasheets for Datasets," Communications of the ACM, 2021  
- Pushkarna et al., "Data Cards: Purposeful and Transparent Dataset Documentation," FAccT 2022  
**Date**: 2026-04-29  
**Author**: Kemeriya  
**Status**: Common reading — informs datasheet.md and Act V publication

---

## Core Claims

Gebru et al. propose a standardized documentation template for datasets covering seven sections: motivation, composition, collection process, preprocessing/cleaning/labeling, uses, distribution, and maintenance. The goal is to enable dataset consumers to make informed decisions, prevent misuse, and identify potential harms before deployment.

Pushkarna et al. extend this with a layered "data card" format: telescopic (one-line overview), periscopic (key facts table), and microscopic (full details). The innovation is that a single dataset gets multiple views for different audiences — a practitioner who needs a quick check gets the periscopic layer; an auditor who needs full provenance gets the microscopic layer.

---

## Where I Agree

The seven-section structure maps cleanly onto evaluation datasets for agent systems, and I applied it directly in datasheet.md. The "intended uses" and "out-of-scope uses" sections are particularly valuable for Tenacious-Bench: the rubric weights (action_correctness: 0.5 for dual_control tasks vs. 0.7 for ICP tasks) make the benchmark inappropriate for any use case that doesn't understand which dimension is being weighted. A practitioner who reads only the "score ≥ 0.70 = PASS" headline without the composition section will misinterpret results on signal_reliability (only 8 tasks, no held-out coverage).

The layered detail from Pushkarna is the right approach for a benchmark that will be used at multiple levels — an evaluator running ablations needs the quick reference table, a reviewer auditing the leakage controls needs the preprocessing section.

---

## Where I Disagree

Gebru et al.'s "distribution" section assumes the dataset will be distributed widely and asks creators to document licensing terms, third-party rights, and applicable regulations. The implicit model is a static artifact published once for broad community use — think ImageNet or GLUE.

**Tenacious-Bench is a living production asset, not a static research artifact.** The held-out partition is intentionally kept back until the leaderboard is published (v1.0); the train/dev partitions evolve as new failure modes are discovered. This creates a two-tier distribution model that Gebru et al.'s section does not anticipate: the "public" artifact (train + dev + evaluation scripts) is shareable under CC-BY-4.0, but the sealed held-out is not distributable until after the trained component is locked.

**My evidence**: datasheet.md Section 6 originally read "Distribution decision pending legal review" — a stub that correctly flagged this uncertainty but didn't resolve it. The honest answer is that Tenacious-Bench has a different distribution lifecycle than ImageNet. The training partition is public now; the held-out is public only after the model card is published and the training run is committed. Gebru et al.'s framework treats "will the dataset be distributed?" as a binary question; the correct answer here is "yes, partially, on a staggered timeline tied to training-run completion."

**The fix applied**: datasheet.md Section 6 now documents the staggered distribution model explicitly. This is a case where the framework needs to be adapted rather than directly applied.

---

## Implication for Tenacious-Bench

The datasheet format is appropriate and applied correctly. The one structural adaptation needed — staggered distribution tied to the held-out leaderboard — is not a failure to follow Gebru et al. but a deliberate departure justified by the production-evaluation context that the paper was not designed for.
