# OpenViking Golden Retrieval Gate Protocol

> **Status:** Arbiter-approved (2026-10-12, P-20261011-04 residual decision)  
> **Source:** Kai (ZCode / Reviewer) evaluation protocol recommendation  
> **Target Subsystem:** OpenViking retrieval benchmark probes (`probes/golden_set.yaml`, `probes/golden_retrieval_eval.py`)

---

## 1. Core Principles

The OpenViking golden retrieval benchmark measures semantic search, intent analysis, and tiered reranking accuracy across a frozen query set. Because the underlying knowledge corpus grows dynamically over time, benchmark evaluations must discriminate between **genuine algorithmic/configuration regressions** and **benign corpus-epoch shifts**.

---

## 2. Protocol Standards

### 2.1 Corpus-Epoch Marker Per Run
Every benchmark evaluation run MUST capture a **corpus-epoch marker**:
- Execution timestamp and live server target.
- Total indexed document count or latest ingested memory/resource URI.
- Landmark state (e.g. post-audit ingestion, post-task archive).

When a query's top rank shifts, the evaluator checks whether a document ingested after the golden set was authored has legitimately taken top rank.

### 2.2 Per-Query Differential on Metric Drops
If any metric drops below gate thresholds or historical certified levels:
- **Do not evaluate aggregate numbers in isolation.**
- Generate a per-query differential (`ΔRank`, `ΔScore`, `Top-1 URI`) against the certified baseline.
- Inspect the specific query that flipped:
  - If the new top-1 document is a valid, high-fidelity ground truth created as part of subsequent operational work (a corpus-epoch event), the issue is routed for **re-baselining**, not treated as a code regression.
  - If the new top-1 document is an irrelevant decoy or negative noise query exceeds threshold, it is treated as a **regression defect**.

### 2.3 Tolerance Bands
Due to stochastic variations in embedding provider outputs and minor corpus density shifts, gate acceptance enforces the following tolerance bands vs certified baseline:
- **Recall@1**: **±1 query** (for a 21-positive set, 1 query represents $\approx 4.76\%$).
- **MRR (Mean Reciprocal Rank)**: **±0.024**.
- **Score Averages (Positive Top-1, Score Separation)**: **±0.004**.

Metrics shifting within these bands do not constitute a regression unless an individual query categorically breaks.

### 2.4 Median-of-3 Tie-Breaker
- Reranker and embedding API latencies or micro-network fluctuations can rarely perturb borderline score rankings.
- If a metric falls within a borderline marginal breach ($\le 0.5$ tolerance band width), execute a sequence of **3 consecutive runs**.
- Take the **median** metric across the three runs as the authoritative gate verdict.
- The median-of-3 rule is strictly a **marginal tie-breaker only**; it must never be used to mask systematic or persistent drops.

---

## 3. Re-baselining Governance

When a corpus-epoch event introduces a legitimate new authority document:
1. Append the new URI to the target query's `expected_uris` in `golden_set.yaml`.
2. Bump the golden set `version` (e.g. `1.0.0` → `1.0.1`).
3. Add a dated, attributed changelog comment in `golden_set.yaml` documenting the justification.
4. Obtain Arbiter approval prior to merging the re-baseline into main/carry branches.
