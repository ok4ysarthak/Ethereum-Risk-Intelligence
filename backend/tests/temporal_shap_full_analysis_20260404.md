# Full Analysis: Temporal Decay and SHAP Productivity Validation (2026-04-04 UTC)

## 1. Executive Summary

This validation confirms that the live inference pipeline is functioning for:

- Temporal trust scoring with exposed diagnostics.
- SHAP explanation generation and consistency.
- Operational prediction productivity across synthetic risk cohorts.

The run passed all temporal scenario checks (4/4), achieved full SHAP coverage (100%), and produced a high composite productivity index (0.986111).

Primary evidence source:

- `backend/tests/temporal_shap_productivity_result_20260404.json`
- `backend/tests/temporal_shap_productivity_report_20260404.md`

## 2. Scope and Objective

The objective was to verify three research claims under live backend execution:

1. Temporal time-decay logic is stable and behaviorally correct under long-gap, dormant, burst, and risk-sensitivity conditions.
2. SHAP explanations are present, directionally meaningful, and reproducible.
3. Prediction outputs are operationally useful ("productive") for ranking low, medium, and high-risk cohorts.

## 3. Runtime and Protocol

### 3.1 Environment

- API endpoint: `http://127.0.0.1:5000`
- Health status: `200 (ok)`
- Model loaded: `true`
- Graph ready: `true`

### 3.2 Execution Method

A dedicated script was executed:

- Script: `backend/tests/temporal_shap_productivity_validation.py`
- Output JSON: `backend/tests/temporal_shap_productivity_result_20260404.json`
- Output Markdown: `backend/tests/temporal_shap_productivity_report_20260404.md`

The script performs live `/predict/transaction` requests using controlled scenario profiles and extracts temporal and SHAP payloads directly from response JSON.

## 4. Methods

### 4.1 Temporal Scenario Tests

Four deterministic temporal tests were executed:

1. `low_risk_long_gap_guard`
- Two low-risk updates for same wallet with a 19,800s gap.
- Pass criterion: no sharp trust collapse (`delta >= -0.03`) and temporal normalized formula present.

2. `dormant_spike_detection`
- Same wallet updated after 25,200s gap (>6h dormant threshold).
- Pass criterion: `dormant_spike = true` and expected large time gap.

3. `burst_detection`
- Four high-risk updates within one minute.
- Pass criterion: `burst_detected = true` and `tx_count_last_1_min >= 4`.

4. `risk_sensitivity_ordering`
- Compare benign vs severe first-update trust outcomes.
- Pass criterion: benign temporal score > severe temporal score.

### 4.2 SHAP and Productivity Evaluation

Three synthetic cohorts were sampled (`n=18` each, total `n=54`):

- `benign`
- `suspicious`
- `severe`

Per prediction, the script captured:

- `risk_probability`, `risk_score`
- SHAP `summary`, top features, signed impacts
- Latency

Derived metrics:

- SHAP presence rate: summary present and top-feature count >= 3.
- SHAP sign alignment rate: net top-impact sign aligns with high/low risk region.
- Determinism rate: repeated identical payload returns same top-5 SHAP signature.
- Pairwise ordering rates: probability ranking quality across cohort pairs.
- Macro pairwise ordering rate: mean of pairwise rates.

### 4.3 Composite Productivity Index

The following weighted index was used:

`0.45 * pairwise_macro_ordering_rate + 0.25 * shap_present_rate + 0.15 * determinism_rate + 0.15 * temporal_pass_rate`

This combines ranking utility, explanation availability, explanation stability, and temporal reliability.

## 5. Results

### 5.1 Temporal Results

All temporal checks passed: `4/4`

1. Low-risk long gap guard:
- Previous: `0.545258`
- Current: `0.751940`
- Delta: `+0.206682`
- Gap: `19800s`
- Decay: `0.092922`
- Formula normalized: `0.903601`

2. Dormant spike detection:
- Dormant spike: `true`
- Gap: `25200s`
- Decay: `0.048606`
- Adjusted risk: `1.0`

3. Burst detection:
- Burst detected: `true`
- Count in 1 minute: `4`
- Decay: `0.998801`
- Update alpha: `0.3503`

4. Risk sensitivity ordering:
- Low-risk probability: `0.011232`
- High-risk probability: `0.999248`
- Low-risk temporal score: `0.544321`
- High-risk temporal score: `0.454630`

### 5.2 SHAP Coverage and Quality

- Sample count: `54`
- SHAP presence count: `54/54`
- SHAP presence rate: `1.000000`
- Mean top-feature count: `5`
- SHAP sign alignment rate: `1.000000`
- Determinism rate: `1.000000`

### 5.3 Productivity and Ranking Utility

Profile mean risk probabilities:

- Benign: `0.002125`
- Suspicious: `0.986877`
- Severe: `0.998149`

Ordering checks:

- `benign < suspicious`: `1.000000`
- `benign < severe`: `1.000000`
- `suspicious < severe`: `0.907407`
- Macro ordering rate: `0.969136`

Latency profile:

- Mean: `29.279 ms`
- P95: `32.967 ms`
- Max: `58.630 ms`

Composite productivity index:

- `0.986111`

## 6. Interpretation

1. Temporal logic is not only active but diagnostically transparent.
- Key state variables (`decay`, `time_gap_seconds`, `dormant_spike`, `burst_detected`, `formula_normalized_score`) are consistently returned.

2. Prior low-risk trust-collapse concern is not reproduced in this run.
- The historical long-gap condition (19,800s) produced a positive trust transition in the tested low-risk scenario.

3. SHAP explanations are production-usable.
- Full coverage, stable repetition behavior, and high directional agreement were observed.

4. Model output is operationally productive for triage ordering.
- Cohorts remain strongly rank-separable, supporting downstream prioritization workflows.

## 7. Threats to Validity

1. Synthetic scenario bias.
- Cohorts are generated from controlled distributions and may not capture all real-world adversarial strategies.

2. Single-run snapshot.
- Metrics reflect one live run and should be complemented by repeated time-window studies.

3. Non-causal interpretability.
- SHAP indicates feature contribution patterns for this model; it is not causal proof.

4. Environment caveat.
- Runtime displayed a Python-version warning from external dependency stack (non-blocking in this run).

## 8. Research Reporting Guidance

Suggested paper-safe statement:

"In live system validation, temporal trust updates were stable across long-gap, burst, and dormant scenarios, while SHAP explanations achieved full availability and deterministic consistency. Across synthetic benign/suspicious/severe cohorts, risk outputs preserved strong ordering utility (macro pairwise rate 0.969) with a composite operational productivity score of 0.986."

## 9. Reproducibility

1. Start backend API and confirm `/health` returns status `ok`.
2. Run: `E:/testing/env/Scripts/python.exe backend/tests/temporal_shap_productivity_validation.py`
3. Inspect generated artifacts:
- `backend/tests/temporal_shap_productivity_result_YYYYMMDD.json`
- `backend/tests/temporal_shap_productivity_report_YYYYMMDD.md`
- `backend/tests/temporal_shap_full_analysis_20260404.md`
