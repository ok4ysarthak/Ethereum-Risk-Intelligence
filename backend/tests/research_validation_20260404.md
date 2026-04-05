# Research Validation Note (2026-04-04)

## Objective
Investigate why a low-risk transaction (`risk_score=1/10`) could still reduce trust (`0.50 -> 0.36`), verify backend correctness, and confirm whether SHAP explainability is surfaced in the UI.

## Observed Issue (Pre-Fix)
From live oracle evidence in `oracle_e2e_report_20260404.md`:
- Example tx: `0x3c28874abb6bd2ed6c784532227e78d391f96a79eee66ceed525a002f3ae5223`
- Fraud probability: `0.0889` (low risk)
- Trust changed: `0.50 -> 0.364412`
- Temporal state: `decay=0.092922`, `time_gap_seconds=19800`

This behavior is unintuitive for low-risk updates.

## Root Causes Identified
1. Score compression in temporal formula:
- Raw formula used: `new = old * decay + (1 - adjusted_risk) * weight`
- With small `decay` and fixed `weight=0.35`, output can be compressed toward a low band even for low risk.

2. Timestamp convention mismatch:
- Temporal module mixed UTC-naive writes (`utcfromtimestamp`) with naive `.timestamp()` reads.
- On systems with non-UTC local time, this inflates effective gap and amplifies decay.

## Fixes Applied
1. Temporal score stabilization:
- Keep raw formula signal, then normalize by `(decay + weight)` and apply bounded smoothing.
- Add low-risk guard to prevent sharp trust drops for very low adjusted risk.

2. Time consistency fix:
- Use consistent `datetime.fromtimestamp(...)` for persisted temporal timestamps.

3. API transparency for research/UI:
- Added temporal diagnostics to response:
  - `formula_raw_score`
  - `formula_normalized_score`
  - `update_alpha`

4. UI explainability upgrade:
- Transaction modal now renders:
  - Rule explanation
  - SHAP explanation (summary + top features)
  - Temporal trust explanation (before/after + state diagnostics)

## Post-Fix Validation Evidence

### A) Deterministic DB-layer temporal validation
Executed against real DB/session layer with controlled state:
- Setup: wallet trust `0.5`, gap `19800s`, risk `0.0889`
- Output:
```json
{
  "temporal_score_normalized": 0.686151,
  "temporal_previous_score": 0.5,
  "decay": 0.092922,
  "time_gap_seconds": 19800,
  "dormant_spike": false,
  "adjusted_risk_probability": 0.091567,
  "formula_raw_score": 0.364412,
  "formula_normalized_score": 0.822747,
  "update_alpha": 0.57677
}
```
Result: trust no longer collapses for low-risk update.

### B) API-level sequential validation (`/predict/transaction`)
Same wallet, two sequential calls (`update_state=true`) with same low-risk feature profile:
```json
{
  "first": {
    "risk_score": 2,
    "fraud_probability": 0.1196,
    "temporal_prev": 0.5,
    "temporal_now": 0.534192,
    "shap_summary": "Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk."
  },
  "second": {
    "risk_score": 2,
    "fraud_probability": 0.1196,
    "temporal_prev": 0.534192,
    "temporal_now": 0.689039,
    "temporal_state": {
      "decay": 0.092922,
      "time_gap_seconds": 19800,
      "formula_raw_score": 0.355518,
      "formula_normalized_score": 0.802665,
      "update_alpha": 0.57677
    },
    "shap_top_count": 5
  }
}
```
Result: temporal update remains stable and explainable; SHAP payload is present.

### C) UI validation
Activity modal now contains visible sections:
- `SHAP Explanation`
- `Temporal Trust Explanation`

### D) Live oracle comparison (pre-fix vs strict post-fix)

Computed from generated JSON evidence artifacts:

| Metric | Pre-fix live report | Post-fix STRICT live report |
|---|---:|---:|
| Records analyzed | 20 | 15 |
| Low-risk records (`risk <= 2`) | 17 | 13 |
| Low-risk non-trivial trust drops (`delta < -0.05`) | 16 | 0 |
| SHAP present | 19/20 | 15/15 |
| Temporal state present | 20/20 | 15/15 |
| On-chain tx hash present | 20/20 | 15/15 |
| On-chain wallet hash present | 20/20 | 15/15 |

Interpretation:
- The pre-fix report exhibited frequent low-risk trust-collapse behavior.
- The strict post-fix slice (records containing new temporal diagnostics) shows zero non-trivial low-risk trust drops.
- The mixed post-fix window can still include legacy-format rows from immediately before restart, which is why strict-slice filtering is required for clean post-fix attribution.

## Believability Assessment for Research Use

### Why the backend behavior is believable
- Multi-stage evidence chain exists: live tx fetch -> model scoring -> temporal state transition -> on-chain write hash -> persisted payload.
- SHAP output includes both summary and feature-level directional impacts (`increases_risk`/`reduces_risk`).
- Temporal transition now exposes internal diagnostics (`decay`, `adjusted_risk_probability`, normalized formula score).

### Remaining limitations (must disclose in paper)
- Startup warning exists in current environment due Python 3.9 compatibility noise from external dependency stack.
- This is a production-like validation in a live testnet environment, not a blinded benchmark study.
- Explainability is post-hoc (SHAP) and should not be interpreted as causal proof.

## Recommended citation statement
"We validated the live inference pipeline end-to-end on Sepolia testnet, confirming that each transaction produced (i) risk prediction, (ii) temporal trust update with exposed state diagnostics, and (iii) SHAP feature attributions persisted and rendered in the UI."
