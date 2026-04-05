# Temporal Decay + SHAP Productivity Validation (2026-04-04)

## Objective

Validate temporal trust-decay behavior, SHAP explanation quality, and practical prediction productivity using live /predict/transaction calls.

## Runtime Context

- Base URL: http://127.0.0.1:5000
- Health status: 200 (ok)
- Total API calls: 71

## Temporal Validation

- Passed: 4/4
- [PASS] low_risk_long_gap_guard: {"current": 0.75194, "decay": 0.092922, "delta": 0.206682, "formula_normalized_score": 0.903601, "previous": 0.545258, "time_gap_seconds": 19800}
- [PASS] dormant_spike_detection: {"adjusted_risk_probability": 1.0, "decay": 0.048606, "dormant_spike": true, "time_gap_seconds": 25200}
- [PASS] burst_detection: {"burst_detected": true, "decay": 0.998801, "tx_count_last_1_min": 4, "update_alpha": 0.3503}
- [PASS] risk_sensitivity_ordering: {"high_risk_probability": 0.999247670173645, "high_temporal_score": 0.45463, "low_risk_probability": 0.0112324059009552, "low_temporal_score": 0.544321}

## SHAP Quality + Productivity

- SHAP presence rate (summary + >=3 top features): 1.0
- Mean top-feature count: 5
- SHAP sign alignment rate: 1.0
- Determinism rate (same input repeated): 1.0
- Profile mean risk probabilities: benign=0.002125, suspicious=0.986877, severe=0.998149
- Profile ordering pass (benign < suspicious < severe): True
- Pairwise ordering rates: {"benign_lt_severe": 1.0, "benign_lt_suspicious": 1.0, "suspicious_lt_severe": 0.907407}
- Latency ms (mean/p95/max): 29.279 / 32.967 / 58.63

## Composite Productivity Index

- Formula: 0.45 * pairwise_macro_ordering_rate + 0.25 * shap_present_rate + 0.15 * determinism_rate + 0.15 * temporal_pass_rate
- Value: 0.986111

## Interpretation

- Temporal diagnostics (decay, gap, dormant_spike, burst_detected, formula_normalized_score) are present and testable in live responses.
- SHAP output is consistently available and stable for repeated identical inputs, supporting reproducible interpretation.
- Risk productivity is supported when synthetic benign/suspicious/severe cohorts remain correctly rank-ordered.

## Limitations

- This is a live-system validation against synthetic scenario cohorts, not a blinded benchmark dataset.
- Productivity metrics are operational proxies and should be complemented with dataset-level calibration studies.