# Oracle End-to-End Post-Fix STRICT Report (2026-04-04)

- Generated (UTC): 2026-04-04T17:26:11.0265721Z
- Source report: oracle_e2e_report_postfix_20260404.json
- Selection rule: temporal_state.formula_normalized_score != null and temporal_state.update_alpha != null
- Records in source: 20
- Records in strict report: 15

## Summary Evidence

- SHAP present in strict records: 15/15
- Temporal state present in strict records: 15/15
- On-chain tx hash present: 15/15
- On-chain wallet hash present: 15/15
- Low-risk records (risk <= 2): 13
- Low-risk records with non-trivial trust drop (< -0.05): 0
- Non-trivial low-risk drops: none

## Interpretation for Research Use

- Strict post-fix slice shows no non-trivial trust collapse for low-risk transactions.
- All strict records contain temporal diagnostics, SHAP output, and on-chain write hashes.
- This provides stronger post-fix evidence than mixed-window reporting when legacy-format rows are present.
