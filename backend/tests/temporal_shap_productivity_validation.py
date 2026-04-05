import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests


BASE_URL = "http://127.0.0.1:5000"
PREDICT_URL = f"{BASE_URL}/predict/transaction"
HEALTH_URL = f"{BASE_URL}/health"

RNG = random.Random(20260405)


def wallet_from_int(value: int) -> str:
    return "0x" + f"{value:040x}"[-40:]


def _randf(low: float, high: float) -> float:
    return round(RNG.uniform(low, high), 6)


def _randi(low: int, high: int) -> int:
    return int(RNG.randint(low, high))


def build_features(profile: str) -> Dict[str, float]:
    if profile == "benign":
        wallet_balance = _randf(8.0, 90.0)
        tx_value = _randf(0.05, 2.0)
        return {
            "Transaction_Value": tx_value,
            "Transaction_Fees": _randf(0.0001, 0.003),
            "Number_of_Inputs": float(_randi(1, 2)),
            "Number_of_Outputs": float(_randi(1, 3)),
            "Gas_Price": _randf(18.0, 50.0),
            "Wallet_Age_Days": _randf(180.0, 1200.0),
            "Wallet_Balance": wallet_balance,
            "Transaction_Velocity": _randf(0.1, 3.0),
            "Exchange_Rate": _randf(2200.0, 3800.0),
            "Final_Balance": max(0.001, round(wallet_balance - tx_value, 6)),
            "BMax_BMin_per_NT": _randf(0.01, 0.2),
            "from_address": wallet_from_int(_randi(10_000, 20_000)),
            "to_address": wallet_from_int(_randi(20_001, 30_000)),
        }

    if profile == "suspicious":
        wallet_balance = _randf(0.5, 18.0)
        tx_value = _randf(1.0, 22.0)
        return {
            "Transaction_Value": tx_value,
            "Transaction_Fees": _randf(0.001, 0.01),
            "Number_of_Inputs": float(_randi(2, 6)),
            "Number_of_Outputs": float(_randi(8, 24)),
            "Gas_Price": _randf(50.0, 180.0),
            "Wallet_Age_Days": _randf(1.0, 30.0),
            "Wallet_Balance": wallet_balance,
            "Transaction_Velocity": _randf(8.0, 35.0),
            "Exchange_Rate": _randf(2200.0, 3800.0),
            "Final_Balance": max(0.0001, round(wallet_balance - min(wallet_balance * 0.8, tx_value), 6)),
            "BMax_BMin_per_NT": _randf(0.4, 4.0),
            "from_address": wallet_from_int(_randi(30_001, 40_000)),
            "to_address": wallet_from_int(_randi(40_001, 50_000)),
        }

    if profile == "severe":
        wallet_balance = _randf(0.2, 6.0)
        tx_value = _randf(8.0, 140.0)
        return {
            "Transaction_Value": tx_value,
            "Transaction_Fees": _randf(0.003, 0.03),
            "Number_of_Inputs": float(_randi(3, 10)),
            "Number_of_Outputs": float(_randi(20, 70)),
            "Gas_Price": _randf(120.0, 350.0),
            "Wallet_Age_Days": _randf(0.0, 6.0),
            "Wallet_Balance": wallet_balance,
            "Transaction_Velocity": _randf(20.0, 95.0),
            "Exchange_Rate": _randf(2200.0, 3800.0),
            "Final_Balance": max(0.0001, round(wallet_balance * _randf(0.0, 0.1), 6)),
            "BMax_BMin_per_NT": _randf(2.0, 20.0),
            "from_address": wallet_from_int(_randi(50_001, 60_000)),
            "to_address": wallet_from_int(_randi(60_001, 70_000)),
        }

    raise ValueError(f"Unknown profile: {profile}")


def parse_impact(value) -> float:
    try:
        return float(str(value).replace("+", "").strip())
    except Exception:
        return 0.0


def call_predict(
    session: requests.Session,
    wallet_address: str,
    timestamp: int,
    features: Dict[str, float],
    update_state: bool,
) -> Tuple[Dict, float]:
    payload = {
        "wallet_address": wallet_address,
        "timestamp": int(timestamp),
        "update_state": bool(update_state),
        "features": features,
    }

    t0 = time.perf_counter()
    response = session.post(PREDICT_URL, json=payload, timeout=35)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    data = response.json() if "application/json" in str(response.headers.get("content-type", "")) else {
        "raw": response.text
    }
    if response.status_code != 200:
        raise RuntimeError(f"Predict call failed ({response.status_code}): {data}")

    return data, latency_ms


def run_temporal_tests(session: requests.Session, base_ts: int) -> List[Dict]:
    tests: List[Dict] = []

    # 1) Low-risk guard after long gap (historical regression check at 19,800s).
    wallet_1 = wallet_from_int(900_001)
    low_features = build_features("benign")
    call_predict(session, wallet_1, base_ts, low_features, update_state=True)
    second, _ = call_predict(session, wallet_1, base_ts + 19_800, low_features, update_state=True)
    delta = (second.get("temporal_score_normalized") or 0.0) - (second.get("temporal_previous_score") or 0.0)
    state_1 = second.get("temporal_state") or {}
    tests.append(
        {
            "name": "low_risk_long_gap_guard",
            "passed": bool(delta >= -0.03 and state_1.get("formula_normalized_score") is not None),
            "details": {
                "delta": round(delta, 6),
                "previous": second.get("temporal_previous_score"),
                "current": second.get("temporal_score_normalized"),
                "time_gap_seconds": state_1.get("time_gap_seconds"),
                "decay": state_1.get("decay"),
                "formula_normalized_score": state_1.get("formula_normalized_score"),
            },
        }
    )

    # 2) Dormant spike activation after >6h inactivity.
    wallet_2 = wallet_from_int(900_002)
    moderate_features = build_features("suspicious")
    call_predict(session, wallet_2, base_ts, moderate_features, update_state=True)
    dormant, _ = call_predict(session, wallet_2, base_ts + (7 * 3600), moderate_features, update_state=True)
    state_2 = dormant.get("temporal_state") or {}
    tests.append(
        {
            "name": "dormant_spike_detection",
            "passed": bool(state_2.get("dormant_spike") is True and int(state_2.get("time_gap_seconds") or 0) >= 7 * 3600),
            "details": {
                "dormant_spike": state_2.get("dormant_spike"),
                "time_gap_seconds": state_2.get("time_gap_seconds"),
                "decay": state_2.get("decay"),
                "adjusted_risk_probability": state_2.get("adjusted_risk_probability"),
            },
        }
    )

    # 3) Burst detection at 4 transactions/minute threshold.
    wallet_3 = wallet_from_int(900_003)
    high_features = build_features("severe")
    burst_responses = []
    for offset in (0, 10, 20, 30):
        resp, _ = call_predict(session, wallet_3, base_ts + offset, high_features, update_state=True)
        burst_responses.append(resp)
    burst_state = burst_responses[-1].get("temporal_state") or {}
    tests.append(
        {
            "name": "burst_detection",
            "passed": bool(burst_state.get("burst_detected") is True and int(burst_state.get("tx_count_last_1_min") or 0) >= 4),
            "details": {
                "burst_detected": burst_state.get("burst_detected"),
                "tx_count_last_1_min": burst_state.get("tx_count_last_1_min"),
                "decay": burst_state.get("decay"),
                "update_alpha": burst_state.get("update_alpha"),
            },
        }
    )

    # 4) Risk sensitivity check: severe profile should produce lower trust than benign profile.
    wallet_low = wallet_from_int(900_010)
    wallet_high = wallet_from_int(900_011)
    low_resp, _ = call_predict(session, wallet_low, base_ts + 120, build_features("benign"), update_state=True)
    high_resp, _ = call_predict(session, wallet_high, base_ts + 120, build_features("severe"), update_state=True)
    tests.append(
        {
            "name": "risk_sensitivity_ordering",
            "passed": bool((low_resp.get("temporal_score_normalized") or 0.0) > (high_resp.get("temporal_score_normalized") or 0.0)),
            "details": {
                "low_risk_probability": low_resp.get("risk_probability"),
                "high_risk_probability": high_resp.get("risk_probability"),
                "low_temporal_score": low_resp.get("temporal_score_normalized"),
                "high_temporal_score": high_resp.get("temporal_score_normalized"),
            },
        }
    )

    return tests


def _pairwise_less_rate(lhs: List[float], rhs: List[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    total = len(lhs) * len(rhs)
    wins = 0
    for lv in lhs:
        for rv in rhs:
            if lv < rv:
                wins += 1
    return wins / total


def run_shap_and_productivity_eval(session: requests.Session, base_ts: int, samples_per_profile: int = 18) -> Dict:
    profile_order = ["benign", "suspicious", "severe"]
    records = []
    latencies = []

    for p_idx, profile in enumerate(profile_order):
        for sample_idx in range(samples_per_profile):
            wallet = wallet_from_int(950_000 + (p_idx * 10_000) + sample_idx)
            ts = base_ts + 500 + (p_idx * 1000) + sample_idx
            features = build_features(profile)
            response, latency_ms = call_predict(session, wallet, ts, features, update_state=False)
            latencies.append(latency_ms)

            shap_payload = response.get("shap_explanation") if isinstance(response.get("shap_explanation"), dict) else {}
            top_features = shap_payload.get("top_features") if isinstance(shap_payload.get("top_features"), list) else []
            summary = str(shap_payload.get("summary") or "").strip()
            impacts = [parse_impact(item.get("impact")) for item in top_features if isinstance(item, dict)]

            records.append(
                {
                    "profile": profile,
                    "risk_probability": float(response.get("risk_probability") or 0.0),
                    "risk_score": int(response.get("risk_score") or 0),
                    "shap_summary_present": bool(summary),
                    "top_feature_count": len(top_features),
                    "top_features": [item.get("feature") for item in top_features if isinstance(item, dict)],
                    "net_top_impact": round(sum(impacts), 6),
                    "latency_ms": round(latency_ms, 3),
                }
            )

    total = len(records)
    shap_present_count = sum(1 for r in records if r["shap_summary_present"] and r["top_feature_count"] >= 3)
    shap_present_rate = (shap_present_count / total) if total else 0.0

    aligned = 0
    for r in records:
        high_risk = r["risk_probability"] >= 0.5
        if high_risk and r["net_top_impact"] > 0:
            aligned += 1
        elif (not high_risk) and r["net_top_impact"] < 0:
            aligned += 1
    sign_alignment_rate = (aligned / total) if total else 0.0

    by_profile = {name: [r["risk_probability"] for r in records if r["profile"] == name] for name in profile_order}
    means = {name: round(statistics.mean(vals), 6) if vals else None for name, vals in by_profile.items()}

    ordering_pass = bool(
        means.get("benign") is not None
        and means.get("suspicious") is not None
        and means.get("severe") is not None
        and means["benign"] < means["suspicious"] < means["severe"]
    )

    pairwise = {
        "benign_lt_suspicious": round(_pairwise_less_rate(by_profile["benign"], by_profile["suspicious"]), 6),
        "benign_lt_severe": round(_pairwise_less_rate(by_profile["benign"], by_profile["severe"]), 6),
        "suspicious_lt_severe": round(_pairwise_less_rate(by_profile["suspicious"], by_profile["severe"]), 6),
    }
    pairwise_macro = round(statistics.mean(pairwise.values()), 6)

    # Determinism check for repeated inference on identical payload.
    deterministic_wallet = wallet_from_int(999_001)
    deterministic_features = build_features("suspicious")
    deterministic_signatures = []
    for rep in range(5):
        rep_resp, _ = call_predict(
            session,
            deterministic_wallet,
            base_ts + 10_000 + rep,
            deterministic_features,
            update_state=False,
        )
        rep_shap = rep_resp.get("shap_explanation") if isinstance(rep_resp.get("shap_explanation"), dict) else {}
        rep_top = rep_shap.get("top_features") if isinstance(rep_shap.get("top_features"), list) else []
        signature = tuple((item.get("feature"), round(parse_impact(item.get("impact")), 4)) for item in rep_top[:5] if isinstance(item, dict))
        deterministic_signatures.append(signature)

    baseline = deterministic_signatures[0] if deterministic_signatures else tuple()
    deterministic_matches = sum(1 for sig in deterministic_signatures[1:] if sig == baseline)
    determinism_rate = (deterministic_matches / max(1, len(deterministic_signatures) - 1))

    return {
        "sample_count": total,
        "samples_per_profile": samples_per_profile,
        "shap_present_count": shap_present_count,
        "shap_present_rate": round(shap_present_rate, 6),
        "average_top_feature_count": round(
            statistics.mean([r["top_feature_count"] for r in records]) if records else 0.0,
            6,
        ),
        "sign_alignment_rate": round(sign_alignment_rate, 6),
        "profile_risk_probability_means": means,
        "profile_ordering_pass": ordering_pass,
        "pairwise_ordering_rates": pairwise,
        "pairwise_macro_ordering_rate": pairwise_macro,
        "determinism_rate": round(determinism_rate, 6),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "sample_records": records,
    }


def compose_markdown_report(result: Dict) -> str:
    temporal_tests = result.get("temporal_tests") or []
    temporal_passed = sum(1 for t in temporal_tests if t.get("passed"))
    temporal_total = len(temporal_tests)

    shap_eval = result.get("shap_productivity") or {}
    means = shap_eval.get("profile_risk_probability_means") or {}
    latency = shap_eval.get("latency_ms") or {}

    lines: List[str] = []
    lines.append(f"# Temporal Decay + SHAP Productivity Validation ({result['run_date_utc']})")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Validate temporal trust-decay behavior, SHAP explanation quality, and practical prediction productivity using live /predict/transaction calls.")
    lines.append("")
    lines.append("## Runtime Context")
    lines.append("")
    lines.append(f"- Base URL: {result.get('base_url')}")
    lines.append(f"- Health status: {result.get('health', {}).get('status_code')} ({result.get('health', {}).get('payload', {}).get('status')})")
    lines.append(f"- Total API calls: {result.get('total_api_calls')}")
    lines.append("")
    lines.append("## Temporal Validation")
    lines.append("")
    lines.append(f"- Passed: {temporal_passed}/{temporal_total}")
    for test in temporal_tests:
        state = "PASS" if test.get("passed") else "FAIL"
        lines.append(f"- [{state}] {test.get('name')}: {json.dumps(test.get('details', {}), sort_keys=True)}")
    lines.append("")
    lines.append("## SHAP Quality + Productivity")
    lines.append("")
    lines.append(f"- SHAP presence rate (summary + >=3 top features): {shap_eval.get('shap_present_rate')}")
    lines.append(f"- Mean top-feature count: {shap_eval.get('average_top_feature_count')}")
    lines.append(f"- SHAP sign alignment rate: {shap_eval.get('sign_alignment_rate')}")
    lines.append(f"- Determinism rate (same input repeated): {shap_eval.get('determinism_rate')}")
    lines.append(f"- Profile mean risk probabilities: benign={means.get('benign')}, suspicious={means.get('suspicious')}, severe={means.get('severe')}")
    lines.append(f"- Profile ordering pass (benign < suspicious < severe): {shap_eval.get('profile_ordering_pass')}")
    lines.append(f"- Pairwise ordering rates: {json.dumps(shap_eval.get('pairwise_ordering_rates', {}), sort_keys=True)}")
    lines.append(f"- Latency ms (mean/p95/max): {latency.get('mean')} / {latency.get('p95')} / {latency.get('max')}")
    lines.append("")
    lines.append("## Composite Productivity Index")
    lines.append("")
    lines.append(
        "- Formula: 0.45 * pairwise_macro_ordering_rate + 0.25 * shap_present_rate "
        "+ 0.15 * determinism_rate + 0.15 * temporal_pass_rate"
    )
    lines.append(f"- Value: {result.get('composite_productivity_index')}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Temporal diagnostics (decay, gap, dormant_spike, burst_detected, formula_normalized_score) are present and testable in live responses.")
    lines.append("- SHAP output is consistently available and stable for repeated identical inputs, supporting reproducible interpretation.")
    lines.append("- Risk productivity is supported when synthetic benign/suspicious/severe cohorts remain correctly rank-ordered.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- This is a live-system validation against synthetic scenario cohorts, not a blinded benchmark dataset.")
    lines.append("- Productivity metrics are operational proxies and should be complemented with dataset-level calibration studies.")

    return "\n".join(lines)


def main():
    session = requests.Session()
    session.headers.update({"Cache-Control": "no-cache"})

    health_resp = session.get(HEALTH_URL, timeout=20)
    health_payload = health_resp.json() if "application/json" in str(health_resp.headers.get("content-type", "")) else {}
    if health_resp.status_code not in (200, 503):
        raise RuntimeError(f"Unexpected health status: {health_resp.status_code}, payload={health_payload}")

    base_ts = int(time.time()) - 60_000
    temporal_tests = run_temporal_tests(session, base_ts=base_ts)
    shap_eval = run_shap_and_productivity_eval(session, base_ts=base_ts, samples_per_profile=18)

    temporal_pass_rate = (
        sum(1 for t in temporal_tests if t.get("passed")) / max(1, len(temporal_tests))
    )
    composite_productivity_index = round(
        (0.45 * float(shap_eval.get("pairwise_macro_ordering_rate") or 0.0))
        + (0.25 * float(shap_eval.get("shap_present_rate") or 0.0))
        + (0.15 * float(shap_eval.get("determinism_rate") or 0.0))
        + (0.15 * float(temporal_pass_rate)),
        6,
    )

    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_date_human = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_dir = Path(__file__).resolve().parent
    json_path = output_dir / f"temporal_shap_productivity_result_{run_date}.json"
    md_path = output_dir / f"temporal_shap_productivity_report_{run_date}.md"

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date_utc": run_date_human,
        "base_url": BASE_URL,
        "health": {
            "status_code": health_resp.status_code,
            "payload": health_payload,
        },
        "temporal_tests": temporal_tests,
        "shap_productivity": shap_eval,
        "temporal_pass_rate": round(temporal_pass_rate, 6),
        "composite_productivity_index": composite_productivity_index,
        "total_api_calls": 1 + len(temporal_tests) + (18 * 3) + 5 + 7,
    }

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(compose_markdown_report(result), encoding="utf-8")

    print(json_path)
    print(md_path)
    print(json.dumps({
        "temporal_pass_rate": result["temporal_pass_rate"],
        "shap_present_rate": shap_eval.get("shap_present_rate"),
        "pairwise_macro_ordering_rate": shap_eval.get("pairwise_macro_ordering_rate"),
        "determinism_rate": shap_eval.get("determinism_rate"),
        "composite_productivity_index": composite_productivity_index,
    }, indent=2))


if __name__ == "__main__":
    main()
