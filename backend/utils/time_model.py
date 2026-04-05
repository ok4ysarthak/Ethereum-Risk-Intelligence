import math
from datetime import datetime
from typing import Any, Dict, List, Optional

INITIAL_TEMPORAL_SCORE = 0.5
DEFAULT_DECAY_LAMBDA = 0.00012
DEFAULT_WEIGHT = 0.35
DEFAULT_BURST_WINDOW_SECONDS = 60
DEFAULT_BURST_THRESHOLD = 4
DEFAULT_BURST_PENALTY = 0.12
DEFAULT_DORMANT_GAP_SECONDS = 6 * 60 * 60
DEFAULT_DORMANT_PENALTY = 0.10
DEFAULT_ROLLING_ALPHA = 0.2
DEFAULT_UPDATE_ALPHA = 0.35
MAX_RECENT_TIMESTAMPS = 120


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _normalize_address(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    value = str(address).strip().lower()
    if not value:
        return None
    if not value.startswith("0x"):
        value = f"0x{value}"
    return value


def _coerce_timestamp(timestamp: Optional[Any]) -> int:
    if timestamp is None:
        return int(datetime.utcnow().timestamp())

    if isinstance(timestamp, datetime):
        return int(timestamp.timestamp())

    try:
        return int(float(timestamp))
    except Exception:
        return int(datetime.utcnow().timestamp())


def compute_decay(time_gap_seconds: int, decay_lambda: float = DEFAULT_DECAY_LAMBDA) -> float:
    """Exponential time-decay for stateful trust memory."""
    safe_gap = max(0, int(time_gap_seconds))
    safe_lambda = max(0.0, float(decay_lambda))
    return float(math.exp(-safe_lambda * safe_gap))


def detect_burst_activity(
    recent_tx_timestamps: List[int],
    current_timestamp: int,
    window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS,
    threshold: int = DEFAULT_BURST_THRESHOLD,
) -> Dict[str, Any]:
    """Detect transaction bursts within a short time window."""
    safe_now = _coerce_timestamp(current_timestamp)
    safe_window = max(1, int(window_seconds))
    safe_threshold = max(1, int(threshold))

    in_window = [
        int(ts)
        for ts in recent_tx_timestamps
        if 0 <= (safe_now - int(ts)) <= safe_window
    ]

    return {
        "is_burst": len(in_window) >= safe_threshold,
        "count": len(in_window),
        "window_seconds": safe_window,
        "timestamps": in_window,
    }


def get_wallet_state(session, wallet_address: str, now_timestamp: Optional[Any] = None) -> Dict[str, Any]:
    """Load or initialize wallet temporal state persisted in PostgreSQL."""
    from database.models_db import Wallet

    normalized_address = _normalize_address(wallet_address)
    if not normalized_address:
        raise ValueError("wallet_address is required for temporal scoring")

    now_ts = _coerce_timestamp(now_timestamp)
    now_dt = datetime.fromtimestamp(now_ts)

    wallet = session.query(Wallet).filter(Wallet.address == normalized_address).one_or_none()

    if wallet is None:
        wallet = Wallet(
            address=normalized_address,
            first_seen=now_dt,
            last_seen=now_dt,
            trust_score=INITIAL_TEMPORAL_SCORE,
            avg_risk=0.0,
            metadata_json={},
        )
        session.add(wallet)
        session.flush()

    metadata = wallet.metadata_json if isinstance(wallet.metadata_json, dict) else {}

    raw_recent = metadata.get("recent_tx_timestamps", [])
    recent_tx_timestamps: List[int] = []
    if isinstance(raw_recent, list):
        for item in raw_recent:
            try:
                ts = int(item)
                if ts > 0:
                    recent_tx_timestamps.append(ts)
            except Exception:
                continue

    if wallet.last_seen is not None:
        last_tx_ts = int(wallet.last_seen.timestamp())
    elif recent_tx_timestamps:
        last_tx_ts = int(recent_tx_timestamps[-1])
    else:
        last_tx_ts = now_ts

    return {
        "wallet": wallet,
        "metadata": metadata,
        "recent_tx_timestamps": recent_tx_timestamps,
        "last_score": _clamp(wallet.trust_score if wallet.trust_score is not None else INITIAL_TEMPORAL_SCORE),
        "last_transaction_ts": int(last_tx_ts),
        "rolling_avg_risk": _clamp(wallet.avg_risk if wallet.avg_risk is not None else metadata.get("rolling_avg_risk", 0.0)),
        "tx_count_total": int(metadata.get("tx_count_total", 0) or 0),
    }


def update_wallet_score(
    session,
    wallet_address: str,
    risk_probability: float,
    current_timestamp: Optional[Any] = None,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    weight: float = DEFAULT_WEIGHT,
    burst_threshold: int = DEFAULT_BURST_THRESHOLD,
    burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS,
    burst_penalty: float = DEFAULT_BURST_PENALTY,
    dormant_gap_seconds: int = DEFAULT_DORMANT_GAP_SECONDS,
    dormant_penalty: float = DEFAULT_DORMANT_PENALTY,
    rolling_alpha: float = DEFAULT_ROLLING_ALPHA,
    update_alpha: float = DEFAULT_UPDATE_ALPHA,
) -> Dict[str, Any]:
    """Update trust score with temporal memory and anomaly-aware adjustments."""
    now_ts = _coerce_timestamp(current_timestamp)
    state = get_wallet_state(session, wallet_address, now_timestamp=now_ts)

    previous_score = _clamp(state["last_score"])
    previous_ts = int(state["last_transaction_ts"])
    time_gap_seconds = max(0, now_ts - previous_ts)
    decay = compute_decay(time_gap_seconds, decay_lambda=decay_lambda)

    recent_tx_timestamps = list(state["recent_tx_timestamps"])
    recent_tx_timestamps.append(now_ts)

    burst_meta = detect_burst_activity(
        recent_tx_timestamps,
        current_timestamp=now_ts,
        window_seconds=burst_window_seconds,
        threshold=burst_threshold,
    )

    has_history = bool(state["tx_count_total"] > 0 or time_gap_seconds > 0)
    dormant_spike = time_gap_seconds >= max(1, int(dormant_gap_seconds)) and has_history

    adjusted_risk = _clamp(risk_probability)
    if burst_meta["is_burst"]:
        adjusted_risk = _clamp(adjusted_risk + float(burst_penalty))
    if dormant_spike:
        adjusted_risk = _clamp(adjusted_risk + float(dormant_penalty))

    alpha = _clamp(rolling_alpha)
    rolling_avg_risk = _clamp(((1.0 - alpha) * state["rolling_avg_risk"]) + (alpha * adjusted_risk))

    adjusted_with_history = _clamp(adjusted_risk + (0.15 * rolling_avg_risk))

    # Keep the requested base temporal equation, then normalize and smooth to
    # avoid compressing trust into a narrow 0.0-0.4 band during long gaps.
    formula_raw_score = (previous_score * decay) + ((1.0 - adjusted_with_history) * float(weight))
    formula_norm_den = max((decay + float(weight)), 1e-8)
    formula_normalized_score = _clamp(formula_raw_score / formula_norm_den)

    alpha_update = _clamp(float(update_alpha))
    dynamic_alpha = _clamp(alpha_update + ((1.0 - decay) * 0.25), 0.15, 0.85)
    new_score = previous_score + (dynamic_alpha * (formula_normalized_score - previous_score))

    # Low-risk transactions should not trigger sharp trust drops.
    if adjusted_with_history <= 0.15 and new_score < previous_score:
        new_score = max(new_score, previous_score - 0.03)

    new_score = _clamp(new_score)

    wallet = state["wallet"]
    current_dt = datetime.fromtimestamp(now_ts)

    if wallet.first_seen is None:
        wallet.first_seen = current_dt

    wallet.last_seen = current_dt
    wallet.trust_score = new_score
    wallet.avg_risk = rolling_avg_risk

    try:
        wallet.age_days = max((wallet.last_seen - wallet.first_seen).days, 0)
    except Exception:
        wallet.age_days = wallet.age_days

    retention_window = max(int(dormant_gap_seconds), int(burst_window_seconds) * 5)
    recent_pruned = [
        int(ts)
        for ts in recent_tx_timestamps
        if 0 <= (now_ts - int(ts)) <= retention_window
    ][-MAX_RECENT_TIMESTAMPS:]

    wallet.metadata_json = {
        **state["metadata"],
        "recent_tx_timestamps": recent_pruned,
        "tx_count_last_1_min": int(burst_meta["count"]),
        "tx_count_total": int(state["tx_count_total"]) + 1,
        "rolling_avg_risk": round(float(rolling_avg_risk), 6),
        "last_decay": round(float(decay), 6),
        "last_time_gap_seconds": int(time_gap_seconds),
        "last_adjusted_risk": round(float(adjusted_with_history), 6),
        "last_formula_raw_score": round(float(formula_raw_score), 6),
        "last_formula_normalized_score": round(float(formula_normalized_score), 6),
        "last_update_alpha": round(float(dynamic_alpha), 6),
        "burst_detected": bool(burst_meta["is_burst"]),
        "dormant_spike": bool(dormant_spike),
    }

    session.add(wallet)
    session.flush()

    return {
        "temporal_score_normalized": round(float(new_score), 6),
        "temporal_score": round(float(new_score) * 10.0, 3),
        "temporal_previous_score": round(float(previous_score), 6),
        "decay": round(float(decay), 6),
        "time_gap_seconds": int(time_gap_seconds),
        "burst_detected": bool(burst_meta["is_burst"]),
        "tx_count_last_1_min": int(burst_meta["count"]),
        "dormant_spike": bool(dormant_spike),
        "rolling_avg_risk": round(float(rolling_avg_risk), 6),
        "adjusted_risk_probability": round(float(adjusted_with_history), 6),
        "formula_raw_score": round(float(formula_raw_score), 6),
        "formula_normalized_score": round(float(formula_normalized_score), 6),
        "update_alpha": round(float(dynamic_alpha), 6),
    }
