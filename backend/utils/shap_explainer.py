import logging
import importlib
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    shap = importlib.import_module("shap")
except Exception:  # pragma: no cover - runtime dependency guard
    shap = None

logger = logging.getLogger(__name__)

FEATURE_MEANINGS = {
    "Transaction_Velocity": "High transaction frequency",
    "Wallet_Age_Days": "New wallet",
    "Transaction_Value": "Large transaction amount",
    "Wallet_Balance": "Low remaining balance",
    "Transaction_Fees": "Unusual transaction fee pattern",
    "Gas_Price": "Aggressive gas bidding",
    "Number_of_Inputs": "Complex transaction inputs",
    "Number_of_Outputs": "Mass-distribution output pattern",
    "Final_Balance": "Post-transaction balance change",
    "BMax_BMin_per_NT": "Balance volatility pattern",
}

_explainer = None
_explainer_lock = threading.Lock()
_feature_order: List[str] = []


def load_explainer(model: Any, feature_order: Optional[List[str]] = None):
    """Initialize a singleton SHAP TreeExplainer for the loaded model."""
    global _explainer, _feature_order

    if shap is None:
        raise RuntimeError("shap package is not installed")

    if model is None:
        raise ValueError("A trained tree model is required to initialize SHAP")

    with _explainer_lock:
        if _explainer is not None:
            return _explainer

        _explainer = shap.TreeExplainer(model)

        if feature_order:
            _feature_order = list(feature_order)
        elif hasattr(model, "feature_names_in_"):
            _feature_order = list(model.feature_names_in_)

        logger.info("SHAP explainer initialized with %d features", len(_feature_order))

    return _explainer


def _get_feature_order(features: Dict[str, Any], feature_order: Optional[List[str]] = None) -> List[str]:
    if feature_order:
        return list(feature_order)
    if _feature_order:
        return list(_feature_order)
    return list(features.keys())


def compute_shap_values(
    tx_features: Dict[str, Any],
    feature_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute SHAP values for a single transaction feature dictionary."""
    if _explainer is None:
        raise RuntimeError("SHAP explainer has not been initialized")

    ordered_features = _get_feature_order(tx_features, feature_order)
    row = {name: float(tx_features.get(name, 0.0) or 0.0) for name in ordered_features}
    X = pd.DataFrame([row], columns=ordered_features)

    shap_values = None
    try:
        shap_values = _explainer.shap_values(X)
    except Exception:
        raw_values = _explainer(X)
        shap_values = raw_values.values

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 2:
        shap_row = shap_array[0]
    elif shap_array.ndim == 1:
        shap_row = shap_array
    else:
        shap_row = shap_array.reshape(-1)

    expected_value = _explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = float(np.asarray(expected_value).reshape(-1)[-1])
    else:
        expected_value = float(expected_value)

    contributions = {
        ordered_features[idx]: float(shap_row[idx])
        for idx in range(min(len(ordered_features), len(shap_row)))
    }

    return {
        "base_value": expected_value,
        "contributions": contributions,
    }


def _build_summary(positive_drivers: List[str], negative_drivers: List[str]) -> str:
    if positive_drivers and negative_drivers:
        return (
            f"Risk is primarily increased by {positive_drivers[0]}"
            + (f" and {positive_drivers[1]}" if len(positive_drivers) > 1 else "")
            + f", while {negative_drivers[0]} partially offsets the risk."
        )

    if positive_drivers:
        return (
            f"The score is driven upward mainly by {positive_drivers[0]}"
            + (f" and {positive_drivers[1]}." if len(positive_drivers) > 1 else ".")
        )

    if negative_drivers:
        return (
            f"Risk remains contained because {negative_drivers[0]}"
            + (f" and {negative_drivers[1]} reduce the model output." if len(negative_drivers) > 1 else " reduces the model output.")
        )

    return "No strong SHAP drivers were identified for this prediction."


def format_explanation(
    shap_payload: Dict[str, Any],
    top_k: int = 5,
    feature_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convert raw SHAP values to API-safe top features and summary text."""
    contributions = shap_payload.get("contributions") or {}
    mapping = feature_mapping or FEATURE_MEANINGS

    sorted_items = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    if not sorted_items:
        return {"top_features": [], "summary": "SHAP explanation unavailable."}

    top_n = min(max(3, top_k), len(sorted_items))
    top_items = sorted_items[:top_n]

    top_features = []
    positive = []
    negative = []

    for feature, impact_value in top_items:
        meaning = mapping.get(feature, feature.replace("_", " "))
        top_features.append(
            {
                "feature": feature,
                "meaning": meaning,
                "impact": f"{impact_value:+.4f}",
                "absolute_impact": round(abs(float(impact_value)), 4),
                "direction": "increases_risk" if impact_value >= 0 else "reduces_risk",
            }
        )

        if impact_value >= 0:
            positive.append(meaning.lower())
        else:
            negative.append(meaning.lower())

    summary = _build_summary(positive, negative)

    return {
        "top_features": top_features,
        "summary": summary,
    }


def explain_prediction(
    tx_features: Dict[str, Any],
    feature_order: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Best-effort wrapper used by API handlers."""
    try:
        shap_payload = compute_shap_values(tx_features, feature_order=feature_order)
        explanation = format_explanation(shap_payload, top_k=top_k)
        explanation["base_value"] = round(float(shap_payload.get("base_value", 0.0)), 6)
        return explanation
    except Exception as exc:
        logger.exception("Failed to compute SHAP explanation: %s", exc)
        return {
            "top_features": [],
            "summary": "SHAP explanation unavailable due to runtime error.",
        }
