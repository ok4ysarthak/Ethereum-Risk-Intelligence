import json
import time
from datetime import datetime
from uuid import uuid4

import requests

BASE = "http://127.0.0.1:5000"

A = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
B = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
C = "0xcccccccccccccccccccccccccccccccccccccccc"
D = "0xdddddddddddddddddddddddddddddddddddddddd"

session = requests.Session()

# 1) Verify predict path is functional with model + SHAP + temporal
predict_payload = {
    "wallet_address": A,
    "timestamp": int(time.time()),
    "update_state": True,
    "features": {
        "Transaction_Value": 12.5,
        "Transaction_Fees": 0.003,
        "Number_of_Inputs": 4,
        "Number_of_Outputs": 12,
        "Gas_Price": 75,
        "Wallet_Age_Days": 5,
        "Wallet_Balance": 1.2,
        "Transaction_Velocity": 22,
        "Exchange_Rate": 3200,
        "Final_Balance": 1.18,
        "BMax_BMin_per_NT": 0.72,
        "from_address": A,
        "to_address": B,
    },
}

pred = session.post(f"{BASE}/predict/transaction", json=predict_payload, timeout=20)
print("predict_status", pred.status_code)
pred_js = pred.json() if pred.headers.get("content-type", "").startswith("application/json") else {"raw": pred.text}
print("predict_keys", sorted(list(pred_js.keys()))[:12])
print("predict_risk", pred_js.get("fraud_probability"), pred_js.get("risk_score"))
print("predict_temporal_keys", sorted(list((pred_js.get("temporal_state") or {}).keys())))
print("predict_has_shap", isinstance(pred_js.get("shap_explanation"), dict))

if pred.status_code != 200:
    raise SystemExit("predict_failed")

# 2) Inject a transaction sequence to trigger graph behaviors
seq = [
    # A -> B high risk
    {"from": A, "to": B, "fraud_probability": 0.91, "risk_score": 10, "value": 8.0},
    # B -> C moderate risk
    {"from": B, "to": C, "fraud_probability": 0.62, "risk_score": 7, "value": 4.0},
    # C -> A creates cycle A->B->C->A
    {"from": C, "to": A, "fraud_probability": 0.58, "risk_score": 6, "value": 2.0},
    # Repeated A -> B to trigger funding threshold
    {"from": A, "to": B, "fraud_probability": 0.88, "risk_score": 9, "value": 3.0},
    {"from": A, "to": B, "fraud_probability": 0.85, "risk_score": 9, "value": 2.5},
    # Additional edges for propagation spread
    {"from": B, "to": D, "fraud_probability": 0.44, "risk_score": 5, "value": 1.1},
    {"from": D, "to": C, "fraud_probability": 0.22, "risk_score": 3, "value": 0.9},
]

ingest_results = []
for i, tx in enumerate(seq, 1):
    txh = "0x" + uuid4().hex + uuid4().hex
    payload = {
        "transaction_hash": txh,
        "from_address": tx["from"],
        "to_address": tx["to"],
        "wallet_address": tx["from"],
        "fraud_probability": tx["fraud_probability"],
        "risk_probability": tx["fraud_probability"],
        "risk_score": tx["risk_score"],
        "wallet_trust_score": max(0.0, 1.0 - tx["fraud_probability"]),
        "temporal_score_normalized": max(0.0, 1.0 - tx["fraud_probability"]),
        "temporal_score": round((1.0 - tx["fraud_probability"]) * 10.0, 3),
        "timestamp": int(time.time()) + i,
        "value": tx["value"],
        "Transaction_Value": tx["value"],
        "raw_details": {
            "risk_score": tx["risk_score"],
            "fraud_probability": tx["fraud_probability"],
            "shap_explanation": pred_js.get("shap_explanation"),
            "rule_explanation": pred_js.get("explanation"),
        },
        "shap_explanation": pred_js.get("shap_explanation"),
        "rule_explanation": pred_js.get("explanation"),
        "onchain_tx_update": "0x" + uuid4().hex + uuid4().hex,
    }
    r = session.post(f"{BASE}/transactions", json=payload, timeout=20)
    js = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    ingest_results.append({"status": r.status_code, "graph": js.get("graph")})

print("ingest_statuses", [x["status"] for x in ingest_results])
print("ingest_graph_flags", [bool((x.get("graph") or {}).get("updated")) for x in ingest_results])
print("ingest_affected_wallets", [int((x.get("graph") or {}).get("affected_wallets", 0)) for x in ingest_results])

# 3) Query graph endpoints
wallet_graph = session.get(f"{BASE}/graph/wallet/{A}?depth=3&min_risk=0.0&include_trace=1", timeout=20)
trace = session.get(f"{BASE}/graph/trace/{A}?depth=3", timeout=20)
path = session.get(f"{BASE}/graph/path?from={A}&to={C}&max_depth=6&directed=1", timeout=20)
high = session.get(f"{BASE}/graph/high-risk?limit=20&min_score=0.4", timeout=20)

wg = wallet_graph.json()
tr = trace.json()
pa = path.json() if path.status_code == 200 else {}
hi = high.json()

print("wallet_graph_status", wallet_graph.status_code, "nodes", len(wg.get("nodes", [])), "edges", len(wg.get("edges", [])))
print("wallet_graph_metrics", wg.get("metrics", {}))
print("wallet_alert_types", sorted({a.get("type") for a in wg.get("alerts", []) if isinstance(a, dict)}))
print("trace_status", trace.status_code, "propagation_paths", len(tr.get("propagation", [])))
print("path_status", path.status_code, "path", pa.get("path"))
print("high_status", high.status_code, "count", hi.get("count"), "top3", [n.get("address") for n in hi.get("nodes", [])[:3]])

# 4) UI accessibility checks
ui = session.get(f"{BASE}/graph", timeout=20)
index_page = session.get(f"{BASE}/", timeout=20)
print("ui_graph_status", ui.status_code, "contains_title", "Graph Intelligence - DeTrust ETH" in ui.text)
print("ui_graph_controls", all(token in ui.text for token in ["wallet-search", "btn-load", "btn-trace", "btn-path"]))
print("index_has_graph_nav", "Graph Intelligence" in index_page.text)

# Persist machine-readable quick result
summary = {
    "predict_status": pred.status_code,
    "predict_has_shap": isinstance(pred_js.get("shap_explanation"), dict),
    "ingest_statuses": [x["status"] for x in ingest_results],
    "ingest_graph_flags": [bool((x.get("graph") or {}).get("updated")) for x in ingest_results],
    "wallet_graph_status": wallet_graph.status_code,
    "wallet_nodes": len(wg.get("nodes", [])),
    "wallet_edges": len(wg.get("edges", [])),
    "wallet_alert_types": sorted({a.get("type") for a in wg.get("alerts", []) if isinstance(a, dict)}),
    "trace_status": trace.status_code,
    "trace_paths": len(tr.get("propagation", [])),
    "path_status": path.status_code,
    "path": pa.get("path"),
    "high_risk_status": high.status_code,
    "high_risk_count": hi.get("count"),
    "ui_graph_status": ui.status_code,
    "ui_contains_title": "Graph Intelligence - DeTrust ETH" in ui.text,
}

out_path = "e:/testing/backend/tests/graph_e2e_result_20260404.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("saved", out_path)
