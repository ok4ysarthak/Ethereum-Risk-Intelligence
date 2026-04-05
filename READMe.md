# DeTrust Protocol

DeTrust is a live Ethereum fraud intelligence system. It scores each transaction (risk) and each wallet (trust), then exposes those signals to a real-time dashboard and graph explorer.

## What This Solves

- Detects suspicious transaction behavior in near real time.
- Tracks wallet trust drift over time instead of using one-time static labels.
- Propagates risk across transfer relationships to expose connected threats.
- Persists enriched transaction records for analyst review.

## Active Product Stack

- Backend API and realtime stream: Flask + Socket.IO in [backend/app.py](backend/app.py)
- Live oracle processor and on-chain writer: [backend/oracle.py](backend/oracle.py)
- Graph intelligence engine: [backend/graph/graph_builder.py](backend/graph/graph_builder.py), [backend/graph/risk_engine.py](backend/graph/risk_engine.py), [backend/graph/graph_api.py](backend/graph/graph_api.py)
- Frontend currently used in product: static pages in [frontend/static](frontend/static)

## Core Routes

- Dashboard: `GET /`
- Live feed: `GET /live`
- Shield/analyst view: `GET /shield`
- Graph console: `GET /graph`
- Health: `GET /health`
- Predict transaction risk: `POST /predict/transaction`
- Read transactions: `GET /transactions`
- Ingest enriched transaction from oracle: `POST /transactions`
- Graph subgraph: `GET /graph/wallet/<address>`
- Graph trace: `GET /graph/trace/<address>`
- Graph path: `GET /graph/path?from=<wallet>&to=<wallet>`
- High risk wallets: `GET /graph/high-risk`

## Local Run (Business Demo Path)

### 1) Backend API

From repository root:

```bash
cd backend
python app.py
```

Expected: API on `http://127.0.0.1:5000`

### 2) Open UI

Use browser directly:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/live`
- `http://127.0.0.1:5000/graph`

### 3) Start Oracle (optional for live chain ingestion)

In a second terminal:

```bash
cd backend
python oracle.py
```

The oracle watches Sepolia blocks, computes features, requests model prediction, writes on-chain updates, and posts enriched payloads back to backend.

## Example Prediction Request

```json
{
	"wallet_address": "0x1111111111111111111111111111111111111111",
	"timestamp": 1712300000,
	"update_state": true,
	"features": {
		"Transaction_Value": 2.1,
		"Transaction_Fees": 0.003,
		"Number_of_Inputs": 1,
		"Number_of_Outputs": 2,
		"Gas_Price": 45,
		"Wallet_Age_Days": 14,
		"Wallet_Balance": 3.5,
		"Transaction_Velocity": 8,
		"Exchange_Rate": 3100,
		"Final_Balance": 3.3,
		"BMax_BMin_per_NT": 0.22
	}
}
```

## Graph Intelligence Model

Pipeline:

`New transaction -> DB write -> graph update -> risk propagation -> graph APIs -> graph dashboard`

Graph types:

- Wallet nodes: `wallet:<address>`
- Transaction nodes: `tx:<hash>`
- Edge semantics: `INITIATED`, `RECEIVED`, `TRANSFER`, `FUNDING`, `RELATED`

## Environment Variables Used by Runtime

Backend/oracle require values similar to:

- `SEPOLIA_RPC_URL`
- `ETHERSCAN_API_KEY`
- `CONTRACT_ADDRESS`
- `ORACLE_PRIVATE_KEY`
- `SEPOLIA_PRIVATE_KEY`
- `DATABASE_URL`
- `GOOGLE_API_KEY` (optional for AI explanation endpoint)

Optional graph knobs:

- `GRAPH_BOOTSTRAP_LIMIT`
- `GRAPH_PROPAGATION_DEPTH`
- `GRAPH_ALPHA`
- `GRAPH_HOP_DECAY`
- `GRAPH_FUNDING_THRESHOLD`
- `GRAPH_RELATED_THRESHOLD`

## Business Demo Checklist

1. Start backend and open `/health`.
2. Open `/live` and verify incoming rows update in real time.
3. Open `/graph` and load a wallet subgraph.
4. Trigger one `POST /predict/transaction` and verify:
	 - `risk_probability`, `risk_score`
	 - `temporal_score_normalized`
	 - `shap_explanation`
5. Show `/graph/high-risk` output to demonstrate network-level threat surfacing.

## Current Scope Notes

- Product UI in active use is static HTML under [frontend/static](frontend/static).
- Vite React scaffolding exists in [frontend/vite-project](frontend/vite-project) but is not the primary product surface today.
